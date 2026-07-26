import unittest
from unittest import mock

import torch

from scripts.train import TrainingJob, _restore_job_state
from t2c_reid.loops import _checkpoint_payload
from t2c_reid.precision import PrecisionController, PrecisionPolicy, resolve_precision


class PrecisionResolutionTest(unittest.TestCase):
    def test_auto_cpu_resolves_to_fp32(self):
        policy = resolve_precision("auto", torch.device("cpu"))
        self.assertEqual(policy.resolved, "fp32")

    def test_auto_cuda_prefers_bf16_when_supported(self):
        policy = resolve_precision(
            "auto", torch.device("cuda"), is_bf16_supported=lambda: True
        )
        self.assertEqual(policy.resolved, "bf16")

    def test_auto_cuda_falls_back_to_fp16(self):
        policy = resolve_precision(
            "auto", torch.device("cuda"), is_bf16_supported=lambda: False
        )
        self.assertEqual(policy.resolved, "fp16")

    def test_explicit_unsupported_bf16_fails(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            resolve_precision(
                "bf16", torch.device("cuda"), is_bf16_supported=lambda: False
            )

    def test_low_precision_on_cpu_fails_instead_of_falling_back(self):
        for requested in ("bf16", "fp16"):
            with self.subTest(requested=requested), self.assertRaisesRegex(
                ValueError, "requires a CUDA"
            ):
                resolve_precision(requested, torch.device("cpu"))

    def test_unknown_precision_fails(self):
        with self.assertRaisesRegex(ValueError, "unsupported precision"):
            resolve_precision("tf32", torch.device("cuda"))


class GradientClippingTest(unittest.TestCase):
    """Clipping must work on the scaler-disabled (BF16/FP32) path too."""

    @staticmethod
    def _model_with_grad(scale: float) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        model = torch.nn.Linear(4, 4, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model.weight.grad = torch.full_like(model.weight, scale)
        return model, optimizer

    def test_clipping_rescales_an_oversized_gradient(self):
        controller = PrecisionController(PrecisionPolicy("fp32", "fp32", "cpu"))
        model, optimizer = self._model_with_grad(1.0)
        # 16 entries of 1.0 -> norm 4.0
        total_norm = controller.clip_grad_norm(optimizer, model.parameters(), 1.0)

        self.assertAlmostEqual(float(total_norm), 4.0, places=5)
        self.assertAlmostEqual(float(model.weight.grad.norm()), 1.0, places=5)

    def test_healthy_gradient_is_left_alone(self):
        controller = PrecisionController(PrecisionPolicy("fp32", "fp32", "cpu"))
        model, optimizer = self._model_with_grad(0.1)
        before = model.weight.grad.clone()
        controller.clip_grad_norm(optimizer, model.parameters(), 5.0)

        self.assertTrue(torch.allclose(model.weight.grad, before))

    def test_non_positive_max_norm_disables_clipping(self):
        controller = PrecisionController(PrecisionPolicy("fp32", "fp32", "cpu"))
        model, optimizer = self._model_with_grad(1.0)
        before = model.weight.grad.clone()

        self.assertIsNone(controller.clip_grad_norm(optimizer, model.parameters(), 0.0))
        self.assertTrue(torch.allclose(model.weight.grad, before))

    def test_fp16_path_unscales_before_clipping(self):
        with mock.patch("t2c_reid.precision.torch.amp.GradScaler", _FakeGradScaler):
            controller = PrecisionController(PrecisionPolicy("fp16", "fp16", "cuda"))
            model, optimizer = self._model_with_grad(1.0)
            controller.clip_grad_norm(optimizer, model.parameters(), 1.0)

        self.assertEqual(controller.scaler.unscaled, [optimizer])


class PrecisionCheckpointTest(unittest.TestCase):
    def test_fp16_scaler_state_round_trips_through_training_checkpoint(self):
        with mock.patch("t2c_reid.precision.torch.amp.GradScaler", _FakeGradScaler):
            source = PrecisionController(PrecisionPolicy("fp16", "fp16", "cuda"))
            source.scaler.scale_value = 2048.0
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            metadata = {
                "schema_version": 2,
                "backbone_family": "siglip2",
                "model_id": "google/siglip2-so400m-patch14-384",
            }
            payload = _checkpoint_payload(
                model,
                optimizer,
                epoch=3,
                metrics=None,
                best_map=0.4,
                stage="stage2",
                checkpoint_metadata=metadata,
                auxiliary_state=source,
            )

            restored = PrecisionController(PrecisionPolicy("fp16", "fp16", "cuda"))
            job = TrainingJob(
                model=model,
                optimizer=optimizer,
                train_one_epoch=lambda epoch, reporter: {},
                validate=lambda epoch: None,
                checkpoint_metadata=metadata,
                auxiliary_state=restored,
            )
            _restore_job_state(job, payload)

        self.assertEqual(restored.scaler.scale_value, 2048.0)
        self.assertEqual(payload["checkpoint_metadata"], metadata)

    def test_siglip_job_rejects_checkpoint_without_schema_metadata(self):
        model = torch.nn.Linear(1, 1)
        job = TrainingJob(
            model=model,
            optimizer=None,
            train_one_epoch=lambda epoch, reporter: {},
            validate=lambda epoch: None,
            checkpoint_metadata={"schema_version": 2, "backbone_family": "siglip2"},
            auxiliary_state=_DictState(),
        )
        legacy = {"model_state": model.state_dict()}

        with self.assertRaisesRegex(ValueError, "old T2C-CLIP"):
            _restore_job_state(job, legacy)

    def test_siglip_job_rejects_mismatched_model_metadata_before_weights(self):
        model = torch.nn.Linear(1, 1)
        job = TrainingJob(
            model=model,
            optimizer=None,
            train_one_epoch=lambda epoch, reporter: {},
            validate=lambda epoch: None,
            checkpoint_metadata={"schema_version": 2, "model_id": "expected"},
        )
        state = {
            "checkpoint_metadata": {"schema_version": 2, "model_id": "other"},
            "model_state": model.state_dict(),
        }

        with self.assertRaisesRegex(ValueError, "model_id"):
            _restore_job_state(job, state)


class _FakeGradScaler:
    def __init__(self, device="cuda", enabled=True):
        self.device = device
        self.enabled = enabled
        self.scale_value = 65536.0
        self.unscaled: list[object] = []

    def is_enabled(self):
        return self.enabled

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        if not self.enabled:
            return
        self.unscaled.append(optimizer)

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def get_scale(self):
        return self.scale_value

    def state_dict(self):
        return {"scale": self.scale_value} if self.enabled else {}

    def load_state_dict(self, state):
        if state:
            self.scale_value = float(state["scale"])


class _DictState:
    def __init__(self):
        self.value = {}

    def state_dict(self):
        return dict(self.value)

    def load_state_dict(self, state):
        self.value = dict(state)


if __name__ == "__main__":
    unittest.main()
