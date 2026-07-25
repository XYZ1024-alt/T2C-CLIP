from pathlib import Path
import contextlib
import copy
import io
import tempfile
import unittest
from unittest import mock

import torch

from scripts.train import (
    StageMetadata,
    TrainingJob,
    _build_parser,
    _validate_checkpoint_metadata,
    main,
)
from t2c_reid.evaluation import ReIDMetrics
from t2c_reid.wandb import WandbConfig, WandbTracker

RECORDED_ARGS = None
RECORDED_RNG_STATE = None
RECORDED_EPOCHS: list[int] = []
RECORDED_MODEL_STATE_AT_FIRST_EPOCH: dict = {}
RECORDED_OPTIMIZER_STATE_AT_FIRST_EPOCH: dict = {}


class TrainScriptTest(unittest.TestCase):
    def test_schema2_and_tfc_fingerprint_mismatches_are_rejected(self):
        expected = {
            "schema_version": 3,
            "pid_camera_count_fingerprint": "new-fingerprint",
        }
        with self.assertRaisesRegex(ValueError, "schema_version"):
            _validate_checkpoint_metadata(
                expected,
                {
                    "checkpoint_metadata": {
                        "schema_version": 2,
                        "pid_camera_count_fingerprint": "new-fingerprint",
                    }
                },
            )
        with self.assertRaisesRegex(ValueError, "pid_camera_count_fingerprint"):
            _validate_checkpoint_metadata(
                expected,
                {
                    "checkpoint_metadata": {
                        "schema_version": 3,
                        "pid_camera_count_fingerprint": "old-fingerprint",
                    }
                },
            )

        objective_expected = {
            "schema_version": 3,
            "tfc_weight": 1.0,
            "beta": 0.1,
            "beta_warmup_epochs": 10,
            "stage1_epochs": 20,
            "stage2_first_epoch": 21,
        }
        for key, changed in (
            ("tfc_weight", 0.0),
            ("beta", 0.2),
            ("beta_warmup_epochs", 0),
            ("stage1_epochs", 0),
            ("stage2_first_epoch", 1),
        ):
            actual = dict(objective_expected)
            actual[key] = changed
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    _validate_checkpoint_metadata(
                        objective_expected,
                        {"checkpoint_metadata": actual},
                    )

    def test_seed_arg_is_accepted_and_seeds_torch(self):
        global RECORDED_ARGS, RECORDED_RNG_STATE
        RECORDED_ARGS = None
        RECORDED_RNG_STATE = None
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            main(
                _recording_job_args(checkpoint_dir) + ["--seed", "42"],
                progress_factory=lambda iterable, **kwargs: iterable,
            )

        self.assertEqual(RECORDED_ARGS.seed, 42)
        # main() must seed torch before calling the job builder, so the RNG
        # state observed inside the builder matches a freshly-seeded generator.
        ref = torch.Generator().manual_seed(42).get_state()
        self.assertTrue(torch.equal(RECORDED_RNG_STATE, ref))

    def test_single_stage_resume_rejects_stage1_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            main(
                [
                    "--job-builder", f"{__name__}:build_training_job",
                    "--epochs", "1",
                    "--validation-interval", "1",
                    "--checkpoint-dir", str(checkpoint_dir),
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            payload = torch.load(
                checkpoint_dir / "last.pth",
                map_location="cpu",
                weights_only=True,
            )
            payload["stage"] = "stage1"
            bad_checkpoint = checkpoint_dir / "stage1.pth"
            torch.save(payload, bad_checkpoint)

            with self.assertRaisesRegex(ValueError, "only stage2 checkpoints"):
                main(
                    [
                        "--job-builder", f"{__name__}:build_training_job",
                        "--epochs", "2",
                        "--checkpoint-dir", str(checkpoint_dir),
                        "--resume", str(bad_checkpoint),
                    ],
                    progress_factory=lambda iterable, **kwargs: iterable,
                )

    def test_resume_restarts_from_saved_checkpoint_epoch(self):
        RECORDED_EPOCHS.clear()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            # Run 2 epochs to produce a last.pth
            main(
                [
                    "--job-builder", f"{__name__}:build_training_job",
                    "--epochs", "2",
                    "--validation-interval", "1",
                    "--checkpoint-dir", str(checkpoint_dir),
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )

            exit_code = main(
                [
                    "--job-builder", f"{__name__}:recording_epoch_builder",
                    "--epochs", "4",
                    "--validation-interval", "1",
                    "--checkpoint-dir", str(checkpoint_dir),
                    "--resume", str(checkpoint_dir / "last.pth"),
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(RECORDED_EPOCHS, [3, 4])

    def test_resume_restores_model_and_optimizer_state(self):
        RECORDED_EPOCHS.clear()
        RECORDED_MODEL_STATE_AT_FIRST_EPOCH.clear()
        RECORDED_OPTIMIZER_STATE_AT_FIRST_EPOCH.clear()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            main(
                [
                    "--job-builder", f"{__name__}:build_training_job",
                    "--epochs", "2",
                    "--validation-interval", "1",
                    "--checkpoint-dir", str(checkpoint_dir),
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            saved = torch.load(checkpoint_dir / "last.pth", map_location="cpu", weights_only=True)

            main(
                [
                    "--job-builder", f"{__name__}:recording_epoch_builder",
                    "--epochs", "3",
                    "--validation-interval", "1",
                    "--checkpoint-dir", str(checkpoint_dir),
                    "--resume", str(checkpoint_dir / "last.pth"),
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )

        for key, value in saved["model_state"].items():
            self.assertTrue(torch.equal(RECORDED_MODEL_STATE_AT_FIRST_EPOCH[key], value))
        saved_momentum = saved["optimizer_state"]["state"][0]["momentum_buffer"]
        recorded_momentum = RECORDED_OPTIMIZER_STATE_AT_FIRST_EPOCH["state"][0]["momentum_buffer"]
        self.assertTrue(torch.equal(recorded_momentum, saved_momentum))

    def test_resume_preserves_best_checkpoint_from_previous_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            # declining mAP: epoch 1 scores 0.9 and stays the all-time best.
            main(
                [
                    "--job-builder", f"{__name__}:declining_map_builder",
                    "--epochs", "2",
                    "--validation-interval", "1",
                    "--checkpoint-dir", str(checkpoint_dir),
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )

            main(
                [
                    "--job-builder", f"{__name__}:declining_map_builder",
                    "--epochs", "3",
                    "--validation-interval", "1",
                    "--checkpoint-dir", str(checkpoint_dir),
                    "--resume", str(checkpoint_dir / "last.pth"),
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            best = torch.load(checkpoint_dir / "best.pth", map_location="cpu", weights_only=True)

        self.assertEqual(best["epoch"], 1)
        self.assertEqual(best["metrics"]["mAP"], 0.9)

    def test_parser_defaults_to_the_project_training_recipe(self):
        args = _build_parser().parse_args([])

        self.assertEqual(
            args.job_builder,
            "t2c_reid.jobs.siglip2_reid:build_training_job",
        )
        self.assertEqual(args.dataset, "msmt17")
        self.assertEqual(args.data_root, Path("data/MSMT17_V1"))
        self.assertEqual(args.stage1_epochs, 20)
        self.assertEqual(args.epochs, 120)
        self.assertEqual(
            args.checkpoint_dir,
            Path("checkpoints/msmt17-siglip2-tfc"),
        )
        self.assertEqual(args.run_name, "msmt17-siglip2-camera-tfc")

    def test_main_runs_builder_training_job_and_saves_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            exit_code = main(
                [
                    "--job-builder",
                    f"{__name__}:build_training_job",
                    "--epochs",
                    "2",
                    "--validation-interval",
                    "1",
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            best_payload = torch.load(checkpoint_dir / "best.pth", map_location="cpu", weights_only=True)
            last_payload = torch.load(checkpoint_dir / "last.pth", map_location="cpu", weights_only=True)

        self.assertEqual(exit_code, 0)
        self.assertEqual(best_payload["epoch"], 2)
        self.assertEqual(last_payload["epoch"], 2)
        self.assertEqual(last_payload["metrics"]["mAP"], 0.2)

    def test_main_passes_project_training_args_to_builder(self):
        global RECORDED_ARGS
        RECORDED_ARGS = None
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            exit_code = main(
                _recording_job_args(checkpoint_dir),
                progress_factory=lambda iterable, **kwargs: iterable,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(RECORDED_ARGS.dataset, "msmt17")
        self.assertEqual(RECORDED_ARGS.data_root, Path("MSMT17_V1"))
        self.assertEqual(
            RECORDED_ARGS.siglip2_model_name,
            "google/siglip2-so400m-patch14-384",
        )
        self.assertEqual(RECORDED_ARGS.batch_size, 8)
        self.assertEqual(RECORDED_ARGS.eval_batch_size, 16)
        self.assertEqual(RECORDED_ARGS.gradient_accumulation_steps, 4)
        self.assertEqual(RECORDED_ARGS.precision, "auto")
        self.assertEqual(RECORDED_ARGS.num_workers, 2)
        self.assertEqual(RECORDED_ARGS.data_backend, "rust")
        self.assertEqual(RECORDED_ARGS.prefetch_factor, 2)
        self.assertIsNone(RECORDED_ARGS.pin_memory)
        self.assertIsNone(RECORDED_ARGS.persistent_workers)
        self.assertEqual(RECORDED_ARGS.rust_data_threads, 1)
        self.assertEqual(RECORDED_ARGS.evaluation_backend, "rust")
        self.assertEqual(RECORDED_ARGS.evaluation_chunk_size, 256)
        self.assertEqual(RECORDED_ARGS.lr, 0.001)
        self.assertEqual(RECORDED_ARGS.triplet_metric, "euclidean")
        self.assertEqual(RECORDED_ARGS.device, "cpu")

    def test_default_hyperparameters_follow_siglip2_24gb_recipe(self):
        global RECORDED_ARGS
        RECORDED_ARGS = None
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            main(
                [
                    "--job-builder", f"{__name__}:recording_training_job",
                    "--epochs", "1",
                    "--validation-interval", "1",
                    "--checkpoint-dir", str(checkpoint_dir),
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )

        self.assertEqual(RECORDED_ARGS.image_encoder_lr, 5e-6)
        self.assertEqual(RECORDED_ARGS.stage1_epochs, 20)
        self.assertEqual(RECORDED_ARGS.dataset, "msmt17")
        self.assertEqual(RECORDED_ARGS.data_root, Path("data/MSMT17_V1"))
        self.assertEqual(RECORDED_ARGS.alignment_weight, 1.0)
        self.assertEqual(RECORDED_ARGS.tfc_tail_momentum, 0.9)
        self.assertEqual(RECORDED_ARGS.tfc_class_balance_beta, 0.9999)
        self.assertEqual(RECORDED_ARGS.tfc_local_weight, 1.0)
        self.assertEqual(RECORDED_ARGS.tfc_global_weight, 1.0)
        self.assertEqual(RECORDED_ARGS.tfc_cross_modal_weight, 0.5)
        self.assertEqual(RECORDED_ARGS.tfc_cross_camera_weight, 0.1)
        self.assertEqual(RECORDED_ARGS.tfc_contrast_temperature, 0.07)
        self.assertEqual(RECORDED_ARGS.tfc_transfer_reg_weight, 0.01)
        self.assertTrue(RECORDED_ARGS.gradient_checkpointing)

    def test_main_logs_validation_metrics_when_wandb_enabled(self):
        run = FakeWandbRun()
        captured: list[tuple[WandbConfig, str]] = []

        @contextlib.contextmanager
        def fake_start_wandb_run(config: WandbConfig, run_name: str):
            captured.append((config, run_name))
            yield WandbTracker(run)

        with tempfile.TemporaryDirectory() as tmp:
            wandb_dir = Path(tmp) / "wandb-data"
            checkpoint_dir = Path(tmp) / "checkpoints"
            with mock.patch("scripts.train.start_wandb_run", new=fake_start_wandb_run):
                exit_code = main(
                    [
                        "--job-builder",
                        f"{__name__}:build_training_job",
                        "--epochs",
                        "1",
                        "--validation-interval",
                        "1",
                        "--checkpoint-dir",
                        str(checkpoint_dir),
                        "--enable-wandb",
                        "--wandb-project",
                        "T2C-ReID-TrainScript-Test",
                        "--wandb-entity",
                        "test-team",
                        "--wandb-mode",
                        "offline",
                        "--wandb-dir",
                        str(wandb_dir),
                        "--run-name",
                        "train-script-test",
                    ],
                    progress_factory=lambda iterable, **kwargs: iterable,
                )

        step_history = [payload for payload in run.logged if "stage2_train_step" in payload]
        epoch_history = [payload for payload in run.logged if "stage2_epoch" in payload]
        validation_history = [payload for payload in run.logged if "validation_epoch" in payload]
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            captured,
            [
                (
                    WandbConfig(
                        project="T2C-ReID-TrainScript-Test",
                        entity="test-team",
                        mode="offline",
                        directory=wandb_dir,
                    ),
                    "train-script-test",
                )
            ],
        )
        self.assertEqual([payload["stage2_train_step"] for payload in step_history], [1, 2])
        self.assertEqual([payload["stage2_train_step_loss"] for payload in step_history], [1.1, 1.2])
        self.assertEqual(epoch_history[0]["stage2_train_loss"], 1.0)
        self.assertEqual(epoch_history[0]["stage2_lr"], 0.1)
        self.assertEqual(validation_history[0]["mAP"], 0.1)
        self.assertEqual(validation_history[0]["rank_1"], 0.1)
        self.assertEqual(run.config["dataset"], "fixture")

    def test_tracking_disabled_does_not_load_wandb(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("t2c_reid.wandb._load_wandb") as load_wandb:
                exit_code = main(
                    [
                        "--job-builder",
                        f"{__name__}:build_training_job",
                        "--epochs",
                        "1",
                        "--validation-interval",
                        "1",
                        "--checkpoint-dir",
                        str(Path(tmp) / "checkpoints"),
                    ],
                    progress_factory=lambda iterable, **kwargs: iterable,
                )

        self.assertEqual(exit_code, 0)
        load_wandb.assert_not_called()

    def test_naflex_model_override_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                main(
                    [
                        "--job-builder",
                        f"{__name__}:build_training_job",
                        "--siglip2-model-name",
                        "google/siglip2-so400m-patch16-naflex",
                    ]
                )
        self.assertNotEqual(context.exception.code, 0)

    def test_removed_clip_model_flag_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                main(
                    [
                        "--job-builder",
                        f"{__name__}:build_training_job",
                        "--clip-model-name",
                        "openai/clip-vit-base-patch16",
                    ]
                )
        self.assertNotEqual(context.exception.code, 0)

    def test_legacy_tracking_flag_is_rejected(self):
        legacy_flag = "--enable-" + "ml" + "flow"
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                main(
                    [
                        "--job-builder",
                        f"{__name__}:build_training_job",
                        legacy_flag,
                    ]
                )

        self.assertNotEqual(context.exception.code, 0)


def build_training_job(args) -> TrainingJob:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    def train_one_epoch(epoch: int, reporter) -> dict[str, float]:
        for batch_number in reporter.batches([1, 2]):
            optimizer.zero_grad()
            loss = model(torch.eye(2)).sum() * (epoch + batch_number / 10.0)
            loss.backward()
            optimizer.step()
            reporter.report_batch({"loss": epoch + batch_number / 10.0, "lr": 0.1})
        return {"loss": float(epoch), "lr": 0.1}

    def validate(epoch: int) -> ReIDMetrics:
        return ReIDMetrics(map=epoch / 10.0, cmc={1: epoch / 10.0})

    return TrainingJob(
        model,
        optimizer,
        train_one_epoch,
        validate,
        stage_metadata=StageMetadata({"dataset": "fixture", "unused": None}),
    )


def recording_training_job(args) -> TrainingJob:
    global RECORDED_ARGS, RECORDED_RNG_STATE
    RECORDED_ARGS = args
    RECORDED_RNG_STATE = torch.get_rng_state()
    return build_training_job(args)


def recording_epoch_builder(args) -> TrainingJob:
    job = build_training_job(args)

    def train_one_epoch(epoch: int, reporter) -> dict[str, float]:
        if not RECORDED_EPOCHS:
            RECORDED_MODEL_STATE_AT_FIRST_EPOCH.update(
                {key: value.clone() for key, value in job.model.state_dict().items()}
            )
            RECORDED_OPTIMIZER_STATE_AT_FIRST_EPOCH.update(copy.deepcopy(job.optimizer.state_dict()))
        RECORDED_EPOCHS.append(epoch)
        return job.train_one_epoch(epoch, reporter)

    return TrainingJob(job.model, job.optimizer, train_one_epoch, job.validate)


def declining_map_builder(args) -> TrainingJob:
    job = build_training_job(args)

    def validate(epoch: int) -> ReIDMetrics:
        return ReIDMetrics(map=1.0 - epoch / 10.0, cmc={1: 0.5})

    return TrainingJob(job.model, job.optimizer, job.train_one_epoch, validate)


def _recording_job_args(checkpoint_dir: Path) -> list[str]:
    return [
        "--job-builder",
        f"{__name__}:recording_training_job",
        "--epochs",
        "1",
        "--validation-interval",
        "1",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--dataset",
        "msmt17",
        "--data-root",
        "MSMT17_V1",
        "--siglip2-model-name",
        "google/siglip2-so400m-patch14-384",
        "--batch-size",
        "8",
        "--num-workers",
        "2",
        "--lr",
        "0.001",
        "--device",
        "cpu",
    ]


class FakeWandbConfig(dict):
    def update(self, values=None, *, allow_val_change: bool = False):
        super().update({} if values is None else values)


class FakeWandbRun:
    def __init__(self):
        self.config = FakeWandbConfig()
        self.logged: list[dict] = []
        self.defined_metrics: list[tuple[str, dict]] = []

    def define_metric(self, name: str, **kwargs) -> None:
        self.defined_metrics.append((name, dict(kwargs)))

    def log(self, payload) -> None:
        self.logged.append(dict(payload))
