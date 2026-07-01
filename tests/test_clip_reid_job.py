from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from t2c_clip.datasets import ReIDImageBatch
from t2c_clip.jobs.clip_reid import (
    CLIPLoadResult,
    CLIPReIDTrainingModel,
    JobDataConfig,
    _extract_features,
    BetaSchedule,
    StageLRScheduler,
    build_training_job,
    load_dataset_bundle,
)
from t2c_clip.retrieval import IMAGE_ONLY_RETRIEVAL
from tests._clip_fakes import FakeCLIP, ImageAwareFakeImageProcessor


class CLIPReIDJobTest(unittest.TestCase):
    def test_load_dataset_bundle_rejects_missing_root(self):
        config = JobDataConfig("market1501", Path("missing"))

        with self.assertRaises(FileNotFoundError):
            load_dataset_bundle(config, ImageAwareFakeImageProcessor())

    def test_build_training_job_returns_real_callbacks_with_fake_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            job = build_training_job(_training_args(root), clip_loader=_load_fake_clip)
            reporter = TrainBatchReporterRecorder()

            train_metrics = job.train_one_epoch(1, reporter)
            metrics = job.validate(1)

        self.assertEqual(len(reporter.batch_reports), 1)
        self.assertIn("loss", train_metrics)
        self.assertIn("clip_loss", train_metrics)
        self.assertIn("reid_loss", train_metrics)
        self.assertIn("triplet_loss", train_metrics)
        self.assertIn("tfc_loss", train_metrics)
        self.assertIn("lr", train_metrics)
        self.assertGreaterEqual(metrics.map, 0.0)
        self.assertIn(1, metrics.cmc)

    def test_build_training_job_returns_two_stage_job_when_stage1_epochs_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.stage1_epochs = 1
            job = build_training_job(args, clip_loader=_load_fake_clip)

        from scripts.train import TwoStageTrainingJob
        self.assertIsInstance(job, TwoStageTrainingJob)

    def test_build_training_job_rejects_training_split_without_positive_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture_without_positive_pairs(Path(tmp))

            with self.assertRaises(ValueError):
                build_training_job(_training_args(root), clip_loader=_load_fake_clip)

    def test_extract_features_passes_configured_retrieval_mode(self):
        model = RetrievalModeRecorder()
        batch = ReIDImageBatch(
            images=torch.ones(2, 3, 2, 2),
            person_ids=torch.tensor([0, 1]),
            camera_ids=torch.tensor([1, 2]),
            original_person_ids=(10, 20),
            original_camera_ids=(1, 2),
        )

        features = _extract_features(
            model,
            [batch],
            torch.device("cpu"),
            retrieval_mode=IMAGE_ONLY_RETRIEVAL,
        )

        self.assertEqual(model.retrieval_modes, [IMAGE_ONLY_RETRIEVAL])
        self.assertEqual(features.person_ids, (10, 20))
        self.assertEqual(features.camera_ids, (1, 2))
        self.assertEqual(tuple(features.features.shape), (2, 4))

    def test_validation_reports_rerank_metrics_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.report_rerank = True

            job = build_training_job(args, clip_loader=_load_fake_clip)
            metrics = job.validate(1)

        self.assertIn(1, metrics.cmc)
        self.assertIn("rerank_mAP", metrics.extras)
        self.assertIn("rerank_rank_1", metrics.extras)

    def test_local_clip_checkpoint_is_loaded_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            checkpoint = Path(tmp) / "clip_state.pth"
            model = FakeCLIP(hidden_size=8, projection_dim=4)
            state = model.state_dict()
            key = "visual_projection.weight"
            state[key] = torch.full_like(state[key], 0.25)
            torch.save(state, checkpoint)
            args = _training_args(root)
            args.clip_checkpoint = checkpoint

            job = build_training_job(args, clip_loader=_load_fake_clip)

        loaded = job.model.retrieval_model.image_encoder.clip_model.visual_projection.weight
        self.assertTrue(torch.allclose(loaded, torch.full_like(loaded, 0.25)))

    def test_missing_clip_checkpoint_fails_at_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.clip_checkpoint = Path(tmp) / "missing.pth"

            with self.assertRaises(FileNotFoundError):
                build_training_job(args, clip_loader=_load_fake_clip)

    def test_clip_checkpoint_with_unexpected_keys_fails_at_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            checkpoint = Path(tmp) / "bad_clip_state.pth"
            torch.save({"not_a_clip_weight": torch.ones(1)}, checkpoint)
            args = _training_args(root)
            args.clip_checkpoint = checkpoint

            with self.assertRaisesRegex(ValueError, "unexpected CLIP checkpoint keys"):
                build_training_job(args, clip_loader=_load_fake_clip)

    def test_no_freeze_image_encoder_stage2_reenables_encoder_after_stage1_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.stage1_epochs = 1
            args.freeze_image_encoder_stage1 = True
            args.freeze_image_encoder_stage2 = False

            job = build_training_job(args, clip_loader=_load_fake_clip)

        clip_model = job.stage2.model.retrieval_model.image_encoder.clip_model
        self.assertGreater(_trainable_parameter_count(clip_model.visual_projection), 0)

    def test_stage1_training_reapplies_stage1_freezing_before_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.stage1_epochs = 1
            args.freeze_image_encoder_stage1 = True
            args.freeze_image_encoder_stage2 = False
            job = build_training_job(args, clip_loader=_load_fake_clip)
            reporter = TrainBatchReporterRecorder()

            job.stage1.train_one_epoch(1, reporter)

        clip_model = job.stage1.model.retrieval_model.image_encoder.clip_model
        self.assertEqual(_trainable_parameter_count(clip_model.visual_projection), 0)

    def test_no_freeze_image_encoder_stage2_works_without_stage1_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.stage1_epochs = 0
            args.freeze_image_encoder_stage1 = True
            args.freeze_image_encoder_stage2 = False

            job = build_training_job(args, clip_loader=_load_fake_clip)

        clip_model = job.model.retrieval_model.image_encoder.clip_model
        self.assertGreater(_trainable_parameter_count(clip_model.visual_projection), 0)

    def test_stage2_can_freeze_prompt_bank_after_stage1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.stage1_epochs = 1
            args.freeze_prompt_bank_stage2 = True

            job = build_training_job(args, clip_loader=_load_fake_clip)
            job.stage2.train_one_epoch(2, TrainBatchReporterRecorder())

        prompt_bank = job.stage2.model.retrieval_model.prompt_bank
        self.assertEqual(_trainable_parameter_count(prompt_bank), 0)

    def test_stage2_prompt_bank_remains_trainable_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)

            job = build_training_job(args, clip_loader=_load_fake_clip)

        prompt_bank = job.model.retrieval_model.prompt_bank
        self.assertGreater(_trainable_parameter_count(prompt_bank), 0)

    def test_bnneck_adds_trainable_batch_norm_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.reid_head = "bnneck"

            job = build_training_job(args, clip_loader=_load_fake_clip)

        self.assertTrue(hasattr(job.model, "feature_head"))
        self.assertGreater(_trainable_parameter_count(job.model.feature_head), 0)

    def test_bnneck_keeps_batch_norm_bias_frozen_in_stage2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.reid_head = "bnneck"

            job = build_training_job(args, clip_loader=_load_fake_clip)

        self.assertFalse(job.model.feature_head.bn.bias.requires_grad)

    def test_clipreid_model_encode_retrieval_applies_feature_head(self):
        # Retrieval/validation must pass the base feature through the same
        # feature_head (e.g. BNNeck) the Stage-2 ID classifier is trained on.
        # Otherwise the ID signal shapes BN(f) while retrieval uses raw f.
        class StubRetrieval(torch.nn.Module):
            def encode_retrieval(self, images, camera_ids, retrieval_mode="fused"):
                return torch.ones(images.shape[0], 4)

        head = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.eye(4) * 2.0)
        model = CLIPReIDTrainingModel(
            retrieval_model=StubRetrieval(),
            classifier=torch.nn.Linear(4, 2),
            tfc_bank=torch.nn.Module(),
            feature_head=head,
        )

        output = model.encode_retrieval(torch.zeros(3, 3, 2, 2), torch.zeros(3, dtype=torch.long))

        self.assertTrue(torch.allclose(output, torch.full((3, 4), 2.0)))

    def test_validation_extracts_features_through_bnneck_head(self):
        # The built validation path must route retrieval through the BNNeck head,
        # so extracted features differ from the raw pre-head retrieval feature.
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.reid_head = "bnneck"

            job = build_training_job(args, clip_loader=_load_fake_clip)

        head = job.model.feature_head
        with torch.no_grad():
            head.bn.weight.copy_(torch.full_like(head.bn.weight, 3.0))
            head.bn.running_var.copy_(torch.full_like(head.bn.running_var, 4.0))
        job.model.eval()

        batch = ReIDImageBatch(
            images=torch.ones(2, 3, 2, 2),
            person_ids=torch.tensor([0, 1]),
            camera_ids=torch.tensor([0, 0]),
            original_person_ids=(10, 20),
            original_camera_ids=(1, 1),
        )
        device = torch.device("cpu")

        with_head = _extract_features(job.model, [batch], device, "fused")
        without_head = _extract_features(job.model.retrieval_model, [batch], device, "fused")

        self.assertFalse(torch.allclose(with_head.features, without_head.features))

    def test_tfc_center_update_uses_feature_head_output(self):
        from t2c_clip.jobs.clip_reid import _update_tfc_centers
        from t2c_clip.tfc import TFCCenterBank
        from t2c_clip.training import TrainingBatch

        class StubRetrieval(torch.nn.Module):
            def forward_stage2(self, images, camera_ids, person_ids):
                return {"retrieval": images}

        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        model = CLIPReIDTrainingModel(
            retrieval_model=StubRetrieval(),
            classifier=torch.nn.Linear(2, 2),
            tfc_bank=TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5),
            feature_head=head,
        )
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.0, 0.8]]),
            camera_ids=torch.tensor([0, 0, 1, 1]),
            person_ids=torch.tensor([0, 0, 1, 1]),
        )

        _update_tfc_centers(model, batch)

        self.assertTrue(torch.allclose(model.tfc_bank.centers[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(model.tfc_bank.centers[1], torch.tensor([0.0, 0.0])))

    def test_default_args_freeze_image_encoder_stage2_is_false_when_attr_absent(self):
        # The job config must default Stage-2 image encoder to UNFROZEN (matching
        # CLIP-ReID's standard Stage-2 recipe) when the caller passes no explicit flag.
        args = Namespace(
            dataset="market1501",
            data_root=Path("."),
            clip_model_name="fake-clip",
            batch_size=4,
            num_workers=0,
            lr=0.001,
            device="cpu",
            beta=0.1,
            context_length=2,
            tfc_momentum=0.5,
            triplet_margin=0.3,
            tfc_weight=1.0,
            clip_weight=0.1,
            label_smoothing=0.0,
            stage1_epochs=0,
            epochs=1,
            validation_interval=1,
            freeze_image_encoder_stage1=True,
            # freeze_image_encoder_stage2 deliberately absent
            freeze_text_encoder=True,
            freeze_prompt_bank_stage2=False,
            reid_head="linear",
            retrieval_mode="fused",
        )
        from t2c_clip.jobs.clip_reid import _job_config_from_args

        config = _job_config_from_args(args)
        self.assertFalse(config.freeze_image_encoder_stage2)

    def test_image_encoder_lr_creates_separate_param_group_for_unfrozen_stage2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.stage1_epochs = 0
            args.freeze_image_encoder_stage1 = True
            args.freeze_image_encoder_stage2 = False  # default unfrozen
            args.lr = 0.001
            args.image_encoder_lr = 5e-5

            job = build_training_job(args, clip_loader=_load_fake_clip)

        optimizer = job.optimizer
        backbone_group, new_group = _lookup_param_groups(optimizer)
        self.assertAlmostEqual(backbone_group["lr"], 5e-5)
        self.assertAlmostEqual(new_group["lr"], 0.001)
        # Filtered by name: backbone group should hold only visual_projection params.
        backbone_names = [_element_name(model=job.model, parameter=parameter) for parameter in backbone_group["params"]]
        self.assertTrue(
            all("visual_projection" in name or "vision_model" in name for name in backbone_names),
            f"backbone group contains non-backbone params: {backbone_names}",
        )

    def test_beta_schedule_ramps_from_zero_to_beta_over_warmup(self):
        schedule = BetaSchedule(beta=0.1, warmup_epochs=5)
        self.assertAlmostEqual(schedule.effective_beta(1), 0.0)
        self.assertAlmostEqual(schedule.effective_beta(2), 0.02)
        self.assertAlmostEqual(schedule.effective_beta(5), 0.08)
        self.assertAlmostEqual(schedule.effective_beta(6), 0.1)
        self.assertAlmostEqual(schedule.effective_beta(120), 0.1)

    def test_beta_schedule_uses_stage_local_epoch_offset(self):
        schedule = BetaSchedule(beta=0.1, warmup_epochs=5)

        self.assertAlmostEqual(schedule.effective_beta(stage_epoch=1), 0.0)
        self.assertAlmostEqual(schedule.effective_beta(stage_epoch=2), 0.02)
        self.assertAlmostEqual(schedule.effective_beta(stage_epoch=5), 0.08)
        self.assertAlmostEqual(schedule.effective_beta(stage_epoch=6), 0.1)

    def test_beta_schedule_zero_warmup_returns_constant_beta(self):
        schedule = BetaSchedule(beta=0.1, warmup_epochs=0)
        self.assertAlmostEqual(schedule.effective_beta(1), 0.1)
        self.assertAlmostEqual(schedule.effective_beta(10), 0.1)

    def test_beta_schedule_applies_to_model_retrieval_beta(self):
        class CLIPReIDTrainingModelStub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.retrieval_model = torch.nn.Module()
                self.retrieval_model.beta = 999.0

        stub = CLIPReIDTrainingModelStub()
        warmup_schedule = BetaSchedule(beta=0.3, warmup_epochs=2)
        warmup_schedule.apply(stub, epoch=1)
        self.assertEqual(stub.retrieval_model.beta, 0.0)
        warmup_schedule.apply(stub, epoch=3)
        self.assertAlmostEqual(stub.retrieval_model.beta, 0.3)

    def test_beta_schedule_apply_uses_stage_first_epoch(self):
        class CLIPReIDTrainingModelStub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.retrieval_model = torch.nn.Module()
                self.retrieval_model.beta = 999.0

        stub = CLIPReIDTrainingModelStub()
        schedule = BetaSchedule(beta=0.1, warmup_epochs=5, first_epoch=11)

        schedule.apply(stub, epoch=11)
        self.assertAlmostEqual(stub.retrieval_model.beta, 0.0)

        schedule.apply(stub, epoch=15)
        self.assertAlmostEqual(stub.retrieval_model.beta, 0.08)

        schedule.apply(stub, epoch=16)
        self.assertAlmostEqual(stub.retrieval_model.beta, 0.1)

    def test_stage_lr_scheduler_warmup_then_cosine(self):
        scheduler = StageLRScheduler(base_lrs=(1.0,), total_epochs=10, warmup_epochs=2)

        self.assertAlmostEqual(scheduler.scale(1), 0.5)
        self.assertAlmostEqual(scheduler.scale(2), 1.0)
        self.assertAlmostEqual(scheduler.scale(3), 1.0)
        self.assertLess(scheduler.scale(10), 0.05)
        self.assertGreaterEqual(scheduler.scale(10), 0.0)

    def test_stage_lr_scheduler_apply_scales_groups_by_stage_epoch(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 1e-4, "name": "new"}])
        scheduler = StageLRScheduler(base_lrs=(1e-4,), total_epochs=10, warmup_epochs=2, first_epoch=11)

        scheduler.apply(optimizer, epoch=11)  # stage epoch 1 -> 0.5x warmup

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.5e-4)

    def test_stage2_cosine_scheduler_changes_lr_across_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.stage2_lr_scheduler = "cosine"
            args.stage2_warmup_epochs = 2
            args.epochs = 10

            job = build_training_job(args, clip_loader=_load_fake_clip)
            first = job.train_one_epoch(1, TrainBatchReporterRecorder())["lr"]
            second = job.train_one_epoch(2, TrainBatchReporterRecorder())["lr"]

        self.assertNotEqual(first, second)

    def test_stage2_scheduler_none_keeps_lr_constant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.stage2_lr_scheduler = "none"
            args.epochs = 10

            job = build_training_job(args, clip_loader=_load_fake_clip)
            first = job.train_one_epoch(1, TrainBatchReporterRecorder())["lr"]
            second = job.train_one_epoch(2, TrainBatchReporterRecorder())["lr"]

        self.assertEqual(first, second)

    def test_job_config_reads_num_instances(self):
        from t2c_clip.jobs.clip_reid import _job_config_from_args

        args = _training_args(Path("."))
        args.num_instances = 4

        config = _job_config_from_args(args)

        self.assertEqual(config.num_instances, 4)

    def test_job_config_reads_id_logit_scale(self):
        from t2c_clip.jobs.clip_reid import _job_config_from_args

        args = _training_args(Path("."))
        args.id_logit_scale = 10.0

        config = _job_config_from_args(args)

        self.assertEqual(config.id_logit_scale, 10.0)

    def test_job_config_num_instances_defaults_to_two_when_absent(self):
        from t2c_clip.jobs.clip_reid import _job_config_from_args

        args = _training_args(Path("."))

        config = _job_config_from_args(args)

        self.assertEqual(config.num_instances, 2)

    def test_train_loader_uses_configured_num_instances(self):
        from t2c_clip.jobs.clip_reid import _job_config_from_args, _train_loader

        class _StubDataset(torch.utils.data.Dataset):
            def __init__(self, person_ids):
                self._person_ids = tuple(person_ids)

            @property
            def person_ids(self):
                return self._person_ids

            def __len__(self):
                return len(self._person_ids)

            def __getitem__(self, index):
                return index

        args = _training_args(Path("."))
        args.num_instances = 4
        args.batch_size = 8
        config = _job_config_from_args(args)
        dataset = _StubDataset([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

        loader = _train_loader(dataset, config)

        self.assertEqual(loader.batch_sampler._instances_per_identity, 4)
        self.assertEqual(loader.batch_sampler._identities_per_batch, 2)


def _load_fake_clip(model_name: str) -> CLIPLoadResult:
    return CLIPLoadResult(FakeCLIP(hidden_size=8, projection_dim=4), ImageAwareFakeImageProcessor(), tokenizer=None)


def _trainable_parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _lookup_param_groups(optimizer: torch.optim.Optimizer) -> tuple[dict, dict]:
    """Return the (backbone, new) param groups used by the grouped-learning-rate optimizer."""
    by_name = {group.get("name", ""): group for group in optimizer.param_groups}
    for required in ("backbone", "new"):
        if required not in by_name:
            raise AssertionError(
                f"optimizer is missing required param group {required!r}; "
                f"found names: {sorted(by_name)}"
            )
    return by_name["backbone"], by_name["new"]


def _element_name(model: torch.nn.Module, parameter: torch.nn.Parameter) -> str:
    for name, candidate in model.named_parameters():
        if candidate is parameter:
            return name
    raise AssertionError("parameter is not present on model.named_parameters()")


class TrainBatchReporterRecorder:
    def __init__(self):
        self.batch_reports: list[dict[str, float]] = []

    def batches(self, iterable):
        return iterable

    def report_batch(self, metrics):
        self.batch_reports.append(dict(metrics))


class RetrievalModeRecorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.retrieval_modes: list[str] = []

    def encode_retrieval(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        retrieval_mode: str = "fused",
    ) -> torch.Tensor:
        self.retrieval_modes.append(retrieval_mode)
        return torch.ones(images.shape[0], 4)


def _training_args(root: Path) -> Namespace:
    return Namespace(
        dataset="market1501",
        data_root=root,
        clip_model_name="fake-clip",
        batch_size=4,
        num_workers=0,
        lr=0.001,
        image_encoder_lr=5e-5,
        device="cpu",
        beta=0.1,
        context_length=2,
        tfc_momentum=0.5,
        triplet_margin=0.3,
        tfc_weight=1.0,
        clip_weight=0.1,
        label_smoothing=0.0,
        stage1_epochs=0,
        epochs=1,
        validation_interval=1,
        freeze_image_encoder_stage1=True,
        freeze_image_encoder_stage2=False,
        freeze_text_encoder=True,
        freeze_prompt_bank_stage2=False,
        reid_head="linear",
        clip_checkpoint=None,
        retrieval_mode="fused",
        beta_warmup_epochs=0,
        report_rerank=False,
    )


def _build_market_fixture(root: Path) -> Path:
    _write_market_image(root / "bounding_box_train" / "0001_c1s1_000001_01.jpg", "red")
    _write_market_image(root / "bounding_box_train" / "0001_c2s1_000002_01.jpg", "red")
    _write_market_image(root / "bounding_box_train" / "0002_c1s1_000003_01.jpg", "blue")
    _write_market_image(root / "bounding_box_train" / "0002_c2s1_000004_01.jpg", "blue")
    _write_market_image(root / "query" / "0003_c1s1_000004_01.jpg", "green")
    _write_market_image(root / "bounding_box_test" / "0003_c2s1_000005_01.jpg", "green")
    _write_market_image(root / "bounding_box_test" / "0004_c1s1_000006_01.jpg", "blue")
    return root


def _build_market_fixture_without_positive_pairs(root: Path) -> Path:
    _write_market_image(root / "bounding_box_train" / "0001_c1s1_000001_01.jpg", "red")
    _write_market_image(root / "bounding_box_train" / "0002_c1s1_000002_01.jpg", "blue")
    _write_market_image(root / "query" / "0003_c1s1_000003_01.jpg", "green")
    _write_market_image(root / "bounding_box_test" / "0003_c2s1_000004_01.jpg", "green")
    return root


def _write_market_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=color).save(path)
