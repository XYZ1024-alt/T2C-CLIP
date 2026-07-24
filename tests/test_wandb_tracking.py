from __future__ import annotations

from enum import Enum
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from t2c_clip.evaluation import ReIDMetrics
from t2c_clip.wandb import (
    WandbConfig,
    WandbTracker,
    _load_wandb,
    start_wandb_run,
)


class ExampleMode(Enum):
    VALUE = "serialized"


class FakeConfig(dict):
    def __init__(self):
        super().__init__()
        self.allow_val_change: list[bool] = []

    def update(self, values=None, *, allow_val_change: bool = False):
        self.allow_val_change.append(allow_val_change)
        super().update({} if values is None else values)


class FakeRun:
    def __init__(self):
        self.config = FakeConfig()
        self.defined_metrics: list[tuple[str, dict[str, str]]] = []
        self.logged: list[dict[str, float | int]] = []
        self.finish_codes: list[int] = []

    def define_metric(self, name: str, **kwargs) -> None:
        self.defined_metrics.append((name, dict(kwargs)))

    def log(self, payload) -> None:
        self.logged.append(dict(payload))

    def finish(self, exit_code: int = 0) -> None:
        self.finish_codes.append(exit_code)


class FakeWandb:
    def __init__(self, run: FakeRun | None = None):
        self.run = FakeRun() if run is None else run
        self.init_calls: list[dict] = []

    def init(self, **kwargs):
        self.init_calls.append(dict(kwargs))
        return self.run


class WandbConfigTest(unittest.TestCase):
    def test_defaults_to_online_t2c_clip_project(self):
        config = WandbConfig()

        self.assertEqual(config.project, "T2C-CLIP")
        self.assertEqual(config.mode, "online")
        self.assertIsNone(config.entity)
        self.assertIsNone(config.directory)

    def test_rejects_empty_project_and_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "project"):
            WandbConfig(project="  ")
        with self.assertRaisesRegex(ValueError, "unsupported wandb mode"):
            WandbConfig(mode="disabled")  # type: ignore[arg-type]


class WandbRunContextTest(unittest.TestCase):
    def test_init_receives_configured_values_and_success_finishes_run(self):
        fake_wandb = FakeWandb()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "tracking"
            config = WandbConfig(
                project="T2C-CLIP-Test",
                entity="test-team",
                mode="offline",
                directory=directory,
            )
            with mock.patch("t2c_clip.wandb._load_wandb", return_value=fake_wandb):
                with start_wandb_run(config, "test-run") as tracker:
                    self.assertIs(tracker.run, fake_wandb.run)

            expected_directory = str(directory.resolve())
            self.assertTrue(directory.exists())

        self.assertEqual(
            fake_wandb.init_calls,
            [
                {
                    "project": "T2C-CLIP-Test",
                    "entity": "test-team",
                    "name": "test-run",
                    "mode": "offline",
                    "dir": expected_directory,
                }
            ],
        )
        self.assertEqual(fake_wandb.run.finish_codes, [0])

    def test_init_omits_optional_entity_and_directory(self):
        fake_wandb = FakeWandb()
        with mock.patch("t2c_clip.wandb._load_wandb", return_value=fake_wandb):
            with start_wandb_run(WandbConfig(), "train"):
                pass

        self.assertEqual(
            fake_wandb.init_calls,
            [{"project": "T2C-CLIP", "name": "train", "mode": "online"}],
        )

    def test_training_exception_finishes_failed_run_and_is_not_swallowed(self):
        fake_wandb = FakeWandb()
        with mock.patch("t2c_clip.wandb._load_wandb", return_value=fake_wandb):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with start_wandb_run(WandbConfig(mode="offline"), "failed-run"):
                    raise RuntimeError("training failed")

        self.assertEqual(fake_wandb.run.finish_codes, [1])

    def test_init_returning_no_run_fails_explicitly(self):
        fake_wandb = FakeWandb()
        fake_wandb.run = None
        with mock.patch("t2c_clip.wandb._load_wandb", return_value=fake_wandb):
            with self.assertRaisesRegex(RuntimeError, "returned no run"):
                with start_wandb_run(WandbConfig(mode="offline"), "missing-run"):
                    pass

    def test_missing_dependency_has_actionable_error(self):
        with mock.patch.dict(sys.modules, {"wandb": None}):
            with self.assertRaisesRegex(ImportError, "pip install wandb"):
                _load_wandb()


class WandbTrackerTest(unittest.TestCase):
    def setUp(self):
        self.run = FakeRun()
        self.tracker = WandbTracker(self.run)

    def test_stage_config_filters_none_and_serializes_structured_values(self):
        self.tracker.log_stage_config(
            {
                "dataset": "msmt17",
                "clip_checkpoint": None,
                "checkpoint_dir": Path("checkpoints"),
                "shape": (256, 128),
                "mode": ExampleMode.VALUE,
                "nested": {"path": Path("weights/model.pth")},
            }
        )

        self.assertEqual(
            self.run.config,
            {
                "checkpoint_dir": "checkpoints",
                "dataset": "msmt17",
                "mode": "serialized",
                "nested": {"path": "weights/model.pth"},
                "shape": [256, 128],
            },
        )
        self.assertEqual(self.run.config.allow_val_change, [True])

    def test_stage_loggers_keep_metric_names_and_use_independent_axes(self):
        stage1_epoch, stage1_step = self.tracker.make_stage_metric_loggers("stage1")
        stage2_epoch, stage2_step = self.tracker.make_stage_metric_loggers("stage2")

        stage1_step(7, {"loss": 0.7, "clip_loss": 0.2, "lr": 0.001})
        stage1_epoch(3, {"loss": 0.6, "clip_loss": 0.1, "lr": 0.001})
        stage2_step(1, {"loss": 1.2, "reid_loss": 0.8, "lr": 0.0001})
        stage2_epoch(21, {"loss": 1.0, "reid_loss": 0.7, "lr": 0.0001})

        self.assertEqual(
            self.run.logged,
            [
                {
                    "stage1_train_step": 7,
                    "stage1_train_step_loss": 0.7,
                    "stage1_train_step_clip_loss": 0.2,
                    "stage1_train_step_lr": 0.001,
                },
                {
                    "stage1_epoch": 3,
                    "stage1_train_loss": 0.6,
                    "stage1_train_clip_loss": 0.1,
                    "stage1_lr": 0.001,
                },
                {
                    "stage2_train_step": 1,
                    "stage2_train_step_loss": 1.2,
                    "stage2_train_step_reid_loss": 0.8,
                    "stage2_train_step_lr": 0.0001,
                },
                {
                    "stage2_epoch": 21,
                    "stage2_train_loss": 1.0,
                    "stage2_train_reid_loss": 0.7,
                    "stage2_lr": 0.0001,
                },
            ],
        )
        definitions = _definitions(self.run)
        self.assertEqual(
            definitions["stage1_train_step_loss"],
            {"step_metric": "stage1_train_step"},
        )
        self.assertEqual(
            definitions["stage1_train_loss"],
            {"step_metric": "stage1_epoch"},
        )
        self.assertEqual(
            definitions["stage2_train_step_loss"],
            {"step_metric": "stage2_train_step"},
        )
        self.assertEqual(
            definitions["stage2_train_loss"],
            {"step_metric": "stage2_epoch"},
        )

    def test_validation_logs_cmc_extras_and_max_summaries(self):
        self.tracker.log_reid_metrics(
            5,
            ReIDMetrics(
                map=0.4,
                cmc={1: 0.6, 5: 0.8},
                extras={"rerank_mAP": 0.5},
            ),
            best_map=0.4,
            is_best=True,
        )

        self.assertEqual(
            self.run.logged,
            [
                {
                    "validation_epoch": 5,
                    "mAP": 0.4,
                    "is_best": 1.0,
                    "rank_1": 0.6,
                    "rank_5": 0.8,
                    "rerank_mAP": 0.5,
                    "best_mAP": 0.4,
                }
            ],
        )
        definitions = _definitions(self.run)
        self.assertEqual(
            definitions["mAP"],
            {"step_metric": "validation_epoch", "summary": "max"},
        )
        self.assertEqual(
            definitions["best_mAP"],
            {"step_metric": "validation_epoch", "summary": "max"},
        )
        self.assertEqual(
            definitions["rank_1"],
            {"step_metric": "validation_epoch"},
        )

    def test_unknown_stage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown training stage"):
            self.tracker.make_stage_metric_loggers("stage3")


def _definitions(run: FakeRun) -> dict[str, dict[str, str]]:
    return {name: kwargs for name, kwargs in run.defined_metrics}


if __name__ == "__main__":
    unittest.main()
