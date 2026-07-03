from pathlib import Path
import contextlib
import io
import tempfile
import unittest

from mlflow.tracking import MlflowClient
import torch

from scripts.train import TrainingJob, TwoStageTrainingJob, main
from t2c_clip.evaluation import ReIDMetrics
from t2c_clip.mlflow import sqlite_tracking_uri

RECORDED_STAGE_EPOCHS: dict[str, list[int]] = {"stage1": [], "stage2": []}


class TwoStageTrainingScriptTest(unittest.TestCase):
    def test_main_runs_stage1_then_stage2_and_logs_stage_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracking_db = Path(tmp) / "mlflow" / "tracking.db"
            artifact_root = Path(tmp) / "mlruns"
            checkpoint_dir = Path(tmp) / "checkpoints"
            exit_code = main(
                [
                    "--job-builder",
                    f"{__name__}:build_two_stage_training_job",
                    "--stage1-epochs",
                    "2",
                    "--epochs",
                    "1",
                    "--validation-interval",
                    "1",
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--enable-mlflow",
                    "--tracking-db",
                    str(tracking_db),
                    "--artifact-root",
                    str(artifact_root),
                    "--experiment-name",
                    "T2C-CLIP-TwoStage-Script-Test",
                    "--run-name",
                    "two-stage-test",
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            runs = _runs_for_experiment(tracking_db, "T2C-CLIP-TwoStage-Script-Test")
            self.assertEqual(exit_code, 0)
            self.assertTrue(runs)
            stage1_epochs = runs[0].data.metrics.get("stage1_train_loss")
            stage2_losses = runs[0].data.metrics.get("stage2_train_loss")
            validation_map = runs[0].data.metrics.get("mAP")
            self.assertIsNotNone(stage1_epochs, "stage1_train_loss must be logged")
            self.assertIsNotNone(stage2_losses, "stage2_train_loss must be logged")
            self.assertIsNotNone(validation_map, "mAP must be logged for validation")
            stage1_last_path = checkpoint_dir / "stage1_last.pth"
            best_path = checkpoint_dir / "best.pth"
            last_path = checkpoint_dir / "last.pth"
            self.assertTrue(stage1_last_path.exists(), "stage1 checkpoint file missing")
            self.assertTrue(best_path.exists(), "best.pth should exist (single Stage-2 epoch)")
            self.assertTrue(last_path.exists(), "last.pth should exist (Stage-2 checkpoint)")
            best_payload = _load_checkpoint(best_path)
            self.assertEqual(best_payload["stage"], "stage2")
            last_payload = _load_checkpoint(last_path)
            self.assertEqual(last_payload["stage"], "stage2")
            stage1_payload = _load_checkpoint(stage1_last_path)
            self.assertEqual(stage1_payload["stage"], "stage1")

    def test_resume_skips_stage1_and_completed_stage2_epochs(self):
        RECORDED_STAGE_EPOCHS["stage1"].clear()
        RECORDED_STAGE_EPOCHS["stage2"].clear()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            # Global epochs: stage1 = [1], stage2 = [2, 3]; last.pth records epoch 3.
            main(
                [
                    "--job-builder", f"{__name__}:build_two_stage_training_job",
                    "--stage1-epochs", "1",
                    "--epochs", "2",
                    "--validation-interval", "1",
                    "--checkpoint-dir", str(checkpoint_dir),
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )

            exit_code = main(
                [
                    "--job-builder", f"{__name__}:recording_two_stage_builder",
                    "--stage1-epochs", "1",
                    "--epochs", "4",
                    "--validation-interval", "1",
                    "--checkpoint-dir", str(checkpoint_dir),
                    "--resume", str(checkpoint_dir / "last.pth"),
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(RECORDED_STAGE_EPOCHS["stage1"], [])
        self.assertEqual(RECORDED_STAGE_EPOCHS["stage2"], [4, 5])


def build_two_stage_training_job(args) -> TwoStageTrainingJob:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def make_train_one_epoch(stage: str):
        def train_one_epoch(epoch: int, reporter) -> dict[str, float]:
            for batch_number in reporter.batches([1]):
                optimizer.zero_grad()
                loss = model(torch.eye(2)).sum() * epoch
                loss.backward()
                optimizer.step()
                reporter.report_batch({"loss": float(epoch), "lr": 0.1})
            return {"loss": float(epoch), "lr": 0.1}

        return train_one_epoch

    def stage1_validate(epoch: int) -> ReIDMetrics:
        # Stage-1 has no ReID classifier, so we return NaN metrics on purpose.
        return ReIDMetrics(map=float("nan"), cmc={rank: float("nan") for rank in (1, 5, 10)})

    def stage2_validate(epoch: int) -> ReIDMetrics:
        return ReIDMetrics(map=0.5, cmc={1: 0.5})

    return TwoStageTrainingJob(
        stage1=TrainingJob(
            model=model,
            optimizer=optimizer,
            train_one_epoch=make_train_one_epoch("stage1"),
            validate=stage1_validate,
        ),
        stage2=TrainingJob(
            model=model,
            optimizer=optimizer,
            train_one_epoch=make_train_one_epoch("stage2"),
            validate=stage2_validate,
        ),
        stage_metadata=None,
    )


def recording_two_stage_builder(args) -> TwoStageTrainingJob:
    job = build_two_stage_training_job(args)

    def wrap(stage: str, inner):
        def train_one_epoch(epoch: int, reporter):
            RECORDED_STAGE_EPOCHS[stage].append(epoch)
            return inner(epoch, reporter)

        return train_one_epoch

    return TwoStageTrainingJob(
        stage1=TrainingJob(
            model=job.stage1.model,
            optimizer=job.stage1.optimizer,
            train_one_epoch=wrap("stage1", job.stage1.train_one_epoch),
            validate=job.stage1.validate,
        ),
        stage2=TrainingJob(
            model=job.stage2.model,
            optimizer=job.stage2.optimizer,
            train_one_epoch=wrap("stage2", job.stage2.train_one_epoch),
            validate=job.stage2.validate,
        ),
        stage_metadata=None,
    )


def _runs_for_experiment(tracking_db: Path, experiment_name: str):
    client = MlflowClient(tracking_uri=sqlite_tracking_uri(tracking_db))
    experiment = client.get_experiment_by_name(experiment_name)
    return client.search_runs([experiment.experiment_id])


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)