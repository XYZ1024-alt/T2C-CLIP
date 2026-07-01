import json
from pathlib import Path
import tempfile
import unittest

from mlflow.tracking import MlflowClient

from t2c_clip.evaluation import ReIDMetrics
from t2c_clip.cli.mlflow import main
from t2c_clip.mlflow import (
    DEFAULT_MLFLOW_UI_PORT,
    MLflowSQLiteConfig,
    initialize_mlflow_sqlite,
    log_stage_params_to_mlflow,
    log_reid_metrics_to_mlflow,
    log_training_step_metrics_to_mlflow,
    log_training_metrics_to_mlflow,
    mlflow_ui_command,
    sqlite_tracking_uri,
    start_mlflow_sqlite_run,
)


class MLflowSQLiteTest(unittest.TestCase):
    def test_sqlite_tracking_uri_uses_sqlite_scheme(self):
        uri = sqlite_tracking_uri(Path("mlflow") / "t2c_clip.db")
        self.assertEqual(uri, "sqlite:///mlflow/t2c_clip.db")

    def test_initialize_creates_sqlite_store_experiment_and_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = MLflowSQLiteConfig(
                tracking_db=Path(tmp) / "tracking" / "t2c_clip.db",
                artifact_root=Path(tmp) / "artifacts",
                experiment_name="T2C-CLIP-Test",
            )

            result = initialize_mlflow_sqlite(config, run_name="init-test")
            client = MlflowClient(tracking_uri=result.tracking_uri)
            run = client.get_run(result.run_id)
            database_exists = config.tracking_db.exists()
            artifact_root_exists = config.artifact_root.exists()

        self.assertTrue(database_exists)
        self.assertTrue(artifact_root_exists)
        self.assertEqual(run.info.experiment_id, result.experiment_id)
        self.assertEqual(run.data.tags["t2c_clip.role"], "mlflow_sqlite_init")
        self.assertEqual(run.data.params["tracking_backend"], "sqlite")

    def test_ui_command_uses_default_port_6006(self):
        config = MLflowSQLiteConfig(
            tracking_db=Path("mlflow") / "t2c_clip.db",
            artifact_root=Path("mlruns"),
            experiment_name="T2C-CLIP",
        )

        command = mlflow_ui_command(config)

        self.assertEqual(DEFAULT_MLFLOW_UI_PORT, 6006)
        self.assertIn("--port 6006", command)
        self.assertIn("sqlite:///mlflow/t2c_clip.db", command)

    def test_cli_initializes_sqlite_store_and_writes_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "mlflow.json"
            exit_code = main(
                [
                    "--tracking-db",
                    str(Path(tmp) / "tracking.db"),
                    "--artifact-root",
                    str(Path(tmp) / "artifacts"),
                    "--experiment-name",
                    "T2C-CLIP-CLI-Test",
                    "--run-name",
                    "cli-init",
                    "--output",
                    str(output),
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["experiment_name"], "T2C-CLIP-CLI-Test")
        self.assertIn("--port 6006", payload["ui_command"])
        self.assertTrue(payload["tracking_uri"].startswith("sqlite:///"))

    def test_training_run_logs_reid_metrics_to_sqlite_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = MLflowSQLiteConfig(
                tracking_db=Path(tmp) / "tracking.db",
                artifact_root=Path(tmp) / "artifacts",
                experiment_name="T2C-CLIP-Training-Test",
            )
            with start_mlflow_sqlite_run(config, run_name="train-test") as run:
                log_reid_metrics_to_mlflow(
                    5,
                    ReIDMetrics(map=0.4, cmc={1: 0.6}, extras={"rerank_mAP": 0.5}),
                    0.4,
                    True,
                )
                run_id = run.run_id
            client = MlflowClient(tracking_uri=run.tracking_uri)
            logged = client.get_run(run_id)

        self.assertEqual(logged.data.metrics["mAP"], 0.4)
        self.assertEqual(logged.data.metrics["best_mAP"], 0.4)
        self.assertEqual(logged.data.metrics["rank_1"], 0.6)
        self.assertEqual(logged.data.metrics["rerank_mAP"], 0.5)
        self.assertEqual(logged.data.metrics["is_best"], 1.0)
        self.assertEqual(logged.data.tags["t2c_clip.role"], "training")

    def test_training_run_logs_train_metrics_to_sqlite_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = MLflowSQLiteConfig(
                tracking_db=Path(tmp) / "tracking.db",
                artifact_root=Path(tmp) / "artifacts",
                experiment_name="T2C-CLIP-TrainMetric-Test",
            )
            with start_mlflow_sqlite_run(config, run_name="train-metric-test") as run:
                log_training_metrics_to_mlflow(
                    1,
                    {
                        "loss": 0.7,
                        "clip_loss": 0.2,
                        "reid_loss": 0.3,
                        "triplet_loss": 0.1,
                        "tfc_loss": 0.1,
                        "lr": 0.001,
                    },
                )
                run_id = run.run_id
            client = MlflowClient(tracking_uri=run.tracking_uri)
            logged = client.get_run(run_id)

        self.assertEqual(logged.data.metrics["train_loss"], 0.7)
        self.assertEqual(logged.data.metrics["train_clip_loss"], 0.2)
        self.assertEqual(logged.data.metrics["train_reid_loss"], 0.3)
        self.assertEqual(logged.data.metrics["train_triplet_loss"], 0.1)
        self.assertEqual(logged.data.metrics["train_tfc_loss"], 0.1)
        self.assertEqual(logged.data.metrics["lr"], 0.001)

    def test_training_run_logs_stage_identity_logit_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = MLflowSQLiteConfig(
                tracking_db=Path(tmp) / "tracking.db",
                artifact_root=Path(tmp) / "artifacts",
                experiment_name="T2C-CLIP-StageParam-Test",
            )
            with start_mlflow_sqlite_run(config, run_name="stage-param-test") as run:
                log_stage_params_to_mlflow(
                    {
                        "retrieval_mode": "image",
                        "id_logit_scale": 10.0,
                    }
                )
                run_id = run.run_id
            client = MlflowClient(tracking_uri=run.tracking_uri)
            logged = client.get_run(run_id)

        self.assertEqual(logged.data.params["id_logit_scale"], "10.0")
        self.assertEqual(logged.data.tags["t2c_clip.retrieval_mode"], "image")

    def test_training_run_logs_step_train_metrics_to_sqlite_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = MLflowSQLiteConfig(
                tracking_db=Path(tmp) / "tracking.db",
                artifact_root=Path(tmp) / "artifacts",
                experiment_name="T2C-CLIP-TrainStepMetric-Test",
            )
            with start_mlflow_sqlite_run(config, run_name="train-step-metric-test") as run:
                log_training_step_metrics_to_mlflow(1, {"loss": 0.7, "lr": 0.001})
                log_training_step_metrics_to_mlflow(2, {"loss": 0.6, "lr": 0.001})
                run_id = run.run_id
            client = MlflowClient(tracking_uri=run.tracking_uri)
            loss_history = client.get_metric_history(run_id, "train_step_loss")
            lr_history = client.get_metric_history(run_id, "train_step_lr")

        self.assertEqual([point.step for point in loss_history], [1, 2])
        self.assertEqual([point.value for point in loss_history], [0.7, 0.6])
        self.assertEqual([point.step for point in lr_history], [1, 2])
