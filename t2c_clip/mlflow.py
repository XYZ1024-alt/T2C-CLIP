"""MLflow SQLite tracking support."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mlflow
from mlflow.tracking import MlflowClient

from t2c_clip.evaluation import ReIDMetrics

DEFAULT_TRACKING_DB = Path("mlflow") / "t2c_clip.db"
DEFAULT_ARTIFACT_ROOT = Path("mlruns")
DEFAULT_EXPERIMENT_NAME = "T2C-CLIP"
DEFAULT_INIT_RUN_NAME = "mlflow-sqlite-init"
DEFAULT_MLFLOW_UI_HOST = "127.0.0.1"
DEFAULT_MLFLOW_UI_PORT = 6006
TRAIN_STAGES = ("stage1", "stage2")


@dataclass(frozen=True)
class MLflowSQLiteConfig:
    tracking_db: Path = DEFAULT_TRACKING_DB
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    experiment_name: str = DEFAULT_EXPERIMENT_NAME


@dataclass(frozen=True)
class MLflowInitialization:
    tracking_uri: str
    artifact_uri: str
    experiment_id: str
    run_id: str
    experiment_name: str
    ui_command: str


def sqlite_tracking_uri(database_path: Path) -> str:
    normalized = database_path.expanduser().as_posix()
    return f"sqlite:///{normalized}"


def file_artifact_uri(artifact_root: Path) -> str:
    return artifact_root.expanduser().resolve().as_uri()


def initialize_mlflow_sqlite(
    config: MLflowSQLiteConfig,
    run_name: str = DEFAULT_INIT_RUN_NAME,
    tags: Mapping[str, str] | None = None,
) -> MLflowInitialization:
    _prepare_paths(config)
    tracking_uri = sqlite_tracking_uri(config.tracking_db)
    artifact_uri = file_artifact_uri(config.artifact_root)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _ensure_experiment(client, config.experiment_name, artifact_uri)
    run_id = _start_initialization_run(experiment_id, run_name, artifact_uri, tags)
    return MLflowInitialization(
        tracking_uri=tracking_uri,
        artifact_uri=artifact_uri,
        experiment_id=experiment_id,
        run_id=run_id,
        experiment_name=config.experiment_name,
        ui_command=mlflow_ui_command(config),
    )


@contextmanager
def start_mlflow_sqlite_run(
    config: MLflowSQLiteConfig,
    run_name: str,
    tags: Mapping[str, str] | None = None,
) -> Iterator[MLflowInitialization]:
    tracking_uri, artifact_uri, experiment_id = _prepare_tracking_context(config)
    run_tags = _run_tags("training", tags)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tags(run_tags)
        mlflow.log_param("tracking_backend", "sqlite")
        mlflow.log_param("artifact_root", artifact_uri)
        yield MLflowInitialization(
            tracking_uri=tracking_uri,
            artifact_uri=artifact_uri,
            experiment_id=experiment_id,
            run_id=run.info.run_id,
            experiment_name=config.experiment_name,
            ui_command=mlflow_ui_command(config),
        )


def log_reid_metrics_to_mlflow(
    epoch: int,
    metrics: ReIDMetrics,
    best_map: float | None,
    is_best: bool,
) -> None:
    mlflow.log_metric("mAP", metrics.map, step=epoch)
    if best_map is not None:
        mlflow.log_metric("best_mAP", best_map, step=epoch)
    mlflow.log_metric("is_best", float(is_best), step=epoch)
    for rank, value in metrics.cmc.items():
        mlflow.log_metric(f"rank_{rank}", value, step=epoch)


def log_training_metrics_to_mlflow(epoch: int, metrics: Mapping[str, float]) -> None:
    for name, value in metrics.items():
        mlflow.log_metric(_training_metric_name(name), float(value), step=epoch)


def log_training_step_metrics_to_mlflow(train_step: int, metrics: Mapping[str, float]) -> None:
    for name, value in metrics.items():
        mlflow.log_metric(_training_step_metric_name(name), float(value), step=train_step)


def make_stage_metric_loggers(stage: str) -> tuple["TrainMetricLogger", "TrainStepMetricLogger"]:
    """Build stage-aware epoch/step MLflow loggers.

    ``stage`` must be one of :data:`TRAIN_STAGES` (``"stage1"`` or
    ``"stage2"``). The returned loggers prefix each metric with
    ``{stage}_train_`` / ``{stage}_train_step_`` so MLflow histories are
    disambiguated across the two training phases.
    """
    if stage not in TRAIN_STAGES:
        raise ValueError(f"unknown training stage: {stage!r}; expected one of {TRAIN_STAGES}")

    def epoch_logger(epoch: int, metrics: Mapping[str, float]) -> None:
        for name, value in metrics.items():
            mlflow.log_metric(f"{stage}_{_training_metric_name(name)}", float(value), step=epoch)

    def step_logger(train_step: int, metrics: Mapping[str, float]) -> None:
        for name, value in metrics.items():
            mlflow.log_metric(f"{stage}_{_training_step_metric_name(name)}", float(value), step=train_step)

    return epoch_logger, step_logger


def log_stage_params_to_mlflow(metadata: Mapping[str, Any]) -> None:
    """Record the two-stage training configuration as MLflow params/tags."""
    mlflow.set_tag("t2c_clip.retrieval_mode", str(metadata.get("retrieval_mode", "fused")))
    for key in (
        "stage1_epochs",
        "stage2_epochs",
        "validation_interval",
        "freeze_image_encoder_stage1",
        "freeze_image_encoder_stage2",
        "freeze_text_encoder",
        "clip_weight",
        "tfc_weight",
        "beta",
        "beta_warmup_epochs",
        "lr",
        "image_encoder_lr",
        "retrieval_mode",
    ):
        if key in metadata:
            mlflow.log_param(key, metadata[key])


def mlflow_ui_command(
    config: MLflowSQLiteConfig,
    host: str = DEFAULT_MLFLOW_UI_HOST,
    port: int = DEFAULT_MLFLOW_UI_PORT,
) -> str:
    return (
        "mlflow ui "
        f"--backend-store-uri {sqlite_tracking_uri(config.tracking_db)} "
        f"--default-artifact-root {config.artifact_root.as_posix()} "
        f"--host {host} "
        f"--port {port}"
    )


def _prepare_paths(config: MLflowSQLiteConfig) -> None:
    config.tracking_db.expanduser().parent.mkdir(parents=True, exist_ok=True)
    config.artifact_root.expanduser().mkdir(parents=True, exist_ok=True)


def _prepare_tracking_context(config: MLflowSQLiteConfig) -> tuple[str, str, str]:
    _prepare_paths(config)
    tracking_uri = sqlite_tracking_uri(config.tracking_db)
    artifact_uri = file_artifact_uri(config.artifact_root)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _ensure_experiment(client, config.experiment_name, artifact_uri)
    return tracking_uri, artifact_uri, experiment_id


def _ensure_experiment(client: MlflowClient, name: str, artifact_uri: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        return client.create_experiment(name=name, artifact_location=artifact_uri)
    if experiment.lifecycle_stage != "active":
        raise RuntimeError(f"MLflow experiment is not active: {name}")
    return experiment.experiment_id


def _start_initialization_run(
    experiment_id: str,
    run_name: str,
    artifact_uri: str,
    tags: Mapping[str, str] | None,
) -> str:
    run_tags = _run_tags("mlflow_sqlite_init", tags)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tags(run_tags)
        mlflow.log_param("tracking_backend", "sqlite")
        mlflow.log_param("artifact_root", artifact_uri)
        return run.info.run_id


def _run_tags(role: str, tags: Mapping[str, str] | None) -> dict[str, str]:
    run_tags = {"t2c_clip.role": role}
    if tags is not None:
        run_tags.update(tags)
    return run_tags


def _training_metric_name(name: str) -> str:
    if name == "lr":
        return name
    return f"train_{name}"


def _training_step_metric_name(name: str) -> str:
    return f"train_step_{name}"
