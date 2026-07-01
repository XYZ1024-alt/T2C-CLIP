"""Training entrypoint for project-specific T2C-CLIP jobs.

Supports two-stage training (Stage-1 prompt alignment then Stage-2 ReID
training) when the job builder returns a :class:`TwoStageTrainingJob`.

For single-stage job builders (e.g. unit test fixtures), falls back to the
legacy single-loop path with stage-aware MLflow logging treated as the
Stage-2 metrics.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any, Sequence

import torch

from t2c_clip.evaluation import ReIDMetrics
from t2c_clip.loops import (
    MetricLogger,
    TrainMetricLogger,
    TrainStepMetricLogger,
    TrainingEpochReporter,
    TrainingLoopConfig,
    run_training_loop,
)
from t2c_clip.mlflow import (
    MLflowSQLiteConfig,
    log_reid_metrics_to_mlflow,
    log_stage_params_to_mlflow,
    make_stage_metric_loggers,
    start_mlflow_sqlite_run,
)
from t2c_clip.retrieval import SUPPORTED_RETRIEVAL_MODES

DEFAULT_TOTAL_EPOCHS = 120
DEFAULT_VALIDATION_INTERVAL = 5
DEFAULT_STAGE1_EPOCHS = 0
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_TRACKING_DB = Path("mlflow") / "t2c_clip.db"
DEFAULT_ARTIFACT_ROOT = Path("mlruns")
DEFAULT_EXPERIMENT_NAME = "T2C-CLIP"
DEFAULT_RUN_NAME = "train"
DEFAULT_CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_INSTANCES = 2
DEFAULT_NUM_WORKERS = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_IMAGE_ENCODER_LR = 5e-5
DEFAULT_BETA_WARMUP_EPOCHS = 0
DEFAULT_STAGE2_LR_SCHEDULER = "none"
DEFAULT_STAGE2_WARMUP_EPOCHS = 0
DEFAULT_SANITY_GATE_EPOCHS = 0
DEFAULT_SANITY_GATE_FACTOR = 1.5
DEFAULT_DEVICE = "cuda"
DEFAULT_BETA = 0.1
DEFAULT_CONTEXT_LENGTH = 4
DEFAULT_TFC_MOMENTUM = 0.5
DEFAULT_TRIPLET_MARGIN = 0.3
DEFAULT_TFC_WEIGHT = 1.0
DEFAULT_CLIP_WEIGHT = 0.1
DEFAULT_RETRIEVAL_MODE = "fused"
SUPPORTED_DATASETS = ("market1501", "msmt17")

TrainOneEpoch = Callable[[int, TrainingEpochReporter], dict[str, float] | None]
ValidateEpoch = Callable[[int], ReIDMetrics]
JobBuilder = Callable[[argparse.Namespace], "TrainingJob"]
ProgressFactory = Callable[[Iterable[int]], Iterable[int]]

# Stage-1 never performs mAP validation; setting a huge value disables validation.
STAGE1_DISABLE_VALIDATION_INTERVAL = 10**9


@dataclass(frozen=True)
class TrainingJob:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer | None
    train_one_epoch: TrainOneEpoch
    validate: ValidateEpoch


@dataclass(frozen=True)
class StageMetadata:
    values: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)


@dataclass(frozen=True)
class TwoStageTrainingJob:
    stage1: TrainingJob
    stage2: TrainingJob
    stage_metadata: StageMetadata | None = None


def main(argv: Sequence[str] | None = None, progress_factory: ProgressFactory | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.stage2_first_epoch = args.stage1_epochs + 1
    with _mlflow_context_if_requested(args):
        job = _load_job_builder(args.job_builder)(args)
        # Use a structural check rather than ``isinstance(job, TwoStageTrainingJob)``.
        # Under ``python -m scripts.train`` the entry module is loaded twice (once
        # as ``__main__`` and once as ``scripts.train``), so the ``TwoStageTrainingJob``
        # class object seen here (from ``__main__``) differs from the one the job
        # builder imported (from ``scripts.train``). isinstance would always be
        # False and the two-stage job would be wrongly dispatched to the single
        # loop. Duck-typing on the public stage attributes sidesteps that.
        if _is_two_stage_job(job):
            _run_two_stage_loop(job, args, progress_factory)
        else:
            _run_single_loop(job, args, progress_factory)
    return 0


def _is_two_stage_job(job: Any) -> bool:
    return hasattr(job, "stage1") and hasattr(job, "stage2") and not hasattr(job, "model")


def _stage_metadata_values(metadata: Any) -> dict[str, Any]:
    """Extract the metadata dict from either a ``StageMetadata`` wrapper or a raw dict.

    Job builders historically returned either a ``StageMetadata`` (with a
    ``.values`` dict attribute) or a raw ``dict`` (whose ``.values`` is the
    ``dict.values`` method, not the values themselves). Normalize both shapes
    so the MLflow logger always receives a mapping and never a bound method.
    """
    if isinstance(metadata, Mapping):
        return dict(metadata)
    values = getattr(metadata, "values", None)
    if isinstance(values, Mapping):
        return dict(values)
    if values is None:
        return {}
    raise TypeError(
        f"stage_metadata.values is neither a mapping nor None: {type(values).__name__}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a T2C-CLIP training job.")
    parser.add_argument("--job-builder", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TOTAL_EPOCHS)
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--tracking-db", type=Path, default=DEFAULT_TRACKING_DB)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    _add_project_training_args(parser)
    return parser


def _add_project_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--clip-model-name", default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--clip-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-instances", type=int, default=DEFAULT_NUM_INSTANCES)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--image-encoder-lr", type=float, default=DEFAULT_IMAGE_ENCODER_LR)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--beta-warmup-epochs", type=int, default=DEFAULT_BETA_WARMUP_EPOCHS)
    parser.add_argument("--stage2-lr-scheduler", choices=("none", "cosine"), default=DEFAULT_STAGE2_LR_SCHEDULER)
    parser.add_argument("--stage2-warmup-epochs", type=int, default=DEFAULT_STAGE2_WARMUP_EPOCHS)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--tfc-momentum", type=float, default=DEFAULT_TFC_MOMENTUM)
    parser.add_argument("--triplet-margin", type=float, default=DEFAULT_TRIPLET_MARGIN)
    parser.add_argument("--tfc-weight", type=float, default=DEFAULT_TFC_WEIGHT)
    parser.add_argument("--stage1-epochs", type=int, default=DEFAULT_STAGE1_EPOCHS)
    parser.add_argument("--clip-weight", type=float, default=DEFAULT_CLIP_WEIGHT)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--reid-head", choices=("linear", "bnneck"), default="linear")
    parser.add_argument("--retrieval-mode", choices=SUPPORTED_RETRIEVAL_MODES, default=DEFAULT_RETRIEVAL_MODE)
    parser.add_argument("--report-rerank", action="store_true")
    parser.add_argument(
        "--freeze-prompt-bank-stage2",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--freeze-image-encoder-stage1",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--freeze-image-encoder-stage2",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--freeze-text-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--sanity-gate-epochs", type=int, default=DEFAULT_SANITY_GATE_EPOCHS)
    parser.add_argument("--sanity-gate-factor", type=float, default=DEFAULT_SANITY_GATE_FACTOR)


def _mlflow_context_if_requested(args: argparse.Namespace):
    if not args.enable_mlflow:
        return nullcontext()
    config = MLflowSQLiteConfig(args.tracking_db, args.artifact_root, args.experiment_name)
    return start_mlflow_sqlite_run(config, run_name=args.run_name)


def _run_two_stage_loop(job: TwoStageTrainingJob, args: argparse.Namespace, progress_factory: ProgressFactory | None) -> None:
    if args.enable_mlflow and job.stage_metadata is not None:
        log_stage_params_to_mlflow(_stage_metadata_values(job.stage_metadata))
    stage1_loggers = _stage_metric_loggers_for("stage1", args)
    stage2_loggers = _stage_metric_loggers_for("stage2", args)
    stage1_config = TrainingLoopConfig(
        total_epochs=args.stage1_epochs,
        validation_interval=STAGE1_DISABLE_VALIDATION_INTERVAL,
        checkpoint_dir=args.checkpoint_dir,
        progress_description="stage1",
        checkpoint_prefix="stage1",
        stage="stage1",
    )
    if args.stage1_epochs > 0:
        run_training_loop(
            model=job.stage1.model,
            optimizer=job.stage1.optimizer,
            config=stage1_config,
            train_one_epoch=job.stage1.train_one_epoch,
            validate=job.stage1.validate,
            progress_factory=_progress(progress_factory),
            metric_logger=None,
            train_metric_logger=stage1_loggers[0],
            train_step_metric_logger=stage1_loggers[1],
        )

    stage2_first_epoch = args.stage1_epochs + 1
    stage2_config = TrainingLoopConfig(
        total_epochs=args.epochs,
        validation_interval=args.validation_interval,
        checkpoint_dir=args.checkpoint_dir,
        first_epoch=stage2_first_epoch,
        progress_description="stage2",
        checkpoint_prefix="",
        stage="stage2",
        sanity_check_offset=int(getattr(args, "sanity_gate_epochs", 0)),
        sanity_improvement_factor=float(getattr(args, "sanity_gate_factor", 1.5)),
    )
    run_training_loop(
        model=job.stage2.model,
        optimizer=job.stage2.optimizer,
        config=stage2_config,
        train_one_epoch=job.stage2.train_one_epoch,
        validate=job.stage2.validate,
        progress_factory=_progress(progress_factory),
        metric_logger=_metric_logger_if_requested(args),
        train_metric_logger=stage2_loggers[0],
        train_step_metric_logger=stage2_loggers[1],
    )


def _run_single_loop(job: TrainingJob, args: argparse.Namespace, progress_factory: ProgressFactory | None) -> None:
    loggers = _stage_metric_loggers_for("stage2", args)
    config = TrainingLoopConfig(
        total_epochs=args.epochs,
        validation_interval=args.validation_interval,
        checkpoint_dir=args.checkpoint_dir,
        progress_description="stage2",
        stage="stage2",
        sanity_check_offset=int(getattr(args, "sanity_gate_epochs", 0)),
        sanity_improvement_factor=float(getattr(args, "sanity_gate_factor", 1.5)),
    )
    run_training_loop(
        model=job.model,
        optimizer=job.optimizer,
        config=config,
        train_one_epoch=job.train_one_epoch,
        validate=job.validate,
        progress_factory=_progress(progress_factory),
        metric_logger=_metric_logger_if_requested(args),
        train_metric_logger=loggers[0],
        train_step_metric_logger=loggers[1],
    )


def _progress(progress_factory: ProgressFactory | None) -> ProgressFactory:
    return progress_factory if progress_factory is not None else _default_progress_factory


def _default_progress_factory(iterable: Iterable[int], **kwargs) -> Iterable[int]:
    # Defer tqdm import so tests do not depend on a real terminal backend.
    from tqdm.auto import tqdm

    return tqdm(iterable, **kwargs)


def _stage_metric_loggers_for(stage: str, args: argparse.Namespace) -> tuple[TrainMetricLogger | None, TrainStepMetricLogger | None]:
    if not args.enable_mlflow:
        return None, None
    return make_stage_metric_loggers(stage)


def _metric_logger_if_requested(args: argparse.Namespace) -> MetricLogger | None:
    if not args.enable_mlflow:
        return None
    return log_reid_metrics_to_mlflow


def _load_job_builder(spec: str) -> JobBuilder:
    module_name, function_name = _split_builder_spec(spec)
    module = importlib.import_module(module_name)
    builder = getattr(module, function_name)
    if not callable(builder):
        raise TypeError(f"Job builder is not callable: {spec}")
    return builder


def _split_builder_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError("job builder must use 'module:function' format")
    module_name, function_name = spec.split(":", maxsplit=1)
    if not module_name or not function_name:
        raise ValueError("job builder must include both module and function")
    return module_name, function_name


if __name__ == "__main__":
    raise SystemExit(main())
