"""Weights & Biases tracking support for T2C-ReID training."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING
import warnings

from t2c_reid.evaluation import ReIDMetrics

if TYPE_CHECKING:
    from t2c_reid.loops import TrainMetricLogger, TrainStepMetricLogger

DEFAULT_WANDB_PROJECT = "T2C-ReID"
DEFAULT_WANDB_MODE = "online"
WANDB_MODES = ("online", "offline")
TRAIN_STAGES = ("stage1", "stage2")


@dataclass(frozen=True)
class WandbConfig:
    project: str = DEFAULT_WANDB_PROJECT
    entity: str | None = None
    mode: Literal["online", "offline"] = DEFAULT_WANDB_MODE
    directory: Path | None = None

    def __post_init__(self) -> None:
        if not self.project.strip():
            raise ValueError("wandb project must not be empty")
        if self.mode not in WANDB_MODES:
            raise ValueError(f"unsupported wandb mode: {self.mode!r}; expected one of {WANDB_MODES}")


class WandbTracker:
    """Log stage-aware metrics and configuration to one explicit wandb run."""

    def __init__(self, run: Any):
        self._run = run
        self._defined_metrics: set[tuple[str, str | None, str | None]] = set()

    @property
    def run(self) -> Any:
        return self._run

    def log_stage_config(self, metadata: Mapping[str, Any]) -> None:
        values = {
            key: _serializable_value(value)
            for key, value in sorted(metadata.items())
            if value is not None
        }
        if values:
            self._run.config.update(values, allow_val_change=True)

    def make_stage_metric_loggers(self, stage: str) -> tuple[TrainMetricLogger, TrainStepMetricLogger]:
        if stage not in TRAIN_STAGES:
            raise ValueError(f"unknown training stage: {stage!r}; expected one of {TRAIN_STAGES}")

        epoch_axis = f"{stage}_epoch"
        train_step_axis = f"{stage}_train_step"

        def epoch_logger(epoch: int, metrics: Mapping[str, float]) -> None:
            named_metrics = {
                f"{stage}_{_training_metric_name(name)}": float(value)
                for name, value in metrics.items()
            }
            self._log(epoch_axis, epoch, named_metrics)

        def step_logger(train_step: int, metrics: Mapping[str, float]) -> None:
            named_metrics = {
                f"{stage}_train_step_{name}": float(value)
                for name, value in metrics.items()
            }
            self._log(train_step_axis, train_step, named_metrics)

        return epoch_logger, step_logger

    def log_reid_metrics(
        self,
        epoch: int,
        metrics: ReIDMetrics,
        best_map: float | None,
        is_best: bool,
    ) -> None:
        values = {
            "mAP": float(metrics.map),
            "is_best": float(is_best),
            **{f"rank_{rank}": float(value) for rank, value in metrics.cmc.items()},
            **{name: float(value) for name, value in metrics.extras.items()},
        }
        if best_map is not None:
            values["best_mAP"] = float(best_map)
        self._log(
            "validation_epoch",
            epoch,
            values,
            summaries={"mAP": "max", "best_mAP": "max"},
        )

    def _log(
        self,
        axis_name: str,
        axis_value: int,
        metrics: Mapping[str, float],
        summaries: Mapping[str, str] | None = None,
    ) -> None:
        self._define_metric(axis_name)
        for name in metrics:
            summary = summaries.get(name) if summaries is not None else None
            self._define_metric(name, step_metric=axis_name, summary=summary)
        self._run.log({axis_name: int(axis_value), **metrics})

    def _define_metric(
        self,
        name: str,
        step_metric: str | None = None,
        summary: str | None = None,
    ) -> None:
        definition = (name, step_metric, summary)
        if definition in self._defined_metrics:
            return
        kwargs: dict[str, str] = {}
        if step_metric is not None:
            kwargs["step_metric"] = step_metric
        if summary is not None:
            kwargs["summary"] = summary
        self._run.define_metric(name, **kwargs)
        self._defined_metrics.add(definition)


@contextmanager
def start_wandb_run(config: WandbConfig, run_name: str) -> Iterator[WandbTracker]:
    """Start one wandb run and always finish it without masking training errors."""

    wandb = _load_wandb()
    init_kwargs: dict[str, Any] = {
        "project": config.project,
        "name": run_name,
        "mode": config.mode,
    }
    if config.entity is not None:
        init_kwargs["entity"] = config.entity
    if config.directory is not None:
        directory = config.directory.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        init_kwargs["dir"] = str(directory)

    run = wandb.init(**init_kwargs)
    if run is None:
        raise RuntimeError("wandb.init returned no run")

    try:
        yield WandbTracker(run)
    except BaseException:
        try:
            run.finish(exit_code=1)
        except Exception as finish_error:
            warnings.warn(
                f"wandb run cleanup failed while propagating a training error: {finish_error}",
                RuntimeWarning,
                stacklevel=2,
            )
        raise
    else:
        run.finish(exit_code=0)


def _load_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is required when tracking is enabled; install it with 'pip install wandb'"
        ) from exc
    return wandb


def _training_metric_name(name: str) -> str:
    if name == "lr":
        return name
    return f"train_{name}"


def _serializable_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.expanduser().as_posix()
    if isinstance(value, Enum):
        return _serializable_value(value.value)
    if isinstance(value, Mapping):
        return {str(key): _serializable_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_serializable_value(item) for item in value)
    return value
