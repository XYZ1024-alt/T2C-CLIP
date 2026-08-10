"""Training entrypoint for project-specific T2C-ReID jobs.

Supports two-stage training (Stage-1 prompt alignment then Stage-2 ReID
training) when the job builder returns a :class:`TwoStageTrainingJob`.

For single-stage job builders (e.g. unit test fixtures), falls back to the
legacy single-loop path with stage-aware wandb logging treated as the
Stage-2 metrics.
"""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from t2c_reid.configuration import (
    TrainingConfig,
    compose_training_config,
    training_config_from_dict_config,
)
from t2c_reid.evaluation import ReIDMetrics
from t2c_reid.loops import (
    MetricLogger,
    TrainingEpochReporter,
    TrainingLoopConfig,
    TrainMetricLogger,
    TrainStepMetricLogger,
    run_training_loop,
)
from t2c_reid.wandb import (
    WandbConfig,
    WandbTracker,
    start_wandb_run,
)

TrainOneEpoch = Callable[[int, TrainingEpochReporter], dict[str, float] | None]
ValidateEpoch = Callable[[int], ReIDMetrics]
JobBuilder = Callable[[TrainingConfig], "TrainingJob"]
ProgressFactory = Callable[[Iterable[int]], Iterable[int]]

# Stage-1 never performs mAP validation; setting a huge value disables validation.
STAGE1_DISABLE_VALIDATION_INTERVAL = 10**9


@dataclass(frozen=True)
class TrainingJob:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer | None
    train_one_epoch: TrainOneEpoch
    validate: ValidateEpoch
    stage_metadata: StageMetadata | Mapping[str, Any] | None = None
    checkpoint_metadata: Mapping[str, Any] | None = None
    auxiliary_state: Any | None = None


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


def main(
    overrides: Sequence[str] | None = None,
    progress_factory: ProgressFactory | None = None,
) -> int | None:
    """Run through Hydra, or compose explicit overrides for programmatic callers."""

    if overrides is None and progress_factory is None:
        return _hydra_main()
    config = compose_training_config(() if overrides is None else overrides)
    return _run_training(config, progress_factory)


@hydra.main(config_path="../t2c_reid/configs/training", config_name="train")
def _hydra_main(config: DictConfig) -> int:
    return _run_training(training_config_from_dict_config(config))


def _run_training(
    config: TrainingConfig,
    progress_factory: ProgressFactory | None = None,
) -> int:
    _seed_random_generators(config.seed)
    with _wandb_context_if_requested(config) as tracker:
        job = _load_job_builder(config.job_builder)(config)
        resume_state = _load_resume_state(config.resume)
        # Use a structural check rather than ``isinstance(job, TwoStageTrainingJob)``.
        # Under ``python -m scripts.train`` the entry module is loaded twice (once
        # as ``__main__`` and once as ``scripts.train``), so the ``TwoStageTrainingJob``
        # class object seen here (from ``__main__``) differs from the one the job
        # builder imported (from ``scripts.train``). isinstance would always be
        # False and the two-stage job would be wrongly dispatched to the single
        # loop. Duck-typing on the public stage attributes sidesteps that.
        if _is_two_stage_job(job):
            _run_two_stage_loop(job, config, progress_factory, resume_state, tracker)
        else:
            _run_single_loop(job, config, progress_factory, resume_state, tracker)
    return 0


def _seed_random_generators(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)


def _load_resume_state(checkpoint: Path | None) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    if not checkpoint.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
    return torch.load(checkpoint, map_location="cpu", weights_only=True)


def _restore_job_state(job: TrainingJob, resume_state: dict[str, Any]) -> None:
    _validate_checkpoint_metadata(job.checkpoint_metadata, resume_state)
    auxiliary_state = resume_state.get("auxiliary_state")
    if job.auxiliary_state is not None and not isinstance(auxiliary_state, dict):
        raise ValueError(
            "checkpoint is missing SigLIP 2 precision/scaler state; old "
            "T2C-CLIP checkpoints are incompatible"
        )
    job.model.load_state_dict(resume_state["model_state"])
    if job.optimizer is not None and "optimizer_state" in resume_state:
        job.optimizer.load_state_dict(resume_state["optimizer_state"])
    if job.auxiliary_state is not None:
        job.auxiliary_state.load_state_dict(auxiliary_state)


def _validate_checkpoint_metadata(
    expected: Mapping[str, Any] | None,
    resume_state: Mapping[str, Any],
) -> None:
    if expected is None:
        return
    actual = resume_state.get("checkpoint_metadata")
    if not isinstance(actual, Mapping):
        raise ValueError(  # noqa: TRY004 - malformed checkpoint content
            "checkpoint has no SigLIP 2 compatibility metadata; old T2C-CLIP "
            "checkpoints cannot be resumed"
        )
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={found!r}, run={required!r}"
            for key, (found, required) in sorted(mismatches.items())
        )
        raise ValueError(f"incompatible SigLIP 2 resume checkpoint ({details})")


def _resume_best_map(resume_state: dict[str, Any] | None) -> float | None:
    if resume_state is None:
        return None
    return resume_state.get("best_map")


def _is_two_stage_job(job: Any) -> bool:
    return (
        hasattr(job, "stage1") and hasattr(job, "stage2") and not hasattr(job, "model")
    )


def _stage_metadata_values(metadata: Any) -> dict[str, Any]:
    """Extract the metadata dict from either a ``StageMetadata`` wrapper or a raw dict.

    Job builders historically returned either a ``StageMetadata`` (with a
    ``.values`` dict attribute) or a raw ``dict`` (whose ``.values`` is the
    ``dict.values`` method, not the values themselves). Normalize both shapes
    so the tracking adapter always receives a mapping and never a bound method.
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


def _wandb_context_if_requested(config: TrainingConfig):
    if not config.enable_wandb:
        return nullcontext(None)
    wandb_config = WandbConfig(
        project=config.wandb_project,
        entity=config.wandb_entity,
        mode=config.wandb_mode,
        directory=config.wandb_dir,
    )
    return start_wandb_run(wandb_config, run_name=config.run_name)


def _run_two_stage_loop(
    job: TwoStageTrainingJob,
    config: TrainingConfig,
    progress_factory: ProgressFactory | None,
    resume_state: dict[str, Any] | None = None,
    tracker: WandbTracker | None = None,
) -> None:
    completed_stage2_epochs = 0
    if resume_state is not None:
        resume_stage = resume_state.get("stage")
        if resume_stage != "stage2":
            raise ValueError(
                f"only stage2 checkpoints can be resumed, got stage: {resume_stage!r}"
            )
        _restore_job_state(job.stage2, resume_state)
        completed_stage2_epochs = int(resume_state["epoch"]) - config.stage1_epochs
    if tracker is not None and job.stage_metadata is not None:
        tracker.log_stage_config(_stage_metadata_values(job.stage_metadata))
    stage1_loggers = _stage_metric_loggers_for("stage1", tracker)
    stage2_loggers = _stage_metric_loggers_for("stage2", tracker)
    stage1_config = TrainingLoopConfig(
        total_epochs=config.stage1_epochs,
        validation_interval=STAGE1_DISABLE_VALIDATION_INTERVAL,
        checkpoint_dir=config.checkpoint_dir,
        progress_description="stage1",
        checkpoint_prefix="stage1",
        stage="stage1",
        validate_final_epoch=False,
    )
    if config.stage1_epochs > 0 and resume_state is None:
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
            checkpoint_metadata=job.stage1.checkpoint_metadata,
            auxiliary_state=job.stage1.auxiliary_state,
        )

    stage2_first_epoch = config.stage1_epochs + completed_stage2_epochs + 1
    stage2_config = TrainingLoopConfig(
        total_epochs=config.epochs - completed_stage2_epochs,
        validation_interval=config.validation_interval,
        checkpoint_dir=config.checkpoint_dir,
        first_epoch=stage2_first_epoch,
        progress_description="stage2",
        checkpoint_prefix="",
        stage="stage2",
        sanity_check_offset=config.sanity_gate_epochs,
        sanity_improvement_factor=config.sanity_gate_factor,
    )
    run_training_loop(
        model=job.stage2.model,
        optimizer=job.stage2.optimizer,
        config=stage2_config,
        train_one_epoch=job.stage2.train_one_epoch,
        validate=job.stage2.validate,
        progress_factory=_progress(progress_factory),
        metric_logger=_metric_logger_if_requested(tracker),
        train_metric_logger=stage2_loggers[0],
        train_step_metric_logger=stage2_loggers[1],
        initial_best_map=_resume_best_map(resume_state),
        checkpoint_metadata=job.stage2.checkpoint_metadata,
        auxiliary_state=job.stage2.auxiliary_state,
    )


def _run_single_loop(
    job: TrainingJob,
    config: TrainingConfig,
    progress_factory: ProgressFactory | None,
    resume_state: dict[str, Any] | None = None,
    tracker: WandbTracker | None = None,
) -> None:
    completed_epochs = 0
    if resume_state is not None:
        resume_stage = resume_state.get("stage")
        if resume_stage != "stage2":
            raise ValueError(
                f"only stage2 checkpoints can be resumed, got stage: {resume_stage!r}"
            )
        _restore_job_state(job, resume_state)
        completed_epochs = int(resume_state["epoch"])
    if tracker is not None and job.stage_metadata is not None:
        tracker.log_stage_config(_stage_metadata_values(job.stage_metadata))
    loggers = _stage_metric_loggers_for("stage2", tracker)
    loop_config = TrainingLoopConfig(
        total_epochs=config.epochs - completed_epochs,
        validation_interval=config.validation_interval,
        checkpoint_dir=config.checkpoint_dir,
        first_epoch=completed_epochs + 1,
        progress_description="stage2",
        stage="stage2",
        sanity_check_offset=config.sanity_gate_epochs,
        sanity_improvement_factor=config.sanity_gate_factor,
    )
    run_training_loop(
        model=job.model,
        optimizer=job.optimizer,
        config=loop_config,
        train_one_epoch=job.train_one_epoch,
        validate=job.validate,
        progress_factory=_progress(progress_factory),
        metric_logger=_metric_logger_if_requested(tracker),
        train_metric_logger=loggers[0],
        train_step_metric_logger=loggers[1],
        initial_best_map=_resume_best_map(resume_state),
        checkpoint_metadata=job.checkpoint_metadata,
        auxiliary_state=job.auxiliary_state,
    )


def _progress(progress_factory: ProgressFactory | None) -> ProgressFactory:
    return (
        progress_factory if progress_factory is not None else _default_progress_factory
    )


def _default_progress_factory(iterable: Iterable[int], **kwargs) -> Iterable[int]:
    # Defer tqdm import so tests do not depend on a real terminal backend.
    from tqdm.auto import tqdm

    return tqdm(iterable, **kwargs)


def _stage_metric_loggers_for(
    stage: str,
    tracker: WandbTracker | None,
) -> tuple[TrainMetricLogger | None, TrainStepMetricLogger | None]:
    if tracker is None:
        return None, None
    return tracker.make_stage_metric_loggers(stage)


def _metric_logger_if_requested(tracker: WandbTracker | None) -> MetricLogger | None:
    if tracker is None:
        return None
    return tracker.log_reid_metrics


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
