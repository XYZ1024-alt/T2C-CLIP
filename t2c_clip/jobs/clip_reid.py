"""Real CLIP-backed T2C-CLIP two-stage training job builder.

Stage-1 prompt alignment:

- Trains the learnable prompt bank via real CLIP text encoder + text_projection.
- CLIP image encoder is frozen by default (``--freeze-image-encoder-stage1``).
- Loss is ``bidirectional_contrastive_loss(f_v, f_t_id)`` only.
- No classifier / triplet / TFC gradients flow in Stage-1.

Stage-2 ReID training:

- Trains prompt bank, classifier, TFC centers, and (by default) the CLIP
  vision backbone + visual projection. CLIP-ReID's Stage-2 recipe fine-tunes
  the image encoder so the ReID signal can actually act on ``f_v``; freezing
  it caps the retrieval mAP near the frozen CLIP image-only floor
  (~1% on MSMT17) and is opt-in via ``--freeze-image-encoder-stage2``.
- The unfrozen backbone uses its own (smaller) learning rate
  ``image_encoder_lr`` to avoid catastrophic forgetting of the pretrained
  visual features.
- ``--beta-warmup-epochs`` ramps the fused retrieval beta from ``0`` at
  epoch 1 to ``config.beta`` at ``warmup_epochs + 1``, so the random
  camera-conditioned text feature does not pull ``f_eval`` below the
  image-only floor at startup.
- Total loss is::

      L_id + L_triplet + clip_weight * L_clip_dual + tfc_weight * L_TFC

Validation:

- Uses ``encode_retrieval`` (global + camera prompts only). Identity prompts
  never touch query/gallery retrieval.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from scripts.train import StageMetadata, TrainingJob, TwoStageTrainingJob
from t2c_clip.clip_backbone import (
    TransformersCLIPImageEncoder,
    TransformersCLIPTextEncoder,
    clip_projection_dim,
    clip_text_hidden_dim,
)
from t2c_clip.data import ReIDSample, load_market_split, load_msmt17_manifest
from t2c_clip.datasets import (
    IdentityBalancedBatchSampler,
    ReIDImageBatch,
    ReIDImageDataset,
    ReIDImageDatasetConfig,
    build_camera_id_map,
    build_person_id_map,
    collate_reid_batches,
)
from t2c_clip.evaluation import ReIDMetrics, evaluate_reid
from t2c_clip.model import T2CClipModel
from t2c_clip.prompts import PromptBank, PromptConfig
from t2c_clip.retrieval import require_retrieval_mode
from t2c_clip.tfc import TFCCenterBank
from t2c_clip.training import (
    Stage1LossBreakdown,
    Stage1LossConfig,
    Stage2LossBreakdown,
    Stage2LossConfig,
    Stage2LossInputs,
    TrainingBatch,
    stage1_alignment_loss,
    stage2_loss_breakdown,
)
from t2c_clip.transforms import CLIPImageTransform

DEFAULT_RANKS = (1, 5, 10)
SUPPORTED_DATASETS = ("market1501", "msmt17")
DEFAULT_CLIP_TOKEN_IDS = {"sot": 49406, "eos": 49407, "pad": 0}
STAGE1_TRAIN_LOSS_METRIC_NAMES = ("loss", "clip_loss")
STAGE2_TRAIN_LOSS_METRIC_NAMES = ("loss", "clip_loss", "reid_loss", "triplet_loss", "tfc_loss")
STAGE1 = "stage1"
STAGE2 = "stage2"

ClipLoader = Callable[[str], "CLIPLoadResult"]


@dataclass(frozen=True)
class CLIPLoadResult:
    model: torch.nn.Module
    image_processor: Any
    tokenizer: Any


@dataclass(frozen=True)
class JobDataConfig:
    dataset: str
    root: Path


DEFAULT_IMAGE_ENCODER_LR = 5e-5


@dataclass(frozen=True)
class CLIPReIDJobConfig:
    dataset: str
    data_root: Path
    clip_model_name: str
    batch_size: int
    num_workers: int
    lr: float
    image_encoder_lr: float
    device: torch.device
    beta: float
    context_length: int
    tfc_momentum: float
    triplet_margin: float
    tfc_weight: float
    clip_weight: float
    stage1_epochs: int
    stage2_epochs: int
    validation_interval: int
    freeze_image_encoder_stage1: bool
    freeze_image_encoder_stage2: bool
    freeze_text_encoder: bool
    retrieval_mode: str = "fused"
    beta_warmup_epochs: int = 0


@dataclass(frozen=True)
class DatasetBundle:
    train: ReIDImageDataset
    query: ReIDImageDataset
    gallery: ReIDImageDataset
    num_train_ids: int
    num_cameras: int


@dataclass(frozen=True)
class SplitSamples:
    train: Sequence[ReIDSample]
    query: Sequence[ReIDSample]
    gallery: Sequence[ReIDSample]


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    query: DataLoader
    gallery: DataLoader


@dataclass(frozen=True)
class StageTrainingRuntime:
    model: "CLIPReIDTrainingModel"
    loaders: LoaderBundle
    optimizer: torch.optim.Optimizer
    stage: str
    loss_config: Any
    device: torch.device
    beta_schedule: "BetaSchedule | None" = None


@dataclass(frozen=True)
class ValidationRuntime:
    model: "CLIPReIDTrainingModel"
    loaders: LoaderBundle
    device: torch.device
    retrieval_mode: str
    beta_schedule: "BetaSchedule | None" = None


@dataclass(frozen=True)
class BetaSchedule:
    """Linear ramp of the fused retrieval beta from 0 to ``beta`` over ``warmup_epochs`` Stage-2 epochs.

    The fused text feature ``f_t_eval`` carries only global+camera signal — no identity — so
    blended into ``f_eval`` early on, before the image backbone has learned anything
    discriminative, it actively pushes samples toward their camera cluster and *lowers*
    mAP below the image-only floor. The warmup lets the image encoder learn discriminative
    features first, then blends the camera-conditioned text in once the image stream is
    stable. At ``epoch == 1`` the effective beta is ``0`` (pure image feature); at
    ``epoch == warmup_epochs + 1`` (and every epoch after) the effective beta is ``beta``.
    """

    beta: float
    warmup_epochs: int

    def effective_beta(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return self.beta
        if epoch <= 1:
            return 0.0
        return self.beta * min(1.0, (epoch - 1) / self.warmup_epochs)

    def apply(self, model: "CLIPReIDTrainingModel", epoch: int) -> None:
        model.retrieval_model.beta = self.effective_beta(epoch)


class CLIPReIDTrainingModel(torch.nn.Module):
    def __init__(
        self,
        retrieval_model: T2CClipModel,
        classifier: torch.nn.Module,
        tfc_bank: TFCCenterBank,
    ):
        super().__init__()
        self.retrieval_model = retrieval_model
        self.classifier = classifier
        self.tfc_bank = tfc_bank


def build_training_job(
    args: Any,
    clip_loader: ClipLoader = lambda model_name: load_transformers_clip(model_name),
) -> TwoStageTrainingJob | TrainingJob:
    config = _job_config_from_args(args)
    loaded_clip = clip_loader(config.clip_model_name)
    transform = CLIPImageTransform(loaded_clip.image_processor)
    data = load_dataset_bundle(JobDataConfig(config.dataset, config.data_root), transform)
    shared_model = _build_training_model(config, loaded_clip.model, data).to(config.device)
    loaders = _build_loaders(data, config)
    stage1_runtime, stage2_runtime, optimizer_stage1, optimizer_stage2, stage2_beta_schedule = _build_runtimes(
        config, shared_model, loaders
    )
    stage2_job = TrainingJob(
        model=shared_model,
        optimizer=optimizer_stage2,
        train_one_epoch=_train_one_epoch(stage2_runtime),
        validate=_validate(ValidationRuntime(
            shared_model,
            loaders,
            config.device,
            config.retrieval_mode,
            beta_schedule=stage2_beta_schedule,
        )),
    )
    if config.stage1_epochs <= 0:
        return stage2_job
    stage1_job = TrainingJob(
        model=shared_model,
        optimizer=optimizer_stage1,
        train_one_epoch=_train_one_epoch(stage1_runtime),
        validate=_noop_validate(),
    )
    return TwoStageTrainingJob(
        stage1=stage1_job,
        stage2=stage2_job,
        stage_metadata=_stage_metadata(config),
    )


def load_transformers_clip(model_name: str) -> CLIPLoadResult:
    try:
        from transformers import AutoTokenizer, CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError("transformers is required for the CLIP ReID training job") from exc
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        # Tokenizer is only used for sot/eos lookup. Fall back to known CLIP token ids.
        tokenizer = None
    image_processor = getattr(processor, "image_processor", processor)
    return CLIPLoadResult(model, image_processor, tokenizer)


def load_dataset_bundle(config: JobDataConfig, transform) -> DatasetBundle:
    splits = _load_split_samples(config)
    _require_non_empty_splits(splits)
    camera_map = build_camera_id_map([*splits.train, *splits.query, *splits.gallery])
    train_person_map = build_person_id_map(splits.train)
    eval_person_map = build_person_id_map([*splits.query, *splits.gallery])
    return DatasetBundle(
        train=ReIDImageDataset(ReIDImageDatasetConfig(splits.train, train_person_map, camera_map, transform)),
        query=ReIDImageDataset(ReIDImageDatasetConfig(splits.query, eval_person_map, camera_map, transform)),
        gallery=ReIDImageDataset(ReIDImageDatasetConfig(splits.gallery, eval_person_map, camera_map, transform)),
        num_train_ids=len(train_person_map),
        num_cameras=len(camera_map),
    )


def _job_config_from_args(args: Any) -> CLIPReIDJobConfig:
    if args.dataset is None:
        raise ValueError("--dataset is required for t2c_clip.jobs.clip_reid")
    if args.data_root is None:
        raise ValueError("--data-root is required for t2c_clip.jobs.clip_reid")
    return CLIPReIDJobConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        clip_model_name=args.clip_model_name,
        batch_size=int(getattr(args, "batch_size", 64)),
        num_workers=int(getattr(args, "num_workers", 4)),
        lr=float(getattr(args, "lr", 1e-4)),
        image_encoder_lr=float(getattr(args, "image_encoder_lr", DEFAULT_IMAGE_ENCODER_LR)),
        device=torch.device(args.device),
        beta=float(args.beta),
        context_length=int(args.context_length),
        tfc_momentum=float(args.tfc_momentum),
        triplet_margin=float(args.triplet_margin),
        tfc_weight=float(args.tfc_weight),
        clip_weight=float(getattr(args, "clip_weight", 0.1)),
        stage1_epochs=int(getattr(args, "stage1_epochs", 0)),
        stage2_epochs=int(getattr(args, "epochs", 120)),
        validation_interval=int(getattr(args, "validation_interval", 5)),
        freeze_image_encoder_stage1=bool(getattr(args, "freeze_image_encoder_stage1", True)),
        freeze_image_encoder_stage2=bool(getattr(args, "freeze_image_encoder_stage2", False)),
        freeze_text_encoder=bool(getattr(args, "freeze_text_encoder", True)),
        retrieval_mode=require_retrieval_mode(str(getattr(args, "retrieval_mode", "fused"))),
        beta_warmup_epochs=int(getattr(args, "beta_warmup_epochs", 0)),
    )


def _build_runtimes(
    config: CLIPReIDJobConfig,
    model: CLIPReIDTrainingModel,
    loaders: LoaderBundle,
) -> tuple[
    StageTrainingRuntime,
    StageTrainingRuntime,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    BetaSchedule | None,
]:
    _apply_freezing(model, config, stage=STAGE1)
    optimizer_stage1 = _build_optimizer(model, config)
    stage1_runtime = StageTrainingRuntime(
        model=model, loaders=loaders, optimizer=optimizer_stage1, stage=STAGE1,
        loss_config=Stage1LossConfig(), device=config.device,
    )
    _apply_freezing(model, config, stage=STAGE2)
    optimizer_stage2 = _build_optimizer(model, config)
    stage2_loss_config = Stage2LossConfig(
        triplet_margin=config.triplet_margin,
        tfc_weight=config.tfc_weight,
        clip_weight=config.clip_weight,
    )
    stage2_beta_schedule = BetaSchedule(beta=config.beta, warmup_epochs=config.beta_warmup_epochs)
    stage2_runtime = StageTrainingRuntime(
        model=model, loaders=loaders, optimizer=optimizer_stage2, stage=STAGE2,
        loss_config=stage2_loss_config, device=config.device,
        beta_schedule=stage2_beta_schedule,
    )
    return stage1_runtime, stage2_runtime, optimizer_stage1, optimizer_stage2, stage2_beta_schedule


def _apply_freezing(model: CLIPReIDTrainingModel, config: CLIPReIDJobConfig, stage: str) -> None:
    retrieval = model.retrieval_model
    clip_model = _clip_model_for(retrieval)
    image_trainable = _image_encoder_trainable(config, stage)
    text_trainable = not config.freeze_text_encoder
    _set_module_requires_grad(clip_model.vision_model, image_trainable)
    _set_module_requires_grad(clip_model.visual_projection, image_trainable)
    _set_module_requires_grad(clip_model.text_model, text_trainable)
    _set_module_requires_grad(clip_model.text_projection, text_trainable)
    retrieval.prompt_bank.requires_grad_(True)
    model.classifier.requires_grad_(stage == STAGE2)


def _image_encoder_trainable(config: CLIPReIDJobConfig, stage: str) -> bool:
    if stage == STAGE1:
        return not config.freeze_image_encoder_stage1
    if stage == STAGE2:
        return not config.freeze_image_encoder_stage2
    raise ValueError(f"unknown training stage: {stage!r}")


def _clip_model_for(retrieval_model: T2CClipModel) -> torch.nn.Module:
    if not isinstance(retrieval_model.image_encoder, TransformersCLIPImageEncoder):
        raise TypeError("CLIP freezing requires a TransformersCLIPImageEncoder-backed retrieval model")
    return retrieval_model.image_encoder.clip_model


def _set_module_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


# Parameter-name prefixes whose owner is the CLIP vision backbone. These receive the
# smaller image-encoder learning rate so the pretrained visual features are tuned, not
# catastrophically forgotten, when Stage-2 unfreezes the image encoder.
BACKBONE_PARAMETER_PREFIXES = (
    "retrieval_model.image_encoder.clip_model.vision_model.",
    "retrieval_model.image_encoder.clip_model.visual_projection.",
)


def _build_optimizer(model: torch.nn.Module, config: CLIPReIDJobConfig) -> torch.optim.Optimizer:
    backbone_params: list[torch.nn.Parameter] = []
    new_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(BACKBONE_PARAMETER_PREFIXES):
            backbone_params.append(parameter)
        else:
            new_params.append(parameter)
    if not backbone_params and not new_params:
        raise ValueError(
            "no trainable parameters were found for the requested stage; "
            "enable at least one of the prompt_bank/classifier/text_encoder"
        )
    param_groups: list[dict[str, Any]] = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": config.image_encoder_lr, "name": "backbone"})
    if new_params:
        param_groups.append({"params": new_params, "lr": config.lr, "name": "new"})
    return torch.optim.AdamW(param_groups)


def _stage_metadata(config: CLIPReIDJobConfig) -> StageMetadata:
    """Bundle two-stage config into the canonical ``StageMetadata`` container.

    Returning a ``StageMetadata`` (rather than a raw dict) keeps
    ``TwoStageTrainingJob.stage_metadata`` typing consistent and stops
    ``train.py`` from mistaking ``dict.values`` (a bound method) for a
    mapping when it logs stage params to MLflow.
    """
    return StageMetadata(
        values={
            "dataset": config.dataset,
            "clip_model_name": config.clip_model_name,
            "stage1_epochs": config.stage1_epochs,
            "stage2_epochs": config.stage2_epochs,
            "validation_interval": config.validation_interval,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "lr": config.lr,
            "image_encoder_lr": config.image_encoder_lr,
            "beta": config.beta,
            "beta_warmup_epochs": config.beta_warmup_epochs,
            "clip_weight": config.clip_weight,
            "tfc_weight": config.tfc_weight,
            "triplet_margin": config.triplet_margin,
            "tfc_momentum": config.tfc_momentum,
            "context_length": config.context_length,
            "freeze_image_encoder_stage1": config.freeze_image_encoder_stage1,
            "freeze_image_encoder_stage2": config.freeze_image_encoder_stage2,
            "freeze_text_encoder": config.freeze_text_encoder,
            "retrieval_mode": config.retrieval_mode,
        }
    )


def _load_split_samples(config: JobDataConfig) -> SplitSamples:
    if not config.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {config.root}")
    if config.dataset == "market1501":
        return SplitSamples(
            train=load_market_split(config.root, "train"),
            query=load_market_split(config.root, "query"),
            gallery=load_market_split(config.root, "gallery"),
        )
    if config.dataset == "msmt17":
        return SplitSamples(
            train=load_msmt17_manifest(config.root, "train"),
            query=load_msmt17_manifest(config.root, "query"),
            gallery=load_msmt17_manifest(config.root, "gallery"),
        )
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def _require_non_empty_splits(splits: SplitSamples) -> None:
    if not splits.train:
        raise ValueError("training split is empty")
    if not splits.query:
        raise ValueError("query split is empty")
    if not splits.gallery:
        raise ValueError("gallery split is empty")


def _build_training_model(
    config: CLIPReIDJobConfig,
    clip_model: torch.nn.Module,
    data: DatasetBundle,
) -> CLIPReIDTrainingModel:
    text_hidden_dim = clip_text_hidden_dim(clip_model)
    projection_dim = clip_projection_dim(clip_model)
    sot_id, eos_id, pad_id = _resolve_clip_token_ids(clip_model, config)
    prompt_bank = PromptBank(
        PromptConfig(
            num_cameras=data.num_cameras,
            num_train_ids=data.num_train_ids,
            context_length=config.context_length,
            embedding_dim=text_hidden_dim,
        )
    )
    image_encoder = TransformersCLIPImageEncoder(clip_model)
    text_encoder = TransformersCLIPTextEncoder(
        clip_model,
        context_length=config.context_length,
        sot_token_id=sot_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    retrieval = T2CClipModel(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        prompt_bank=prompt_bank,
        beta=config.beta,
    )
    classifier = torch.nn.Linear(projection_dim, data.num_train_ids)
    tfc_bank = TFCCenterBank(data.num_train_ids, projection_dim, config.tfc_momentum)
    return CLIPReIDTrainingModel(retrieval, classifier, tfc_bank)


def _resolve_clip_token_ids(clip_model: torch.nn.Module, config: CLIPReIDJobConfig) -> tuple[int, int, int]:
    config_obj = getattr(clip_model, "config", None)
    text_config = getattr(config_obj, "text_config", None)
    bos = getattr(text_config, "bos_token_id", None)
    eos = getattr(text_config, "eos_token_id", None)
    pad = getattr(text_config, "pad_token_id", None)
    sot = bos if isinstance(bos, int) else DEFAULT_CLIP_TOKEN_IDS["sot"]
    eos_id = eos if isinstance(eos, int) else DEFAULT_CLIP_TOKEN_IDS["eos"]
    pad_id = pad if isinstance(pad, int) else DEFAULT_CLIP_TOKEN_IDS["pad"]
    if eos_id == bos:  # Some CLIP configs reuse BOS as EOS (eos_token_id == 2 historically).
        eos_id = DEFAULT_CLIP_TOKEN_IDS["eos"]
    return sot, eos_id, pad_id


def _build_loaders(data: DatasetBundle, config: CLIPReIDJobConfig) -> LoaderBundle:
    return LoaderBundle(
        train=_train_loader(data.train, config),
        query=_loader(data.query, config, shuffle=False),
        gallery=_loader(data.gallery, config, shuffle=False),
    )


def _train_loader(dataset: ReIDImageDataset, config: CLIPReIDJobConfig) -> DataLoader:
    sampler = IdentityBalancedBatchSampler(dataset.person_ids, batch_size=config.batch_size)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_reid_batches,
    )


def _loader(dataset: ReIDImageDataset, config: CLIPReIDJobConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_reid_batches,
    )


def _train_one_epoch(runtime: StageTrainingRuntime):
    def train(epoch: int, reporter) -> dict[str, float]:
        if runtime.beta_schedule is not None:
            runtime.beta_schedule.apply(runtime.model, epoch)
        runtime.model.train()
        metric_names = _train_metric_names(runtime.stage)
        totals = {name: 0.0 for name in metric_names}
        batch_count = 0
        for batch in reporter.batches(runtime.loaders.train):
            values = _train_batch(runtime, batch, runtime.stage)
            reporter.report_batch(values)
            totals = {name: totals[name] + values[name] for name in metric_names}
            batch_count += 1
        return _average_train_metrics(totals, batch_count, runtime.optimizer, runtime.stage)

    return train


def _noop_validate():
    def validate(epoch: int) -> ReIDMetrics:
        # Stage-1 has no ReID classifier trained yet; reporting random mAP would be misleading.
        return ReIDMetrics(map=float("nan"), cmc={rank: float("nan") for rank in DEFAULT_RANKS})

    return validate


def _validate(runtime: ValidationRuntime):
    def validate(epoch: int) -> ReIDMetrics:
        if runtime.beta_schedule is not None:
            runtime.beta_schedule.apply(runtime.model, epoch)
        runtime.model.eval()
        query = _extract_features(
            runtime.model.retrieval_model,
            runtime.loaders.query,
            runtime.device,
            runtime.retrieval_mode,
        )
        gallery = _extract_features(
            runtime.model.retrieval_model,
            runtime.loaders.gallery,
            runtime.device,
            runtime.retrieval_mode,
        )
        return evaluate_reid(
            query.features,
            gallery.features,
            query.person_ids,
            gallery.person_ids,
            query.camera_ids,
            gallery.camera_ids,
            ranks=DEFAULT_RANKS,
        )

    return validate


@dataclass(frozen=True)
class FeatureSet:
    features: torch.Tensor
    person_ids: tuple[int, ...]
    camera_ids: tuple[int, ...]


def _extract_features(
    model: T2CClipModel,
    loader: DataLoader,
    device: torch.device,
    retrieval_mode: str,
) -> FeatureSet:
    feature_parts: list[torch.Tensor] = []
    person_ids: list[int] = []
    camera_ids: list[int] = []
    with torch.no_grad():
        for batch in loader:
            images = batch.images.to(device)
            cameras = batch.camera_ids.to(device)
            features = model.encode_retrieval(images, cameras, retrieval_mode=retrieval_mode)
            feature_parts.append(features.cpu())
            person_ids.extend(batch.original_person_ids)
            camera_ids.extend(batch.original_camera_ids)
    if not feature_parts:
        raise ValueError("eval loader produced no samples; cannot extract query/gallery features")
    return FeatureSet(torch.cat(feature_parts), tuple(person_ids), tuple(camera_ids))


def _training_batch(batch: ReIDImageBatch, device: torch.device) -> TrainingBatch:
    return TrainingBatch(
        images=batch.images.to(device),
        camera_ids=batch.camera_ids.to(device),
        person_ids=batch.person_ids.to(device),
    )


def _train_batch(runtime: StageTrainingRuntime, batch: ReIDImageBatch, stage: str) -> dict[str, float]:
    training_batch = _training_batch(batch, runtime.device)
    runtime.optimizer.zero_grad()
    if stage == STAGE1:
        breakdown = _stage1_step(runtime, training_batch)
        values = _stage1_metric_values(breakdown)
    else:
        breakdown = _stage2_step(runtime, training_batch)
        values = _stage2_metric_values(breakdown)
    breakdown.total.backward()
    runtime.optimizer.step()
    values["lr"] = _optimizer_lr(runtime.optimizer)
    return values


def _stage1_step(runtime: StageTrainingRuntime, batch: TrainingBatch) -> Stage1LossBreakdown:
    return stage1_alignment_loss(runtime.model.retrieval_model, batch, runtime.loss_config)


def _stage2_step(runtime: StageTrainingRuntime, batch: TrainingBatch) -> Stage2LossBreakdown:
    _update_tfc_centers(runtime.model, batch)
    inputs = Stage2LossInputs(
        classifier=runtime.model.classifier,
        tfc_bank=runtime.model.tfc_bank,
        config=runtime.loss_config,
    )
    return stage2_loss_breakdown(runtime.model.retrieval_model, batch, inputs)


def _train_metric_names(stage: str) -> tuple[str, ...]:
    if stage == STAGE1:
        return STAGE1_TRAIN_LOSS_METRIC_NAMES
    return STAGE2_TRAIN_LOSS_METRIC_NAMES


def _stage1_metric_values(breakdown: Stage1LossBreakdown) -> dict[str, float]:
    return {
        "loss": _tensor_metric_value(breakdown.total),
        "clip_loss": _tensor_metric_value(breakdown.clip_dual),
    }


def _stage2_metric_values(breakdown: Stage2LossBreakdown) -> dict[str, float]:
    return {
        "loss": _tensor_metric_value(breakdown.total),
        "reid_loss": _tensor_metric_value(breakdown.identity),
        "triplet_loss": _tensor_metric_value(breakdown.triplet),
        "clip_loss": _tensor_metric_value(breakdown.clip_dual),
        "tfc_loss": _tensor_metric_value(breakdown.tfc),
    }


def _average_train_metrics(
    totals: dict[str, float],
    batch_count: int,
    optimizer: torch.optim.Optimizer,
    stage: str,
) -> dict[str, float]:
    if batch_count < 1:
        raise ValueError(f"{stage} training loader produced no batches")
    averaged = {name: totals[name] / batch_count for name in totals}
    averaged["lr"] = _optimizer_lr(optimizer)
    return averaged


def _tensor_metric_value(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        raise ValueError("optimizer has no parameter groups")
    return float(optimizer.param_groups[0]["lr"])


def _update_tfc_centers(model: CLIPReIDTrainingModel, batch: TrainingBatch) -> None:
    with torch.no_grad():
        outputs = model.retrieval_model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)
        model.tfc_bank.update(outputs["retrieval"], batch.person_ids)
