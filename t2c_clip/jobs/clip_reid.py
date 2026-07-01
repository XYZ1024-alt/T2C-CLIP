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

import math

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
    DEFAULT_INSTANCES_PER_IDENTITY,
    IdentityBalancedBatchSampler,
    ReIDImageBatch,
    ReIDImageDataset,
    ReIDImageDatasetConfig,
    build_camera_id_map,
    build_person_id_map,
    collate_reid_batches,
)
from t2c_clip.evaluation import ReIDMetrics, evaluate_reid, evaluate_reid_with_rerank
from t2c_clip.model import T2CClipModel
from t2c_clip.prompts import PromptBank, PromptConfig
from t2c_clip.retrieval import FUSED_RETRIEVAL, require_retrieval_mode
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
    clip_checkpoint: Path | None
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
    id_logit_scale: float
    label_smoothing: float
    stage1_epochs: int
    stage2_epochs: int
    validation_interval: int
    freeze_image_encoder_stage1: bool
    freeze_image_encoder_stage2: bool
    freeze_text_encoder: bool
    stage2_first_epoch: int = 1
    freeze_prompt_bank_stage2: bool = False
    reid_head: str = "linear"
    retrieval_mode: str = "fused"
    beta_warmup_epochs: int = 0
    report_rerank: bool = False
    stage2_lr_scheduler: str = "none"
    stage2_warmup_epochs: int = 0
    num_instances: int = DEFAULT_INSTANCES_PER_IDENTITY


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
    freeze_config: "CLIPReIDJobConfig | None" = None
    lr_scheduler: "StageLRScheduler | None" = None


@dataclass(frozen=True)
class ValidationRuntime:
    model: "CLIPReIDTrainingModel"
    loaders: LoaderBundle
    device: torch.device
    retrieval_mode: str
    model_config: "CLIPReIDJobConfig"
    beta_schedule: "BetaSchedule | None" = None
    report_rerank: bool = False


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
    first_epoch: int = 1

    def effective_beta(self, stage_epoch: int) -> float:
        if stage_epoch < 1:
            raise ValueError("stage_epoch must be positive")
        if self.warmup_epochs <= 0:
            return self.beta
        if stage_epoch <= 1:
            return 0.0
        return self.beta * min(1.0, (stage_epoch - 1) / self.warmup_epochs)

    def apply(self, model: "CLIPReIDTrainingModel", epoch: int) -> None:
        stage_epoch = epoch - self.first_epoch + 1
        model.retrieval_model.beta = self.effective_beta(stage_epoch)


@dataclass(frozen=True)
class StageLRScheduler:
    """Warmup + cosine-decay of Stage-2 learning rates over Stage-2-local epochs.

    Linear warmup from ``base_lr / warmup_epochs`` at stage epoch 1 up to the full
    ``base_lr`` at ``warmup_epochs``, then a cosine decay toward ~0 by ``total_epochs``.
    Every param group is scaled by the same factor so the grouped backbone/new
    learning rates keep their ratio. ``warmup_epochs == 0`` disables warmup and
    applies pure cosine decay from stage epoch 1.
    """

    base_lrs: tuple[float, ...]
    total_epochs: int
    warmup_epochs: int
    first_epoch: int = 1

    def scale(self, stage_epoch: int) -> float:
        if stage_epoch < 1:
            raise ValueError("stage_epoch must be positive")
        if self.warmup_epochs > 0 and stage_epoch <= self.warmup_epochs:
            return stage_epoch / self.warmup_epochs
        decay_start = self.warmup_epochs + 1
        decay_epochs = max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, (stage_epoch - decay_start) / decay_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def apply(self, optimizer: torch.optim.Optimizer, epoch: int) -> None:
        stage_epoch = epoch - self.first_epoch + 1
        factor = self.scale(stage_epoch)
        for group, base_lr in zip(optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor


class CLIPReIDTrainingModel(torch.nn.Module):
    def __init__(
        self,
        retrieval_model: T2CClipModel,
        classifier: torch.nn.Module,
        tfc_bank: TFCCenterBank,
        *,
        feature_head: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.retrieval_model = retrieval_model
        self.classifier = classifier
        self.tfc_bank = tfc_bank
        self.feature_head = torch.nn.Identity() if feature_head is None else feature_head

    def encode_retrieval(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        retrieval_mode: str = FUSED_RETRIEVAL,
    ) -> torch.Tensor:
        """Validation / inference retrieval feature.

        Route the base retrieval feature through the same ``feature_head`` the
        Stage-2 ID classifier is trained on, so retrieval uses the BNNeck-normalized
        feature rather than the raw pre-head feature. For the default ``linear``
        head (``Identity``) this is unchanged; ``evaluate_reid`` L2-normalizes the
        result before scoring either way.
        """
        features = self.retrieval_model.encode_retrieval(
            images, camera_ids, retrieval_mode=retrieval_mode
        )
        return self.feature_head(features)


class BNNeck(torch.nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(feature_dim)
        self.freeze_bias()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.bn(features)

    def freeze_bias(self) -> None:
        self.bn.bias.requires_grad_(False)


def build_training_job(
    args: Any,
    clip_loader: ClipLoader = lambda model_name: load_transformers_clip(model_name),
) -> TwoStageTrainingJob | TrainingJob:
    config = _job_config_from_args(args)
    loaded_clip = clip_loader(config.clip_model_name)
    _load_clip_checkpoint_if_requested(loaded_clip.model, config.clip_checkpoint, config.device)
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
            config,
            beta_schedule=stage2_beta_schedule,
            report_rerank=config.report_rerank,
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
        clip_checkpoint=getattr(args, "clip_checkpoint", None),
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
        id_logit_scale=float(getattr(args, "id_logit_scale", 1.0)),
        label_smoothing=float(getattr(args, "label_smoothing", 0.0)),
        stage1_epochs=int(getattr(args, "stage1_epochs", 0)),
        stage2_epochs=int(getattr(args, "epochs", 120)),
        validation_interval=int(getattr(args, "validation_interval", 5)),
        freeze_image_encoder_stage1=bool(getattr(args, "freeze_image_encoder_stage1", True)),
        freeze_image_encoder_stage2=bool(getattr(args, "freeze_image_encoder_stage2", False)),
        freeze_text_encoder=bool(getattr(args, "freeze_text_encoder", True)),
        stage2_first_epoch=int(getattr(args, "stage2_first_epoch", int(getattr(args, "stage1_epochs", 0)) + 1)),
        freeze_prompt_bank_stage2=bool(getattr(args, "freeze_prompt_bank_stage2", False)),
        reid_head=str(getattr(args, "reid_head", "linear")),
        retrieval_mode=require_retrieval_mode(str(getattr(args, "retrieval_mode", "fused"))),
        beta_warmup_epochs=int(getattr(args, "beta_warmup_epochs", 0)),
        report_rerank=bool(getattr(args, "report_rerank", False)),
        stage2_lr_scheduler=str(getattr(args, "stage2_lr_scheduler", "none")),
        stage2_warmup_epochs=int(getattr(args, "stage2_warmup_epochs", 0)),
        num_instances=int(getattr(args, "num_instances", DEFAULT_INSTANCES_PER_IDENTITY)),
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
        freeze_config=config,
    )
    _apply_freezing(model, config, stage=STAGE2)
    optimizer_stage2 = _build_optimizer(model, config)
    stage2_loss_config = Stage2LossConfig(
        triplet_margin=config.triplet_margin,
        tfc_weight=config.tfc_weight,
        clip_weight=config.clip_weight,
        id_logit_scale=config.id_logit_scale,
        label_smoothing=config.label_smoothing,
    )
    stage2_beta_schedule = BetaSchedule(
        beta=config.beta,
        warmup_epochs=config.beta_warmup_epochs,
        first_epoch=config.stage2_first_epoch,
    )
    stage2_lr_scheduler = _build_stage2_lr_scheduler(optimizer_stage2, config)
    stage2_runtime = StageTrainingRuntime(
        model=model, loaders=loaders, optimizer=optimizer_stage2, stage=STAGE2,
        loss_config=stage2_loss_config, device=config.device,
        beta_schedule=stage2_beta_schedule,
        freeze_config=config,
        lr_scheduler=stage2_lr_scheduler,
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
    prompt_trainable = stage == STAGE1 or not config.freeze_prompt_bank_stage2
    retrieval.prompt_bank.requires_grad_(prompt_trainable)
    model.classifier.requires_grad_(stage == STAGE2)
    model.feature_head.requires_grad_(stage == STAGE2)
    if isinstance(model.feature_head, BNNeck):
        model.feature_head.freeze_bias()


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


def _build_stage2_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: CLIPReIDJobConfig,
) -> "StageLRScheduler | None":
    if config.stage2_lr_scheduler == "none":
        return None
    if config.stage2_lr_scheduler != "cosine":
        raise ValueError(f"unsupported stage2_lr_scheduler: {config.stage2_lr_scheduler!r}")
    return StageLRScheduler(
        base_lrs=tuple(float(group["lr"]) for group in optimizer.param_groups),
        total_epochs=config.stage2_epochs,
        warmup_epochs=config.stage2_warmup_epochs,
        first_epoch=config.stage2_first_epoch,
    )


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
            "clip_checkpoint": str(config.clip_checkpoint) if config.clip_checkpoint is not None else None,
            "stage1_epochs": config.stage1_epochs,
            "stage2_epochs": config.stage2_epochs,
            "stage2_first_epoch": config.stage2_first_epoch,
            "validation_interval": config.validation_interval,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "lr": config.lr,
            "image_encoder_lr": config.image_encoder_lr,
            "beta": config.beta,
            "beta_warmup_epochs": config.beta_warmup_epochs,
            "clip_weight": config.clip_weight,
            "id_logit_scale": config.id_logit_scale,
            "label_smoothing": config.label_smoothing,
            "tfc_weight": config.tfc_weight,
            "triplet_margin": config.triplet_margin,
            "tfc_momentum": config.tfc_momentum,
            "context_length": config.context_length,
            "freeze_image_encoder_stage1": config.freeze_image_encoder_stage1,
            "freeze_image_encoder_stage2": config.freeze_image_encoder_stage2,
            "freeze_text_encoder": config.freeze_text_encoder,
            "freeze_prompt_bank_stage2": config.freeze_prompt_bank_stage2,
            "reid_head": config.reid_head,
            "retrieval_mode": config.retrieval_mode,
            "report_rerank": config.report_rerank,
            "stage2_lr_scheduler": config.stage2_lr_scheduler,
            "stage2_warmup_epochs": config.stage2_warmup_epochs,
            "num_instances": config.num_instances,
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
    feature_head = _build_feature_head(config.reid_head, projection_dim)
    return CLIPReIDTrainingModel(retrieval, classifier, tfc_bank, feature_head=feature_head)


def _load_clip_checkpoint_if_requested(
    model: torch.nn.Module,
    checkpoint: Path | None,
    device: torch.device,
) -> None:
    if checkpoint is None:
        return
    if not checkpoint.exists():
        raise FileNotFoundError(f"CLIP checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError("CLIP checkpoint must be a state_dict or contain a state_dict key")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"unexpected CLIP checkpoint keys: {unexpected}")
    if missing:
        raise ValueError(f"missing CLIP checkpoint keys: {missing}")


def _build_feature_head(reid_head: str, projection_dim: int) -> torch.nn.Module:
    if reid_head == "linear":
        return torch.nn.Identity()
    if reid_head == "bnneck":
        return BNNeck(projection_dim)
    raise ValueError(f"unsupported reid_head: {reid_head!r}")


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
    sampler = IdentityBalancedBatchSampler(
        dataset.person_ids,
        batch_size=config.batch_size,
        instances_per_identity=config.num_instances,
    )
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
        if runtime.freeze_config is not None:
            _apply_freezing(runtime.model, runtime.freeze_config, runtime.stage)
        if runtime.beta_schedule is not None:
            runtime.beta_schedule.apply(runtime.model, epoch)
        if runtime.lr_scheduler is not None:
            runtime.lr_scheduler.apply(runtime.optimizer, epoch)
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
        _apply_freezing(runtime.model, runtime.model_config, STAGE2)
        if runtime.beta_schedule is not None:
            runtime.beta_schedule.apply(runtime.model, epoch)
        runtime.model.eval()
        query = _extract_features(
            runtime.model,
            runtime.loaders.query,
            runtime.device,
            runtime.retrieval_mode,
        )
        gallery = _extract_features(
            runtime.model,
            runtime.loaders.gallery,
            runtime.device,
            runtime.retrieval_mode,
        )
        metrics = evaluate_reid(
            query.features,
            gallery.features,
            query_ids=query.person_ids,
            gallery_ids=gallery.person_ids,
            query_cams=query.camera_ids,
            gallery_cams=gallery.camera_ids,
            ranks=DEFAULT_RANKS,
        )
        if not runtime.report_rerank:
            return metrics
        rerank = evaluate_reid_with_rerank(
            query.features,
            gallery.features,
            query_ids=query.person_ids,
            gallery_ids=gallery.person_ids,
            query_cams=query.camera_ids,
            gallery_cams=gallery.camera_ids,
            ranks=DEFAULT_RANKS,
        )
        return ReIDMetrics(
            map=metrics.map,
            cmc=metrics.cmc,
            extras={
                "rerank_mAP": rerank.map,
                "rerank_rank_1": rerank.cmc[1],
            },
        )

    return validate


@dataclass(frozen=True)
class FeatureSet:
    features: torch.Tensor
    person_ids: tuple[int, ...]
    camera_ids: tuple[int, ...]


def _extract_features(
    model: CLIPReIDTrainingModel,
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
        feature_head=runtime.model.feature_head,
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
