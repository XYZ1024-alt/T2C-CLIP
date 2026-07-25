"""Stage-1 and Stage-2 loss wiring for T2C-ReID.

Stage-1 aligns image features with identity-aware prompt features using the
native all-pairs SigLIP sigmoid objective. Stage-2 keeps the ReID identity,
triplet, and TFC losses and adds the same SigLIP objective against the text
anchor of every training identity::

    L_total = L_id + L_triplet + alignment_weight * L_alignment + tfc_weight * L_TFC
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from t2c_reid.losses import (
    batch_hard_triplet_loss,
    siglip_identity_anchor_loss,
    supervised_siglip_loss,
)
from t2c_reid.model import T2CReIDModel
from t2c_reid.tfc import TFCCenterBank

DEFAULT_LOGIT_SCALE = 1.0
DEFAULT_LOGIT_BIAS = 0.0
DEFAULT_TRIPLET_MARGIN = 0.3
DEFAULT_TRIPLET_METRIC = "euclidean"
DEFAULT_TFC_WEIGHT = 1.0
DEFAULT_ALIGNMENT_WEIGHT = 1.0
DEFAULT_LABEL_SMOOTHING = 0.0
DEFAULT_ID_LOGIT_SCALE = 1.0


@dataclass(frozen=True)
class TrainingBatch:
    images: torch.Tensor
    camera_ids: torch.Tensor
    person_ids: torch.Tensor


@dataclass(frozen=True)
class Stage1LossConfig:
    logit_scale: float = DEFAULT_LOGIT_SCALE
    logit_bias: float = DEFAULT_LOGIT_BIAS


@dataclass(frozen=True)
class Stage2LossConfig:
    logit_scale: float = DEFAULT_LOGIT_SCALE
    logit_bias: float = DEFAULT_LOGIT_BIAS
    triplet_margin: float = DEFAULT_TRIPLET_MARGIN
    triplet_metric: str = DEFAULT_TRIPLET_METRIC
    tfc_weight: float = DEFAULT_TFC_WEIGHT
    alignment_weight: float = DEFAULT_ALIGNMENT_WEIGHT
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING
    id_logit_scale: float = DEFAULT_ID_LOGIT_SCALE


@dataclass(frozen=True)
class Stage2LossInputs:
    classifier: torch.nn.Module
    tfc_bank: TFCCenterBank
    anchors: torch.Tensor
    config: Stage2LossConfig = field(default_factory=Stage2LossConfig)


@dataclass(frozen=True)
class Stage1LossBreakdown:
    alignment: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.alignment


@dataclass(frozen=True)
class Stage2LossBreakdown:
    alignment: torch.Tensor
    identity: torch.Tensor
    triplet: torch.Tensor
    tfc: torch.Tensor
    tfc_weight: float
    alignment_weight: float

    @property
    def total(self) -> torch.Tensor:
        return (
            self.identity
            + self.triplet
            + self.alignment_weight * self.alignment
            + self.tfc_weight * self.tfc
        )


def stage1_alignment_loss(
    model: T2CReIDModel,
    batch: TrainingBatch,
    config: Stage1LossConfig,
) -> Stage1LossBreakdown:
    visual = model.encode_visual(batch.images, batch.camera_ids)
    return stage1_alignment_loss_from_visual(
        model,
        visual,
        camera_ids=batch.camera_ids,
        person_ids=batch.person_ids,
        config=config,
    )


def stage1_alignment_loss_from_visual(
    model: T2CReIDModel,
    visual: torch.Tensor,
    camera_ids: torch.Tensor,
    person_ids: torch.Tensor,
    config: Stage1LossConfig,
) -> Stage1LossBreakdown:
    """Stage-1 alignment from a precomputed normalized image feature."""

    text = model.encode_training_text(camera_ids, person_ids)
    alignment = supervised_siglip_loss(
        visual,
        text,
        person_ids,
        logit_scale=config.logit_scale,
        logit_bias=config.logit_bias,
    )
    return Stage1LossBreakdown(alignment=alignment)


def stage2_loss_breakdown(
    model: T2CReIDModel,
    batch: TrainingBatch,
    inputs: Stage2LossInputs,
) -> Stage2LossBreakdown:
    """Compute Stage-2 losses from one image forward.

    Label smoothing is deliberately limited to the ReID identity classifier;
    the alignment term retains SigLIP's native binary targets.
    """

    outputs = model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)
    retrieval = outputs["retrieval"]
    inputs.tfc_bank.update(retrieval, batch.person_ids)
    logits = inputs.classifier(outputs["bn"]) * inputs.config.id_logit_scale
    return Stage2LossBreakdown(
        alignment=siglip_identity_anchor_loss(
            outputs["visual"],
            inputs.anchors,
            batch.person_ids,
            logit_scale=inputs.config.logit_scale,
            logit_bias=inputs.config.logit_bias,
        ),
        identity=F.cross_entropy(
            logits,
            batch.person_ids,
            label_smoothing=inputs.config.label_smoothing,
        ),
        triplet=batch_hard_triplet_loss(
            outputs["visual_raw"],
            batch.person_ids,
            inputs.config.triplet_margin,
            metric=inputs.config.triplet_metric,
        ),
        tfc=inputs.tfc_bank.loss(retrieval, batch.person_ids),
        tfc_weight=inputs.config.tfc_weight,
        alignment_weight=inputs.config.alignment_weight,
    )
