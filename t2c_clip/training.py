"""Stage-1 and Stage-2 loss wiring for T2C-CLIP.

Stage-1 only computes the bidirectional image-text contrastive loss between
``f_v`` and ``f_t_id`` (training identity prompt).

Stage-2 total loss is::

    L_total = L_id + L_triplet + clip_weight * L_i2t + tfc_weight * L_TFC

``L_i2t`` is CLIP-ReID's image-to-text cross entropy: the L2-normalized image
feature is classified against the (detached) identity anchor text matrix over
ALL train identities, not just the batch's.

``clip_weight`` is a configurable hyperparameter (default 0.1) — it must not
be hard-coded to 1.0, since an oversized CLIP alignment signal suppresses
ReID signals early in Stage-2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from t2c_clip.losses import (
    batch_hard_triplet_loss,
    image_to_text_cross_entropy,
    supervised_bidirectional_contrastive_loss,
)
from t2c_clip.model import T2CClipModel
from t2c_clip.tfc import TFCCenterBank

DEFAULT_LOGIT_SCALE = 1.0
DEFAULT_TRIPLET_MARGIN = 0.3
DEFAULT_TRIPLET_METRIC = "euclidean"
DEFAULT_TFC_WEIGHT = 1.0
DEFAULT_CLIP_WEIGHT = 0.1
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


@dataclass(frozen=True)
class Stage2LossConfig:
    logit_scale: float = DEFAULT_LOGIT_SCALE
    triplet_margin: float = DEFAULT_TRIPLET_MARGIN
    triplet_metric: str = DEFAULT_TRIPLET_METRIC
    tfc_weight: float = DEFAULT_TFC_WEIGHT
    clip_weight: float = DEFAULT_CLIP_WEIGHT
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
    clip_dual: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.clip_dual


@dataclass(frozen=True)
class Stage2LossBreakdown:
    i2t: torch.Tensor
    identity: torch.Tensor
    triplet: torch.Tensor
    tfc: torch.Tensor
    tfc_weight: float
    clip_weight: float

    @property
    def total(self) -> torch.Tensor:
        return (
            self.identity
            + self.triplet
            + self.clip_weight * self.i2t
            + self.tfc_weight * self.tfc
        )


def stage1_alignment_loss(
    model: T2CClipModel,
    batch: TrainingBatch,
    config: Stage1LossConfig,
) -> Stage1LossBreakdown:
    outputs = model.forward_stage1(batch.images, batch.camera_ids, batch.person_ids)
    clip_dual = supervised_bidirectional_contrastive_loss(
        outputs["visual"], outputs["text"], batch.person_ids, logit_scale=config.logit_scale
    )
    return Stage1LossBreakdown(clip_dual=clip_dual)


def stage2_loss_breakdown(
    model: T2CClipModel,
    batch: TrainingBatch,
    inputs: Stage2LossInputs,
) -> Stage2LossBreakdown:
    """Compute the Stage-2 losses from a single forward pass.

    CLIP-ReID-standard discriminative losses act on the image feature: the ID
    cross-entropy on the BNNeck output (``bn``), the triplet loss on the
    BN-pre feature (``visual_raw``). The image-to-text cross entropy
    classifies the L2-normalized image feature (``visual``, pre-BN — the
    space CLIP pretraining aligned with text) against ``inputs.anchors``, the
    detached identity anchor matrix over all train identities. The TFC loss
    keeps acting on the fused retrieval feature. TFC centers are updated
    (detached, no-grad) from this forward's retrieval feature before scoring,
    so no separate center-update forward is needed and the BNNeck running
    stats see each batch exactly once.
    """
    outputs = model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)
    retrieval = outputs["retrieval"]
    inputs.tfc_bank.update(retrieval, batch.person_ids)
    logits = inputs.classifier(outputs["bn"]) * inputs.config.id_logit_scale
    return Stage2LossBreakdown(
        i2t=image_to_text_cross_entropy(
            outputs["visual"],
            inputs.anchors,
            batch.person_ids,
            logit_scale=inputs.config.logit_scale,
            label_smoothing=inputs.config.label_smoothing,
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
        clip_weight=inputs.config.clip_weight,
    )
