"""Loss functions used by the T2C-CLIP training design."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from t2c_clip.features import l2_normalize

DEFAULT_LOGIT_SCALE = 1.0
DEFAULT_MARGIN = 0.3
DEFAULT_TRIPLET_METRIC = "euclidean"


def bidirectional_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> torch.Tensor:
    if image_features.shape != text_features.shape:
        raise ValueError("image_features and text_features must have identical shapes")
    image = l2_normalize(image_features)
    text = l2_normalize(text_features)
    logits = logit_scale * image @ text.T
    targets = torch.arange(logits.shape[0], device=logits.device)
    image_loss = F.cross_entropy(logits, targets)
    text_loss = F.cross_entropy(logits.T, targets)
    return (image_loss + text_loss) / 2.0


def supervised_bidirectional_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    person_ids: torch.Tensor,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> torch.Tensor:
    """CLIP-ReID-style multi-positive bidirectional contrastive loss.

    Identity-balanced PK batches always contain several samples of the same
    person. The plain ``targets=arange`` InfoNCE treats those rows as
    negatives and actively pushes same-person pairs apart; here every row
    sharing a person id is a positive, and the per-anchor loss averages the
    log-probabilities over all positives (SupCon "out" formulation). With
    unique ids per batch this reduces exactly to
    :func:`bidirectional_contrastive_loss`.
    """
    if image_features.shape != text_features.shape:
        raise ValueError("image_features and text_features must have identical shapes")
    _validate_person_ids(person_ids, image_features.shape[0])
    image = l2_normalize(image_features)
    text = l2_normalize(text_features)
    logits = logit_scale * image @ text.T
    positives = (person_ids.unsqueeze(0) == person_ids.unsqueeze(1)).to(logits.dtype)
    image_loss = _multi_positive_cross_entropy(logits, positives)
    text_loss = _multi_positive_cross_entropy(logits.T, positives)
    return (image_loss + text_loss) / 2.0


def _multi_positive_cross_entropy(logits: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    positive_log_probs = (log_probs * positives).sum(dim=1) / positives.sum(dim=1)
    return -positive_log_probs.mean()


def _validate_person_ids(person_ids: torch.Tensor, batch_size: int) -> None:
    if person_ids.dtype != torch.long or person_ids.ndim != 1:
        raise ValueError("person_ids must be a rank-1 torch.long tensor")
    if person_ids.shape[0] != batch_size:
        raise ValueError("person_ids must match the contrastive batch size")


def batch_hard_triplet_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    margin: float = DEFAULT_MARGIN,
    metric: str = DEFAULT_TRIPLET_METRIC,
) -> torch.Tensor:
    _validate_triplet_inputs(features, labels)
    distances = _pairwise_distances(features, metric)
    losses = _valid_anchor_losses(distances, labels, margin)
    if not losses:
        raise ValueError("batch_hard_triplet_loss requires at least one valid positive and negative")
    return torch.stack(losses).mean()


def _pairwise_distances(features: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "euclidean":
        squared_norms = features.pow(2).sum(dim=1)
        squared = squared_norms.unsqueeze(1) + squared_norms.unsqueeze(0) - 2.0 * features @ features.T
        return squared.clamp_min(1e-12).sqrt()
    if metric == "cosine":
        return 1.0 - l2_normalize(features) @ l2_normalize(features).T
    raise ValueError(f"unsupported triplet metric: {metric!r}")


def _valid_anchor_losses(distances: torch.Tensor, labels: torch.Tensor, margin: float) -> list[torch.Tensor]:
    losses: list[torch.Tensor] = []
    for index in range(labels.shape[0]):
        positive_mask = labels == labels[index]
        negative_mask = labels != labels[index]
        positive_mask[index] = False
        if bool(torch.any(positive_mask)) and bool(torch.any(negative_mask)):
            hardest_positive = torch.max(distances[index][positive_mask])
            hardest_negative = torch.min(distances[index][negative_mask])
            losses.append(F.relu(hardest_positive - hardest_negative + margin))
    return losses


def _validate_triplet_inputs(features: torch.Tensor, labels: torch.Tensor) -> None:
    if features.ndim != 2:
        raise ValueError("features must be a rank-2 tensor")
    if labels.dtype != torch.long or labels.ndim != 1:
        raise ValueError("labels must be a rank-1 torch.long tensor")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same batch size")
