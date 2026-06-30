"""No-rerank Image-to-Image ReID evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from t2c_clip.features import l2_normalize

DEFAULT_RANKS = (1, 5, 10)


@dataclass(frozen=True)
class ReIDMetrics:
    map: float
    cmc: dict[int, float]


def evaluate_reid(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_ids: Sequence[int],
    gallery_ids: Sequence[int],
    query_cams: Sequence[int],
    gallery_cams: Sequence[int],
    ranks: Sequence[int] = DEFAULT_RANKS,
) -> ReIDMetrics:
    _validate_inputs(query_features, gallery_features, query_ids, gallery_ids, query_cams, gallery_cams)
    similarity = l2_normalize(query_features) @ l2_normalize(gallery_features).T
    per_query = [
        _evaluate_query(similarity[index], query_ids[index], gallery_ids, query_cams[index], gallery_cams, ranks)
        for index in range(query_features.shape[0])
    ]
    return _mean_metrics(per_query, ranks)


def _evaluate_query(
    scores: torch.Tensor,
    query_id: int,
    gallery_ids: Sequence[int],
    query_cam: int,
    gallery_cams: Sequence[int],
    ranks: Sequence[int],
) -> tuple[float, dict[int, float]]:
    order = torch.argsort(scores, descending=True).tolist()
    matches = _ordered_matches(order, query_id, gallery_ids, query_cam, gallery_cams)
    return _average_precision(matches), _cmc(matches, ranks)


def _ordered_matches(
    order: list[int],
    query_id: int,
    gallery_ids: Sequence[int],
    query_cam: int,
    gallery_cams: Sequence[int],
) -> list[bool]:
    valid_matches: list[bool] = []
    for index in order:
        same_identity = gallery_ids[index] == query_id
        same_camera = gallery_cams[index] == query_cam
        if same_identity and same_camera:
            continue
        valid_matches.append(same_identity)
    return valid_matches


def _average_precision(matches: list[bool]) -> float:
    hit_count = 0
    precision_sum = 0.0
    for rank, is_match in enumerate(matches, start=1):
        if is_match:
            hit_count += 1
            precision_sum += hit_count / rank
    if hit_count == 0:
        return 0.0
    return precision_sum / hit_count


def _cmc(matches: list[bool], ranks: Sequence[int]) -> dict[int, float]:
    if True not in matches:
        return {rank: 0.0 for rank in ranks}
    first_match = matches.index(True)
    return {rank: float(first_match < rank) for rank in ranks}


def _mean_metrics(per_query: list[tuple[float, dict[int, float]]], ranks: Sequence[int]) -> ReIDMetrics:
    denominator = len(per_query)
    mean_ap = sum(ap for ap, _ in per_query) / denominator
    cmc = {rank: sum(row[rank] for _, row in per_query) / denominator for rank in ranks}
    return ReIDMetrics(map=mean_ap, cmc=cmc)


def _validate_inputs(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_ids: Sequence[int],
    gallery_ids: Sequence[int],
    query_cams: Sequence[int],
    gallery_cams: Sequence[int],
) -> None:
    if query_features.ndim != 2 or gallery_features.ndim != 2:
        raise ValueError("query_features and gallery_features must be rank-2 tensors")
    if query_features.shape[0] < 1:
        raise ValueError("query_features must contain at least one row")
    if gallery_features.shape[0] < 1:
        raise ValueError("gallery_features must contain at least one row")
    if query_features.shape[1] != gallery_features.shape[1]:
        raise ValueError("query and gallery feature dimensions must match")
    if len(query_ids) != query_features.shape[0] or len(query_cams) != query_features.shape[0]:
        raise ValueError("query metadata lengths must match query feature rows")
    if len(gallery_ids) != gallery_features.shape[0] or len(gallery_cams) != gallery_features.shape[0]:
        raise ValueError("gallery metadata lengths must match gallery feature rows")
