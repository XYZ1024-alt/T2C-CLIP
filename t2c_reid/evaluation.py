"""Image-to-Image ReID evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
import torch

from t2c_reid.native import native_extension as _native
from t2c_reid.features import l2_normalize

DEFAULT_RANKS = (1, 5, 10)
DEFAULT_RERANK_K1 = 20
DEFAULT_RERANK_K2 = 6
DEFAULT_RERANK_LAMBDA = 0.3
DEFAULT_QUERY_CHUNK_SIZE = 256
RUST_EVALUATION_BACKEND = "rust"
PYTHON_EVALUATION_BACKEND = "python"
SUPPORTED_EVALUATION_BACKENDS = (RUST_EVALUATION_BACKEND, PYTHON_EVALUATION_BACKEND)
RERANK_TIE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class RerankConfig:
    k1: int = DEFAULT_RERANK_K1
    k2: int = DEFAULT_RERANK_K2
    lambda_value: float = DEFAULT_RERANK_LAMBDA


@dataclass(frozen=True)
class ReIDMetrics:
    map: float
    cmc: dict[int, float]
    extras: dict[str, float] = field(default_factory=dict)


def evaluate_reid(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    *,
    query_ids: Sequence[int],
    gallery_ids: Sequence[int],
    query_cams: Sequence[int],
    gallery_cams: Sequence[int],
    ranks: Sequence[int] = DEFAULT_RANKS,
    backend: str = RUST_EVALUATION_BACKEND,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
) -> ReIDMetrics:
    _validate_inputs(query_features, gallery_features, query_ids, gallery_ids, query_cams, gallery_cams)
    normalized_ranks = _validate_evaluation_options(ranks, backend, query_chunk_size)
    if backend == PYTHON_EVALUATION_BACKEND:
        return _evaluate_reid_python(
            query_features,
            gallery_features,
            query_ids=query_ids,
            gallery_ids=gallery_ids,
            query_cams=query_cams,
            gallery_cams=gallery_cams,
            ranks=normalized_ranks,
        )
    return _evaluate_reid_rust(
        query_features,
        gallery_features,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        query_cams=query_cams,
        gallery_cams=gallery_cams,
        ranks=normalized_ranks,
        query_chunk_size=query_chunk_size,
    )


def _evaluate_reid_python(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    *,
    query_ids: Sequence[int],
    gallery_ids: Sequence[int],
    query_cams: Sequence[int],
    gallery_cams: Sequence[int],
    ranks: tuple[int, ...],
) -> ReIDMetrics:
    similarity = l2_normalize(query_features) @ l2_normalize(gallery_features).T
    per_query = [
        _evaluate_query(similarity[index], query_ids[index], gallery_ids, query_cams[index], gallery_cams, ranks)
        for index in range(query_features.shape[0])
    ]
    return _mean_metrics(per_query, ranks)


def _evaluate_reid_rust(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    *,
    query_ids: Sequence[int],
    gallery_ids: Sequence[int],
    query_cams: Sequence[int],
    gallery_cams: Sequence[int],
    ranks: tuple[int, ...],
    query_chunk_size: int,
) -> ReIDMetrics:
    normalized_query = l2_normalize(query_features)
    normalized_gallery = l2_normalize(gallery_features)
    query_id_array = np.ascontiguousarray(query_ids, dtype=np.int64)
    gallery_id_array = np.ascontiguousarray(gallery_ids, dtype=np.int64)
    query_cam_array = np.ascontiguousarray(query_cams, dtype=np.int64)
    gallery_cam_array = np.ascontiguousarray(gallery_cams, dtype=np.int64)

    average_precision_sum = 0.0
    cmc_counts = [0] * len(ranks)
    evaluated_queries = 0
    for start in range(0, query_features.shape[0], query_chunk_size):
        end = min(start + query_chunk_size, query_features.shape[0])
        scores = (normalized_query[start:end] @ normalized_gallery.T).detach().float().cpu().contiguous()
        chunk_ap, chunk_cmc, chunk_count = _native.evaluate_scores(
            scores.numpy(),
            query_id_array[start:end],
            gallery_id_array,
            query_cam_array[start:end],
            gallery_cam_array,
            list(ranks),
        )
        average_precision_sum += float(chunk_ap)
        cmc_counts = [total + int(value) for total, value in zip(cmc_counts, chunk_cmc)]
        evaluated_queries += int(chunk_count)
    return ReIDMetrics(
        map=average_precision_sum / evaluated_queries,
        cmc={rank: count / evaluated_queries for rank, count in zip(ranks, cmc_counts)},
    )


def evaluate_reid_with_rerank(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    *,
    query_ids: Sequence[int],
    gallery_ids: Sequence[int],
    query_cams: Sequence[int],
    gallery_cams: Sequence[int],
    ranks: Sequence[int] = DEFAULT_RANKS,
    config: RerankConfig = RerankConfig(),
    backend: str = RUST_EVALUATION_BACKEND,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
) -> ReIDMetrics:
    _validate_inputs(query_features, gallery_features, query_ids, gallery_ids, query_cams, gallery_cams)
    normalized_ranks = _validate_evaluation_options(ranks, backend, query_chunk_size)
    _validate_rerank_config(config)
    if backend == PYTHON_EVALUATION_BACKEND:
        distance = _rerank_distance(query_features, gallery_features, config)
        per_query = [
            _evaluate_distance_query(
                distance[index],
                query_ids[index],
                gallery_ids,
                query_cams[index],
                gallery_cams,
                normalized_ranks,
            )
            for index in range(query_features.shape[0])
        ]
        return _mean_metrics(per_query, normalized_ranks)
    return _evaluate_reid_with_rerank_rust(
        query_features,
        gallery_features,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        query_cams=query_cams,
        gallery_cams=gallery_cams,
        ranks=normalized_ranks,
        config=config,
        query_chunk_size=query_chunk_size,
    )


def _evaluate_reid_with_rerank_rust(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    *,
    query_ids: Sequence[int],
    gallery_ids: Sequence[int],
    query_cams: Sequence[int],
    gallery_cams: Sequence[int],
    ranks: tuple[int, ...],
    config: RerankConfig,
    query_chunk_size: int,
) -> ReIDMetrics:
    query_count = query_features.shape[0]
    features = l2_normalize(torch.cat([query_features, gallery_features], dim=0)).float()
    sample_count = features.shape[0]
    expansion_count = max(1, round(config.k1 / 2)) + 1
    retained_neighbors = min(
        sample_count,
        max(config.k1 + 1, config.k2, expansion_count),
    )

    top_index_parts: list[np.ndarray] = []
    maximum_parts: list[np.ndarray] = []
    for start in range(0, sample_count, query_chunk_size):
        end = min(start + query_chunk_size, sample_count)
        distances = _pairwise_distance_between(features[start:end], features)
        top_indices, _, maxima = _native.select_topk_distances(
            distances.detach().cpu().contiguous().numpy(),
            retained_neighbors,
        )
        top_index_parts.append(np.asarray(top_indices))
        maximum_parts.append(np.asarray(maxima))

    top_indices = np.ascontiguousarray(np.concatenate(top_index_parts), dtype=np.int64)
    max_distances = np.ascontiguousarray(np.concatenate(maximum_parts), dtype=np.float32)
    affinity_offsets, affinity_indices = _native.expanded_reciprocal_sets(
        top_indices, config.k1
    )
    affinity_offsets = np.ascontiguousarray(affinity_offsets, dtype=np.int64)
    affinity_indices = np.ascontiguousarray(affinity_indices, dtype=np.int64)
    affinity_distances = np.empty(affinity_indices.shape[0], dtype=np.float32)
    for start in range(0, sample_count, query_chunk_size):
        end = min(start + query_chunk_size, sample_count)
        distance_block = (
            _pairwise_distance_between(features[start:end], features)
            .detach()
            .cpu()
            .contiguous()
            .numpy()
        )
        for row_index in range(start, end):
            edge_start = int(affinity_offsets[row_index])
            edge_end = int(affinity_offsets[row_index + 1])
            affinity_distances[edge_start:edge_end] = distance_block[
                row_index - start,
                affinity_indices[edge_start:edge_end],
            ]
    index = _native.SparseRerankIndex(
        top_indices,
        max_distances,
        affinity_offsets,
        affinity_indices,
        affinity_distances,
        query_count,
        config.k2,
    )

    query_id_array = np.ascontiguousarray(query_ids, dtype=np.int64)
    gallery_id_array = np.ascontiguousarray(gallery_ids, dtype=np.int64)
    query_cam_array = np.ascontiguousarray(query_cams, dtype=np.int64)
    gallery_cam_array = np.ascontiguousarray(gallery_cams, dtype=np.int64)
    average_precision_sum = 0.0
    cmc_counts = [0] * len(ranks)
    evaluated_queries = 0
    gallery_features_normalized = features[query_count:]
    for start in range(0, query_count, query_chunk_size):
        end = min(start + query_chunk_size, query_count)
        original_distances = _pairwise_distance_between(
            features[start:end], gallery_features_normalized
        )
        chunk_ap, chunk_cmc, chunk_count = index.evaluate_block(
            original_distances.detach().cpu().contiguous().numpy(),
            start,
            query_id_array[start:end],
            gallery_id_array,
            query_cam_array[start:end],
            gallery_cam_array,
            list(ranks),
            config.lambda_value,
        )
        average_precision_sum += float(chunk_ap)
        cmc_counts = [total + int(value) for total, value in zip(cmc_counts, chunk_cmc)]
        evaluated_queries += int(chunk_count)
    return ReIDMetrics(
        map=average_precision_sum / evaluated_queries,
        cmc={rank: count / evaluated_queries for rank, count in zip(ranks, cmc_counts)},
    )


def _validate_rerank_config(config: RerankConfig) -> None:
    if config.k1 < 1:
        raise ValueError("k1 must be positive")
    if config.k2 < 1:
        raise ValueError("k2 must be positive")
    if not 0.0 <= config.lambda_value <= 1.0:
        raise ValueError("lambda_value must be in [0, 1]")


def _pairwise_distance_between(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_squared = torch.sum(left * left, dim=1, keepdim=True)
    right_squared = torch.sum(right * right, dim=1).unsqueeze(0)
    distance = left_squared + right_squared - 2.0 * (left @ right.T)
    return distance.clamp_min(0.0)


def _rerank_distance(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    config: RerankConfig,
) -> torch.Tensor:
    _validate_rerank_config(config)
    features = l2_normalize(torch.cat([query_features, gallery_features], dim=0)).cpu()
    original_distance = _pairwise_distance(features)
    original_distance = original_distance / original_distance.max(dim=0, keepdim=True).values.clamp_min(1e-12)
    transposed_distance = original_distance.T.contiguous()
    initial_rank = torch.argsort(transposed_distance, dim=1, stable=True)
    affinity = _k_reciprocal_affinity(transposed_distance, initial_rank, config.k1)
    expanded_affinity = _query_expansion(affinity, initial_rank, config.k2)
    jaccard_distance = _jaccard_distance(expanded_affinity, query_features.shape[0])
    query_count = query_features.shape[0]
    # Zhong et al. take the original-distance term from the TRANSPOSED
    # column-normalized matrix so each query row is scaled by one constant.
    gallery_slice = transposed_distance[:query_count, query_count:]
    return (1.0 - config.lambda_value) * jaccard_distance + config.lambda_value * gallery_slice


def _pairwise_distance(features: torch.Tensor) -> torch.Tensor:
    squared = torch.sum(features * features, dim=1, keepdim=True)
    distance = squared + squared.T - 2.0 * (features @ features.T)
    return distance.clamp_min(0.0)


def _k_reciprocal_affinity(distance: torch.Tensor, initial_rank: torch.Tensor, k1: int) -> torch.Tensor:
    sample_count = distance.shape[0]
    affinity = torch.zeros_like(distance)
    neighbor_count = min(k1 + 1, sample_count)
    expansion_count = max(1, round(k1 / 2)) + 1
    for index in range(sample_count):
        reciprocal = _k_reciprocal_indices(initial_rank, index, neighbor_count)
        reciprocal = _expand_reciprocal_indices(initial_rank, reciprocal, expansion_count)
        weights = torch.exp(-distance[index, reciprocal])
        affinity[index, reciprocal] = weights / weights.sum().clamp_min(1e-12)
    return affinity


def _k_reciprocal_indices(initial_rank: torch.Tensor, index: int, neighbor_count: int) -> torch.Tensor:
    forward = initial_rank[index, :neighbor_count]
    backward = initial_rank[forward, :neighbor_count]
    matches = torch.any(backward == index, dim=1)
    return forward[matches]


def _expand_reciprocal_indices(
    initial_rank: torch.Tensor,
    reciprocal: torch.Tensor,
    expansion_count: int,
) -> torch.Tensor:
    selected = [reciprocal]
    base = set(reciprocal.tolist())
    for candidate in reciprocal.tolist():
        candidate_reciprocal = _k_reciprocal_indices(initial_rank, candidate, expansion_count)
        overlap = base.intersection(candidate_reciprocal.tolist())
        if len(overlap) > (2.0 / 3.0) * len(candidate_reciprocal):
            selected.append(candidate_reciprocal)
    return torch.unique(torch.cat(selected))


def _query_expansion(affinity: torch.Tensor, initial_rank: torch.Tensor, k2: int) -> torch.Tensor:
    if k2 <= 1:
        return affinity
    sample_count = affinity.shape[0]
    neighbor_count = min(k2, sample_count)
    expanded = torch.zeros_like(affinity)
    for index in range(sample_count):
        expanded[index] = affinity[initial_rank[index, :neighbor_count]].mean(dim=0)
    return expanded


def _jaccard_distance(affinity: torch.Tensor, query_count: int) -> torch.Tensor:
    sample_count = affinity.shape[0]
    gallery_start = query_count
    jaccard = torch.zeros(query_count, sample_count - gallery_start, dtype=affinity.dtype)
    non_zero_columns = [torch.nonzero(affinity[:, index] > 0, as_tuple=False).flatten() for index in range(sample_count)]
    for query_index in range(query_count):
        non_zero = torch.nonzero(affinity[query_index] > 0, as_tuple=False).flatten()
        minima = torch.zeros(sample_count, dtype=affinity.dtype)
        for column in non_zero.tolist():
            rows = non_zero_columns[column]
            minima[rows] += torch.minimum(affinity[query_index, column], affinity[rows, column])
        jaccard[query_index] = 1.0 - minima[gallery_start:] / (2.0 - minima[gallery_start:]).clamp_min(1e-12)
    return jaccard


def _evaluate_distance_query(
    distances: torch.Tensor,
    query_id: int,
    gallery_ids: Sequence[int],
    query_cam: int,
    gallery_cams: Sequence[int],
    ranks: Sequence[int],
) -> tuple[float, dict[int, float]]:
    values = distances.detach().float().cpu().tolist()
    if any(not math.isfinite(value) for value in values):
        raise ValueError("distances must contain only finite values")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    start = 0
    while start < len(order):
        anchor = values[order[start]]
        end = start + 1
        while end < len(order) and values[order[end]] - anchor <= RERANK_TIE_TOLERANCE:
            end += 1
        order[start:end] = sorted(order[start:end])
        start = end
    matches = _ordered_matches(order, query_id, gallery_ids, query_cam, gallery_cams)
    return _average_precision(matches), _cmc(matches, ranks)


def _evaluate_query(
    scores: torch.Tensor,
    query_id: int,
    gallery_ids: Sequence[int],
    query_cam: int,
    gallery_cams: Sequence[int],
    ranks: Sequence[int],
) -> tuple[float, dict[int, float]]:
    values = scores.detach().float().cpu().tolist()
    if any(not math.isfinite(value) for value in values):
        raise ValueError("scores must contain only finite values")
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
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


def require_evaluation_backend(backend: str) -> str:
    if backend not in SUPPORTED_EVALUATION_BACKENDS:
        raise ValueError(
            f"unsupported evaluation backend {backend!r}; expected one of {SUPPORTED_EVALUATION_BACKENDS}"
        )
    return backend


def _validate_evaluation_options(
    ranks: Sequence[int],
    backend: str,
    query_chunk_size: int,
) -> tuple[int, ...]:
    normalized_ranks = tuple(ranks)
    if not normalized_ranks or any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 for rank in normalized_ranks):
        raise ValueError("ranks must contain positive integers")
    require_evaluation_backend(backend)
    if isinstance(query_chunk_size, bool) or not isinstance(query_chunk_size, int) or query_chunk_size < 1:
        raise ValueError("query_chunk_size must be a positive integer")
    return normalized_ranks


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
