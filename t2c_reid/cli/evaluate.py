"""Evaluate primary and optional reranked ReID metrics from an NPZ feature file."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from t2c_reid.configuration import (
    EvaluationConfig,
    compose_evaluation_config,
    evaluation_config_from_dict_config,
)
from t2c_reid.evaluation import (
    DEFAULT_QUERY_CHUNK_SIZE,
    RUST_EVALUATION_BACKEND,
    ReIDMetrics,
    RerankConfig,
    evaluate_reid,
    evaluate_reid_with_rerank,
)

REQUIRED_KEYS = (
    "query_features",
    "gallery_features",
    "query_ids",
    "gallery_ids",
    "query_cams",
    "gallery_cams",
)


def main(overrides: Sequence[str] | None = None) -> int | None:
    if overrides is None:
        return _hydra_main()
    return _run_evaluation(compose_evaluation_config(overrides))


@hydra.main(config_path="../configs/evaluation", config_name="evaluate")
def _hydra_main(config: DictConfig) -> int:
    return _run_evaluation(evaluation_config_from_dict_config(config))


def _run_evaluation(config: EvaluationConfig) -> int:
    primary, rerank = _evaluate_npz(
        config.features,
        tuple(config.ranks),
        backend=config.evaluation_backend,
        query_chunk_size=config.evaluation_chunk_size,
        rerank_config=(
            RerankConfig(config.rerank_k1, config.rerank_k2, config.rerank_lambda)
            if config.report_rerank
            else None
        ),
    )
    payload = _metrics_payload(primary)
    if rerank is not None:
        payload["rerank"] = _metrics_payload(rerank)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if config.output is None:
        print(text)
        return 0
    config.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _evaluate_npz(
    path: Path,
    ranks: tuple[int, ...],
    *,
    backend: str = RUST_EVALUATION_BACKEND,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    rerank_config: RerankConfig | None = None,
) -> tuple[ReIDMetrics, ReIDMetrics | None]:
    with np.load(path) as data:
        _validate_keys(data.files, path)
        query_features = torch.as_tensor(data["query_features"], dtype=torch.float32)
        gallery_features = torch.as_tensor(
            data["gallery_features"], dtype=torch.float32
        )
        metadata = {
            "query_ids": data["query_ids"].astype(int).tolist(),
            "gallery_ids": data["gallery_ids"].astype(int).tolist(),
            "query_cams": data["query_cams"].astype(int).tolist(),
            "gallery_cams": data["gallery_cams"].astype(int).tolist(),
        }
        primary = evaluate_reid(
            query_features,
            gallery_features,
            **metadata,
            ranks=ranks,
            backend=backend,
            query_chunk_size=query_chunk_size,
        )
        rerank = None
        if rerank_config is not None:
            rerank = evaluate_reid_with_rerank(
                query_features,
                gallery_features,
                **metadata,
                ranks=ranks,
                config=rerank_config,
                backend=backend,
                query_chunk_size=query_chunk_size,
            )
        return primary, rerank


def _validate_keys(keys: Sequence[str], path: Path) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in keys]
    if missing:
        raise KeyError(f"{path} is missing required arrays: {', '.join(missing)}")


def _metrics_payload(metrics: ReIDMetrics) -> dict[str, object]:
    payload: dict[str, object] = {
        "cmc": {str(rank): value for rank, value in metrics.cmc.items()},
        "mAP": metrics.map,
    }
    if metrics.extras:
        payload["extras"] = dict(metrics.extras)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
