"""Evaluate primary and optional reranked ReID metrics from an NPZ feature file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from t2c_clip.evaluation import (
    DEFAULT_QUERY_CHUNK_SIZE,
    DEFAULT_RANKS,
    DEFAULT_RERANK_K1,
    DEFAULT_RERANK_K2,
    DEFAULT_RERANK_LAMBDA,
    RUST_EVALUATION_BACKEND,
    SUPPORTED_EVALUATION_BACKENDS,
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    primary, rerank = _evaluate_npz(
        args.features,
        tuple(args.ranks),
        backend=args.evaluation_backend,
        query_chunk_size=args.evaluation_chunk_size,
        rerank_config=(
            RerankConfig(args.rerank_k1, args.rerank_k2, args.rerank_lambda)
            if args.report_rerank
            else None
        ),
    )
    payload = _metrics_payload(primary)
    if rerank is not None:
        payload["rerank"] = _metrics_payload(rerank)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
        return 0
    args.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate primary and optional reranked ReID metrics from .npz features."
    )
    parser.add_argument("features", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ranks", nargs="+", type=int, default=list(DEFAULT_RANKS))
    parser.add_argument(
        "--evaluation-backend",
        choices=SUPPORTED_EVALUATION_BACKENDS,
        default=RUST_EVALUATION_BACKEND,
    )
    parser.add_argument(
        "--evaluation-chunk-size",
        type=int,
        default=DEFAULT_QUERY_CHUNK_SIZE,
    )
    parser.add_argument("--report-rerank", action="store_true")
    parser.add_argument("--rerank-k1", type=int, default=DEFAULT_RERANK_K1)
    parser.add_argument("--rerank-k2", type=int, default=DEFAULT_RERANK_K2)
    parser.add_argument("--rerank-lambda", type=float, default=DEFAULT_RERANK_LAMBDA)
    return parser


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
        gallery_features = torch.as_tensor(data["gallery_features"], dtype=torch.float32)
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
