"""Evaluate no-rerank ReID metrics from an NPZ feature file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from t2c_clip.evaluation import DEFAULT_RANKS, ReIDMetrics, evaluate_reid

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
    payload = _metrics_payload(_evaluate_npz(args.features, tuple(args.ranks)))
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
        return 0
    args.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate no-rerank ReID metrics from .npz features.")
    parser.add_argument("features", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ranks", nargs="+", type=int, default=list(DEFAULT_RANKS))
    return parser


def _evaluate_npz(path: Path, ranks: tuple[int, ...]) -> ReIDMetrics:
    with np.load(path) as data:
        _validate_keys(data.files, path)
        return evaluate_reid(
            torch.as_tensor(data["query_features"], dtype=torch.float32),
            torch.as_tensor(data["gallery_features"], dtype=torch.float32),
            query_ids=data["query_ids"].astype(int).tolist(),
            gallery_ids=data["gallery_ids"].astype(int).tolist(),
            query_cams=data["query_cams"].astype(int).tolist(),
            gallery_cams=data["gallery_cams"].astype(int).tolist(),
            ranks=ranks,
        )


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
