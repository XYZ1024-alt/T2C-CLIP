"""Inspect Stage-2 batch-hard triplet geometry on real training batches."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import _build_parser  # noqa: E402
from t2c_clip.features import l2_normalize  # noqa: E402
from t2c_clip.jobs.clip_reid import build_training_job  # noqa: E402

DEFAULT_MAX_BATCHES = 2


@dataclass(frozen=True)
class RuntimeParts:
    model: torch.nn.Module
    loader: object
    device: torch.device
    margin: float


@dataclass(frozen=True)
class TripletStats:
    valid_anchors: int
    positive_pairs: int
    negative_pairs: int
    positive_mean: float
    negative_mean: float
    hardest_positive_mean: float
    hardest_negative_mean: float
    raw_mean: float
    relu_mean: float
    active_fraction: float


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    parser.add_argument("--diagnostic-checkpoint", type=Path)
    parser.add_argument("--diagnostic-max-batches", type=int, default=DEFAULT_MAX_BATCHES)
    args = parser.parse_args(argv)
    args.stage2_first_epoch = args.stage1_epochs + 1
    parts = _runtime_parts(build_training_job(args))
    _load_checkpoint_if_requested(parts.model, args.diagnostic_checkpoint, parts.device)
    parts.model.eval()
    for batch_index, batch in enumerate(parts.loader, start=1):
        if batch_index > args.diagnostic_max_batches:
            break
        labels = batch.person_ids.to(parts.device)
        cameras = batch.camera_ids.to(parts.device)
        images = batch.images.to(parts.device)
        with torch.no_grad():
            outputs = parts.model.retrieval_model.forward_stage2(images, cameras, labels)
            retrieval = outputs["retrieval"]
            headed = parts.model.feature_head(retrieval)
            logits = parts.model.classifier(headed) * args.id_logit_scale
        _print_batch(batch_index, labels, retrieval, headed, logits, parts.margin)
    return 0


def _runtime_parts(job) -> RuntimeParts:
    stage2 = job.stage2 if hasattr(job, "stage2") else job
    runtime = _stage_runtime(stage2)
    margin = float(getattr(runtime.loss_config, "triplet_margin", 0.3))
    return RuntimeParts(stage2.model, runtime.loaders.train, runtime.device, margin)


def _stage_runtime(stage2):
    closure = stage2.train_one_epoch.__closure__
    if closure is None:
        raise ValueError("train_one_epoch has no closure")
    for cell in closure:
        value = cell.cell_contents
        if hasattr(value, "loaders") and hasattr(value, "loss_config"):
            return value
    raise ValueError("could not find StageTrainingRuntime in train_one_epoch closure")


def _load_checkpoint_if_requested(
    model: torch.nn.Module,
    checkpoint: Path | None,
    device: torch.device,
) -> None:
    if checkpoint is None:
        print("checkpoint=none")
        return
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError("checkpoint must be a state dict or contain model_state")
    model.load_state_dict(state)
    epoch = payload.get("epoch", "unknown") if isinstance(payload, dict) else "unknown"
    best_map = payload.get("best_map", "unknown") if isinstance(payload, dict) else "unknown"
    print(f"checkpoint={checkpoint} epoch={epoch} best_map={best_map}")


def _print_batch(
    batch_index: int,
    labels: torch.Tensor,
    retrieval: torch.Tensor,
    headed: torch.Tensor,
    logits: torch.Tensor,
    margin: float,
) -> None:
    stats = _triplet_stats(retrieval, labels, margin)
    headed_stats = _triplet_stats(headed, labels, margin)
    print(f"batch={batch_index}")
    print(_label_summary(labels))
    print(
        "retrieval "
        f"norm_mean={_norm_mean(retrieval):.6f} "
        f"pos_mean={stats.positive_mean:.6f} neg_mean={stats.negative_mean:.6f} "
        f"hard_pos={stats.hardest_positive_mean:.6f} hard_neg={stats.hardest_negative_mean:.6f} "
        f"raw={stats.raw_mean:.6f} relu={stats.relu_mean:.6f} active={stats.active_fraction:.4f}"
    )
    print(
        "headed "
        f"norm_mean={_norm_mean(headed):.6f} "
        f"pos_mean={headed_stats.positive_mean:.6f} neg_mean={headed_stats.negative_mean:.6f} "
        f"hard_pos={headed_stats.hardest_positive_mean:.6f} hard_neg={headed_stats.hardest_negative_mean:.6f} "
        f"raw={headed_stats.raw_mean:.6f} relu={headed_stats.relu_mean:.6f} active={headed_stats.active_fraction:.4f}"
    )
    print(
        "classifier "
        f"loss={float(F.cross_entropy(logits, labels).detach().cpu()):.6f} "
        f"acc={_accuracy(logits, labels):.4f} "
        f"logit_std={float(logits.detach().std().cpu()):.6f}"
    )


def _triplet_stats(features: torch.Tensor, labels: torch.Tensor, margin: float) -> TripletStats:
    distances = 1.0 - l2_normalize(features) @ l2_normalize(features).T
    positive_values: list[torch.Tensor] = []
    negative_values: list[torch.Tensor] = []
    hard_positive: list[torch.Tensor] = []
    hard_negative: list[torch.Tensor] = []
    raw_losses: list[torch.Tensor] = []
    relu_losses: list[torch.Tensor] = []
    positive_pairs = 0
    negative_pairs = 0
    for index in range(labels.shape[0]):
        positive_mask = labels == labels[index]
        negative_mask = labels != labels[index]
        positive_mask[index] = False
        positives = distances[index][positive_mask]
        negatives = distances[index][negative_mask]
        positive_pairs += int(positive_mask.sum().item())
        negative_pairs += int(negative_mask.sum().item())
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        positive_values.append(positives.mean())
        negative_values.append(negatives.mean())
        hardest_positive = positives.max()
        hardest_negative = negatives.min()
        raw = hardest_positive - hardest_negative + margin
        hard_positive.append(hardest_positive)
        hard_negative.append(hardest_negative)
        raw_losses.append(raw)
        relu_losses.append(F.relu(raw))
    return TripletStats(
        valid_anchors=len(raw_losses),
        positive_pairs=positive_pairs,
        negative_pairs=negative_pairs,
        positive_mean=_mean_or_nan(positive_values),
        negative_mean=_mean_or_nan(negative_values),
        hardest_positive_mean=_mean_or_nan(hard_positive),
        hardest_negative_mean=_mean_or_nan(hard_negative),
        raw_mean=_mean_or_nan(raw_losses),
        relu_mean=_mean_or_nan(relu_losses),
        active_fraction=_active_fraction(raw_losses),
    )


def _label_summary(labels: torch.Tensor) -> str:
    unique, counts = torch.unique(labels.detach().cpu(), return_counts=True)
    count_values = [int(value) for value in counts.tolist()]
    return (
        f"labels batch_size={labels.numel()} ids={unique.numel()} "
        f"min_count={min(count_values)} max_count={max(count_values)} counts={count_values}"
    )


def _mean_or_nan(values: list[torch.Tensor]) -> float:
    if not values:
        return float("nan")
    return float(torch.stack(values).mean().detach().cpu())


def _active_fraction(values: list[torch.Tensor]) -> float:
    if not values:
        return float("nan")
    active = torch.stack(values) > 0.0
    return float(active.float().mean().detach().cpu())


def _norm_mean(features: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(features.detach(), dim=1).mean().cpu())


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits.detach(), dim=1)
    return float((predictions == labels).float().mean().cpu())


if __name__ == "__main__":
    raise SystemExit(main())
