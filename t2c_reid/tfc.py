"""Camera-aware cross-modal Training-time Feature Centralization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from t2c_reid.features import l2_normalize

MIN_MOMENTUM = 0.0
MAX_MOMENTUM = 1.0
DEFAULT_TAIL_MOMENTUM = 0.9
DEFAULT_CLASS_BALANCE_BETA = 0.9999
DEFAULT_LOCAL_WEIGHT = 1.0
DEFAULT_GLOBAL_WEIGHT = 1.0
DEFAULT_CROSS_MODAL_WEIGHT = 0.5
DEFAULT_CROSS_CAMERA_WEIGHT = 0.1
DEFAULT_CONTRAST_TEMPERATURE = 0.07
DEFAULT_TRANSFER_REG_WEIGHT = 0.01
CAMERA_PRIOR_SMOOTHING = 1.0


@dataclass(frozen=True)
class CameraAwareTFCLossConfig:
    local_weight: float = DEFAULT_LOCAL_WEIGHT
    global_weight: float = DEFAULT_GLOBAL_WEIGHT
    cross_modal_weight: float = DEFAULT_CROSS_MODAL_WEIGHT
    cross_camera_weight: float = DEFAULT_CROSS_CAMERA_WEIGHT
    contrast_temperature: float = DEFAULT_CONTRAST_TEMPERATURE
    transfer_reg_weight: float = DEFAULT_TRANSFER_REG_WEIGHT

    def __post_init__(self) -> None:
        weights = (
            self.local_weight,
            self.global_weight,
            self.cross_modal_weight,
            self.cross_camera_weight,
            self.transfer_reg_weight,
        )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("TFC loss weights must be finite and non-negative")
        if not math.isfinite(self.contrast_temperature) or self.contrast_temperature <= 0.0:
            raise ValueError("TFC contrast temperature must be finite and positive")

    @property
    def main_weight_sum(self) -> float:
        return (
            self.local_weight
            + self.global_weight
            + self.cross_modal_weight
            + self.cross_camera_weight
        )


@dataclass(frozen=True)
class CameraAwareTFCLossBreakdown:
    local: torch.Tensor
    global_center: torch.Tensor
    cross_modal: torch.Tensor
    cross_camera: torch.Tensor
    transfer_regularization: torch.Tensor
    cross_camera_coverage: torch.Tensor
    config: CameraAwareTFCLossConfig

    @property
    def total(self) -> torch.Tensor:
        config = self.config
        if config.main_weight_sum == 0.0:
            main = self.local * 0.0
        else:
            main = (
                config.local_weight * self.local
                + config.global_weight * self.global_center
                + config.cross_modal_weight * self.cross_modal
                + config.cross_camera_weight * self.cross_camera
            ) / config.main_weight_sum
        return main + config.transfer_reg_weight * self.transfer_regularization


@dataclass
class _PairSnapshot:
    visual_center: torch.Tensor
    text_center: torch.Tensor
    visual_initialized: torch.Tensor
    text_initialized: torch.Tensor


@dataclass
class _IdentitySnapshot:
    visual_center: torch.Tensor
    text_center: torch.Tensor
    visual_initialized: torch.Tensor
    text_initialized: torch.Tensor


class CameraAwareTFCBank(torch.nn.Module):
    """EMA visual/text prototypes indexed by training identity and camera."""

    def __init__(
        self,
        identity_counts: torch.Tensor,
        identity_camera_counts: torch.Tensor,
        feature_dim: int,
        head_momentum: float,
        tail_momentum: float = DEFAULT_TAIL_MOMENTUM,
        class_balance_beta: float = DEFAULT_CLASS_BALANCE_BETA,
    ):
        super().__init__()
        counts, pair_counts = _validate_statistics(identity_counts, identity_camera_counts)
        _validate_shape(counts.shape[0], pair_counts.shape[1], feature_dim)
        _validate_momenta(head_momentum, tail_momentum)
        _validate_class_balance_beta(class_balance_beta)

        num_train_ids = counts.shape[0]
        num_cameras = pair_counts.shape[1]
        local_shape = (num_train_ids, num_cameras, feature_dim)
        global_shape = (num_train_ids, feature_dim)
        momenta = _identity_momenta(counts, head_momentum, tail_momentum)
        class_weights = _effective_number_weights(counts, class_balance_beta)
        camera_active = pair_counts.sum(dim=0) > 0
        camera_prior = _camera_cooccurrence_prior(pair_counts, camera_active)

        self.feature_dim = int(feature_dim)
        self.head_momentum = float(head_momentum)
        self.tail_momentum = float(tail_momentum)
        self.class_balance_beta = float(class_balance_beta)
        self.register_buffer("identity_counts", counts)
        self.register_buffer("identity_camera_counts", pair_counts)
        self.register_buffer("identity_momenta", momenta)
        self.register_buffer("class_weights", class_weights)
        self.register_buffer("train_camera_active", camera_active)
        self.register_buffer("camera_prior", camera_prior)
        self.register_buffer("visual_local_centers", torch.zeros(local_shape, dtype=torch.float32))
        self.register_buffer("text_local_centers", torch.zeros(local_shape, dtype=torch.float32))
        self.register_buffer(
            "visual_local_initialized", torch.zeros(num_train_ids, num_cameras, dtype=torch.bool)
        )
        self.register_buffer(
            "text_local_initialized", torch.zeros(num_train_ids, num_cameras, dtype=torch.bool)
        )
        self.register_buffer("visual_global_centers", torch.zeros(global_shape, dtype=torch.float32))
        self.register_buffer("text_global_centers", torch.zeros(global_shape, dtype=torch.float32))
        self.register_buffer("visual_global_initialized", torch.zeros(num_train_ids, dtype=torch.bool))
        self.register_buffer("text_global_initialized", torch.zeros(num_train_ids, dtype=torch.bool))

        initial_logits = torch.zeros((num_cameras, num_cameras), dtype=torch.float32)
        positive_prior = camera_prior > 0
        initial_logits[positive_prior] = torch.log(camera_prior[positive_prior])
        self.camera_transfer_logits = torch.nn.Parameter(initial_logits)

        self._window_active = False
        self._pair_snapshots: dict[tuple[int, int], _PairSnapshot] = {}
        self._identity_snapshots: dict[int, _IdentitySnapshot] = {}

    @property
    def num_train_ids(self) -> int:
        return int(self.identity_counts.shape[0])

    @property
    def num_cameras(self) -> int:
        return int(self.identity_camera_counts.shape[1])

    def begin_update_window(self) -> None:
        if self._window_active:
            raise RuntimeError("a TFC update window is already active")
        self._window_active = True
        self._pair_snapshots.clear()
        self._identity_snapshots.clear()

    def commit_update_window(self) -> None:
        if not self._window_active:
            return
        self._clear_update_window()

    @torch.no_grad()
    def rollback_update_window(self) -> None:
        if not self._window_active:
            return
        for (person_id, camera_id), snapshot in self._pair_snapshots.items():
            self.visual_local_centers[person_id, camera_id].copy_(snapshot.visual_center)
            self.text_local_centers[person_id, camera_id].copy_(snapshot.text_center)
            self.visual_local_initialized[person_id, camera_id].copy_(snapshot.visual_initialized)
            self.text_local_initialized[person_id, camera_id].copy_(snapshot.text_initialized)
        for person_id, snapshot in self._identity_snapshots.items():
            self.visual_global_centers[person_id].copy_(snapshot.visual_center)
            self.text_global_centers[person_id].copy_(snapshot.text_center)
            self.visual_global_initialized[person_id].copy_(snapshot.visual_initialized)
            self.text_global_initialized[person_id].copy_(snapshot.text_initialized)
        self._clear_update_window()

    @torch.no_grad()
    def update(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        person_ids: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> None:
        _validate_batch(
            visual_features,
            text_features,
            person_ids,
            camera_ids,
            self.num_train_ids,
            self.num_cameras,
            self.feature_dim,
        )
        if torch.any(self.identity_camera_counts[person_ids, camera_ids] <= 0):
            raise ValueError("TFC batch contains a pid-camera pair absent from the training split")
        visual = l2_normalize(visual_features.detach().float())
        text = l2_normalize(text_features.detach().float())
        pairs = torch.stack((person_ids, camera_ids), dim=1)
        affected_ids: set[int] = set()
        for pair in torch.unique(pairs, dim=0):
            person_id = int(pair[0].item())
            camera_id = int(pair[1].item())
            self._snapshot_pair(person_id, camera_id)
            mask = (person_ids == person_id) & (camera_ids == camera_id)
            visual_mean = l2_normalize(visual[mask].mean(dim=0, keepdim=True))[0]
            text_mean = l2_normalize(text[mask].mean(dim=0, keepdim=True))[0]
            momentum = float(self.identity_momenta[person_id].item())
            self._update_local_center(
                self.visual_local_centers,
                self.visual_local_initialized,
                person_id,
                camera_id,
                visual_mean,
                momentum,
            )
            self._update_local_center(
                self.text_local_centers,
                self.text_local_initialized,
                person_id,
                camera_id,
                text_mean,
                momentum,
            )
            affected_ids.add(person_id)

        for person_id in affected_ids:
            self._snapshot_identity(person_id)
            self._refresh_global_center(person_id)

    def loss(
        self,
        retrieval_features: torch.Tensor,
        visual_features: torch.Tensor,
        person_ids: torch.Tensor,
        camera_ids: torch.Tensor,
        beta: float,
        config: CameraAwareTFCLossConfig,
    ) -> CameraAwareTFCLossBreakdown:
        _validate_loss_inputs(
            retrieval_features,
            visual_features,
            person_ids,
            camera_ids,
            self.num_train_ids,
            self.num_cameras,
            self.feature_dim,
            beta,
        )
        if not bool(torch.all(self.visual_local_initialized[person_ids, camera_ids])):
            raise RuntimeError("TFC loss requested before visual local centers were initialized")
        if not bool(torch.all(self.text_local_initialized[person_ids, camera_ids])):
            raise RuntimeError("TFC loss requested before text local centers were initialized")
        if not bool(torch.all(self.visual_global_initialized[person_ids])):
            raise RuntimeError("TFC loss requested before visual global centers were initialized")
        if not bool(torch.all(self.text_global_initialized[person_ids])):
            raise RuntimeError("TFC loss requested before text global centers were initialized")
        with torch.autocast(device_type=retrieval_features.device.type, enabled=False):
            retrieval = l2_normalize(retrieval_features.float())
            visual = l2_normalize(visual_features.float())
            sample_weights = self.class_weights[person_ids].float()
            local_visual = self.visual_local_centers[person_ids, camera_ids].detach()
            local_text = self.text_local_centers[person_ids, camera_ids].detach()
            global_visual = self.visual_global_centers[person_ids].detach()
            global_text = self.text_global_centers[person_ids].detach()
            local_prototypes = _fuse_prototypes(local_visual, local_text, beta)
            global_prototypes = _fuse_prototypes(global_visual, global_text, beta)

            local_losses = 1.0 - torch.sum(retrieval * local_prototypes, dim=1)
            global_losses = 1.0 - torch.sum(retrieval * global_prototypes, dim=1)
            cross_modal_losses = 1.0 - torch.sum(visual * local_text, dim=1)
            local_loss = _weighted_mean(local_losses, sample_weights)
            global_loss = _weighted_mean(global_losses, sample_weights)
            cross_modal_loss = _weighted_mean(cross_modal_losses, sample_weights)
            cross_camera_loss, coverage = self._cross_camera_loss(
                retrieval,
                person_ids,
                camera_ids,
                sample_weights,
                beta,
                config.contrast_temperature,
            )
            transfer_regularization = self.transfer_regularization()

        return CameraAwareTFCLossBreakdown(
            local=local_loss,
            global_center=global_loss,
            cross_modal=cross_modal_loss,
            cross_camera=cross_camera_loss,
            transfer_regularization=transfer_regularization,
            cross_camera_coverage=coverage,
            config=config,
        )

    def camera_transfer_log_probabilities(self) -> torch.Tensor:
        log_probabilities = torch.full_like(
            self.camera_transfer_logits, -torch.inf, dtype=torch.float32
        )
        if self.num_cameras < 2:
            return log_probabilities
        for source in range(self.num_cameras):
            targets = self.train_camera_active.clone()
            targets[source] = False
            if not bool(torch.any(targets)) or not bool(self.train_camera_active[source]):
                continue
            log_probabilities[source, targets] = torch.log_softmax(
                self.camera_transfer_logits[source, targets].float(), dim=0
            )
        return log_probabilities

    def camera_transfer_probabilities(self) -> torch.Tensor:
        return torch.exp(self.camera_transfer_log_probabilities())

    def transfer_regularization(self) -> torch.Tensor:
        log_probabilities = self.camera_transfer_log_probabilities()
        valid = self.camera_prior > 0
        if not bool(torch.any(valid)):
            return self.camera_transfer_logits.float().sum() * 0.0
        terms = self.camera_prior[valid] * (
            torch.log(self.camera_prior[valid]) - log_probabilities[valid]
        )
        valid_rows = torch.any(valid, dim=1).sum().clamp_min(1)
        return terms.sum() / valid_rows

    def _cross_camera_loss(
        self,
        retrieval: torch.Tensor,
        person_ids: torch.Tensor,
        camera_ids: torch.Tensor,
        sample_weights: torch.Tensor,
        beta: float,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initialized = self.visual_local_initialized & self.text_local_initialized
        pair_indices = torch.nonzero(initialized, as_tuple=False)
        zero = retrieval.sum() * 0.0 + self.camera_transfer_logits.float().sum() * 0.0
        if pair_indices.shape[0] == 0:
            return zero, zero.detach()

        pair_person_ids = pair_indices[:, 0]
        pair_camera_ids = pair_indices[:, 1]
        prototypes = _fuse_prototypes(
            self.visual_local_centers[initialized].detach(),
            self.text_local_centers[initialized].detach(),
            beta,
        )
        transfer_log_probabilities = self.camera_transfer_log_probabilities()
        losses: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for index in range(retrieval.shape[0]):
            person_id = person_ids[index]
            camera_id = camera_ids[index]
            positives = (pair_person_ids == person_id) & (pair_camera_ids != camera_id)
            negatives = pair_person_ids != person_id
            if not bool(torch.any(positives)) or not bool(torch.any(negatives)):
                continue
            positive_cameras = pair_camera_ids[positives]
            positive_log_weights = transfer_log_probabilities[camera_id, positive_cameras]
            if not bool(torch.all(torch.isfinite(positive_log_weights))):
                continue
            positive_log_weights = positive_log_weights - torch.logsumexp(
                positive_log_weights, dim=0
            )
            candidate_mask = positives | negatives
            logits = (retrieval[index] @ prototypes.T) / float(temperature)
            numerator = torch.logsumexp(
                logits[positives] + positive_log_weights, dim=0
            )
            denominator = torch.logsumexp(logits[candidate_mask], dim=0)
            losses.append(denominator - numerator)
            weights.append(sample_weights[index])
        if not losses:
            return zero, zero.detach()
        loss_tensor = torch.stack(losses)
        weight_tensor = torch.stack(weights)
        coverage = torch.tensor(
            len(losses) / retrieval.shape[0],
            dtype=torch.float32,
            device=retrieval.device,
        )
        return _weighted_mean(loss_tensor, weight_tensor), coverage

    @staticmethod
    def _update_local_center(
        centers: torch.Tensor,
        initialized: torch.Tensor,
        person_id: int,
        camera_id: int,
        observation: torch.Tensor,
        momentum: float,
    ) -> None:
        if bool(initialized[person_id, camera_id]):
            updated = momentum * centers[person_id, camera_id] + (1.0 - momentum) * observation
            centers[person_id, camera_id] = l2_normalize(updated.unsqueeze(0))[0]
        else:
            centers[person_id, camera_id] = observation
            initialized[person_id, camera_id] = True

    def _refresh_global_center(self, person_id: int) -> None:
        visual_mask = self.visual_local_initialized[person_id]
        text_mask = self.text_local_initialized[person_id]
        if bool(torch.any(visual_mask)):
            visual_mean = self.visual_local_centers[person_id, visual_mask].mean(dim=0, keepdim=True)
            self.visual_global_centers[person_id] = l2_normalize(visual_mean)[0]
            self.visual_global_initialized[person_id] = True
        if bool(torch.any(text_mask)):
            text_mean = self.text_local_centers[person_id, text_mask].mean(dim=0, keepdim=True)
            self.text_global_centers[person_id] = l2_normalize(text_mean)[0]
            self.text_global_initialized[person_id] = True

    def _snapshot_pair(self, person_id: int, camera_id: int) -> None:
        key = (person_id, camera_id)
        if not self._window_active or key in self._pair_snapshots:
            return
        self._pair_snapshots[key] = _PairSnapshot(
            visual_center=self.visual_local_centers[person_id, camera_id].clone(),
            text_center=self.text_local_centers[person_id, camera_id].clone(),
            visual_initialized=self.visual_local_initialized[person_id, camera_id].clone(),
            text_initialized=self.text_local_initialized[person_id, camera_id].clone(),
        )

    def _snapshot_identity(self, person_id: int) -> None:
        if not self._window_active or person_id in self._identity_snapshots:
            return
        self._identity_snapshots[person_id] = _IdentitySnapshot(
            visual_center=self.visual_global_centers[person_id].clone(),
            text_center=self.text_global_centers[person_id].clone(),
            visual_initialized=self.visual_global_initialized[person_id].clone(),
            text_initialized=self.text_global_initialized[person_id].clone(),
        )

    def _clear_update_window(self) -> None:
        self._window_active = False
        self._pair_snapshots.clear()
        self._identity_snapshots.clear()

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "feature_dim": self.feature_dim,
            "head_momentum": self.head_momentum,
            "tail_momentum": self.tail_momentum,
            "class_balance_beta": self.class_balance_beta,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        expected = self.get_extra_state()
        if state != expected:
            raise ValueError(f"incompatible Camera-aware TFC state: checkpoint={state!r}, run={expected!r}")


def _identity_momenta(
    counts: torch.Tensor,
    head_momentum: float,
    tail_momentum: float,
) -> torch.Tensor:
    log_counts = torch.log(counts.to(dtype=torch.float64))
    minimum = log_counts.min()
    maximum = log_counts.max()
    if bool(maximum == minimum):
        rarity = torch.zeros_like(log_counts)
    else:
        rarity = 1.0 - (log_counts - minimum) / (maximum - minimum)
    return (head_momentum + rarity * (tail_momentum - head_momentum)).float()


def _effective_number_weights(counts: torch.Tensor, beta: float) -> torch.Tensor:
    if beta == 0.0:
        return torch.ones_like(counts, dtype=torch.float32)
    beta64 = torch.tensor(beta, dtype=torch.float64)
    denominator = -torch.expm1(counts.to(dtype=torch.float64) * torch.log(beta64))
    weights = (1.0 - beta64) / denominator
    weights = weights / weights.mean()
    return weights.float()


def _camera_cooccurrence_prior(
    pair_counts: torch.Tensor,
    camera_active: torch.Tensor,
) -> torch.Tensor:
    num_cameras = pair_counts.shape[1]
    prior = torch.zeros((num_cameras, num_cameras), dtype=torch.float32)
    if int(camera_active.sum().item()) < 2:
        return prior
    observed = (pair_counts > 0).to(dtype=torch.float64)
    cooccurrence = observed.T @ observed
    for source in range(num_cameras):
        if not bool(camera_active[source]):
            continue
        targets = camera_active.clone()
        targets[source] = False
        values = cooccurrence[source, targets] + CAMERA_PRIOR_SMOOTHING
        prior[source, targets] = (values / values.sum()).float()
    return prior


def _fuse_prototypes(
    visual_centers: torch.Tensor,
    text_centers: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return l2_normalize(visual_centers.float() + float(beta) * text_centers.float())


def _weighted_mean(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(losses.float() * weights.float()) / torch.sum(weights.float()).clamp_min(1e-12)


def _validate_statistics(
    identity_counts: torch.Tensor,
    identity_camera_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if identity_counts.dtype != torch.long or identity_counts.ndim != 1:
        raise ValueError("identity_counts must be a rank-1 torch.long tensor")
    if identity_camera_counts.dtype != torch.long or identity_camera_counts.ndim != 2:
        raise ValueError("identity_camera_counts must be a rank-2 torch.long tensor")
    if identity_camera_counts.shape[0] != identity_counts.shape[0]:
        raise ValueError("identity count tensors must share the identity dimension")
    if torch.any(identity_counts <= 0):
        raise ValueError("identity_counts must be positive")
    if torch.any(identity_camera_counts < 0):
        raise ValueError("identity_camera_counts must be non-negative")
    if not torch.equal(identity_camera_counts.sum(dim=1), identity_counts):
        raise ValueError("identity_camera_counts rows must sum to identity_counts")
    return identity_counts.detach().cpu().clone(), identity_camera_counts.detach().cpu().clone()


def _validate_shape(num_train_ids: int, num_cameras: int, feature_dim: int) -> None:
    if num_train_ids < 1 or num_cameras < 1 or feature_dim < 1:
        raise ValueError("num_train_ids, num_cameras, and feature_dim must be positive")


def _validate_momenta(head_momentum: float, tail_momentum: float) -> None:
    if not (
        math.isfinite(head_momentum)
        and math.isfinite(tail_momentum)
        and MIN_MOMENTUM <= head_momentum <= tail_momentum < MAX_MOMENTUM
    ):
        raise ValueError("TFC momentum must satisfy 0.0 <= head_momentum <= tail_momentum < 1.0")


def _validate_class_balance_beta(beta: float) -> None:
    if not math.isfinite(beta) or not 0.0 <= beta < 1.0:
        raise ValueError("TFC class-balance beta must satisfy 0.0 <= beta < 1.0")


def _validate_batch(
    visual_features: torch.Tensor,
    text_features: torch.Tensor,
    person_ids: torch.Tensor,
    camera_ids: torch.Tensor,
    num_train_ids: int,
    num_cameras: int,
    feature_dim: int,
) -> None:
    if visual_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("visual_features and text_features must be rank-2 tensors")
    if visual_features.shape != text_features.shape:
        raise ValueError("visual_features and text_features must have identical shapes")
    if visual_features.shape[1] != feature_dim:
        raise ValueError("TFC features do not match the center feature dimension")
    _validate_indices(person_ids, camera_ids, visual_features.shape[0], num_train_ids, num_cameras)


def _validate_loss_inputs(
    retrieval_features: torch.Tensor,
    visual_features: torch.Tensor,
    person_ids: torch.Tensor,
    camera_ids: torch.Tensor,
    num_train_ids: int,
    num_cameras: int,
    feature_dim: int,
    beta: float,
) -> None:
    if retrieval_features.ndim != 2 or visual_features.ndim != 2:
        raise ValueError("retrieval_features and visual_features must be rank-2 tensors")
    if retrieval_features.shape != visual_features.shape:
        raise ValueError("retrieval_features and visual_features must have identical shapes")
    if retrieval_features.shape[1] != feature_dim:
        raise ValueError("TFC loss features do not match the center feature dimension")
    if not math.isfinite(beta):
        raise ValueError("TFC prototype beta must be finite")
    _validate_indices(person_ids, camera_ids, retrieval_features.shape[0], num_train_ids, num_cameras)


def _validate_indices(
    person_ids: torch.Tensor,
    camera_ids: torch.Tensor,
    batch_size: int,
    num_train_ids: int,
    num_cameras: int,
) -> None:
    if person_ids.dtype != torch.long or person_ids.ndim != 1:
        raise ValueError("person_ids must be a rank-1 torch.long tensor")
    if camera_ids.dtype != torch.long or camera_ids.ndim != 1:
        raise ValueError("camera_ids must be a rank-1 torch.long tensor")
    if person_ids.shape[0] != batch_size or camera_ids.shape[0] != batch_size:
        raise ValueError("person_ids and camera_ids must match the feature batch size")
    if torch.any(person_ids < 0) or torch.any(person_ids >= num_train_ids):
        raise ValueError("person_ids contain indices outside the center bank")
    if torch.any(camera_ids < 0) or torch.any(camera_ids >= num_cameras):
        raise ValueError("camera_ids contain indices outside the center bank")
