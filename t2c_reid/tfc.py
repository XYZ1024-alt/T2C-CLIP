"""Training-time Feature Centralization."""

from __future__ import annotations

import torch

from t2c_reid.features import l2_normalize

MIN_MOMENTUM = 0.0
MAX_MOMENTUM = 1.0


class TFCCenterBank(torch.nn.Module):
    def __init__(self, num_train_ids: int, feature_dim: int, momentum: float):
        super().__init__()
        _validate_shape(num_train_ids, feature_dim)
        _validate_momentum(momentum)
        self.momentum = momentum
        self.register_buffer("centers", torch.zeros(num_train_ids, feature_dim))
        self.register_buffer("initialized", torch.zeros(num_train_ids, dtype=torch.bool))

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        _validate_batch(features, labels, self.centers.shape[0])
        normalized = l2_normalize(features.detach())
        for label in torch.unique(labels):
            index = int(label.item())
            mean_feature = l2_normalize(normalized[labels == label].mean(dim=0, keepdim=True))[0]
            self._update_center(index, mean_feature)

    def loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _validate_batch(features, labels, self.centers.shape[0])
        if not bool(torch.all(self.initialized[labels])):
            raise RuntimeError("TFC loss requested before all label centers were initialized")
        normalized = l2_normalize(features)
        label_centers = self.centers[labels].detach()
        cosine = torch.sum(normalized * label_centers, dim=1)
        return torch.mean(1.0 - cosine)

    def _update_center(self, index: int, mean_feature: torch.Tensor) -> None:
        if bool(self.initialized[index]):
            updated = self.momentum * self.centers[index] + (1.0 - self.momentum) * mean_feature
            self.centers[index] = l2_normalize(updated.unsqueeze(0))[0]
            return
        self.centers[index] = mean_feature
        self.initialized[index] = True


def _validate_shape(num_train_ids: int, feature_dim: int) -> None:
    if num_train_ids < 1 or feature_dim < 1:
        raise ValueError("num_train_ids and feature_dim must be positive")


def _validate_momentum(momentum: float) -> None:
    if not MIN_MOMENTUM <= momentum < MAX_MOMENTUM:
        raise ValueError("momentum must satisfy 0.0 <= momentum < 1.0")


def _validate_batch(features: torch.Tensor, labels: torch.Tensor, num_train_ids: int) -> None:
    if features.ndim != 2:
        raise ValueError("features must be a rank-2 tensor")
    if labels.dtype != torch.long or labels.ndim != 1:
        raise ValueError("labels must be a rank-1 torch.long tensor")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same batch size")
    if torch.any(labels < 0) or torch.any(labels >= num_train_ids):
        raise ValueError("labels contain identity indices outside the center bank")
