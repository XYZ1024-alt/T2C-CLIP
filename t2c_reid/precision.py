"""Mixed-precision policy and optimizer-step control."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ContextManager

import torch

SUPPORTED_PRECISIONS = ("auto", "fp32", "bf16", "fp16")


@dataclass(frozen=True)
class PrecisionPolicy:
    requested: str
    resolved: str
    device_type: str

    @property
    def dtype(self) -> torch.dtype | None:
        if self.resolved == "bf16":
            return torch.bfloat16
        if self.resolved == "fp16":
            return torch.float16
        return None

    def autocast(self) -> ContextManager[Any]:
        if self.resolved == "fp32":
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)


class PrecisionController:
    """One precision/scaler controller shared by both training stages."""

    def __init__(self, policy: PrecisionPolicy):
        self.policy = policy
        self.scaler = torch.amp.GradScaler(
            device="cuda",
            enabled=policy.resolved == "fp16",
        )

    def autocast(self) -> ContextManager[Any]:
        return self.policy.autocast()

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def clip_grad_norm(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: Iterable[torch.nn.Parameter],
        max_norm: float,
    ) -> torch.Tensor | None:
        """Unscale FP16 gradients in place, then clip the global gradient norm.

        ``GradScaler.unscale_`` returns immediately while the scaler is
        disabled, so the BF16 and FP32 paths reach ``clip_grad_norm_`` with raw
        gradients. A later :meth:`step` sees the optimizer already unscaled and
        skips its own unscale, which is the documented AMP ordering.

        Returns the pre-clip norm, or ``None`` when clipping is disabled.
        """

        if max_norm <= 0.0:
            return None
        self.scaler.unscale_(optimizer)
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Apply one optimizer update and report whether FP16 overflow skipped it."""

        if not self.scaler.is_enabled():
            optimizer.step()
            return True
        scale_before = self.scaler.get_scale()
        self.scaler.step(optimizer)
        self.scaler.update()
        return self.scaler.get_scale() >= scale_before

    def state_dict(self) -> dict[str, Any]:
        return {
            "precision": self.policy.resolved,
            "grad_scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        precision = state.get("precision")
        if precision != self.policy.resolved:
            raise ValueError(
                "checkpoint precision does not match this run "
                f"({precision!r} != {self.policy.resolved!r})"
            )
        scaler_state = state.get("grad_scaler", {})
        if not isinstance(scaler_state, dict):
            raise TypeError("checkpoint grad_scaler state must be a dictionary")
        self.scaler.load_state_dict(scaler_state)


def resolve_precision(
    requested: str,
    device: torch.device,
    *,
    is_bf16_supported: Callable[[], bool] | None = None,
) -> PrecisionPolicy:
    if requested not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"unsupported precision: {requested!r}; expected one of {SUPPORTED_PRECISIONS}"
        )
    device_type = device.type
    bf16_supported = is_bf16_supported or torch.cuda.is_bf16_supported
    if requested == "auto":
        if device_type != "cuda":
            resolved = "fp32"
        else:
            resolved = "bf16" if bf16_supported() else "fp16"
        return PrecisionPolicy(requested, resolved, device_type)
    if requested == "fp32":
        return PrecisionPolicy(requested, requested, device_type)
    if device_type != "cuda":
        raise ValueError(f"precision {requested} requires a CUDA device")
    if requested == "bf16" and not bf16_supported():
        raise ValueError("bf16 precision was requested but this CUDA device does not support BF16")
    return PrecisionPolicy(requested, requested, device_type)
