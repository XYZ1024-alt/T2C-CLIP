"""Retrieval feature mode constants and validation."""

from __future__ import annotations

FUSED_RETRIEVAL = "fused"
IMAGE_ONLY_RETRIEVAL = "image_only"
SUPPORTED_RETRIEVAL_MODES = (FUSED_RETRIEVAL, IMAGE_ONLY_RETRIEVAL)


def require_retrieval_mode(retrieval_mode: str) -> str:
    if retrieval_mode not in SUPPORTED_RETRIEVAL_MODES:
        raise ValueError(
            f"unsupported retrieval mode: {retrieval_mode!r}; "
            f"expected one of {SUPPORTED_RETRIEVAL_MODES}"
        )
    return retrieval_mode
