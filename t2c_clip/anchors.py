"""Identity anchor matrix provider for the Stage-2 image-to-text cross entropy.

CLIP-ReID's Stage-2 classifies each image feature against the text features
of ALL train identities, not just the batch's. The anchors are
camera-agnostic identity prototypes (``global + identity`` prompts, no camera
component) encoded through the model's text tower in chunks, always under
``torch.no_grad()`` and detached — no gradient reaches the prompt bank or the
text encoder through this term.

Freshness semantics:

- ``frozen=True`` (prompt bank frozen in Stage-2): the matrix is computed
  once lazily at first use and cached for the whole stage — true CLIP-ReID
  text-feature fixation. Epoch boundaries are ignored.
- ``frozen=False``: :meth:`start_epoch` invalidates the cache, so the matrix
  is re-encoded at the start of every Stage-2 epoch and acts as a
  slowly-moving teacher.
"""

from __future__ import annotations

import torch

from t2c_clip.model import T2CClipModel

DEFAULT_CHUNK_SIZE = 256


class IdentityAnchorProvider:
    """Lazily encodes and caches the ``(num_train_ids, projection_dim)`` anchor matrix."""

    def __init__(
        self,
        model: T2CClipModel,
        num_train_ids: int,
        frozen: bool,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        if num_train_ids < 1:
            raise ValueError("num_train_ids must be positive")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self._model = model
        self._num_train_ids = num_train_ids
        self._frozen = frozen
        self._chunk_size = chunk_size
        self._cache: torch.Tensor | None = None

    def start_epoch(self) -> None:
        """Stage-2 epoch boundary hook: drop the cache unless the prompts are frozen."""
        if not self._frozen:
            self._cache = None

    def anchors(self) -> torch.Tensor:
        if self._cache is None:
            self._cache = self._encode_all()
        return self._cache

    def _encode_all(self) -> torch.Tensor:
        device = self._model.prompt_bank.identity_prompts.device
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, self._num_train_ids, self._chunk_size):
                stop = min(start + self._chunk_size, self._num_train_ids)
                person_ids = torch.arange(start, stop, device=device)
                prompts = self._model.prompt_bank.identity_anchor_prompts(person_ids)
                chunks.append(self._model.encode_text(prompts))
        return torch.cat(chunks, dim=0).detach()
