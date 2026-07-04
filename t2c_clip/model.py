"""Injectable dual-stream T2C-CLIP model wiring.

The model has three explicit forward paths:

- ``forward_stage1``: prompt alignment with identity-aware training text.
- ``forward_stage2``: ReID training. Discriminative losses act on the image
  feature (``visual_raw``) and its ``feature_head`` (BNNeck) output; the
  image-to-text term classifies ``visual`` against a precomputed identity
  anchor matrix, so no per-sample training text is encoded here.
- ``encode_retrieval``: validation and inference retrieval with either fused
  global + camera prompts or image-only features, both routed through the
  same ``feature_head``.

``set_inference_text_cache`` installs a precomputed ``(num_cameras, dim)``
retrieval text matrix; while installed, ``encode_inference_text`` indexes it
instead of running the text tower. The caller owns the guarantee that the
camera text is constant (prompt bank and text encoder both frozen).
"""

from __future__ import annotations

import torch

from t2c_clip.features import fuse_features, l2_normalize
from t2c_clip.prompts import PromptBank, _validate_index_tensor
from t2c_clip.retrieval import FUSED_RETRIEVAL, IMAGE_ONLY_RETRIEVAL, require_retrieval_mode


class T2CClipModel(torch.nn.Module):
    def __init__(
        self,
        image_encoder: torch.nn.Module,
        text_encoder: torch.nn.Module,
        prompt_bank: PromptBank,
        beta: float,
        feature_head: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.prompt_bank = prompt_bank
        self.beta = float(beta)
        self.feature_head = torch.nn.Identity() if feature_head is None else feature_head
        self.inference_text_cache: torch.Tensor | None = None

    def encode_retrieval(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        retrieval_mode: str = FUSED_RETRIEVAL,
    ) -> torch.Tensor:
        """Inference / validation retrieval feature."""
        bn_features = self.feature_head(self.encode_visual_raw(images))
        mode = require_retrieval_mode(retrieval_mode)
        if mode == IMAGE_ONLY_RETRIEVAL:
            return l2_normalize(bn_features)
        text = self.encode_inference_text(camera_ids)
        return fuse_features(bn_features, text, self.beta)

    def encode_visual_raw(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.encode_visual_raw(images))

    def encode_inference_text(self, camera_ids: torch.Tensor) -> torch.Tensor:
        if self.inference_text_cache is not None:
            _validate_index_tensor(camera_ids, self.inference_text_cache.shape[0], "camera_ids")
            return self.inference_text_cache[camera_ids]
        prompts = self.prompt_bank.inference_prompts(camera_ids)
        return self.encode_text(prompts)

    def set_inference_text_cache(self, cache: torch.Tensor | None) -> None:
        """Install (or clear with ``None``) the precomputed per-camera retrieval text.

        Only valid while the camera text is provably constant — the prompt
        bank and the text encoder must both be frozen; the caller owns that
        guarantee.
        """
        if cache is not None:
            if cache.ndim != 2:
                raise ValueError("inference text cache must be a rank-2 (num_cameras, dim) tensor")
            if cache.shape[0] != self.prompt_bank.camera_prompts.shape[0]:
                raise ValueError("inference text cache must have exactly one row per camera prompt")
        self.inference_text_cache = cache

    def encode_training_text(self, camera_ids: torch.Tensor, person_ids: torch.Tensor) -> torch.Tensor:
        prompts = self.prompt_bank.training_prompts(camera_ids, person_ids)
        return self.encode_text(prompts)

    def forward_stage1(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        person_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Stage-1 prompt alignment forward."""
        visual = self.encode_visual(images)
        text = self.encode_training_text(camera_ids, person_ids)
        return {"visual": visual, "text": text}

    def forward_stage2(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        person_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Stage-2 ReID training forward.

        ``person_ids`` no longer selects per-sample training text: the Stage-2
        image-to-text loss classifies ``visual`` against the precomputed
        identity anchor matrix instead. The parameter is kept so both stage
        forwards share one signature.
        """
        _ = person_ids
        visual_raw = self.encode_visual_raw(images)
        bn_features = self.feature_head(visual_raw)
        retrieval_text = self.encode_inference_text(camera_ids)
        retrieval = fuse_features(bn_features, retrieval_text, self.beta)
        return {
            "visual_raw": visual_raw,
            "bn": bn_features,
            "visual": l2_normalize(visual_raw),
            "retrieval": retrieval,
        }

    def forward_training(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        person_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Backwards-compatible alias for Stage-2 training forward."""
        return self.forward_stage2(images, camera_ids, person_ids)

    def encode_text(self, prompts: torch.Tensor) -> torch.Tensor:
        """Encode prompt embeddings through the configured text encoder."""
        text_features = self.text_encoder(prompts)
        return l2_normalize(text_features)
