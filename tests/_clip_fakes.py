"""Reusable fakes for CLIP-backed T2C-CLIP unit tests.

The fakes mirror the structure (not the semantics) of ``CLIPModel`` — enough
to drive :class:`TransformersCLIPImageEncoder` and
:class:`TransformersCLIPTextEncoder` without loading real weights.
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput


DEFAULT_VOCAB_SIZE = 49408  # CLIP standard vocab so default SOT/EOS token IDs are valid.


class FakeClipTextEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        seq = input_ids.shape[-1]
        if position_ids is None:
            position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class FakeClipTextEncoder(nn.Module):
    """Identity-transformer fake — returns inputs unchanged through the encoder."""

    def __init__(
        self,
        hidden_size: int,
        max_position_embeddings: int,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
    ):
        super().__init__()
        self.embeddings = FakeClipTextEmbeddings(vocab_size, hidden_size, max_position_embeddings)
        self.encoder = _FakeClipEncoder()
        self.final_layer_norm = nn.Identity()
        self.config = Namespace(
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            bos_token_id=49406,
            eos_token_id=49407,
            pad_token_id=0,
            vocab_size=vocab_size,
            is_causal=True,
            _attn_implementation="sdpa",
        )
        self.eos_token_id = 49407


class _FakeClipEncoder(nn.Module):
    def forward(self, inputs_embeds: torch.Tensor, attention_mask: Any = None, is_causal: bool = True, **kwargs) -> BaseModelOutput:
        _ = attention_mask, is_causal, kwargs
        self.last_inputs_embeds = inputs_embeds
        return BaseModelOutput(last_hidden_state=inputs_embeds)


class FakeCLIPTokenizer:
    """Word-level stand-in for the CLIP BPE tokenizer.

    Knows only the prompt-template words and wraps encodings in SOT/EOS like
    the real tokenizer, so job-builder tests can verify special-token stripping.
    """

    bos_token_id = 49406
    eos_token_id = 49407

    _word_ids = {"a": 320, "photo": 1125, "of": 539, "person": 2533, ".": 269}

    def __call__(self, text: str) -> dict[str, list[int]]:
        ids = [self._word_ids[word] for word in text.split()]
        return {"input_ids": [self.bos_token_id, *ids, self.eos_token_id]}


class FakeCLIP(nn.Module):
    """Minimal CLIP-shaped test double.

    - ``get_image_features`` mean-pools a (B, C, H, W) pixel batch and maps
      to the shared projection space via ``visual_projection``.
    - ``text_model`` drives :class:`TransformersCLIPTextEncoder` with the
      standard CLIP token embedding table size so SOT/EOS IDs are valid.
    - ``text_projection`` maps text hidden states to ``projection_dim``.
    """

    def __init__(
        self,
        hidden_size: int = 8,
        projection_dim: int = 2,
        max_position_embeddings: int = 64,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
    ):
        super().__init__()
        self.config = Namespace(
            projection_dim=projection_dim,
            text_config=Namespace(
                hidden_size=hidden_size,
                max_position_embeddings=max_position_embeddings,
                bos_token_id=49406,
                eos_token_id=49407,
                pad_token_id=0,
                vocab_size=vocab_size,
            ),
            vision_config=Namespace(hidden_size=projection_dim),
        )
        self.text_model = FakeClipTextEncoder(hidden_size, max_position_embeddings, vocab_size)
        self.vision_model = nn.Identity()
        self.text_projection = nn.Linear(hidden_size, projection_dim, bias=False)
        self.visual_projection = nn.Linear(projection_dim, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.last_interpolate_pos_encoding: bool | None = None
        self.image_feature_calls = 0

    def get_image_features(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        self.last_interpolate_pos_encoding = interpolate_pos_encoding
        self.image_feature_calls += 1
        pooled = pixel_values.mean(dim=(2, 3))  # [B, channels]
        target_dim = self.config.vision_config.hidden_size
        if pooled.shape[1] < target_dim:
            zeros = torch.zeros(
                pooled.shape[0],
                target_dim - pooled.shape[1],
                device=pooled.device,
                dtype=pooled.dtype,
            )
            pooled = torch.cat([pooled, zeros], dim=1)
        elif pooled.shape[1] > target_dim:
            pooled = pooled[:, :target_dim]
        return self.visual_projection(pooled) * self.scale


class FakeImageProcessor:
    """Mimics a CLIP image processor returning ``{"pixel_values": ...}``.

    Always returns a constant pixel tensor (default ones) independent of the
    input image so tests can assert exact tensor equality. Exposes identity
    ``image_mean``/``image_std`` so ReID transforms can read normalization
    stats without changing pixel values.
    """

    def __init__(self, fill_value: float = 1.0):
        self.call_count = 0
        self.fill_value = fill_value
        self.image_mean = [0.0, 0.0, 0.0]
        self.image_std = [1.0, 1.0, 1.0]

    def __call__(self, images, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        self.call_count += 1
        self.return_tensors = return_tensors
        self.image_mode = getattr(images, "mode", None)
        values = torch.full((1, 3, 2, 2), float(self.fill_value))
        return {"pixel_values": values}


class ImageAwareFakeImageProcessor:
    """Returns pixel values derived from the actual pixel at coordinate (0, 0)."""

    image_mean = [0.0, 0.0, 0.0]
    image_std = [1.0, 1.0, 1.0]

    def __call__(self, images, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        pixel = torch.tensor(list(images.getpixel((0, 0))), dtype=torch.float32) / 255.0
        values = pixel.view(1, 3, 1, 1).expand(1, 3, 2, 2)
        return {"pixel_values": values}