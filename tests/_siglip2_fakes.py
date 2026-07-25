"""Reusable SigLIP 2-shaped fakes for unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from transformers import Siglip2TextConfig, Siglip2VisionConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling


class FakeSiglip2Tokenizer:
    bos_token_id = 2
    eos_token_id = 1
    pad_token_id = 0
    padding_side = "left"

    _word_ids = {"a": 5, "photo": 6, "of": 7, "person": 8, ".": 9}

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        **_: Any,
    ) -> dict[str, list[int]]:
        ids = [self._word_ids[word] for word in text.split()]
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
        return {"input_ids": ids}


class FakeTextEmbeddings(nn.Module):
    def __init__(self, config: Siglip2TextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            inputs_embeds = self.token_embedding(input_ids)
        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)
        return inputs_embeds + self.position_embedding(position_ids)


class CountingIdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.last_inputs_embeds: torch.Tensor | None = None
        self.last_attention_mask: torch.Tensor | None = None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> BaseModelOutput:
        self.call_count += 1
        self.last_inputs_embeds = inputs_embeds
        self.last_attention_mask = attention_mask
        return BaseModelOutput(last_hidden_state=inputs_embeds)


class FakeTextModel(nn.Module):
    def __init__(self, config: Siglip2TextConfig):
        super().__init__()
        self.config = config
        self.embeddings = FakeTextEmbeddings(config)
        self.encoder = CountingIdentityEncoder()
        self.final_layer_norm = nn.Identity()
        self.head = nn.Linear(config.hidden_size, config.projection_size, bias=True)


class FakeVisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches
        self.patch_embedding = nn.Linear(
            config.num_channels * config.patch_size * config.patch_size,
            config.hidden_size,
        )
        self.position_embedding = nn.Embedding(config.num_patches, config.hidden_size)

    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        _ = spatial_shapes
        positions = torch.arange(
            pixel_values.shape[1], device=pixel_values.device
        ).unsqueeze(0)
        return self.patch_embedding(pixel_values) + self.position_embedding(positions)


class FakeVisionHead(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pixel_attention_mask is None:
            return hidden_states.mean(dim=1)
        weights = pixel_attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1) / weights.sum(dim=1)


class FakeVisionModel(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = FakeVisionEmbeddings(config)
        self.encoder = CountingIdentityEncoder()
        self.post_layernorm = nn.Identity()
        self.use_head = True
        self.head = FakeVisionHead()
        self.call_count = 0

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> BaseModelOutputWithPooling:
        self.call_count += 1
        hidden = self.embeddings(pixel_values, spatial_shapes)
        encoded = self.encoder(
            inputs_embeds=hidden,
            attention_mask=pixel_attention_mask,
        ).last_hidden_state
        pooled = self.head(self.post_layernorm(encoded), pixel_attention_mask)
        return BaseModelOutputWithPooling(
            last_hidden_state=encoded,
            pooler_output=pooled,
        )


class FakeSiglip2(nn.Module):
    """Small SigLIP 2-shaped model with a 20x20 positional patch budget."""

    def __init__(
        self,
        hidden_size: int = 8,
        feature_dim: int | None = None,
        max_position_embeddings: int = 16,
        patch_size: int = 14,
        num_patches: int = 400,
        vocab_size: int = 32,
    ):
        super().__init__()
        feature_dim = hidden_size if feature_dim is None else feature_dim
        text_config = Siglip2TextConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=1,
            projection_size=feature_dim,
        )
        vision_config = Siglip2VisionConfig(
            hidden_size=feature_dim,
            intermediate_size=feature_dim * 2,
            num_hidden_layers=1,
            num_attention_heads=2,
            patch_size=patch_size,
            num_patches=num_patches,
        )
        self.config = SimpleNamespace(
            model_type="siglip",
            text_config=text_config,
            vision_config=vision_config,
        )
        self.text_model = FakeTextModel(text_config)
        self.vision_model = FakeVisionModel(vision_config)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_bias = nn.Parameter(torch.tensor(-1.0))
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    @property
    def image_feature_calls(self) -> int:
        return self.vision_model.call_count

    @image_feature_calls.setter
    def image_feature_calls(self, value: int) -> None:
        self.vision_model.call_count = int(value)


class FakeSiglip2ImageProcessor:
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]

    def __init__(self, patch_size: int = 14, max_num_patches: int = 400):
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches


class ImageAwareFakeImageProcessor(FakeSiglip2ImageProcessor):
    pass
