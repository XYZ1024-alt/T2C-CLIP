"""Transformers CLIP adapters for T2C-CLIP training.

The text encoder injects learnable prompts into the CLIP token embedding
space and runs the real CLIP text transformer plus ``text_projection`` so
that ``f_t`` lives in the same projection space as ``CLIP_ImageEncoder``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class TransformersCLIPImageEncoder(torch.nn.Module):
    def __init__(self, clip_model: torch.nn.Module):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # ReID inputs are non-square (e.g. 256x128); the pretrained CLIP ViT
        # only accepts them with position-embedding interpolation enabled.
        output = self.clip_model.get_image_features(
            pixel_values=images, interpolate_pos_encoding=True
        )
        return _image_features_tensor(output)


class TransformersCLIPTextEncoder(torch.nn.Module):
    """Real CLIP text encoder that injects learnable prompt embeddings.

    The learnable prompt parameters replace token embeddings at a fixed
    context block between the SOT and EOS tokens, then the CLIP text
    transformer + ``text_projection`` are run as usual. Output features are
    L2-normalized so callers can fuse them with image features directly.
    """

    def __init__(
        self,
        clip_model: torch.nn.Module,
        context_length: int,
        sot_token_id: int,
        eos_token_id: int,
        pad_token_id: int = 0,
        trainable_embeddings: bool = True,
    ):
        super().__init__()
        _require_positive(context_length, "context_length")
        self.clip_model = clip_model
        self.context_length = context_length
        self.sot_token_id = int(sot_token_id)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self._freeze_unfreeze(trainable_embeddings)

    def forward(self, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        if prompt_embeddings.ndim != 3:
            raise ValueError("prompt_embeddings must have shape [batch, context_length, hidden_dim]")
        batch_size, context_length, _ = prompt_embeddings.shape
        if context_length != self.context_length:
            raise ValueError(
                f"prompt_embeddings context length {context_length} does not match encoder context_length {self.context_length}"
            )
        hidden_states = self._build_hidden_states(prompt_embeddings, batch_size)
        encoded = self._run_text_transformer(hidden_states)
        pooled = self._pool_eos(encoded, batch_size)
        projected = self.clip_model.text_projection(pooled)
        return F.normalize(projected, p=2.0, dim=-1, eps=DEFAULT_EPS)

    def embedding_dim(self) -> int:
        return int(self.clip_model.text_model.embeddings.token_embedding.weight.shape[1])

    def _build_hidden_states(self, prompt_embeddings: torch.Tensor, batch_size: int) -> torch.Tensor:
        device = prompt_embeddings.device
        seq_len = self.context_length + 2
        template = torch.full(
            (batch_size, seq_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        template[:, 0] = self.sot_token_id
        template[:, self.context_length + 1] = self.eos_token_id
        token_embedding = self.clip_model.text_model.embeddings.token_embedding
        position_embedding = self.clip_model.text_model.embeddings.position_embedding
        token_embeds = token_embedding(template)
        token_embeds = token_embeds.clone()
        prompt_block = slice(1, self.context_length + 1)
        token_embeds[:, prompt_block, :] = prompt_embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        position_embeds = position_embedding(position_ids)
        return token_embeds + position_embeds

    def _run_text_transformer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        text_model = self.clip_model.text_model
        # ``is_causal=True`` with ``attention_mask=None`` lets the CLIP attention
        # dispatch realize causal masking via SDPA's native ``is_causal`` argument
        # without needing a separately materialized 4D causal mask.
        encoder_outputs = text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            is_causal=True,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        if hasattr(last_hidden_state, "last_hidden_state"):
            last_hidden_state = last_hidden_state.last_hidden_state
        return text_model.final_layer_norm(last_hidden_state)

    def _pool_eos(self, hidden_states: torch.Tensor, batch_size: int) -> torch.Tensor:
        device = hidden_states.device
        seq_len = self.context_length + 2
        # EOS sits at the last position of our fixed template.
        eos_position = torch.full((batch_size,), self.context_length + 1, dtype=torch.long, device=device)
        return hidden_states[torch.arange(batch_size, device=device), eos_position]

    def _freeze_unfreeze(self, trainable: bool) -> None:
        for parameter in self.clip_model.text_model.parameters():
            parameter.requires_grad_(trainable)
        for parameter in self.clip_model.text_projection.parameters():
            parameter.requires_grad_(trainable)


class PromptTextEncoder(torch.nn.Module):
    """Deprecated random projection text encoder.

    Kept only for backwards-compatible test imports. New training paths
    must use :class:`TransformersCLIPTextEncoder`.
    """

    def __init__(self, prompt_embedding_dim: int, output_dim: int):
        super().__init__()
        _require_positive(prompt_embedding_dim, "prompt_embedding_dim")
        _require_positive(output_dim, "output_dim")
        self.projection = torch.nn.Linear(prompt_embedding_dim, output_dim)

    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        if prompts.ndim != 3:
            raise ValueError("prompts must have shape [batch, context_length, embedding_dim]")
        return self.projection(prompts.mean(dim=1))


def clip_projection_dim(clip_model: Any) -> int:
    projection_dim = getattr(getattr(clip_model, "config", None), "projection_dim", None)
    if not isinstance(projection_dim, int):
        raise ValueError("CLIP model config must expose integer projection_dim")
    _require_positive(projection_dim, "projection_dim")
    return projection_dim


def clip_text_hidden_dim(clip_model: Any) -> int:
    embedding = (
        getattr(getattr(getattr(clip_model, "text_model", None), "embeddings", None), "token_embedding", None)
    )
    weight = getattr(embedding, "weight", None) if embedding is not None else None
    if weight is None or weight.ndim != 2:
        raise ValueError("CLIP model must expose text_model.embeddings.token_embedding with weight [vocab, hidden]")
    hidden = int(weight.shape[1])
    _require_positive(hidden, "text_hidden_dim")
    return hidden


def _image_features_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state
    raise TypeError("CLIP image features must be a tensor or expose tensor pooler_output / last_hidden_state")


def _require_positive(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


DEFAULT_EPS = 1e-12