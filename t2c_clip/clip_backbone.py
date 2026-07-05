"""Transformers CLIP adapters for T2C-CLIP training.

The text encoder injects learnable prompts into the CLIP token embedding
space and runs the real CLIP text transformer plus ``text_projection`` so
that ``f_t`` lives in the same projection space as ``CLIP_ImageEncoder``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from t2c_clip.prompts import validate_index_tensor


class TransformersCLIPImageEncoder(torch.nn.Module):
    """Real CLIP image encoder with an optional SIE camera embedding.

    With ``sie_coe == 0.0`` or ``num_cameras is None`` the encoder is a plain
    wrapper around ``clip_model.get_image_features``. When enabled, a learnable
    per-camera embedding (TransReID's Side Information Embedding) is added to
    every vision token before the pre-layernorm, which requires manually
    replicating the HF CLIP vision forward.
    """

    def __init__(
        self,
        clip_model: torch.nn.Module,
        num_cameras: int | None = None,
        sie_coe: float = 0.0,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.sie_coe = float(sie_coe)
        self.sie_embedding: torch.nn.Embedding | None = None
        if self.sie_coe != 0.0 and num_cameras is not None:
            _require_positive(num_cameras, "num_cameras")
            self.sie_embedding = torch.nn.Embedding(num_cameras, clip_vision_hidden_dim(clip_model))
            torch.nn.init.trunc_normal_(self.sie_embedding.weight, std=SIE_INIT_STD)

    def forward(self, images: torch.Tensor, camera_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.sie_embedding is None:
            # ReID inputs are non-square (e.g. 256x128); the pretrained CLIP ViT
            # only accepts them with position-embedding interpolation enabled.
            output = self.clip_model.get_image_features(
                pixel_values=images, interpolate_pos_encoding=True
            )
            return _image_features_tensor(output)
        if camera_ids is None:
            raise ValueError("camera_ids is required when the SIE camera embedding is enabled")
        validate_index_tensor(camera_ids, self.sie_embedding.num_embeddings, "camera_ids")
        return self._forward_with_sie(images, camera_ids)

    def _forward_with_sie(self, images: torch.Tensor, camera_ids: torch.Tensor) -> torch.Tensor:
        # Manual replica of transformers v5 ``CLIPVisionModel.forward`` (embeddings ->
        # pre_layrnorm -> encoder -> post_layernorm on CLS -> visual_projection)
        # with the SIE term broadcast over all tokens before the pre-layernorm.
        # The zeroed-SIE bit-exact test against ``get_image_features`` is the
        # canary for transformers upgrades changing this pipeline.
        vision_model = self.clip_model.vision_model
        hidden_states = vision_model.embeddings(images, interpolate_pos_encoding=True)
        hidden_states = hidden_states + self.sie_coe * self.sie_embedding(camera_ids).unsqueeze(1)
        hidden_states = vision_model.pre_layrnorm(hidden_states)  # HF spells it 'pre_layrnorm'
        encoder_outputs = vision_model.encoder(inputs_embeds=hidden_states)
        last_hidden_state = encoder_outputs.last_hidden_state
        if hasattr(last_hidden_state, "last_hidden_state"):
            last_hidden_state = last_hidden_state.last_hidden_state
        pooled = vision_model.post_layernorm(last_hidden_state[:, 0, :])
        return self.clip_model.visual_projection(pooled)


class TransformersCLIPTextEncoder(torch.nn.Module):
    """Real CLIP text encoder that injects learnable prompt embeddings.

    The learnable prompt parameters replace token embeddings at a fixed
    context block between the SOT and EOS tokens, then the CLIP text
    transformer + ``text_projection`` are run as usual. Output features are
    L2-normalized so callers can fuse them with image features directly.

    ``prefix_token_ids`` / ``suffix_token_ids`` optionally frame the learnable
    block in a natural-language template::

        [SOT] + prefix tokens + <context_length slots> + suffix tokens + [EOS]

    Empty tuples (the default) keep the bare ``[SOT] + slots + [EOS]`` sequence.
    """

    def __init__(
        self,
        clip_model: torch.nn.Module,
        context_length: int,
        sot_token_id: int,
        eos_token_id: int,
        pad_token_id: int = 0,
        trainable_embeddings: bool = True,
        prefix_token_ids: tuple[int, ...] = (),
        suffix_token_ids: tuple[int, ...] = (),
    ):
        super().__init__()
        _require_positive(context_length, "context_length")
        self.clip_model = clip_model
        self.context_length = context_length
        self.sot_token_id = int(sot_token_id)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.prefix_token_ids = tuple(int(token_id) for token_id in prefix_token_ids)
        self.suffix_token_ids = tuple(int(token_id) for token_id in suffix_token_ids)
        seq_len = 2 + len(self.prefix_token_ids) + context_length + len(self.suffix_token_ids)
        max_positions = int(
            clip_model.text_model.embeddings.position_embedding.weight.shape[0]
        )
        if seq_len > max_positions:
            raise ValueError(
                f"prompt sequence length {seq_len} (SOT + template + {context_length} "
                f"slots + EOS) exceeds the CLIP text position embeddings ({max_positions})"
            )
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
        template_ids = (
            self.sot_token_id,
            *self.prefix_token_ids,
            *((self.pad_token_id,) * self.context_length),
            *self.suffix_token_ids,
            self.eos_token_id,
        )
        seq_len = len(template_ids)
        template = torch.tensor(template_ids, dtype=torch.long, device=device)
        template = template.unsqueeze(0).expand(batch_size, seq_len)
        token_embedding = self.clip_model.text_model.embeddings.token_embedding
        position_embedding = self.clip_model.text_model.embeddings.position_embedding
        token_embeds = token_embedding(template)
        token_embeds = token_embeds.clone()
        prompt_start = 1 + len(self.prefix_token_ids)
        prompt_block = slice(prompt_start, prompt_start + self.context_length)
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
        # EOS sits at the last position of our fixed template.
        eos_index = 1 + len(self.prefix_token_ids) + self.context_length + len(self.suffix_token_ids)
        eos_position = torch.full((batch_size,), eos_index, dtype=torch.long, device=device)
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


def clip_vision_hidden_dim(clip_model: Any) -> int:
    hidden = getattr(
        getattr(getattr(clip_model, "config", None), "vision_config", None), "hidden_size", None
    )
    if not isinstance(hidden, int):
        raise ValueError("CLIP model config must expose integer vision_config.hidden_size")
    _require_positive(hidden, "vision_hidden_dim")
    return hidden


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
SIE_INIT_STD = 0.02