"""Transformers adapters for the fixed and patchified SigLIP 2 formats.

Hugging Face publishes ``google/siglip2-so400m-patch14-384`` as a fixed
``SiglipModel`` that consumes BCHW images. Native ``Siglip2Model`` checkpoints
consume patch vectors plus a patch mask and spatial shape. The image adapter
keeps the ReID boundary on BCHW tensors and dispatches to the loaded model's
actual format.

The text adapter injects learnable prompts into token-embedding space and
preserves the loaded checkpoint's fixed-length padding and final-position
pooling semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F
from transformers.masking_utils import create_bidirectional_mask

from t2c_clip.prompts import validate_index_tensor


SIGLIP2_MODEL_ID = "google/siglip2-so400m-patch14-384"


@dataclass(frozen=True)
class Siglip2VisionInputs:
    pixel_values: torch.Tensor
    pixel_attention_mask: torch.Tensor
    spatial_shapes: torch.Tensor


class TransformersSiglip2ImageEncoder(torch.nn.Module):
    """SigLIP 2 image encoder with optional SIE camera conditioning."""

    def __init__(
        self,
        siglip2_model: torch.nn.Module,
        num_cameras: int | None = None,
        sie_coe: float = 0.0,
    ):
        super().__init__()
        self.siglip2_model = siglip2_model
        self.patch_size = siglip2_patch_size(siglip2_model)
        self.max_num_patches = siglip2_max_num_patches(siglip2_model)
        self.uses_patchified_inputs = siglip2_uses_patchified_inputs(siglip2_model)
        self.sie_coe = float(sie_coe)
        self.sie_embedding: torch.nn.Embedding | None = None
        if self.sie_coe != 0.0 and num_cameras is not None:
            _require_positive(num_cameras, "num_cameras")
            self.sie_embedding = torch.nn.Embedding(
                num_cameras, siglip2_vision_hidden_dim(siglip2_model)
            )
            torch.nn.init.trunc_normal_(self.sie_embedding.weight, std=SIE_INIT_STD)

    def forward(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.uses_patchified_inputs:
            inputs = patchify_siglip2_images(
                images,
                patch_size=self.patch_size,
                max_num_patches=self.max_num_patches,
            )
            if self.sie_embedding is None:
                output = self.siglip2_model.vision_model(
                    pixel_values=inputs.pixel_values,
                    pixel_attention_mask=inputs.pixel_attention_mask,
                    spatial_shapes=inputs.spatial_shapes,
                )
                return _pooler_output(output, "SigLIP 2 vision model")
        else:
            _validate_image_tensor_size(
                images,
                patch_size=self.patch_size,
                max_num_patches=self.max_num_patches,
                require_divisible=False,
            )
            if self.sie_embedding is None:
                output = self.siglip2_model.vision_model(
                    pixel_values=images,
                    interpolate_pos_encoding=True,
                )
                return _pooler_output(output, "SigLIP 2 fixed vision model")
            inputs = images
        if camera_ids is None:
            raise ValueError("camera_ids is required when the SIE camera embedding is enabled")
        validate_index_tensor(camera_ids, self.sie_embedding.num_embeddings, "camera_ids")
        if self.uses_patchified_inputs:
            return self._forward_patchified_with_sie(inputs, camera_ids)
        return self._forward_fixed_with_sie(inputs, camera_ids)

    def _forward_patchified_with_sie(
        self,
        inputs: Siglip2VisionInputs,
        camera_ids: torch.Tensor,
    ) -> torch.Tensor:
        vision_model = self.siglip2_model.vision_model
        hidden_states = vision_model.embeddings(
            inputs.pixel_values,
            inputs.spatial_shapes,
        )
        hidden_states = hidden_states + self.sie_coe * self.sie_embedding(camera_ids).unsqueeze(1)
        attention_mask = create_bidirectional_mask(
            config=vision_model.config,
            inputs_embeds=hidden_states,
            attention_mask=inputs.pixel_attention_mask,
        )
        encoder_outputs = vision_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )
        last_hidden_state = _last_hidden_state(encoder_outputs)
        last_hidden_state = vision_model.post_layernorm(last_hidden_state)
        if not getattr(vision_model, "use_head", False):
            raise ValueError("SigLIP 2 vision model must expose its attention pooling head")
        return vision_model.head(last_hidden_state, inputs.pixel_attention_mask)

    def _forward_fixed_with_sie(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> torch.Tensor:
        vision_model = self.siglip2_model.vision_model
        hidden_states = vision_model.embeddings(
            images,
            interpolate_pos_encoding=True,
        )
        hidden_states = hidden_states + self.sie_coe * self.sie_embedding(camera_ids).unsqueeze(1)
        encoder_outputs = vision_model.encoder(inputs_embeds=hidden_states)
        last_hidden_state = vision_model.post_layernorm(
            _last_hidden_state(encoder_outputs)
        )
        if not getattr(vision_model, "use_head", False):
            raise ValueError("SigLIP 2 vision model must expose its attention pooling head")
        return vision_model.head(last_hidden_state)


class TransformersSiglip2TextEncoder(torch.nn.Module):
    """SigLIP 2 text encoder with fixed-length learnable prompt injection."""

    def __init__(
        self,
        siglip2_model: torch.nn.Module,
        context_length: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        trainable_embeddings: bool = True,
        prefix_token_ids: tuple[int, ...] = (),
        suffix_token_ids: tuple[int, ...] = (),
        left_padding: bool = True,
        include_bos_token: bool = True,
        mask_padding: bool = True,
    ):
        super().__init__()
        _require_positive(context_length, "context_length")
        self.siglip2_model = siglip2_model
        self.context_length = context_length
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.prefix_token_ids = tuple(int(token_id) for token_id in prefix_token_ids)
        self.suffix_token_ids = tuple(int(token_id) for token_id in suffix_token_ids)
        self.left_padding = bool(left_padding)
        self.include_bos_token = bool(include_bos_token)
        self.mask_padding = bool(mask_padding)
        self.max_length = int(
            siglip2_model.text_model.embeddings.position_embedding.weight.shape[0]
        )
        content_length = (
            1
            + int(self.include_bos_token)
            + len(self.prefix_token_ids)
            + context_length
            + len(self.suffix_token_ids)
        )
        if content_length > self.max_length:
            raise ValueError(
                f"prompt content length {content_length} (special tokens + template + "
                f"{context_length} slots) exceeds the SigLIP 2 text length "
                f"({self.max_length})"
            )
        self.padding_length = self.max_length - content_length
        self._freeze_unfreeze(trainable_embeddings)

    def forward(self, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        if prompt_embeddings.ndim != 3:
            raise ValueError(
                "prompt_embeddings must have shape [batch, context_length, hidden_dim]"
            )
        batch_size, context_length, hidden_dim = prompt_embeddings.shape
        if context_length != self.context_length:
            raise ValueError(
                f"prompt_embeddings context length {context_length} does not match "
                f"encoder context_length {self.context_length}"
            )
        expected_hidden_dim = self.embedding_dim()
        if hidden_dim != expected_hidden_dim:
            raise ValueError(
                f"prompt_embeddings hidden dim {hidden_dim} does not match SigLIP 2 "
                f"text hidden dim {expected_hidden_dim}"
            )
        hidden_states, attention_mask = self._build_hidden_states(
            prompt_embeddings, batch_size
        )
        encoded = self._run_text_transformer(hidden_states, attention_mask)
        projected = self.siglip2_model.text_model.head(encoded[:, -1, :])
        return F.normalize(projected, p=2.0, dim=-1, eps=DEFAULT_EPS)

    def embedding_dim(self) -> int:
        return int(
            self.siglip2_model.text_model.embeddings.token_embedding.weight.shape[1]
        )

    def _build_hidden_states(
        self,
        prompt_embeddings: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = prompt_embeddings.device
        content_ids = (
            *((self.bos_token_id,) if self.include_bos_token else ()),
            *self.prefix_token_ids,
            *((self.pad_token_id,) * self.context_length),
            *self.suffix_token_ids,
            self.eos_token_id,
        )
        padding_ids = (self.pad_token_id,) * self.padding_length
        if self.left_padding:
            template_ids = (*padding_ids, *content_ids)
            prompt_start = (
                self.padding_length
                + int(self.include_bos_token)
                + len(self.prefix_token_ids)
            )
        else:
            template_ids = (*content_ids, *padding_ids)
            prompt_start = int(self.include_bos_token) + len(self.prefix_token_ids)
        template = torch.tensor(template_ids, dtype=torch.long, device=device)
        template = template.unsqueeze(0).expand(batch_size, self.max_length)
        embeddings = self.siglip2_model.text_model.embeddings
        token_embeds = embeddings.token_embedding(template).clone()
        prompt_block = slice(prompt_start, prompt_start + self.context_length)
        token_embeds[:, prompt_block, :] = prompt_embeddings
        position_ids = torch.arange(self.max_length, device=device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, self.max_length)
        hidden_states = embeddings(inputs_embeds=token_embeds, position_ids=position_ids)
        if not self.mask_padding:
            return hidden_states, None
        attention_mask = torch.ones(
            (batch_size, self.max_length), dtype=torch.long, device=device
        )
        if self.padding_length:
            if self.left_padding:
                attention_mask[:, : self.padding_length] = 0
            else:
                attention_mask[:, -self.padding_length :] = 0
        return hidden_states, attention_mask

    def _run_text_transformer(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        text_model = self.siglip2_model.text_model
        encoder_attention_mask = create_bidirectional_mask(
            config=text_model.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
        )
        encoder_outputs = text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attention_mask,
        )
        return text_model.final_layer_norm(_last_hidden_state(encoder_outputs))

    def _freeze_unfreeze(self, trainable: bool) -> None:
        for parameter in self.siglip2_model.text_model.parameters():
            parameter.requires_grad_(trainable)


def _validate_image_tensor_size(
    images: torch.Tensor,
    *,
    patch_size: int,
    max_num_patches: int,
    require_divisible: bool = True,
) -> tuple[int, int, int, int, int]:
    if images.ndim != 4:
        raise ValueError("images must have shape [batch, channels, height, width]")
    _require_positive(patch_size, "patch_size")
    _require_positive(max_num_patches, "max_num_patches")
    batch_size, channels, height, width = images.shape
    if batch_size < 1 or channels < 1:
        raise ValueError("images must contain at least one image and one channel")
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"image size {height}x{width} must contain at least one "
            f"{patch_size}x{patch_size} patch"
        )
    if require_divisible and (
        height % patch_size != 0 or width % patch_size != 0
    ):
        raise ValueError(
            f"image size {height}x{width} must be divisible by SigLIP 2 patch size "
            f"{patch_size}"
        )
    patches_height = height // patch_size
    patches_width = width // patch_size
    patch_count = patches_height * patches_width
    if patch_count > max_num_patches:
        raise ValueError(
            f"image size {height}x{width} produces {patch_count} patches, exceeding "
            f"the SigLIP 2 budget of {max_num_patches}"
        )
    return batch_size, channels, patches_height, patches_width, patch_count


def patchify_siglip2_images(
    images: torch.Tensor,
    *,
    patch_size: int,
    max_num_patches: int,
) -> Siglip2VisionInputs:
    """Patchify normalized BCHW images in the H-W-C ordering used by SigLIP 2."""

    batch_size, channels, patches_height, patches_width, patch_count = (
        _validate_image_tensor_size(
            images,
            patch_size=patch_size,
            max_num_patches=max_num_patches,
        )
    )
    patches = images.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )
    patches = patches.permute(0, 2, 3, 4, 5, 1).contiguous()
    pixel_values = patches.view(
        batch_size, patch_count, patch_size * patch_size * channels
    )
    pixel_attention_mask = torch.ones(
        (batch_size, patch_count), dtype=torch.long, device=images.device
    )
    spatial_shapes = torch.tensor(
        (patches_height, patches_width), dtype=torch.long, device=images.device
    )
    spatial_shapes = spatial_shapes.unsqueeze(0).expand(batch_size, 2).clone()
    return Siglip2VisionInputs(pixel_values, pixel_attention_mask, spatial_shapes)


def validate_siglip2_image_size(
    image_size: tuple[int, int],
    siglip2_model: Any,
) -> int:
    if len(image_size) != 2:
        raise ValueError("image_size must be a (height, width) pair")
    height, width = (int(image_size[0]), int(image_size[1]))
    _require_positive(height, "image_height")
    _require_positive(width, "image_width")
    patch_size = siglip2_patch_size(siglip2_model)
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"image size {height}x{width} must contain at least one "
            f"{patch_size}x{patch_size} patch"
        )
    if siglip2_uses_patchified_inputs(siglip2_model) and (
        height % patch_size != 0 or width % patch_size != 0
    ):
        raise ValueError(
            f"image size {height}x{width} must be divisible by SigLIP 2 patch size "
            f"{patch_size}"
        )
    patch_count = (height // patch_size) * (width // patch_size)
    max_num_patches = siglip2_max_num_patches(siglip2_model)
    if patch_count > max_num_patches:
        raise ValueError(
            f"image size {height}x{width} produces {patch_count} patches, exceeding "
            f"the SigLIP 2 budget of {max_num_patches}"
        )
    return patch_count


def siglip2_feature_dim(siglip2_model: Any) -> int:
    text_projection = getattr(
        getattr(getattr(siglip2_model, "config", None), "text_config", None),
        "projection_size",
        None,
    )
    vision_hidden = getattr(
        getattr(getattr(siglip2_model, "config", None), "vision_config", None),
        "hidden_size",
        None,
    )
    if not isinstance(text_projection, int):
        raise ValueError("SigLIP 2 config must expose integer text_config.projection_size")
    if not isinstance(vision_hidden, int):
        raise ValueError("SigLIP 2 config must expose integer vision_config.hidden_size")
    _require_positive(text_projection, "text_projection_size")
    _require_positive(vision_hidden, "vision_hidden_dim")
    if text_projection != vision_hidden:
        raise ValueError(
            "SigLIP 2 text projection size and vision feature size must match "
            f"({text_projection} != {vision_hidden})"
        )
    return text_projection


def siglip2_vision_hidden_dim(siglip2_model: Any) -> int:
    hidden = getattr(
        getattr(getattr(siglip2_model, "config", None), "vision_config", None),
        "hidden_size",
        None,
    )
    if not isinstance(hidden, int):
        raise ValueError("SigLIP 2 config must expose integer vision_config.hidden_size")
    _require_positive(hidden, "vision_hidden_dim")
    return hidden


def siglip2_text_hidden_dim(siglip2_model: Any) -> int:
    embedding = getattr(
        getattr(getattr(siglip2_model, "text_model", None), "embeddings", None),
        "token_embedding",
        None,
    )
    weight = getattr(embedding, "weight", None) if embedding is not None else None
    if weight is None or weight.ndim != 2:
        raise ValueError(
            "SigLIP 2 model must expose text_model.embeddings.token_embedding "
            "with weight [vocab, hidden]"
        )
    hidden = int(weight.shape[1])
    _require_positive(hidden, "text_hidden_dim")
    return hidden


def siglip2_patch_size(siglip2_model: Any) -> int:
    patch_size = getattr(
        getattr(getattr(siglip2_model, "config", None), "vision_config", None),
        "patch_size",
        None,
    )
    if isinstance(patch_size, (tuple, list)):
        if len(patch_size) != 2 or int(patch_size[0]) != int(patch_size[1]):
            raise ValueError("SigLIP 2 requires a square vision patch size")
        patch_size = int(patch_size[0])
    if not isinstance(patch_size, int):
        raise ValueError("SigLIP 2 config must expose integer vision_config.patch_size")
    _require_positive(patch_size, "patch_size")
    return patch_size


def siglip2_uses_patchified_inputs(siglip2_model: Any) -> bool:
    patch_embedding = getattr(
        getattr(getattr(siglip2_model, "vision_model", None), "embeddings", None),
        "patch_embedding",
        None,
    )
    weight = getattr(patch_embedding, "weight", None)
    if weight is None:
        raise ValueError("SigLIP 2 vision model must expose patch_embedding.weight")
    if weight.ndim == 2:
        return True
    if weight.ndim == 4:
        return False
    raise ValueError(
        f"unsupported SigLIP 2 patch embedding rank: {weight.ndim}"
    )


def siglip2_max_num_patches(siglip2_model: Any) -> int:
    vision_config = getattr(
        getattr(siglip2_model, "config", None), "vision_config", None
    )
    configured = getattr(vision_config, "num_patches", None)
    position_embedding = getattr(
        getattr(getattr(siglip2_model, "vision_model", None), "embeddings", None),
        "position_embedding",
        None,
    )
    position_weight = getattr(position_embedding, "weight", None)
    if position_weight is None or position_weight.ndim != 2:
        raise ValueError("SigLIP 2 vision model must expose positional embeddings")
    position_count = int(position_weight.shape[0])
    if isinstance(configured, int):
        num_patches = configured
    else:
        image_size = getattr(vision_config, "image_size", None)
        patch_size = siglip2_patch_size(siglip2_model)
        if not isinstance(image_size, int):
            raise ValueError(
                "fixed SigLIP 2 config must expose integer vision_config.image_size"
            )
        num_patches = (image_size // patch_size) ** 2
    _require_positive(num_patches, "num_patches")
    if position_count != num_patches:
        raise ValueError(
            "SigLIP 2 positional embedding count disagrees with the derived patch budget "
            f"({position_count} != {num_patches})"
        )
    side = math.isqrt(num_patches)
    if side * side != num_patches:
        raise ValueError(
            "SigLIP 2 patch budget must be a perfect square for positional "
            "embedding interpolation"
        )
    return num_patches


def _pooler_output(output: Any, owner: str) -> torch.Tensor:
    pooler_output = getattr(output, "pooler_output", None)
    if not isinstance(pooler_output, torch.Tensor):
        raise TypeError(f"{owner} output must expose tensor pooler_output")
    return pooler_output


def _last_hidden_state(output: Any) -> torch.Tensor:
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if not isinstance(last_hidden_state, torch.Tensor):
        raise TypeError("SigLIP 2 encoder output must expose tensor last_hidden_state")
    return last_hidden_state


def _require_positive(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


DEFAULT_EPS = 1e-12
SIE_INIT_STD = 0.02
