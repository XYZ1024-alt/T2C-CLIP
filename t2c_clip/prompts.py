"""Learnable prompt composition for T2C-SigLIP 2.

The ``embedding_dim`` of the prompt bank must equal the SigLIP 2 text model
**token embedding dimension** (``text_model.embeddings.token_embedding``'s
hidden dimension), not the shared output feature dimension. The job
builder obtains this dimension from
:func:`t2c_clip.siglip2_backbone.siglip2_text_hidden_dim`. Each prompt
parameter lives in token-embedding space and is injected by
:class:`~t2c_clip.siglip2_backbone.TransformersSiglip2TextEncoder` into fixed
context slots using the loaded checkpoint's native fixed-length padding
layout.

Training-time prompts combine ``global + camera + identity`` prompts.
Inference-time prompts combine ``global + camera`` prompts only —
identity prompts are excluded from query/gallery retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

MIN_SIZE = 1
DEFAULT_INIT_STD = 0.02


@dataclass(frozen=True)
class PromptConfig:
    num_cameras: int
    num_train_ids: int
    context_length: int
    embedding_dim: int
    init_std: float = DEFAULT_INIT_STD


class PromptBank(torch.nn.Module):
    def __init__(self, config: PromptConfig):
        super().__init__()
        _validate_config(config)
        shape = (config.context_length, config.embedding_dim)
        camera_shape = (config.num_cameras, *shape)
        identity_shape = (config.num_train_ids, *shape)
        self.global_prompt = torch.nn.Parameter(torch.empty(shape))
        self.camera_prompts = torch.nn.Parameter(torch.empty(camera_shape))
        self.identity_prompts = torch.nn.Parameter(torch.empty(identity_shape))
        self.reset_parameters(config.init_std)

    def reset_parameters(self, init_std: float) -> None:
        torch.nn.init.normal_(self.global_prompt, std=init_std)
        torch.nn.init.normal_(self.camera_prompts, std=init_std)
        torch.nn.init.normal_(self.identity_prompts, std=init_std)

    def inference_prompts(self, camera_ids: torch.Tensor) -> torch.Tensor:
        self._validate_camera_ids(camera_ids)
        return self.global_prompt.unsqueeze(0) + self.camera_prompts[camera_ids]

    def training_prompts(self, camera_ids: torch.Tensor, person_ids: torch.Tensor) -> torch.Tensor:
        self._validate_camera_ids(camera_ids)
        self._validate_identity_ids(person_ids)
        base_prompt = self.inference_prompts(camera_ids)
        return base_prompt + self.identity_prompts[person_ids]

    def identity_anchor_prompts(self, person_ids: torch.Tensor) -> torch.Tensor:
        """Camera-agnostic identity prototype prompts: global + identity only.

        Used as the Stage-2 supervised SigLIP identity-anchor bank; the camera
        component is deliberately excluded so camera prompts stay dedicated
        to retrieval fusion.
        """
        self._validate_identity_ids(person_ids)
        return self.global_prompt.unsqueeze(0) + self.identity_prompts[person_ids]

    def _validate_camera_ids(self, camera_ids: torch.Tensor) -> None:
        validate_index_tensor(camera_ids, self.camera_prompts.shape[0], "camera_ids")

    def _validate_identity_ids(self, person_ids: torch.Tensor) -> None:
        validate_index_tensor(person_ids, self.identity_prompts.shape[0], "person_ids")


def _validate_config(config: PromptConfig) -> None:
    values = {
        "num_cameras": config.num_cameras,
        "num_train_ids": config.num_train_ids,
        "context_length": config.context_length,
        "embedding_dim": config.embedding_dim,
    }
    invalid = [name for name, value in values.items() if value < MIN_SIZE]
    if invalid:
        raise ValueError(f"PromptConfig values must be positive: {', '.join(invalid)}")


def validate_index_tensor(indices: torch.Tensor, upper_bound: int, name: str) -> None:
    """Require a torch.long index tensor whose values all fall in ``[0, upper_bound)``."""
    if indices.dtype != torch.long:
        raise ValueError(f"{name} must be a torch.long tensor")
    if torch.any(indices < 0) or torch.any(indices >= upper_bound):
        raise ValueError(f"{name} contains indices outside [0, {upper_bound})")
