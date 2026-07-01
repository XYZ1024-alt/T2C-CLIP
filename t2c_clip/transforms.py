"""Image transforms derived from CLIP processors."""

from __future__ import annotations

from typing import Any

from PIL import Image
import torch
from torchvision import transforms


class CLIPImageTransform:
    def __init__(self, image_processor: Any):
        self.image_processor = image_processor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        encoded = self.image_processor(images=image, return_tensors="pt")
        return encoded["pixel_values"][0]


class CLIPTrainImageTransform:
    def __init__(
        self,
        image_processor: Any,
        *,
        flip_prob: float = 0.5,
        color_jitter: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.05),
        erase_prob: float = 0.5,
        erase_scale: tuple[float, float] = (0.02, 0.2),
        erase_ratio: tuple[float, float] = (0.3, 3.3),
    ):
        self.image_processor = image_processor
        self.pre_processor = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=flip_prob),
                transforms.ColorJitter(
                    brightness=color_jitter[0],
                    contrast=color_jitter[1],
                    saturation=color_jitter[2],
                    hue=color_jitter[3],
                ),
            ]
        )
        self.random_erasing = transforms.RandomErasing(
            p=erase_prob,
            scale=erase_scale,
            ratio=erase_ratio,
            value=0.0,
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        encoded = self.image_processor(images=self.pre_processor(image), return_tensors="pt")
        return self.random_erasing(encoded["pixel_values"][0])
