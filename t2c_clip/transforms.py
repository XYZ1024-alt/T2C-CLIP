"""Image transforms for CLIP-backed person ReID.

CLIP's own image processor resizes the shortest edge to 224 and center-crops
a 224x224 square. On tall person crops (aspect ratio ~1:2) that discards the
head and feet — only the torso survives. These transforms therefore reuse only
the processor's normalization statistics (``image_mean``/``image_std``) and
resize the whole person image to a fixed ReID aspect ratio instead. The CLIP
vision encoder consumes the non-square input through position-embedding
interpolation (see ``TransformersCLIPImageEncoder``).
"""

from __future__ import annotations

from typing import Any

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# (height, width): standard whole-person ReID input resolution.
DEFAULT_IMAGE_SIZE = (256, 128)


def _processor_normalization(image_processor: Any) -> tuple[list[float], list[float]]:
    mean = getattr(image_processor, "image_mean", None)
    std = getattr(image_processor, "image_std", None)
    if mean is None or std is None:
        raise ValueError(
            "image_processor must expose image_mean and image_std so ReID "
            "transforms can normalize like the CLIP pretraining pipeline"
        )
    return [float(value) for value in mean], [float(value) for value in std]


class CLIPImageTransform:
    def __init__(self, image_processor: Any, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE):
        self.image_processor = image_processor
        self.image_size = (int(image_size[0]), int(image_size[1]))
        mean, std = _processor_normalization(image_processor)
        self._pipeline = transforms.Compose(
            [
                transforms.Resize(self.image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self._pipeline(image)


class CLIPTrainImageTransform:
    def __init__(
        self,
        image_processor: Any,
        *,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        flip_prob: float = 0.5,
        color_jitter: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.05),
        crop_padding: int = 10,
        erase_prob: float = 0.5,
        erase_scale: tuple[float, float] = (0.02, 0.2),
        erase_ratio: tuple[float, float] = (0.3, 3.3),
    ):
        if crop_padding < 0:
            raise ValueError("crop_padding must be non-negative")
        self.image_processor = image_processor
        self.image_size = (int(image_size[0]), int(image_size[1]))
        mean, std = _processor_normalization(image_processor)
        steps: list[Any] = [
            transforms.RandomHorizontalFlip(p=flip_prob),
            transforms.ColorJitter(
                brightness=color_jitter[0],
                contrast=color_jitter[1],
                saturation=color_jitter[2],
                hue=color_jitter[3],
            ),
            transforms.Resize(self.image_size, interpolation=InterpolationMode.BICUBIC),
        ]
        if crop_padding > 0:
            # Standard ReID strong-baseline augmentation: zero-pad the resized
            # image and randomly crop back to the target size.
            steps.append(transforms.Pad(crop_padding))
            steps.append(transforms.RandomCrop(self.image_size))
        steps.append(transforms.ToTensor())
        steps.append(transforms.Normalize(mean, std))
        self._pipeline = transforms.Compose(steps)
        self.random_erasing = transforms.RandomErasing(
            p=erase_prob,
            scale=erase_scale,
            ratio=erase_ratio,
            value=0.0,
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.random_erasing(self._pipeline(image))
