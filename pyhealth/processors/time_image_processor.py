# Author: Joshua Steier
# Description: Time-aware image processor for multimodal PyHealth
#     pipelines. Pairs image loading with temporal metadata for
#     unified multimodal embedding models. Designed for tasks
#     where each patient has multiple images taken at different
#     times (e.g., serial chest X-rays during an ICU stay).

from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torchvision.transforms as transforms
from PIL import Image

from . import register_processor
from .base_processor import FeatureProcessor, ModalityType, TemporalFeatureProcessor


def _convert_mode(img: Image.Image, mode: str) -> Image.Image:
    """Convert a PIL image to the requested mode.

    Args:
        img: Input PIL image.
        mode: Target PIL image mode (e.g., "RGB", "L").

    Returns:
        Converted PIL image.
    """
    return img.convert(mode)


@register_processor("time_image")
class TimeImageProcessor(TemporalFeatureProcessor):
    """Feature processor that loads images and pairs them with timestamps.

    Takes a tuple of (image_paths, time_differences) and returns a tuple
    of (stacked_image_tensor, timestamp_tensor, "image") suitable for
    the unified multimodal embedding model.

    The processor sorts images chronologically by timestamp and
    optionally caps the number of images per patient, keeping the most
    recent observations.

    Input:
        - image_paths: List[str | Path]
        - time_diffs: List[float] (e.g., days from first admission)

    Processing:
        1. Sort (path, time) pairs chronologically.
        2. Truncate to max_images most recent if set.
        3. Load, resize, and transform each image.
        4. Stack into a single tensor.

    Output:
        - Tuple of (images, timestamps, "image") where:
            - images: torch.Tensor of shape (N, C, H, W)
            - timestamps: torch.Tensor of shape (N,)
            - "image": str literal for modality routing

    Args:
        image_size: Resize images to (image_size, image_size).
            Defaults to 224.
        to_tensor: Whether to convert images to tensors.
            Defaults to True.
        normalize: Whether to normalize pixel values.
            Defaults to False.
        mean: Per-channel means for normalization. Required if
            normalize is True.
        std: Per-channel standard deviations for normalization.
            Required if normalize is True.
        mode: PIL image mode conversion (e.g., "RGB", "L"). If
            None, keeps the original mode. Defaults to None.
        max_images: Maximum number of images per patient. If a
            patient has more images, the most recent (by timestamp)
            are kept. If None, all images are kept. Defaults to
            None.

    Raises:
        ValueError: If normalize is True but mean or std is missing.
        ValueError: If mean/std are provided but normalize is False.

    Example:
        >>> proc = TimeImageProcessor(
        ...     image_size=224,
        ...     normalize=True,
        ...     mean=[0.485, 0.456, 0.406],
        ...     std=[0.229, 0.224, 0.225],
        ... )
        >>> paths = ["/data/xray1.png", "/data/xray2.png"]
        >>> times = [0.0, 2.5]
        >>> images, timestamps, tag = proc.process((paths, times))
        >>> images.shape
        torch.Size([2, 3, 224, 224])
        >>> timestamps
        tensor([0.0000, 2.5000])
        >>> tag
        'image'
    """

    def __init__(
        self,
        image_size: int = 224,
        to_tensor: bool = True,
        normalize: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        mode: Optional[str] = None,
        max_images: Optional[int] = None,
    ) -> None:
        self.image_size = image_size
        self.to_tensor = to_tensor
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.mode = mode
        self.max_images = max_images
        self.n_channels = None

        if self.normalize and (
            self.mean is None or self.std is None
        ):
            raise ValueError(
                "Normalization requires both mean and std to be "
                "provided."
            )
        if not self.normalize and (
            self.mean is not None or self.std is not None
        ):
            raise ValueError(
                "Mean and std are provided but normalize is set "
                "to False. Either provide normalize=True, or "
                "remove mean and std."
            )

        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """Build the torchvision transform pipeline.

        Returns:
            A composed transform that applies mode conversion,
            resizing, tensor conversion, and normalization as
            configured.
        """
        transform_list = []
        if self.mode is not None:
            transform_list.append(
                transforms.Lambda(
                    partial(_convert_mode, mode=self.mode)
                )
            )
        if self.image_size is not None:
            transform_list.append(
                transforms.Resize(
                    (self.image_size, self.image_size)
                )
            )
        if self.to_tensor:
            transform_list.append(transforms.ToTensor())
        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=self.mean, std=self.std
                )
            )
        return transforms.Compose(transform_list)

    def _load_single_image(
        self, path: Union[str, Path]
    ) -> torch.Tensor:
        """Load and transform a single image from disk.

        Called internally by process() for each image path in
        the input list.

        Args:
            path: Path to the image file.

        Returns:
            Transformed image tensor of shape (C, H, W).

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )
        with Image.open(image_path) as img:
            img.load()
            return self.transform(img)

    def fit(
        self, samples: Iterable[Dict[str, Any]], field: str
    ) -> None:
        """Fit the processor by inferring n_channels from data.

        Scans samples to find the first valid entry for the given
        field and infers the number of image channels from mode.

        Args:
            samples: Iterable of sample dictionaries.
            field: The field name to extract from samples.
        """
        if self.mode == "L":
            self.n_channels = 1
        elif self.mode == "RGBA":
            self.n_channels = 4
        elif self.mode is not None:
            self.n_channels = 3
        else:
            for sample in samples:
                if field in sample and sample[field] is not None:
                    image_paths, _ = sample[field]
                    if len(image_paths) > 0:
                        path = Path(image_paths[0])
                        if path.exists():
                            with Image.open(path) as img:
                                if img.mode == "L":
                                    self.n_channels = 1
                                elif img.mode == "RGBA":
                                    self.n_channels = 4
                                else:
                                    self.n_channels = 3
                            break
            if self.n_channels is None:
                self.n_channels = 3

    def process(
        self,
        value: Tuple[
            List[Union[str, Path]], List[float]
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Process paired image paths and timestamps.

        Takes a tuple of (image_paths, time_differences) where
        each image path corresponds to the time difference at the
        same index. Images are sorted chronologically. If
        max_images is set, only the most recent images are kept.

        This method is called by SampleBuilder.transform during
        dataset processing.

        Args:
            value: A tuple of two lists:
                - image_paths: List of file paths to images.
                - time_diffs: List of float time differences
                  from the patient's first admission (e.g.,
                  in days).
                Both lists must have the same length.

        Returns:
            A tuple of:
                - images: Stacked image tensor of shape
                  (N, C, H, W) where N is the number of images.
                - timestamps: Float tensor of shape (N,)
                  containing the time differences.
                - tag: The literal string "image" for modality
                  routing in the multimodal embedding model.

        Raises:
            ValueError: If image_paths and time_diffs have
                different lengths.
            ValueError: If image_paths is empty.
            FileNotFoundError: If any image file does not exist.
        """
        image_paths, time_diffs = value

        if len(image_paths) != len(time_diffs):
            raise ValueError(
                f"image_paths length ({len(image_paths)}) and "
                f"time_diffs length ({len(time_diffs)}) must "
                f"match."
            )
        if len(image_paths) == 0:
            raise ValueError("image_paths must be non-empty.")

        paired = sorted(
            zip(time_diffs, image_paths), key=lambda x: x[0]
        )

        if (
            self.max_images is not None
            and len(paired) > self.max_images
        ):
            paired = paired[-self.max_images:]

        timestamps = []
        image_tensors = []
        for t, p in paired:
            image_tensors.append(self._load_single_image(p))
            timestamps.append(t)

        images = torch.stack(image_tensors, dim=0)
        timestamps = torch.tensor(
            timestamps, dtype=torch.float32
        )

        if self.n_channels is None:
            self.n_channels = images.shape[1]

        return images, timestamps, "image"

    def size(self) -> Optional[int]:
        """Return number of image channels.

        Mirrors the TimeseriesProcessor.size() pattern. Returns
        None if fit() or process() has not been called yet.

        Returns:
            Number of channels, or None if unknown.
        """
        return self.n_channels

    def modality(self) -> ModalityType:
        """Medical image → IMAGE modality."""
        return ModalityType.IMAGE

    def value_dim(self) -> int:
        """Flattened image size C*H*W (used with CNN encoder).
        Returns None if fit() has not been called yet."""
        if self.n_channels is None:
            return None
        return self.n_channels * self.image_size * self.image_size

    def process_temporal(self, value) -> dict:
        """Return dict output for UnifiedMultimodalEmbeddingModel.

        Returns:
            {"value": FloatTensor (N, C, H, W), "time": FloatTensor (N,)}
        """
        images, timestamps, _tag = self.process(value)
        return {"value": images, "time": timestamps}

    def __repr__(self) -> str:
        return (
            f"TimeImageProcessor("
            f"image_size={self.image_size}, "
            f"to_tensor={self.to_tensor}, "
            f"normalize={self.normalize}, "
            f"mean={self.mean}, "
            f"std={self.std}, "
            f"mode={self.mode}, "
            f"max_images={self.max_images})"
        )