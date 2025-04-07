from pathlib import Path
from typing import Any, List, Optional

import torchvision.transforms as transforms
from PIL import Image

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("image")
class ImageProcessor(FeatureProcessor):
    """
    Feature processor for loading images from disk and converting them to tensors.

    Args:
        image_size (Optional[int]): Desired output image size (will resize to square image).
        to_tensor (bool): Whether to convert image to tensor. Defaults to True.
        normalize (bool): Whether to normalize image values to [0, 1]. Defaults to True.
        mean (Optional[List[float]]): Precomputed mean for normalization.
        std (Optional[List[float]]): Precomputed std for normalization.
    """

    def __init__(
        self,
        image_size: int = 224,
        to_tensor: bool = True,
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        self.image_size = image_size
        self.to_tensor = to_tensor
        self.normalize = normalize

        self.mean = mean
        self.std = std

        if self.normalize and (self.mean is None or self.std is None):
            raise ValueError("Normalization requires both mean and std to be provided.")
        if not self.normalize and (self.mean is not None or self.std is not None):
            raise ValueError("Mean and std are provided but normalize is set to False. Either provide normalize=True, or remove mean and std.")

        self.transform = self._build_transform()

    def _build_transform(self):
        transform_list = []
        if self.image_size is not None:
            transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        if self.to_tensor:
            transform_list.append(transforms.ToTensor())
        if self.normalize:
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))
        return transforms.Compose(transform_list)

    def process(self, value: Any) -> Any:
        """Process a single image path into a transformed tensor image.

        Args:
            value: Path to image file as string or Path object.

        Returns:
            Transformed image tensor.
        """
        image_path = Path(value)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as img:
            img.load()  # Avoid "too many open files" errors
            return self.transform(img)

    def __repr__(self):
        return (
            f"ImageLoadingProcessor(image_size={self.image_size}, "
            f"to_tensor={self.to_tensor}, normalize={self.normalize}, "
            f"mean={self.mean}, std={self.std})"
        )
