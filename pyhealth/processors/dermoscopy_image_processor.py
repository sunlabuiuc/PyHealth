"""Custom image processor for dermoscopic images with mode-based processing.

Supports three processing modes that leverage segmentation masks:
- "whole": Use the full unmodified image
- "lesion": Isolate the lesion region (zero out background using mask)
- "background": Isolate the background region (zero out lesion using inverted mask)

The processor accepts (image_path, mask_path) tuples and applies the selected
mode before resizing and converting to a tensor.

Reuses mask/mode logic from the dermoscopic_artifacts repository.
"""

from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from . import register_processor
from .base_processor import FeatureProcessor


VALID_MODES = ("whole", "lesion", "background")


def _load_and_binarize_mask(mask_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Load a segmentation mask, resize to match the image, and binarize it.

    Args:
        mask_path: Path to the segmentation mask file (PNG or BMP).
        target_size: (width, height) to resize the mask to match the image.

    Returns:
        Binary numpy array of shape (H, W) with values 0 or 1.
    """
    mask_img = Image.open(mask_path).convert("L")
    mask_img = mask_img.resize(target_size, Image.NEAREST)
    mask = np.array(mask_img)
    return (mask > 0).astype(np.uint8)


def apply_mode(image: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """Apply a processing mode to a dermoscopic image using its segmentation mask.

    Args:
        image: RGB image as numpy array of shape (H, W, 3).
        mask: Binary mask as numpy array of shape (H, W) with values 0/1.
        mode: One of "whole", "lesion", "background".

    Returns:
        Processed image as numpy array of shape (H, W, 3).
    """
    if mode == "whole":
        return image
    elif mode == "lesion":
        return image * mask[:, :, np.newaxis]
    elif mode == "background":
        return image * (1 - mask[:, :, np.newaxis])
    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of {VALID_MODES}")


@register_processor("dermoscopy_image")
class DermoscopyImageProcessor(FeatureProcessor):
    """Feature processor for dermoscopic images with mask-based mode processing.

    Accepts (image_path, mask_path) tuples and applies mode-based processing
    before standard image transforms (resize, to_tensor, normalize).

    Args:
        mode: Processing mode — "whole", "lesion", or "background".
            Defaults to "whole".
        image_size: Desired output image size (square). Defaults to 224.
        to_tensor: Whether to convert image to tensor. Defaults to True.
        normalize: Whether to apply ImageNet normalization. Defaults to True.
        mean: Normalization mean. Defaults to ImageNet values.
        std: Normalization std. Defaults to ImageNet values.

    Examples:
        >>> processor = DermoscopyImageProcessor(mode="lesion")
        >>> tensor = processor.process(("/path/to/image.jpg", "/path/to/mask.png"))
        >>> tensor.shape
        torch.Size([3, 224, 224])
    """

    def __init__(
        self,
        mode: str = "whole",
        image_size: int = 224,
        to_tensor: bool = True,
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got '{mode}'")

        self.mode = mode
        self.image_size = image_size
        self.to_tensor = to_tensor
        self.normalize = normalize
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """Build the torchvision transform pipeline applied after mode processing."""
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
        ]
        if self.to_tensor:
            transform_list.append(transforms.ToTensor())
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        return transforms.Compose(transform_list)

    def process(self, value: Union[Tuple[str, str], str]) -> Any:
        """Process a dermoscopic image with optional mask-based mode.

        Args:
            value: Either a tuple of (image_path, mask_path) or a single
                image_path string (mask ignored, treated as "whole" mode).

        Returns:
            Transformed image tensor of shape (3, image_size, image_size).

        Raises:
            FileNotFoundError: If image or mask file does not exist.
        """
        if isinstance(value, (tuple, list)):
            image_path, mask_path = value[0], value[1]
        else:
            image_path = value
            mask_path = None

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image as RGB numpy array
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image = np.array(img)

        # Apply mode-based processing
        if self.mode != "whole" and mask_path is not None:
            mask_path = Path(mask_path)
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            target_size = (image.shape[1], image.shape[0])  # (width, height)
            mask = _load_and_binarize_mask(str(mask_path), target_size)
            image = apply_mode(image, mask, self.mode)

        return self.transform(image)

    def is_token(self) -> bool:
        return False

    def schema(self) -> tuple:
        return ("value",)

    def dim(self) -> tuple:
        return (3,)

    def spatial(self) -> tuple:
        return (False, True, True)

    def __repr__(self) -> str:
        return (
            f"DermoscopyImageProcessor(mode={self.mode!r}, "
            f"image_size={self.image_size}, normalize={self.normalize})"
        )
