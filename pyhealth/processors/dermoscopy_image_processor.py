"""Custom image processor for dermoscopic images with mode-based processing.

Supports spatial and frequency processing modes that leverage segmentation masks:
- "whole": Use the full unmodified image
- "lesion": Isolate the lesion region (zero out background using mask)
- "background": Isolate the background region (zero out lesion using inverted mask)
- "bbox": Apply spatial ablation bounding boxes
- "high_ / low_": Apply Gaussian frequency filters

The processor accepts (image_path, mask_path) tuples and applies the selected
mode before resizing and converting to a tensor.

Reuses mask/mode logic from the dermoscopic_artifacts repository.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import scipy.ndimage

from . import register_processor
from .base_processor import FeatureProcessor

VALID_MODES = (
    "whole", "lesion", "background", 
    "bbox", "bbox70", "bbox90",
    "high_whole", "high_lesion", "high_background",
    "low_whole", "low_lesion", "low_background"
)

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
        mode: One of the VALID_MODES (e.g., "whole", "bbox70", "high_lesion").

    Returns:
        Processed image as numpy array of shape (H, W, 3).
    """
    # Pure Spatial Modes (Exact Matches)
    if mode == "whole":
        return image
    elif mode == "lesion":
        return image * mask[:, :, np.newaxis]
    elif mode == "background":
        return image * (1 - mask[:, :, np.newaxis])
    # Bounding Box Modes
    elif "bbox" in mode:
        y_idxs, x_idxs = np.where(mask > 0)
        if len(y_idxs) > 0 and len(x_idxs) > 0:
            y_min, y_max = y_idxs.min(), y_idxs.max()
            x_min, x_max = x_idxs.min(), x_idxs.max()
            
            if mode == "bbox":
                pass # no expansion, use the tight bounding box
            else:
                if mode == "bbox70":
                    expand_ratio = 0.7
                elif mode == "bbox90":
                    expand_ratio = 0.9
                else:
                    raise ValueError(f"Unknown bbox mode '{mode}'. Must be 'bbox', 'bbox70', or 'bbox90'.")
                
                img_h, img_w = image.shape[:2]
                bbox_h, bbox_w = max(1, y_max - y_min), max(1, x_max - x_min)
                target_area = expand_ratio * img_h * img_w
                bbox_center_y, bbox_center_x = (y_min + y_max) // 2, (x_min + x_max) // 2
                
                new_bbox_h = int(np.sqrt(target_area * (bbox_h / bbox_w)))
                new_bbox_w = int(np.sqrt(target_area * (bbox_w / bbox_h)))
                
                y_min = max(0, bbox_center_y - new_bbox_h // 2)
                y_max = min(img_h, bbox_center_y + new_bbox_h // 2)
                x_min = max(0, bbox_center_x - new_bbox_w // 2)
                x_max = min(img_w, bbox_center_x + new_bbox_w // 2)
            
            # Use a copy to prevent in-place modification of the original array
            bbox_image = image.copy()
            # Add +1 to the max bounds to match OpenCV's inclusive drawing logic (like in cv2.rectangle)
            bbox_image[y_min:y_max+1, x_min:x_max+1] = 0

            return bbox_image
        else:
            return image # Failsafe if mask is entirely empty

    # Base Image Prep for Frequency Ablations
    if "whole" in mode:
        base_image = image.copy() # pass the base_image through without modification
    elif "lesion" in mode:
        base_image = image * mask[:, :, np.newaxis]
    elif "background" in mode:
        base_image = image * (1 - mask[:, :, np.newaxis])
    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of {VALID_MODES}")

    # Frequency Ablation Application
    if mode.startswith("high_"):
        image_gray = np.dot(base_image[..., :3], [0.2989, 0.587, 0.114])
        low_freq = scipy.ndimage.gaussian_filter(image_gray, sigma=1)
        high_freq = image_gray - low_freq
        high_freq = np.clip(high_freq, 0, 255).astype(np.uint8)
        base_image = np.stack((high_freq,)*3, axis=-1)
    elif mode.startswith("low_"):
        base_image = scipy.ndimage.gaussian_filter(base_image, sigma=(1, 1, 0))
    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of {VALID_MODES}")

    return base_image

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
        """Returns False as images are not sequence tokens."""
        return False

    def schema(self) -> tuple:
        """Returns the data schema tuple."""
        return ("value",)

    def dim(self) -> tuple:
        """Returns the channel dimension of the output tensor."""
        return (3,)

    def spatial(self) -> tuple:
        """Returns boolean flags indicating which dimensions are spatial (C, H, W)."""
        return (False, True, True)

    def __repr__(self) -> str:
        """Returns the string representation of the processor."""
        return (
            f"DermoscopyImageProcessor(mode={self.mode!r}, "
            f"image_size={self.image_size}, normalize={self.normalize})"
        )