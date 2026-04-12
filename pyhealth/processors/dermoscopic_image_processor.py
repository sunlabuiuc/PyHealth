"""
Dermoscopic image processor for ISIC 2018 artifact experiments.

Implements the 12 preprocessing modes from 'A Study of Artifacts on Melanoma Classification under
Diffusion-Based Perturbations',
adapted as a PyHealth :class:`~pyhealth.processors.base_processor.FeatureProcessor`.

Modes
-----
``whole``
    Full image, no masking.
``lesion``
    Lesion region only (image multiplied by binary segmentation mask).
``background``
    Background region only (image multiplied by inverted mask).
``bbox``
    Full image with the lesion bounding-box blacked out.
``bbox70``
    Full image with an expanded bounding-box (≈70 % of image area) blacked out.
``bbox90``
    Same as ``bbox70`` but ≈90 % of image area.
``high_whole`` / ``high_lesion`` / ``high_background``
    High-pass–filtered version of the respective region.
``low_whole``  / ``low_lesion``  / ``low_background``
    Low-pass–filtered version of the respective region.
"""

import os
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from .base_processor import FeatureProcessor

#: All valid mode identifiers.
VALID_MODES = (
    "whole",
    "lesion",
    "background",
    "bbox",
    "bbox70",
    "bbox90",
    "high_whole",
    "high_lesion",
    "high_background",
    "low_whole",
    "low_lesion",
    "low_background",
)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _high_pass_filter(
    image: np.ndarray,
    sigma: float = 1,
    filter_size: tuple = (
        0,
        0)) -> np.ndarray:
    """Return a grayscale high-pass–filtered image (3-channel output)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, filter_size, sigma)
    hp = gray - blurred
    hp = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX)
    hp_uint8 = hp.astype(np.uint8)
    return cv2.cvtColor(hp_uint8, cv2.COLOR_GRAY2RGB)


def _low_pass_filter(
    image: np.ndarray,
    sigma: float = 1,
    filter_size: tuple = (
        0,
        0)) -> np.ndarray:
    """Return a Gaussian-blurred (low-pass) image."""
    blurred = cv2.GaussianBlur(image, filter_size, sigma)
    return blurred.astype(np.uint8)


class DermoscopicImageProcessor(FeatureProcessor):
    """Load and preprocess a dermoscopy image according to a named mode.

    Mirrors the ``ISICDataset.__getitem__`` preprocessing logic from the
    ``dermoscopic_artifacts`` experiment codebase so that PyHealth training
    scripts reproduce the same pixel-level transformations.

    .. note::
        The reference implementation (``dermoscopic_artifacts/datasets.py``)
        uses ``scipy.ndimage.gaussian_filter`` for ``high_*`` and ``low_*``
        modes, which defaults to ``truncate=4.0`` (effective kernel 9×9 at
        σ=1).  This implementation uses ``cv2.GaussianBlur`` instead; pass
        ``filter_size=(9, 9)`` to match the scipy kernel size exactly.

    Args:
        mask_dir: Directory containing ``*_segmentation.png`` masks.
            Required for all modes except ``"whole"``.
        mode: One of the 12 valid mode strings (see module docstring).
            Defaults to ``"whole"``.
        image_size: Square resize target.  Defaults to 224.
        filter_size: Kernel size ``(width, height)`` for the Gaussian filter
            used in ``high_*`` and ``low_*`` modes.  Both values must be odd
            positive integers.  Defaults to ``(0, 0)``, letting OpenCV
            auto-compute the kernel size from sigma.
        sigma: Standard deviation for the Gaussian filter used in ``high_*``
            and ``low_*`` modes.  Defaults to ``1.0``.

    Raises:
        ValueError: If *mode* is not in :data:`VALID_MODES`.
    """

    def __init__(
        self,
        mask_dir: str = "",
        mode: str = "whole",
        image_size: int = 224,
        filter_size: tuple = (0, 0),
        sigma: float = 1.0,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose from: {VALID_MODES}"
            )
        self.mask_dir = mask_dir
        self.mode = mode
        self.image_size = image_size
        self.filter_size = filter_size
        self.sigma = sigma

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_image_and_mask(self, image_path: str):
        """Return ``(image_rgb, mask_binary)`` as uint8 numpy arrays."""
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_name = os.path.basename(image_path)
        stem = Path(img_name).stem
        mask_path = os.path.join(self.mask_dir, f"{stem}_segmentation.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        if image.shape[:2] != mask.shape:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask = (mask > 0).astype(np.uint8)
        return image, mask

    def _apply_mode(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply the configured mode and return a uint8 RGB numpy array."""
        if self.mode == "whole":
            return image

        if self.mode == "lesion":
            return image * mask[:, :, np.newaxis]

        if self.mode == "background":
            return image * (1 - mask[:, :, np.newaxis])

        if self.mode in ("bbox", "bbox70", "bbox90"):
            y_idxs, x_idxs = np.where(mask > 0)
            if len(y_idxs) == 0:
                return np.zeros_like(image)

            y_min, y_max = int(y_idxs.min()), int(y_idxs.max())
            x_min, x_max = int(x_idxs.min()), int(x_idxs.max())

            if self.mode == "bbox":
                out = image.copy()
                cv2.rectangle(
                    out, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
                return out

            expand_ratio = 0.7 if self.mode == "bbox70" else 0.9
            img_h, img_w = image.shape[:2]
            bbox_h = max(y_max - y_min, 1)
            bbox_w = max(x_max - x_min, 1)
            target_area = expand_ratio * img_h * img_w
            cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
            new_h = int(np.sqrt(target_area * bbox_h / bbox_w))
            new_w = int(np.sqrt(target_area * bbox_w / bbox_h))
            y_min = max(0, cy - new_h // 2)
            y_max = min(img_h, cy + new_h // 2)
            x_min = max(0, cx - new_w // 2)
            x_max = min(img_w, cx + new_w // 2)
            out = image.copy()
            cv2.rectangle(out, (x_min, y_min), (x_max, y_max),
                          (0, 0, 0), thickness=-1)
            return out

        # Frequency-filter modes
        if "whole" in self.mode:
            base = image
        elif "lesion" in self.mode:
            base = image * mask[:, :, np.newaxis]
        else:  # background
            base = image * (1 - mask[:, :, np.newaxis])

        if self.mode.startswith("high_"):
            return _high_pass_filter(
                base.astype(
                    np.uint8),
                sigma=self.sigma,
                filter_size=self.filter_size)
        return _low_pass_filter(
            base.astype(
                np.uint8),
            sigma=self.sigma,
            filter_size=self.filter_size)

    # ------------------------------------------------------------------
    # FeatureProcessor interface
    # ------------------------------------------------------------------

    def process(self, value: Union[str, Path]) -> Any:
        """Load image at *value*, apply mode preprocessing, return tensor.

        Args:
            value: Absolute path to the dermoscopy image.

        Returns:
            Float32 tensor of shape ``(3, image_size, image_size)``,
            normalised with ImageNet statistics.
        """
        image_path = str(value)

        if self.mode == "whole":
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image, mask = self._load_image_and_mask(image_path)
            image = self._apply_mode(image, mask)

        pil_image = Image.fromarray(image.astype(np.uint8))
        return self.transform(pil_image)

    def is_token(self) -> bool:
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        return (3,)

    def spatial(self) -> tuple[bool, ...]:
        return (False, True, True)

    def __repr__(self) -> str:
        return (
            f"DermoscopicImageProcessor(mode={self.mode!r}, "
            f"image_size={self.image_size}, mask_dir={self.mask_dir!r})"
        )
