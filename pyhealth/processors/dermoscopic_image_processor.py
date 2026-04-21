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
``blur_bg``
    Lesion region kept sharp; background blurred with a Gaussian low-pass
    filter.  Composite of the original lesion pixels and the blurred
    background pixels.
``gray_whole``
    Full image converted to grayscale and broadcast back to 3 channels.
    Removes all colour information while preserving spatial structure.
``whole_norm``
    Full image with per-channel min-max normalisation applied (each channel
    stretched to [0, 255]).

Filter backend
--------------
Uses ``scipy.ndimage.gaussian_filter`` (truncate=4.0, σ=1 → effective ~9×9
kernel).  High-pass output is raw float residuals cast to uint8 with no
normalisation, faithfully replicating the reference implementation
(``dermoscopic_artifacts/datasets.py``).
"""

import os
from pathlib import Path
from typing import Any, Union

import numpy as np
import scipy.ndimage
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
    "blur_bg",
    "gray_whole",
    "whole_norm",
)

#: Modes that operate on the full image and do not require a segmentation mask.
MASK_FREE_MODES = frozenset(
    ("whole", "high_whole", "low_whole", "gray_whole", "whole_norm")
)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _high_pass_filter(
    image: np.ndarray,
    sigma: float = 1,
    grayscale: bool = True,
) -> np.ndarray:
    """Return a high-pass–filtered image (3-channel uint8 output).

    Args:
        sigma: Gaussian sigma for the low-pass kernel.
        grayscale: If ``True`` (default), convert to BT.601 grayscale first,
            apply HPF on the single channel, then stack to 3 channels —
            matches ``high_pass_filter(image, grayscale=True)`` in the
            reference.  If ``False``, apply HPF independently on each RGB
            channel.
    """
    if grayscale:
        image_gray = np.dot(image[..., :3], [0.2989, 0.587, 0.114])
        low_frequencies = scipy.ndimage.gaussian_filter(image_gray, sigma=sigma)
        high_frequencies = image_gray - low_frequencies
        out = np.stack([high_frequencies] * 3, axis=-1)
    else:
        out = np.empty(image.shape[:2] + (3,), dtype=np.float32)
        for c in range(3):
            ch = image[:, :, c].astype(np.float64)
            low_frequencies = scipy.ndimage.gaussian_filter(ch, sigma=sigma)
            out[:, :, c] = ch - low_frequencies
    return out.astype(np.uint8)


def _low_pass_filter(image: np.ndarray, sigma: float = 1) -> np.ndarray:
    """Return a Gaussian-blurred (low-pass) image (uint8 output)."""
    return scipy.ndimage.gaussian_filter(image, sigma=sigma).astype(np.uint8)


class DermoscopicImageProcessor(FeatureProcessor):
    """Load and preprocess a dermoscopy image according to a named mode.

    Mirrors the ``ISICDataset.__getitem__`` preprocessing logic from the
    ``dermoscopic_artifacts`` experiment codebase so that PyHealth training
    scripts reproduce the same pixel-level transformations.

    Args:
        mask_dir: Directory containing ``*_segmentation.png`` masks.
            Required for all modes except ``"whole"``.
        mode: One of the valid mode strings (see module docstring).
            Defaults to ``"whole"``.
        image_size: Square resize target.  Defaults to 224.
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
        sigma: float = 1.0,
        high_grayscale: bool = True,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Choose from: {VALID_MODES}"
            )
        self.mask_dir = mask_dir
        self.mode = mode
        self.image_size = image_size
        self.sigma = sigma
        self.high_grayscale = high_grayscale

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_image_and_mask(self, image_path: str):
        """Return ``(image_rgb, mask_binary)`` as uint8 numpy arrays.

        For mask-free modes (``whole``, ``high_whole``, ``low_whole``,
        ``gray_whole``, ``whole_norm``) the mask is a dummy all-ones array
        and no mask file is read from disk.
        """
        try:
            image = np.array(Image.open(image_path).convert("RGB"))
        except Exception as exc:
            raise FileNotFoundError(f"Image not found: {image_path}") from exc

        if self.mode in MASK_FREE_MODES:
            mask = np.ones(image.shape[:2], dtype=np.uint8)
            return image, mask

        img_name = os.path.basename(image_path)
        stem = Path(img_name).stem
        mask_path = os.path.join(self.mask_dir, f"{stem}_segmentation.png")
        try:
            mask = np.array(Image.open(mask_path).convert("L"))
        except Exception as exc:
            raise FileNotFoundError(f"Mask not found: {mask_path}") from exc

        if image.shape[:2] != mask.shape:
            mask = np.array(
                Image.fromarray(mask).resize(
                    (image.shape[1], image.shape[0]), Image.NEAREST
                )
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
                out[y_min:y_max + 1, x_min:x_max + 1] = 0
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
            out[y_min:y_max + 1, x_min:x_max + 1] = 0
            return out

        # Blur background, keep lesion sharp — alpha blend across the boundary
        if self.mode == "blur_bg":
            blurred = _low_pass_filter(image.astype(np.uint8), sigma=self.sigma)
            alpha = mask[:, :, np.newaxis].astype(np.float32)  # 0.0 or 1.0
            sharp = image.astype(np.float32)
            soft = blurred.astype(np.float32)
            return (alpha * sharp + (1.0 - alpha) * soft).astype(np.uint8)

        # Grayscale whole image — broadcast single channel back to 3
        if self.mode == "gray_whole":
            gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            return np.stack([gray] * 3, axis=-1)

        # Per-channel min-max normalisation of whole image
        if self.mode == "whole_norm":
            out = np.empty_like(image)
            for c in range(3):
                ch = image[:, :, c].astype(np.float32)
                mn, mx = ch.min(), ch.max()
                out[:, :, c] = ((ch - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else ch.astype(np.uint8)
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
                base,
                sigma=self.sigma,
                grayscale=self.high_grayscale,
            )
        # low_* modes
        return _low_pass_filter(base, sigma=self.sigma)

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
            try:
                image = np.array(Image.open(image_path).convert("RGB"))
            except Exception as exc:
                raise FileNotFoundError(f"Image not found: {image_path}") from exc
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
