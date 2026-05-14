from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("nifti_image")
class NiftiImageProcessor(FeatureProcessor):
    """Feature processor for loading NIfTI MRI volumes.

    This processor loads a ``.nii`` / ``.nii.gz`` volume, extracts the middle
    axial slice, scales intensities to ``[0, 1]``, and returns a tensor with
    shape ``(1, H, W)``.
    """

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size

    def process(self, value: Union[str, Path]) -> Any:
        image_path = Path(value)
        if not image_path.exists():
            raise FileNotFoundError(f"NIfTI image file not found: {image_path}")

        try:
            import nibabel as nib  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "NiftiImageProcessor requires the 'nibabel' package. "
                "Install it with `pip install nibabel`."
            ) from e

        volume = nib.load(str(image_path)).get_fdata().astype(np.float32)
        if volume.ndim < 3:
            raise ValueError(
                f"Expected a 3D/4D NIfTI volume, got shape {volume.shape} for {image_path}"
            )
        if volume.ndim > 3:
            volume = volume[..., 0]

        mid_slice_idx = volume.shape[2] // 2
        image_2d = volume[:, :, mid_slice_idx]

        min_val = float(np.min(image_2d))
        max_val = float(np.max(image_2d))
        if max_val > min_val:
            image_2d = (image_2d - min_val) / (max_val - min_val)
        else:
            image_2d = np.zeros_like(image_2d, dtype=np.float32)

        tensor = torch.from_numpy(image_2d).unsqueeze(0).unsqueeze(0)
        if self.image_size is not None:
            tensor = F.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return tensor.squeeze(0)

    def is_token(self) -> bool:
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        return (3,)

    def spatial(self) -> tuple[bool, ...]:
        return (False, True, True)

