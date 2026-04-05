"""Image processor for loading Neuroimaging Informatics Technology Initiative 
(NIftI) images.

Author: Bryan Lau (bryan16@illinois.edu)
Description: Loads and pre-processes NIftI MRI images.
"""
import nibabel as nib
import numpy as np
import torch

from pathlib import Path
from scipy.ndimage import zoom
from typing import Any, Union

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("nifti_image")
class NIftIImageProcessor(FeatureProcessor):
    """Image processor for loading Neuroimaging Informatics Technology 
    Initiative (NIftI) images and converting them to tensors.

    Args:
        input_shape (tuple): Target image size for pre-processing. Default is (121, 145, 121).
        output_shape (tuple): Cropped output dimensions. Default is (96, 96, 96).
    """

    def __init__(
        self,
        input_shape=(121, 145, 121),
        output_shape=(96, 96, 96),
    ) -> None:

        # Parameters
        self.input_shape = input_shape
        self.output_shape = output_shape

    def process(self, value: Union[str, Path]) -> Any:
        """Process a single image path into a transformed tensor image.

        Args:
            value: Path to image file as string or Path object.

        Returns:
            Transformed image tensor.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        # Validate image path
        image_path = Path(value)
        if not image_path.exists():
            raise FileNotFoundError(
                f"NIftI image file not found: {image_path}")

        # Load the image
        image = nib.load(str(value))
        image_data = image.get_fdata().astype(np.float32)

        # Resize to input shape
        factors = [t / c for t, c in zip(self.input_shape, image_data.shape)]
        processed_image = zoom(image_data, factors, order=1)

        # Center crop to output size
        d0, h0, w0 = processed_image.shape
        crop_d, crop_h, crop_w = self.output_shape
        d1 = (d0 - crop_d) // 2
        h1 = (h0 - crop_h) // 2
        w1 = (w0 - crop_w) // 2
        processed_image = processed_image[d1:d1 +
                                          crop_d, h1:h1 + crop_h, w1:w1 + crop_w]

        # Intensity normalization
        # 64-bit precision necessary to avoid overflow on some images
        mean = np.mean(processed_image, dtype=np.float64)
        std = np.std(processed_image, dtype=np.float64)
        processed_image = (processed_image - mean) / (std + 1e-8)

        # Convert image data to tensor
        if processed_image.ndim == 3:
            processed_image = processed_image[np.newaxis, ...]

        return torch.from_numpy(processed_image).float()

    def is_token(self) -> bool:
        """Image data is continuous (float-valued pixel intensities), not discrete tokens.

        Returns:
            False.
        """
        return False

    def schema(self) -> tuple[str, ...]:
        """Single tensor output.

        Returns:
            ("value",)
        """
        return ("value",)
    
    def dim(self) -> tuple[int, ...]:
        """Number of dimensions for each output.
        Output is (1, D, H, W).
        """
        return (4,)
    
    def spatial(self):
        """Which dimensions are spatial
        For NIftI images, the channel is not spatial.
        """
        return (False, True, True, True)

    def __repr__(self) -> str:
        return (
            f"NIftIImageProcessor("
            f"input_shape={self.input_shape}, output_shape={self.output_shape}, "
            f"normalize={self.normalize})"
        )
