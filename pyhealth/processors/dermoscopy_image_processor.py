# Contributor: [Your Name]
# NetID: [Your NetID]

from pathlib import Path
from typing import Any, Tuple, Union

import cv2
import numpy as np
import scipy.ndimage
import torch
import torchvision.transforms as transforms
from PIL import Image

from pyhealth.processors.base_processor import FeatureProcessor

VALID_MODES = ("whole", "lesion", "background", "high_whole")

def high_pass_filter(image: np.ndarray, sigma: int = 1) -> np.ndarray:
    """Isolates high-frequency structural edges in an image using Gaussian blur subtraction.
    
    Args:
        image (np.ndarray): The input RGB image array.
        sigma (int, optional): Standard deviation for Gaussian kernel. Defaults to 1.
        
    Returns:
        np.ndarray: The high-frequency residual image.
    """
    image_gray = np.dot(image[..., :3], [0.2989, 0.587, 0.114])
    low_freq = scipy.ndimage.gaussian_filter(image_gray, sigma=sigma)
    high_freq = image_gray - low_freq
    high_freq = np.clip(high_freq, 0, 255).astype(np.uint8)
    return np.stack((high_freq,)*3, axis=-1)

def _load_and_binarize_mask(mask_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    mask_img = Image.open(mask_path).convert("L")
    mask_img = mask_img.resize(target_size, Image.NEAREST)
    return (np.array(mask_img) > 127).astype(np.uint8)

def apply_mode(image: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """Applies a spatial mask to isolate specific regions of the image."""
    if mode == "lesion":
        return image * np.expand_dims(mask, axis=-1)
    elif mode == "background":
        return image * np.expand_dims(1 - mask, axis=-1)
    return image

class DermoscopyImageProcessor(FeatureProcessor): 
    """Image processor for dermoscopic images implementing frequency and spatial ablation.
    
    This processor applies standard PyTorch vision transforms (resizing to 224x224 
    and ImageNet normalization) while allowing for custom ablation modes to test 
    artifact reliance.
    
    Modes:
        - "whole": Returns the standard, unfiltered image.
        - "lesion": Masks out the background, leaving only the lesion.
        - "background": Masks out the lesion, leaving only the background.
        - "high_whole": Applies a high-pass frequency filter to isolate structural edges.

    Args:
        mode (str, optional): The processing mode to apply. Defaults to "whole".
        **kwargs: Additional keyword arguments passed to FeatureProcessor.
        
    Raises:
        ValueError: If an unsupported mode is provided.
    """
    def __init__(self, mode: str = "whole", **kwargs):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode {mode}. Expected one of {VALID_MODES}")
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process(self, value: Union[str, Tuple[str, str]]) -> torch.Tensor:
        """Processes a single image (and optional mask) path into a Tensor.

        Args:
            value: Either a string (path to image) or a tuple (image_path, mask_path).

        Returns:
            torch.Tensor: A normalized PyTorch tensor of shape (3, 224, 224).
        """
        if isinstance(value, (tuple, list)):
            image_path, mask_path = value[0], value[1]
        else:
            image_path, mask_path = value, None

        image_path = Path(image_path)
        with Image.open(image_path) as img:
            image = np.array(img.convert("RGB"))

        if self.mode == "high_whole":
            image = high_pass_filter(image, sigma=1)
        elif self.mode != "whole" and mask_path is not None and Path(mask_path).exists():
            mask_path = Path(mask_path)
            target_size = (image.shape[1], image.shape[0])
            mask = _load_and_binarize_mask(str(mask_path), target_size)
            image = apply_mode(image, mask, self.mode)

        return self.transform(image)
        
    def is_token(self) -> bool: 
        return False
        
    def schema(self) -> tuple: 
        return ("value",)
        
    def dim(self) -> tuple: 
        return (3,)