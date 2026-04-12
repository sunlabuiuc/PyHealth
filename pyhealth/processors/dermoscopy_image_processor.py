# Contributor: [Your Name]
# NetID: [Your NetID]

"""
Dermoscopy Image Processor with Spatial and Frequency Ablation.

This processor bridges raw skin lesion images and PyHealth models. It implements 
the specific image isolation techniques (bounding boxes, masks, and frequency 
filters) required to replicate the artifact robustness experiments detailed in 
the CHIL 2025 paper.

Paper Reference:
    Jin, Q. (2025). A Study of Artifacts on Melanoma Classification under 
    Diffusion-Based Perturbations. Proceedings of Machine Learning Research 
    287:1-14. Conference on Health, Inference, and Learning (CHIL) 2025.

Supported modes include variations of 'whole', 'lesion', 'background', and 'bbox' 
with optional 'high_' or 'low_' frequency prefixes.
"""

from pathlib import Path
from typing import Any, Tuple, Union

import cv2
import numpy as np
import scipy.ndimage
import torch
import torchvision.transforms as transforms

from pyhealth.processors.base_processor import FeatureProcessor

VALID_MODES = (
    "whole", "lesion", "background", 
    "bbox", "bbox70", "bbox90",
    "high_whole", "high_lesion", "high_background",
    "low_whole", "low_lesion", "low_background"
)

def high_pass_filter(image: np.ndarray, sigma: int = 1) -> np.ndarray:
    """Isolates high-frequency structural edges using Gaussian blur subtraction.
    
    Converts the image to grayscale, applies a Gaussian blur to extract the low 
    frequencies, and subtracts them from the original to leave only sharp edges 
    and textures (e.g., hair, rulers, ink).
    
    Args:
        image: The input RGB image array.
        sigma: Standard deviation for the Gaussian kernel.
        
    Returns:
        A 3-channel numpy array representing the high-frequency residual.
    """
    image_gray = np.dot(image[..., :3], [0.2989, 0.587, 0.114])
    low_freq = scipy.ndimage.gaussian_filter(image_gray, sigma=sigma)
    high_freq = image_gray - low_freq
    # Ensure values remain valid pixel intensities
    high_freq = np.clip(high_freq, 0, 255).astype(np.uint8)
    # Stack back to 3 channels so PyTorch vision models accept it
    return np.stack((high_freq,)*3, axis=-1)

def low_pass_filter(image: np.ndarray, sigma: int = 1) -> np.ndarray:
    """Isolates low-frequency color and shape data using a Gaussian blur.
    
    Args:
        image: The input RGB image array.
        sigma: Standard deviation for the Gaussian kernel.
        
    Returns:
        The blurred numpy array.
    """
    return scipy.ndimage.gaussian_filter(image, sigma=sigma)

def apply_bbox(image: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """Applies a black bounding box over the lesion to isolate outer background artifacts.
    
    Based on Jin (2025), this function identifies the spatial bounds of the lesion 
    mask and draws a black rectangle *over* it. This destroys the lesion boundary 
    information, forcing the model to rely solely on distant corner artifacts.
    
    Args:
        image: The input RGB image array.
        mask: The binarized lesion segmentation mask.
        mode: The bounding box scaling mode ("bbox", "bbox70", or "bbox90").
        
    Returns:
        The image array with a black bounding box obscuring the lesion.
    """
    y_idxs, x_idxs = np.where(mask > 0)
    if len(y_idxs) == 0 or len(x_idxs) == 0:  # Safety fallback if no lesion found
        return np.zeros_like(image)
        
    y_min, y_max = y_idxs.min(), y_idxs.max()
    x_min, x_max = x_idxs.min(), x_idxs.max()
    
    processed_image = image.copy()
    
    if mode == "bbox":
        # Draw a tight black box over the exact bounds of the lesion
        cv2.rectangle(processed_image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
    else:
        # Expand the black box to cover 70% or 90% of the surrounding skin context
        expand_ratio = 0.7 if mode == "bbox70" else 0.9
        img_h, img_w = image.shape[:2]
        bbox_h = y_max - y_min
        bbox_w = x_max - x_min
        
        # Prevent division by zero for bizarre straight-line masks
        if bbox_h == 0: bbox_h = 1
        if bbox_w == 0: bbox_w = 1
        
        target_area = expand_ratio * img_h * img_w
        bbox_center_y, bbox_center_x = (y_min + y_max) // 2, (x_min + x_max) // 2
        
        new_bbox_h = int(np.sqrt(target_area * (bbox_h / bbox_w)))
        new_bbox_w = int(np.sqrt(target_area * (bbox_w / bbox_h)))
        
        y_min = max(0, bbox_center_y - new_bbox_h // 2)
        y_max = min(img_h, bbox_center_y + new_bbox_h // 2)
        x_min = max(0, bbox_center_x - new_bbox_w // 2)
        x_max = min(img_w, bbox_center_x + new_bbox_w // 2)
        
        cv2.rectangle(processed_image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
        
    return processed_image

def _load_and_binarize_mask(mask_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Loads a segmentation mask and thresholds it according to the authors' methodology."""
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img.shape[:2] != target_size:
        mask_img = cv2.resize(mask_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    return (mask_img > 0).astype(np.uint8)

class DermoscopyImageProcessor(FeatureProcessor): 
    """Image processor for dermoscopic images implementing frequency and spatial ablation.
    
    This processor applies standard PyTorch vision transforms (Resize, ToTensor, Normalize) 
    while allowing for custom ablation modes to test artifact reliance based on Jin (2025).

    Args:
        mode (str, optional): The processing mode to apply. Defaults to "whole".
            Must be one of the combinations defined in VALID_MODES.
        **kwargs: Additional keyword arguments passed to FeatureProcessor.
    """
    def __init__(self, mode: str = "whole", **kwargs):
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode {mode}. Expected one of {VALID_MODES}")
        self.mode = mode
        
        # Standard ImageNet normalization pipeline required by PyHealth models
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process(self, value: Union[str, Tuple[str, str]]) -> torch.Tensor:
        """Processes a single image (and optional mask) path into a Tensor.
        
        Executes a two-phase pipeline based on the selected mode:
        1. Spatial Isolation: Applies lesion/background masking or bounding boxes.
        2. Frequency Isolation: Applies high/low pass filters to the masked image.

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
        
        # Match authors' BGR to RGB conversion logic exactly
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ---------------------------------------------------------
        # Phase 1: Spatial Isolation
        # ---------------------------------------------------------
        base_image = image.copy()
        
        # If the mode requires a mask (anything other than whole/high_whole/low_whole)
        if self.mode != "whole" and not self.mode.endswith("whole"):
            if mask_path is not None and Path(mask_path).exists():
                mask = _load_and_binarize_mask(str(mask_path), (image.shape[0], image.shape[1]))
                
                if "lesion" in self.mode:
                    # Black out everything EXCEPT the lesion
                    base_image = image * np.expand_dims(mask, axis=-1)
                elif "background" in self.mode:
                    # Black out the lesion, leaving ONLY the background
                    base_image = image * np.expand_dims(1 - mask, axis=-1)
                elif "bbox" in self.mode:
                    # Apply a black bounding box over the lesion center
                    base_image = apply_bbox(image, mask, self.mode)

        # ---------------------------------------------------------
        # Phase 2: Frequency Isolation
        # ---------------------------------------------------------
        if self.mode.startswith("high_"):
            final_image = high_pass_filter(base_image, sigma=1)
        elif self.mode.startswith("low_"):
            final_image = low_pass_filter(base_image, sigma=1)
        else:
            final_image = base_image

        return self.transform(final_image)
        
    def is_token(self) -> bool: 
        """Indicates that this processor does not return discrete tokens."""
        return False
        
    def schema(self) -> tuple: 
        """Returns the PyHealth schema mapping for this processor's output."""
        return ("value",)
        
    def dim(self) -> tuple: 
        """Returns the dimensions of a single pixel (3 channels)."""
        return (3,)