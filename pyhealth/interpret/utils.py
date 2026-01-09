"""Visualization utilities for interpretability methods.

This module provides visualization functions for interpretability in PyHealth,
particularly useful for medical imaging applications. It includes utilities for:

- **Overlay visualizations**: Show attribution/saliency maps on top of images
- **Attribution normalization**: Prepare raw attributions for visualization
- **Interpolation**: Resize patch-level attributions (e.g., from ViT) to image size

Example Usage
-------------

Basic attribution overlay:

>>> from pyhealth.interpret.utils import show_cam_on_image, normalize_attribution
>>> # Assume we have an image and attribution from an interpreter
>>> attr_normalized = normalize_attribution(attribution)
>>> overlay = show_cam_on_image(image, attr_normalized)

Image attribution visualization:

>>> from pyhealth.interpret.methods import CheferRelevance
>>> from pyhealth.interpret.utils import visualize_image_attr
>>> interpreter = CheferRelevance(model)
>>> attribution = interpreter.get_vit_attribution_map(**batch)
>>> image, attr_map, overlay = visualize_image_attr(
...     image=batch["image"][0],
...     attribution=attribution[0, 0],
...     interpolate=True,  # Resize attribution to match image
... )

See Also
--------
pyhealth.interpret.methods : Attribution methods (DeepLift, IntegratedGradients, etc.)
"""

import numpy as np
from typing import Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def show_cam_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = True,
    colormap: int = None,
    image_weight: float = 0.5,
) -> np.ndarray:
    """Overlay a Class Activation Map (CAM) or attribution map on an image.
    
    This function creates a visualization by blending an attribution/saliency
    map with the original image using a colormap (typically 'jet' for heatmap
    visualization).
    
    Args:
        img: Input image as numpy array with shape (H, W, 3) for RGB or (H, W)
            for grayscale. Values should be in range [0, 1].
        mask: Attribution/saliency map with shape (H, W). Values should be
            in range [0, 1] where higher values indicate more importance.
        use_rgb: If True, return RGB format. If False, return BGR format.
            Default is True.
        colormap: OpenCV colormap constant. If None, uses cv2.COLORMAP_JET.
            Common options: cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_VIRIDIS
        image_weight: Weight of the original image in the blend (0 to 1).
            Default is 0.5 for equal blend.
            
    Returns:
        Blended visualization as uint8 numpy array with shape (H, W, 3) in
        range [0, 255].
        
    Raises:
        ValueError: If inputs are invalid or cv2 is not available.
        
    Examples:
        >>> import numpy as np
        >>> from pyhealth.interpret.utils import show_cam_on_image
        >>> 
        >>> # Create sample image and attribution
        >>> image = np.random.rand(224, 224, 3)  # RGB image
        >>> attribution = np.random.rand(224, 224)  # Saliency map
        >>> 
        >>> # Create visualization
        >>> overlay = show_cam_on_image(image, attribution)
        >>> overlay.shape
        (224, 224, 3)
    """
    if not HAS_CV2:
        # Fallback implementation without cv2
        return _show_cam_fallback(img, mask, image_weight)
    
    if colormap is None:
        colormap = cv2.COLORMAP_JET
    
    # Ensure image is RGB format with 3 channels
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    
    # Validate inputs
    if img.max() > 1.0 + 1e-6:
        raise ValueError(
            f"Image values should be in [0, 1], got max={img.max():.4f}. "
            "Normalize with: img = (img - img.min()) / (img.max() - img.min())"
        )
    
    # Normalize mask to [0, 1]
    mask = mask.astype(np.float32)
    if mask.max() > mask.min():
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    
    # Apply colormap to mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    # Blend image and heatmap
    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / cam.max()  # Normalize
    
    return np.uint8(255 * cam)


def _show_cam_fallback(
    img: np.ndarray,
    mask: np.ndarray,
    image_weight: float = 0.5,
) -> np.ndarray:
    """Fallback implementation of show_cam_on_image without OpenCV.
    
    Uses matplotlib colormaps instead of cv2.applyColorMap.
    """
    try:
        from matplotlib import cm
    except ImportError:
        raise ImportError(
            "Either cv2 (opencv-python) or matplotlib is required for "
            "visualization. Install with: pip install opencv-python matplotlib"
        )
    
    # Ensure image is RGB format with 3 channels
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    
    # Normalize mask to [0, 1]
    mask = mask.astype(np.float32)
    if mask.max() > mask.min():
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    
    # Apply jet colormap
    heatmap = cm.jet(mask)[:, :, :3]  # Remove alpha channel
    
    # Blend image and heatmap
    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / cam.max()  # Normalize
    
    return np.uint8(255 * cam)


def interpolate_attribution_map(
    attribution: np.ndarray,
    target_size: Tuple[int, int],
    mode: str = "bilinear",
) -> np.ndarray:
    """Interpolate attribution map to target size.
    
    This is useful for models where the attribution is computed at a lower
    resolution (e.g., 14x14 patch grid for ViT-B/16) and needs to be 
    upsampled to the original image resolution (e.g., 224x224).
    
    Args:
        attribution: Attribution map as numpy array or torch tensor.
            Shape can be (H, W) or (B, H, W) or (1, 1, H, W).
        target_size: Target (height, width) for interpolation.
        mode: Interpolation mode. Options: "bilinear", "nearest".
            Default is "bilinear" for smooth gradients.
            
    Returns:
        Interpolated attribution map with shape (target_h, target_w).
        
    Examples:
        >>> # For ViT-B/16 with 14x14 patch grid
        >>> attr_patches = np.random.rand(14, 14)
        >>> attr_full = interpolate_attribution_map(attr_patches, (224, 224))
        >>> attr_full.shape
        (224, 224)
    """
    import torch
    import torch.nn.functional as F
    
    # Convert to tensor if needed
    is_numpy = isinstance(attribution, np.ndarray)
    if is_numpy:
        attribution = torch.from_numpy(attribution).float()
    
    # Ensure 4D tensor: (B, C, H, W)
    while attribution.dim() < 4:
        attribution = attribution.unsqueeze(0)
    
    # Interpolate
    interpolated = F.interpolate(
        attribution,
        size=target_size,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    
    # Remove batch and channel dims, convert back to numpy
    result = interpolated.squeeze()
    if is_numpy:
        result = result.numpy()
    
    return result


def normalize_attribution(
    attribution: Union[np.ndarray, "torch.Tensor"],
    method: str = "minmax",
) -> np.ndarray:
    """Normalize attribution values for visualization.
    
    Args:
        attribution: Raw attribution values.
        method: Normalization method. Options:
            - "minmax": Scale to [0, 1] using min-max normalization
            - "abs_max": Scale by absolute maximum, keeping sign
            - "percentile": Clip to [5, 95] percentile then normalize
            
    Returns:
        Normalized attribution as numpy array in [0, 1].
    """
    import torch
    
    if isinstance(attribution, torch.Tensor):
        attribution = attribution.detach().cpu().numpy()
    
    attr = attribution.astype(np.float32)
    
    if method == "minmax":
        if attr.max() > attr.min():
            return (attr - attr.min()) / (attr.max() - attr.min())
        return np.zeros_like(attr)
    
    elif method == "abs_max":
        abs_max = np.abs(attr).max()
        if abs_max > 0:
            return (attr / abs_max + 1) / 2  # Map [-1, 1] to [0, 1]
        return np.zeros_like(attr) + 0.5
    
    elif method == "percentile":
        p5, p95 = np.percentile(attr, [5, 95])
        attr = np.clip(attr, p5, p95)
        if p95 > p5:
            return (attr - p5) / (p95 - p5)
        return np.zeros_like(attr)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def visualize_image_attr(
    image: Union[np.ndarray, "torch.Tensor"],
    attribution: Union[np.ndarray, "torch.Tensor"],
    normalize: bool = True,
    interpolate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate visualization components from an image and attribution map.
    
    This is a convenience function that prepares image and attribution for
    visualization, handling common format conversions, interpolation, and 
    creating an overlay. Works with any image-based model (CNN, ViT, etc.).
    
    Args:
        image: Input image as numpy array or torch tensor.
            Accepted shapes: [H, W], [H, W, C], [C, H, W].
            Values can be in any range (will be normalized to [0, 1]).
        attribution: Attribution map as numpy array or torch tensor.
            Shape should be [H, W]. If different from image size, will be
            interpolated to match when interpolate=True.
        normalize: If True, normalize attribution to [0, 1] range.
            Default is True.
        interpolate: If True, interpolate attribution map to match image
            dimensions if they differ. Default is True.
        
    Returns:
        Tuple of (image, attribution_map, overlay) where:
            - image: Normalized image as numpy array [H, W] or [H, W, C] in [0, 1]
            - attribution_map: Attribution as numpy array [H, W] in [0, 1]
            - overlay: Attribution overlay on image as numpy array [H, W, 3] 
              in [0, 255]
        
    Examples:
        >>> from pyhealth.interpret.methods import CheferRelevance
        >>> from pyhealth.interpret.utils import visualize_image_attr
        >>> 
        >>> # Compute attribution with interpreter
        >>> interpreter = CheferRelevance(model)
        >>> attr_map = interpreter.get_vit_attribution_map(**batch)
        >>> 
        >>> # Generate visualization (auto-interpolates to image size)
        >>> image, attr_display, overlay = visualize_image_attr(
        ...     image=batch["image"][0],
        ...     attribution=attr_map[0, 0],
        ...     interpolate=True,
        ... )
        >>> 
        >>> # Display
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(overlay)
        >>> plt.savefig("attribution.png")
    """
    import torch
    
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # Handle channel dimension - convert [C, H, W] to [H, W, C]
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))
    
    # Handle single-channel images
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image.squeeze(-1)
    
    # Normalize image to [0, 1]
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    
    # Get image spatial dimensions
    img_h, img_w = image.shape[:2]
    
    # Convert attribution to numpy
    if isinstance(attribution, torch.Tensor):
        attribution = attribution.detach().cpu().numpy()
    
    # Ensure attribution is 2D
    attribution = np.squeeze(attribution)
    
    # Interpolate attribution to match image size if needed
    if interpolate and attribution.shape != (img_h, img_w):
        attribution = interpolate_attribution_map(attribution, (img_h, img_w))
    
    # Normalize attribution if requested
    if normalize:
        attribution = normalize_attribution(attribution)
    
    # Create overlay
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image
    overlay = show_cam_on_image(image_rgb, attribution)
    
    return image, attribution, overlay
