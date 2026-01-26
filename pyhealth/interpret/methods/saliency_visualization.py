"""
Saliency Map Visualization Utilities for PyHealth Interpretability Methods.

This module provides visualization tools for various attribution methods including:
- Gradient-based saliency maps
- Layer-wise Relevance Propagation (LRP)
- Integrated Gradients
- Other attribution methods

The visualizations support both grayscale and RGB images with customizable
overlays and color maps.
"""

import numpy as np
import torch
from typing import Optional, Dict, Union, Tuple


class SaliencyVisualizer:
    """Unified visualization class for saliency maps and attribution results.
    
    This class provides methods to visualize attribution results from various
    interpretability methods. It handles tensor-to-image conversion, normalization,
    and overlay generation for intuitive interpretation of model decisions.
    
    Examples:
        >>> from pyhealth.interpret.methods import BasicGradient, SaliencyVisualizer
        >>> import matplotlib.pyplot as plt
        >>> 
        >>> # Initialize visualizer
        >>> visualizer = SaliencyVisualizer()
        >>> 
        >>> # Visualize gradient saliency
        >>> gradient = BasicGradient(model)
        >>> attributions = gradient.attribute(**batch)
        >>> visualizer.plot_saliency_overlay(
        ...     plt, 
        ...     image=batch['image'][0],
        ...     saliency=attributions['image'][0],
        ...     title="Gradient Saliency"
        ... )
    """
    
    def __init__(
        self,
        default_cmap: str = 'hot',
        default_alpha: float = 0.3,
        figure_size: Tuple[int, int] = (15, 7)
    ):
        """Initialize the saliency visualizer.
        
        Args:
            default_cmap: Default colormap for saliency overlay (e.g., 'hot', 'jet', 'viridis')
            default_alpha: Default transparency for overlay (0.0 to 1.0)
            figure_size: Default figure size (width, height) in inches
        """
        self.default_cmap = default_cmap
        self.default_alpha = default_alpha
        self.figure_size = figure_size
    
    def plot_saliency_overlay(
        self,
        plt,
        image: Union[torch.Tensor, np.ndarray],
        saliency: Union[torch.Tensor, np.ndarray],
        title: Optional[str] = None,
        alpha: Optional[float] = None,
        cmap: Optional[str] = None,
        normalize: bool = True,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """Plot image with saliency map overlay.
        
        Args:
            plt: matplotlib.pyplot instance
            image: Input image tensor [C, H, W] or [H, W] or [H, W, C]
            saliency: Saliency map tensor [H, W] or [C, H, W]
            title: Optional title for the plot
            alpha: Transparency of saliency overlay (default: uses self.default_alpha)
            cmap: Colormap for saliency (default: uses self.default_cmap)
            normalize: Whether to normalize saliency values to [0, 1]
            show: Whether to call plt.show()
            save_path: Optional path to save the figure
        """
        if alpha is None:
            alpha = self.default_alpha
        if cmap is None:
            cmap = self.default_cmap
        
        # Convert tensors to numpy
        img_np = self._to_numpy(image)
        sal_np = self._to_numpy(saliency)
        
        # Process image dimensions
        img_np = self._process_image(img_np)
        
        # Process saliency dimensions
        sal_np = self._process_saliency(sal_np)
        
        # Normalize saliency if requested
        if normalize:
            sal_np = self._normalize_saliency(sal_np)
        
        # Create visualization
        plt.figure(figsize=self.figure_size)
        plt.axis('off')
        
        # Display image
        if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1):
            plt.imshow(img_np.squeeze(), cmap='gray')
        else:
            plt.imshow(img_np)
        
        # Overlay saliency
        plt.imshow(sal_np, cmap=cmap, alpha=alpha)
        
        if title:
            plt.title(title, fontsize=14)
        
        plt.colorbar(label='Attribution Magnitude', fraction=0.046, pad=0.04)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if show:
            plt.show()
    
    def plot_multiple_attributions(
        self,
        plt,
        image: Union[torch.Tensor, np.ndarray],
        attributions: Dict[str, Union[torch.Tensor, np.ndarray]],
        method_names: Optional[Dict[str, str]] = None,
        alpha: Optional[float] = None,
        cmap: Optional[str] = None,
        normalize: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """Plot multiple attribution methods side-by-side for comparison.
        
        Args:
            plt: matplotlib.pyplot instance
            image: Input image tensor
            attributions: Dictionary mapping method keys to attribution tensors
            method_names: Optional dictionary mapping keys to display names
            alpha: Transparency of saliency overlay
            cmap: Colormap for saliency
            normalize: Whether to normalize saliency values
            save_path: Optional path to save the figure
        """
        if alpha is None:
            alpha = self.default_alpha
        if cmap is None:
            cmap = self.default_cmap
        
        num_methods = len(attributions)
        fig, axes = plt.subplots(1, num_methods + 1, figsize=(5 * (num_methods + 1), 5))
        
        # Convert image to numpy
        img_np = self._process_image(self._to_numpy(image))
        
        # Display original image
        if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1):
            axes[0].imshow(img_np.squeeze(), cmap='gray')
        else:
            axes[0].imshow(img_np)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # Display each attribution method
        for idx, (key, attribution) in enumerate(attributions.items(), start=1):
            sal_np = self._process_saliency(self._to_numpy(attribution))
            
            if normalize:
                sal_np = self._normalize_saliency(sal_np)
            
            # Show image with overlay
            if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1):
                axes[idx].imshow(img_np.squeeze(), cmap='gray')
            else:
                axes[idx].imshow(img_np)
            
            im = axes[idx].imshow(sal_np, cmap=cmap, alpha=alpha)
            
            # Set title
            title = method_names.get(key, key) if method_names else key
            axes[idx].set_title(title, fontsize=12)
            axes[idx].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        plt.show()
    
    def plot_saliency_heatmap(
        self,
        plt,
        saliency: Union[torch.Tensor, np.ndarray],
        title: Optional[str] = None,
        cmap: Optional[str] = None,
        normalize: bool = True,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """Plot saliency map as a standalone heatmap (no image overlay).
        
        Args:
            plt: matplotlib.pyplot instance
            saliency: Saliency map tensor [H, W] or [C, H, W]
            title: Optional title for the plot
            cmap: Colormap for heatmap (default: uses self.default_cmap)
            normalize: Whether to normalize saliency values
            show: Whether to call plt.show()
            save_path: Optional path to save the figure
        """
        if cmap is None:
            cmap = self.default_cmap
        
        # Convert and process saliency
        sal_np = self._process_saliency(self._to_numpy(saliency))
        
        if normalize:
            sal_np = self._normalize_saliency(sal_np)
        
        # Create heatmap
        plt.figure(figsize=self.figure_size)
        plt.imshow(sal_np, cmap=cmap)
        plt.colorbar(label='Attribution Magnitude')
        
        if title:
            plt.title(title, fontsize=14)
        
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if show:
            plt.show()
    
    def plot_attribution_distribution(
        self,
        plt,
        attributions: Union[torch.Tensor, np.ndarray],
        title: Optional[str] = None,
        bins: int = 50,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """Plot histogram of attribution values.
        
        Useful for understanding the distribution of attribution magnitudes.
        
        Args:
            plt: matplotlib.pyplot instance
            attributions: Attribution tensor of any shape
            title: Optional title for the plot
            bins: Number of histogram bins
            show: Whether to call plt.show()
            save_path: Optional path to save the figure
        """
        # Convert to numpy and flatten
        attr_np = self._to_numpy(attributions).flatten()
        
        plt.figure(figsize=(10, 6))
        plt.hist(attr_np, bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel('Attribution Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        if title:
            plt.title(title, fontsize=14)
        else:
            plt.title('Attribution Value Distribution', fontsize=14)
        
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(attr_np)
        median_val = np.median(attr_np)
        plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.axvline(median_val, color='g', linestyle='--', label=f'Median: {median_val:.4f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if show:
            plt.show()
    
    def plot_top_k_features(
        self,
        plt,
        image: Union[torch.Tensor, np.ndarray],
        attributions: Union[torch.Tensor, np.ndarray],
        k: int = 10,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """Highlight top-k most important pixels/features.
        
        Args:
            plt: matplotlib.pyplot instance
            image: Input image tensor
            attributions: Attribution tensor
            k: Number of top features to highlight
            title: Optional title for the plot
            show: Whether to call plt.show()
            save_path: Optional path to save the figure
        """
        # Convert to numpy
        img_np = self._process_image(self._to_numpy(image))
        attr_np = self._process_saliency(self._to_numpy(attributions))
        
        # Find top-k positions
        flat_attr = attr_np.flatten()
        top_k_indices = np.argsort(np.abs(flat_attr))[-k:]
        
        # Create mask
        mask = np.zeros_like(flat_attr)
        mask[top_k_indices] = 1
        mask = mask.reshape(attr_np.shape)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image with saliency
        if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1):
            ax1.imshow(img_np.squeeze(), cmap='gray')
        else:
            ax1.imshow(img_np)
        ax1.imshow(attr_np, cmap=self.default_cmap, alpha=self.default_alpha)
        ax1.set_title('Full Attribution Map', fontsize=12)
        ax1.axis('off')
        
        # Top-k features
        if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[-1] == 1):
            ax2.imshow(img_np.squeeze(), cmap='gray')
        else:
            ax2.imshow(img_np)
        ax2.imshow(mask, cmap='Reds', alpha=0.5)
        ax2.set_title(f'Top-{k} Most Important Features', fontsize=12)
        ax2.axis('off')
        
        if title:
            fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        if show:
            plt.show()
    
    # Helper methods
    
    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)
    
    def _process_image(self, img: np.ndarray) -> np.ndarray:
        """Process image to HWC or HW format for visualization.
        
        Args:
            img: Image array in various formats
            
        Returns:
            Processed image array ready for visualization
        """
        # Remove batch dimension if present
        if img.ndim == 4:
            img = img[0]
        
        # Convert CHW to HWC for color images
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        
        # Squeeze single channel
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        
        # Ensure values are in reasonable range for display
        if img.max() > 1.0 and img.max() <= 255.0:
            img = img / 255.0
        
        return img
    
    def _process_saliency(self, saliency: np.ndarray) -> np.ndarray:
        """Process saliency map to 2D format.
        
        Args:
            saliency: Saliency array in various formats
            
        Returns:
            2D saliency array
        """
        # Remove batch dimension if present
        if saliency.ndim == 4:
            saliency = saliency[0]
        
        # For multi-channel saliency, aggregate across channels
        if saliency.ndim == 3:
            # Take absolute maximum across channels or sum
            saliency = np.sum(np.abs(saliency), axis=0)
        
        return saliency
    
    def _normalize_saliency(self, saliency: np.ndarray) -> np.ndarray:
        """Normalize saliency values to [0, 1] range.
        
        Args:
            saliency: Saliency array
            
        Returns:
            Normalized saliency array
        """
        min_val = saliency.min()
        max_val = saliency.max()
        
        if max_val - min_val > 1e-8:
            return (saliency - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(saliency)


# Convenience function for quick visualization
def visualize_attribution(
    plt,
    image: Union[torch.Tensor, np.ndarray],
    attribution: Union[torch.Tensor, np.ndarray],
    title: Optional[str] = None,
    method: str = 'overlay',
    **kwargs
) -> None:
    """Quick visualization of attribution results.
    
    Convenience function that creates a SaliencyVisualizer and plots.
    
    Args:
        plt: matplotlib.pyplot instance
        image: Input image
        attribution: Attribution map
        title: Optional title
        method: Visualization method ('overlay', 'heatmap', 'top_k')
        **kwargs: Additional arguments passed to the visualization method
    
    Examples:
        >>> import matplotlib.pyplot as plt
        >>> visualize_attribution(plt, image, attribution, title="Gradient")
    """
    visualizer = SaliencyVisualizer()
    
    if method == 'overlay':
        visualizer.plot_saliency_overlay(plt, image, attribution, title, **kwargs)
    elif method == 'heatmap':
        visualizer.plot_saliency_heatmap(plt, attribution, title, **kwargs)
    elif method == 'top_k':
        visualizer.plot_top_k_features(plt, image, attribution, title=title, **kwargs)
    else:
        raise ValueError(f"Unknown visualization method: {method}")
