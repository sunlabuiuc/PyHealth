from typing import Any, Dict, List
import numpy as np

import torch

from pyhealth.processors import register_processor
from pyhealth.processors.base_processor import FeatureProcessor


@register_processor("ndarray")
class NDArrayProcessor(FeatureProcessor):
    """Processor for numerical feature arrays (preserves float values)."""

    def __init__(self):
        super().__init__()
    
    def process(self, value: Any) -> torch.Tensor:
        """Converts numerical arrays to float tensors without vocabulary mapping.
        
        Args:
            value: Input array (list, numpy array, or similar)
            
        Returns:
            torch.FloatTensor: Converted tensor preserving numerical values
        """
        # Convert to numpy array if not already
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float32)
            
        # Ensure numerical dtype
        if not np.issubdtype(value.dtype, np.number):
            value = value.astype(np.float32)
            
        return torch.from_numpy(value).float()
    
    def size(self):
        """Returns feature dimension instead of vocab size."""
        return 1  # Not applicable for continuous features
    
    def __repr__(self):
        return "NDArrayProcessor()"