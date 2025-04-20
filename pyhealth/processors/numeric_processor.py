from typing import Any
import numpy as np
import torch
from pyhealth.processors import register_processor
from pyhealth.processors.base_processor import FeatureProcessor

@register_processor("NumericToTensor")  
class NumericProcessor(FeatureProcessor): 
    """Processor for numerical feature arrays that converts inputs to PyTorch tensors.

    This processor:
    - Accepts various numerical inputs (lists, numpy arrays, etc.)
    - Converts them to float32 tensors
    - Preserves numerical values exactly
    - Handles type conversion automatically

    Typical use cases:
    - Processing continuous-valued features
    - Converting pre-computed features for neural networks
    - Preparing numerical data for PyTorch models
    """

    def __init__(self):
        super().__init__()
    
    def process(self, value: Any) -> torch.Tensor:
        """Converts numerical arrays to float32 tensors.
        
        Args:
            value: Input numerical data (list, numpy array, torch.Tensor, or similar)
            
        Returns:
            torch.FloatTensor: Converted tensor with dtype=float32
            
        Raises:
            ValueError: If input cannot be converted to numerical tensor
        """
        # If already a tensor, just ensure correct dtype
        if isinstance(value, torch.Tensor):
            return value.float()
            
        try:
            # Convert to numpy array if not already
            if not isinstance(value, np.ndarray):
                value = np.array(value, dtype=np.float32)
                
            # Ensure numerical dtype
            if not np.issubdtype(value.dtype, np.number):
                value = value.astype(np.float32)
                
            return torch.from_numpy(value).float()
        except Exception as e:
            raise ValueError(f"Cannot convert input to tensor: {value}") from e
    
    def size(self) -> int:
        """Returns feature dimension (1 for scalar processors).
        
        Note:
            This processor doesn't maintain a vocabulary size since it handles
            continuous numerical features. The return value of 1 indicates
            it processes single features (though they may be multi-dimensional).
        """
        return 1

    def __repr__(self) -> str:
        return "TensorProcessor()"