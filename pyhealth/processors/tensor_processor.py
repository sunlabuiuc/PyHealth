from typing import Any

import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("tensor")
class TensorProcessor(FeatureProcessor):
    """
    Feature processor for converting numerical lists to tensors.

    Input:
        - List of numbers (int/float) or nested lists of numbers

    Processing:
        - Convert input directly to torch.Tensor using torch.tensor()

    Output:
        - torch.Tensor with appropriate shape and dtype
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        """
        Initialize the TensorProcessor.

        Args:
            dtype: The desired torch data type for the output tensor.
                  Default is torch.float32.
        """
        self.dtype = dtype

    def process(self, value: Any) -> torch.Tensor:
        """
        Process a numerical value or list into a torch.Tensor.

        Args:
            value: Input value (list of numbers or nested lists)

        Returns:
            torch.Tensor: Processed tensor
        """
        return torch.tensor(value, dtype=self.dtype)

    def size(self) -> None:
        """
        Get the feature size of the processor.

        Returns:
            None: Size is not predetermined for tensor processor
        """
        return None

    def __repr__(self) -> str:
        """
        String representation of the processor.

        Returns:
            str: String representation
        """
        return f"TensorProcessor(dtype={self.dtype})"
