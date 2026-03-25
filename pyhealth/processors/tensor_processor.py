from typing import Any, Dict, Iterable, Optional

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

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        spatial_dims: Optional[tuple[bool, ...]] = None,
    ):
        """
        Initialize the TensorProcessor.

        Args:
            dtype: The desired torch data type for the output tensor.
                  Default is torch.float32.
            spatial_dims: Tuple of booleans indicating which dimensions are spatial.
                  If None, defaults to all False. Default is None.
        """
        self.dtype = dtype
        self._n_dim = None
        self._spatial_dims = spatial_dims

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """Infer n_dim from the first valid sample.

        Args:
            samples: Iterable of sample dictionaries.
            field: The field name to extract from samples.
        """
        for sample in samples:
            if field in sample and sample[field] is not None:
                value = sample[field]
                tensor = (
                    value.detach().clone()
                    if isinstance(value, torch.Tensor)
                    else torch.tensor(value, dtype=self.dtype)
                )
                self._n_dim = tensor.dim()
                break

    def process(self, value: Any) -> torch.Tensor:
        """
        Process a numerical value or list into a torch.Tensor.

        Args:
            value: Input value (list of numbers or nested lists)

        Returns:
            torch.Tensor: Processed tensor
        """
        # Prefer to avoid constructing a new tensor from an existing tensor
        # which can trigger a UserWarning. If value is already a tensor,
        # return a detached clone cast to the requested dtype.
        if isinstance(value, torch.Tensor):
            return value.detach().clone().to(dtype=self.dtype)
        return torch.tensor(value, dtype=self.dtype)

    def size(self) -> None:
        """
        Get the feature size of the processor.

        Returns:
            None: Size is not predetermined for tensor processor
        """
        return None

    def is_token(self) -> bool:
        """Whether the output tensor represents discrete token indices, inferred from dtype.

        Returns:
            True if dtype is integer (discrete tokens), False if floating point (continuous).
        """
        return not self.dtype.is_floating_point

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Number of dimensions for the output tensor.

        Returns:
            (n_dim,)

        Raises:
            NotImplementedError: If n_dim was not provided and fit() was not called.
        """
        if self._n_dim is None:
            raise NotImplementedError(
                "TensorProcessor cannot determine n_dim automatically. "
                "Call fit() first."
            )
        return (self._n_dim,)

    def spatial(self) -> tuple[bool, ...]:
        """Whether each dimension of the output tensor is spatial.

        If spatial_dims was provided at init, returns that. Otherwise defaults
        to all False based on n_dim.
        """
        if self._spatial_dims is not None:
            return self._spatial_dims
        if self._n_dim is None:
            raise NotImplementedError(
                "TensorProcessor cannot determine spatial dims. "
                "Call fit() first."
            )
        return tuple(False for _ in range(self._n_dim))

    def __repr__(self) -> str:
        """
        String representation of the processor.

        Returns:
            str: String representation
        """
        return f"TensorProcessor(dtype={self.dtype})"
