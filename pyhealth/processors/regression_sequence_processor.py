from __future__ import annotations

from typing import Any, Dict, Iterable

import torch

from . import register_processor
from .tensor_processor import TensorProcessor


@register_processor("regression_sequence")
class RegressionSequenceProcessor(TensorProcessor):
    """Label processor for variable-length remaining LoS sequences.

    Wraps :class:`TensorProcessor` with ``dtype=torch.float32`` and a temporal
    spatial dimension. Converts per-hour remaining LoS values from the
    ``RemainingLengthOfStayTPC_MIMIC4`` task into a 1-D float tensor.

    Note:
        The constructor takes no arguments; dtype and spatial layout are fixed for the
        TPC label pipeline. Each ``process`` call maps a ``list[float]`` of remaining
        LoS in days to a ``torch.float32`` tensor of shape ``(T,)`` (up to 332 steps for
        a 336-hour stay with predictions from hour 5).

    Examples:
        >>> processor = RegressionSequenceProcessor()
        >>> processor.fit([], "y")
        >>> out = processor.process([2.0, 1.5, 1.0, 0.5])
        >>> out.shape
        torch.Size([4])
        >>> out.dtype
        torch.float32
    """

    def __init__(self) -> None:
        """Initialise float32 regression labels with one spatial (time) axis."""
        super().__init__(dtype=torch.float32, spatial_dims=(True,))
        self._n_dim = 1

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """No fitting is required for regression-sequence labels."""
        return

    def size(self) -> int:
        """Each label timestep is a scalar; processor width is 1."""
        return 1
