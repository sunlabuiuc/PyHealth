"""Dense tensors from numeric records, for the ``"dense"`` task view.

A task with a numeric time series emits it as records with ``NaN`` for a
value that was not measured::

    {"time": [0, 1, 2], "values": [[7.4, nan], [nan, nan], [7.3, 90.0]]}

and a static record as ``{"values": [[63.0, 1.0, nan, 80.0]]}``. The
processors here turn those into plain padded tensors with an explicit mask,
which is what most sequence models read. They apply the same preprocessing
as the ``"jagged"`` view: forward fill (series only), the training mean, and
zero for a feature never seen in training; standardization; and optional
missing-value indicators.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from . import register_processor
from .base_processor import LegacyFeatureProcessor


class _DenseNumeric(LegacyFeatureProcessor):
    """Shared fitting and scaling of the dense numeric processors."""

    stores_tensor = True

    def __init__(
        self,
        forward_fill: bool = False,
        normalize: bool = True,
        missing_indicator: bool = True,
    ) -> None:
        self.forward_fill = forward_fill
        self.normalize = normalize
        self.missing_indicator = missing_indicator
        # Only standardization learns from the data.
        self.learns_statistics = normalize
        self.n_features: int | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    @staticmethod
    def _values(value: Any) -> np.ndarray:
        """The ``[N, F]`` float array of a records value, ``NaN`` if missing."""
        values = value["values"] if isinstance(value, dict) else value
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        return array

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Learns each feature's mean and standard deviation, ignoring ``NaN``.

        Args:
            samples: Sample dictionaries; with a split, the training part.
            field: Key of the field in each sample.
        """
        count = total = squares = None
        for sample in samples:
            values = self._values(sample[field])
            if values.size == 0:
                continue
            if count is None:
                self.n_features = values.shape[1]
                count = np.zeros(self.n_features)
                total = np.zeros(self.n_features)
                squares = np.zeros(self.n_features)
            seen = ~np.isnan(values)
            filled = np.where(seen, values, 0.0)
            count += seen.sum(axis=0)
            total += filled.sum(axis=0)
            squares += (filled**2).sum(axis=0)
        if count is None:
            raise ValueError(f"No values found for field '{field}'.")
        if not self.normalize:
            return
        observed = count > 0
        mean = np.where(observed, total / np.maximum(count, 1), 0.0)
        variance = np.where(observed, squares / np.maximum(count, 1) - mean**2, 1.0)
        std = np.sqrt(np.maximum(variance, 0.0))
        # A constant or never-measured feature is left unscaled.
        self.mean = mean
        self.std = np.where(std > 0, std, 1.0)

    def _dense(self, value: Any) -> tuple[np.ndarray, np.ndarray]:
        """Scaled and imputed ``[N, F]`` values and the ``[N, F]`` missing flags."""
        values = self._values(value)
        missing = np.isnan(values)
        if self.normalize:
            values = (values - self.mean) / self.std
        if self.forward_fill and len(values) > 1:
            # Carry each feature's last measurement forward.
            index = np.where(~missing, np.arange(len(values))[:, None], -1)
            last = np.maximum.accumulate(index, axis=0)
            carried = np.take_along_axis(values, np.maximum(last, 0), axis=0)
            values = np.where(last >= 0, carried, np.nan)
        # With standardization 0 is the training mean; without it, plain zero.
        values = np.nan_to_num(values, nan=0.0)
        return values, missing

    def _tensor(self, value: Any) -> torch.Tensor:
        values, missing = self._dense(value)
        if self.missing_indicator:
            values = np.concatenate([values, missing.astype(np.float64)], axis=1)
        return torch.tensor(values, dtype=torch.float32)

    def size(self) -> int | None:
        """Width of the value tensor's last dimension, or None before fitting."""
        if self.n_features is None:
            return None
        return self.n_features * (2 if self.missing_indicator else 1)

    def is_token(self) -> bool:
        """Values are continuous."""
        return False

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(forward_fill={self.forward_fill}, "
            f"normalize={self.normalize}, "
            f"missing_indicator={self.missing_indicator})"
        )


@register_processor("dense_series")
class DenseSeriesProcessor(_DenseNumeric):
    """A numeric time series as a ``[T, F]`` tensor with a ``[T]`` step mask.

    The output is the tuple ``(value, mask)``. ``mask`` is 1 for every step
    of the series; batches pad it with 0, so a model can tell a padded step
    from a real step whose standardized values happen to be 0.

    Args:
        forward_fill: Carry each feature's last measurement forward before
            filling the rest with the training mean.
        normalize: Standardize each feature with the training mean and
            standard deviation.
        missing_indicator: Append one "was not measured" flag per feature,
            doubling the width.

    Examples:
        >>> from pyhealth.processors import DenseSeriesProcessor
        >>> nan = float("nan")
        >>> samples = [{"x": {"time": [0, 1, 2], "values": [[1.0], [nan], [3.0]]}}]
        >>> processor = DenseSeriesProcessor(normalize=False, forward_fill=True)
        >>> processor.fit(samples, "x")
        >>> value, mask = processor.process(samples[0]["x"])
        >>> value
        tensor([[1., 0.],
                [1., 1.],
                [3., 0.]])
        >>> mask
        tensor([1., 1., 1.])
    """

    def __init__(
        self,
        forward_fill: bool = True,
        normalize: bool = True,
        missing_indicator: bool = True,
    ) -> None:
        super().__init__(forward_fill, normalize, missing_indicator)

    def process(self, value: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(value [T, F'], mask [T])`` for one series."""
        tensor = self._tensor(value)
        return tensor, torch.ones(tensor.shape[0], dtype=torch.float32)

    def schema(self) -> tuple[str, ...]:
        return ("value", "mask")

    def dim(self) -> tuple[int, ...]:
        return (2, 1)

    def spatial(self) -> tuple[bool, ...]:
        # (time, feature): time is spatial, features are not.
        return (True, False)


@register_processor("dense_static")
class DenseStaticProcessor(_DenseNumeric):
    """A single numeric record as an ``[F]`` tensor.

    Args:
        normalize: Standardize each feature with the training mean and
            standard deviation; a missing value then becomes 0, the mean.
        missing_indicator: Append one "was not measured" flag per feature.

    Examples:
        >>> from pyhealth.processors import DenseStaticProcessor
        >>> nan = float("nan")
        >>> samples = [{"s": {"values": [[60.0, nan]]}}]
        >>> processor = DenseStaticProcessor(normalize=False)
        >>> processor.fit(samples, "s")
        >>> processor.process(samples[0]["s"])
        tensor([60.,  0.,  0.,  1.])
    """

    def __init__(self, normalize: bool = True, missing_indicator: bool = True) -> None:
        super().__init__(False, normalize, missing_indicator)

    def process(self, value: Any) -> torch.Tensor:
        """Returns the ``[F']`` tensor of one record."""
        return self._tensor(value)[0]

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)
