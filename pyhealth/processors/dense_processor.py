"""Dense tensors from numeric records, for the ``"dense"`` task view.

A task with a numeric time series emits it as records with ``NaN`` for a
value that was not measured::

    {"time": [0, 1, 2], "values": [[7.4, nan], [nan, nan], [7.3, 90.0]]}

and a static record as ``{"values": [[63.0, 1.0, nan, 80.0]]}``. Most
sequence models read plain padded tensors with a mask instead of the record
structures of :class:`~pyhealth.processors.JaggedFeatureProcessor`. The
processors here give them that layout. The imputation and scaling are done
by a ``JaggedFeatureProcessor`` inside, so the values are exactly those of
the ``"jagged"`` view: forward fill (series only), the training mean, then
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
from .jagged_processor import JaggedFeatureProcessor


class _DenseNumeric(LegacyFeatureProcessor):
    """Builds and fits the inner jagged processor of one numeric level."""

    stores_tensor = True
    learns_statistics = True  # the imputation means, at least
    #: The jagged ``ordering`` of the level: ``"series"`` or ``"single"``.
    ordering: str

    def __init__(
        self,
        forward_fill: bool = False,
        normalize: bool = True,
        missing_indicator: bool = True,
    ) -> None:
        self.forward_fill = forward_fill
        self.normalize = normalize
        self.missing_indicator = missing_indicator
        self.jagged: JaggedFeatureProcessor | None = None
        self.n_features: int | None = None

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Fits the imputation and scaling, on the training part with a split.

        Args:
            samples: Sample dictionaries; with a split, the training part.
            field: Key of the field in each sample.
        """
        for sample in samples:
            values = np.asarray(sample[field]["values"], dtype=np.float64)
            if values.size:
                self.n_features = values.shape[-1]
                break
        if self.n_features is None:
            raise ValueError(f"No values found for field '{field}'.")
        impute = [{"method": "mean", "by_dims": ()}, {"method": "zero"}]
        if self.forward_fill:
            impute.insert(0, {"method": "forward_fill"})
        level: dict[str, Any] = {
            "ordering": self.ordering,
            "values": [self.n_features],
            "impute": impute,
            "missing_indicator": self.missing_indicator,
        }
        if self.normalize:
            level["normalize"] = {"method": "zscore"}
        self.jagged = JaggedFeatureProcessor(levels=[level])
        self.jagged.fit(samples, field)

    def _tensor(self, value: Any) -> torch.Tensor:
        """The ``[N, F']`` values, with the missing indicators appended."""
        roles = (role for _, role in self.jagged.schema())
        parts = dict(zip(roles, self.jagged.process(value)))
        values = parts["values"].float()
        if self.missing_indicator:
            values = torch.cat([values, parts["mask"].float()], dim=-1)
        return values

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
        >>> processor = DenseSeriesProcessor(normalize=False)
        >>> processor.fit(samples, "x")
        >>> value, mask = processor.process(samples[0]["x"])
        >>> value
        tensor([[1., 0.],
                [1., 1.],
                [3., 0.]])
        >>> mask
        tensor([1., 1., 1.])
    """

    ordering = "series"

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
            standard deviation.
        missing_indicator: Append one "was not measured" flag per feature.

    Examples:
        >>> from pyhealth.processors import DenseStaticProcessor
        >>> nan = float("nan")
        >>> samples = [{"s": {"values": [[60.0, nan]]}}, {"s": {"values": [[70.0, 2.0]]}}]
        >>> processor = DenseStaticProcessor(normalize=False)
        >>> processor.fit(samples, "s")
        >>> processor.process(samples[0]["s"])
        tensor([60.,  2.,  0.,  1.])
    """

    ordering = "single"

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
