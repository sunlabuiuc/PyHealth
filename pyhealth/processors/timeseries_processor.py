from datetime import datetime, timedelta
from typing import Any, List, Tuple

import numpy as np
import torch

from . import register_processor
from .base_processor import FeatureProcessor
from .utils import (
    NORMALIZATIONS,
    Normalization,
    fit_normalization,
    grid_steps,
    sample_on_grid,
)


@register_processor("timeseries")
class TimeseriesProcessor(FeatureProcessor):
    """
    Feature processor for irregular time series with missing values.

    Input:
        - timestamps: List[datetime]
        - values: np.ndarray of shape (T, F)

    Processing:
        1. Uniform sampling at fixed intervals.
        2. Optional normalization, with statistics learned in ``fit()`` from
           the observed (non-missing) sampled values.
        3. Optional missing-value indicators, one per feature, 1 where the
           value was missing before imputation.
        4. Imputation for missing values.

    With ``normalization="zscore"`` and ``impute_strategy="forward_fill"``,
    values missing before the first observation are filled with 0, which is
    the training mean after normalization.

    Output:
        - torch.Tensor of shape (S, F), or (S, 2F) with missing-value
          indicators appended, where S is the number of sampled time steps.

    Args:
        sampling_rate: Interval of the sampled time steps.
        impute_strategy: ``"forward_fill"`` (values before the first
            observation are filled with 0) or ``"zero"``.
        normalization: ``None``, ``"zscore"`` (zero mean, unit variance) or
            ``"minmax"`` (scaled to [0, 1] on the fitted range).
        missing_indicators: Whether to append the missing-value indicators.

    Examples:
        >>> from datetime import datetime, timedelta
        >>> from pyhealth.processors import TimeseriesProcessor
        >>> t0 = datetime(2020, 1, 1)
        >>> value = ([t0, t0 + timedelta(hours=2)], [[1.0], [3.0]])
        >>> p = TimeseriesProcessor(sampling_rate=timedelta(hours=1))
        >>> p.fit([{"x": value}], "x")
        >>> p.process(value).shape
        torch.Size([3, 1])
    """

    stores_tensor = True
    # Defaults for processors pickled before these options existed.
    normalization: Normalization = None
    missing_indicators: bool = False

    def __init__(
        self,
        sampling_rate: timedelta = timedelta(hours=1),
        impute_strategy: str = "forward_fill",
        normalization: Normalization = None,
        missing_indicators: bool = False,
    ):
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
        # Configurable sampling rate and imputation method
        self.sampling_rate = sampling_rate
        self.impute_strategy = impute_strategy
        self.normalization = normalization
        self.missing_indicators = missing_indicators
        self.learns_statistics = normalization is not None
        self.n_features = None
        self.center: list[float] = []
        self.scale: list[float] = []

    def fit(self, samples: Any, field: str) -> None:
        """Fit the processor.

        Determines n_features from the first valid sample and, with
        ``normalization``, learns the per-feature statistics from all samples.

        Args:
            samples: Iterable of sample dictionaries.
            field: The field name to extract from samples.
        """
        # Extract n_features from the first valid sample without full processing
        for sample in samples:
            if field in sample and sample[field] is not None:
                _, values = sample[field]
                values = np.asarray(values)
                if values.ndim == 2:
                    self.n_features = values.shape[1]
                    break
                elif values.ndim == 1:
                    self.n_features = 1
                    break

        if self.normalization is not None:
            self.center, self.scale = fit_normalization(
                (
                    self._sample(*sample[field])
                    for sample in samples
                    if sample.get(field) is not None and len(sample[field][0]) > 0
                ),
                self.normalization,
            )

    def _sample(
        self, timestamps: list[datetime], values: np.ndarray
    ) -> np.ndarray:
        return sample_on_grid(timestamps, values, self.sampling_rate)

    def process(self, value: Tuple[List[datetime], np.ndarray]) -> torch.Tensor:
        timestamps, values = value

        if len(timestamps) == 0:
            raise ValueError("Timestamps list is empty.")

        # Step 1: Uniform sampling
        sampled_values = self._sample(timestamps, values)
        total_steps, num_features = sampled_values.shape

        # Step 2: Normalization
        if self.normalization is not None:
            sampled_values = (
                sampled_values - np.asarray(self.center)
            ) / np.asarray(self.scale)

        # Step 3: Missing-value indicators, before imputation
        if self.missing_indicators:
            indicators = np.isnan(sampled_values).astype(float)

        # Step 4: Imputation
        if self.impute_strategy == "forward_fill":
            for f in range(num_features):
                last_value = 0.0
                for t in range(total_steps):
                    if not np.isnan(sampled_values[t, f]):
                        last_value = sampled_values[t, f]
                    else:
                        sampled_values[t, f] = last_value
        elif self.impute_strategy == "zero":
            sampled_values = np.nan_to_num(sampled_values, nan=0.0)
        else:
            raise ValueError(f"Unsupported imputation strategy: {self.impute_strategy}")

        if self.n_features is None:
            self.n_features = sampled_values.shape[1]

        if self.missing_indicators:
            sampled_values = np.concatenate([sampled_values, indicators], axis=1)
        return torch.tensor(sampled_values, dtype=torch.float)

    def size(self):
        # Size equals number of features, unknown until first process
        if self.n_features is None:
            return None
        return self.n_features * (2 if self.missing_indicators else 1)

    def is_token(self) -> bool:
        """Time series values are continuous, not discrete tokens."""
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Output is a 2D tensor (time_steps, features)."""
        return (2,)

    def spatial(self) -> tuple[bool, ...]:
        # Time dimension is spatial; feature dimension is not
        return (True, False)

    def __repr__(self):
        return (
            f"TimeSeriesProcessor(sampling_rate={self.sampling_rate}, "
            f"impute_strategy='{self.impute_strategy}', "
            f"normalization={self.normalization!r}, "
            f"missing_indicators={self.missing_indicators})"
        )


class TimeseriesLabelProcessor(FeatureProcessor):
    """Base processor for labels given per time step.

    The label is a time series ``(timestamps, values)``, placed on the same
    kind of uniform grid as :class:`TimeseriesProcessor`: a label at time
    ``t`` goes to step ``floor((t - t0) / sampling_rate)``, where ``t0`` is the
    first label timestamp. Steps that no label falls into, or whose label is
    NaN, are masked out and never imputed. Two labels in the same step raise
    a ``ValueError`` rather than one silently replacing the other.

    The processor does not see the input time series, so the caller must give
    the labels the same first timestamp and ``sampling_rate`` as the input for
    each label to line up with one input step. A label series longer than its
    input is rejected by the model's loss.

    Output:
        - ``(value, mask)``: two float tensors of shape (S,), where ``mask``
          is 1 for steps with a label. Padding added when batching has mask 0.

    The registered names ``"timeseries_binary"`` and
    ``"timeseries_regression"`` are model modes of their own, so modules that
    assume one prediction per sample (calibration, prediction sets,
    interpretability) do not treat these labels as ``"binary"`` or
    ``"regression"``.

    Args:
        sampling_rate: Interval of the sampled time steps.

    Examples:
        >>> from datetime import datetime, timedelta
        >>> from pyhealth.processors import TimeseriesBinaryLabelProcessor
        >>> t0 = datetime(2020, 1, 1)
        >>> times = [t0, t0 + timedelta(hours=2)]
        >>> TimeseriesBinaryLabelProcessor().process((times, [0, 1]))
        (tensor([0., 0., 1.]), tensor([1., 0., 1.]))
    """

    def __init__(self, sampling_rate: timedelta = timedelta(hours=1)):
        self.sampling_rate = sampling_rate

    def process(
        self, value: tuple[list[datetime], np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestamps, labels = value
        if len(timestamps) == 0:
            raise ValueError("Timestamps list is empty.")
        steps, counts = np.unique(
            grid_steps(timestamps, self.sampling_rate), return_counts=True
        )
        if (counts > 1).any():
            step = int(steps[counts > 1][0])
            raise ValueError(
                f"{int(counts.max())} labels fall into time step {step}; "
                "give at most one label per step of the sampling rate."
            )
        sampled = sample_on_grid(timestamps, labels, self.sampling_rate)
        if sampled.shape[1] != 1:
            raise ValueError("Expected one label per time step.")
        sampled = sampled[:, 0]
        mask = ~np.isnan(sampled)
        self._check(sampled[mask])
        return (
            torch.tensor(np.where(mask, sampled, 0.0), dtype=torch.float),
            torch.tensor(mask, dtype=torch.float),
        )

    def _check(self, labels: np.ndarray) -> None:
        pass

    def size(self) -> int:
        return 1

    def is_token(self) -> bool:
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value", "mask")

    def dim(self) -> tuple[int, ...]:
        return (1, 1)

    def spatial(self) -> tuple[bool, ...]:
        return (True,)

    def __repr__(self):
        return f"{type(self).__name__}(sampling_rate={self.sampling_rate})"


@register_processor("timeseries_binary")
class TimeseriesBinaryLabelProcessor(TimeseriesLabelProcessor):
    """Binary (0/1) label per time step, see :class:`TimeseriesLabelProcessor`.

    Examples:
        >>> from datetime import datetime, timedelta
        >>> from pyhealth.processors import TimeseriesBinaryLabelProcessor
        >>> t0 = datetime(2020, 1, 1)
        >>> times = [t0 + timedelta(hours=h) for h in range(3)]
        >>> TimeseriesBinaryLabelProcessor().process((times, [0, float("nan"), 1]))
        (tensor([0., 0., 1.]), tensor([1., 0., 1.]))
    """

    def _check(self, labels: np.ndarray) -> None:
        if not np.isin(labels, (0.0, 1.0)).all():
            raise ValueError("Binary labels must be 0 or 1.")


@register_processor("timeseries_regression")
class TimeseriesRegressionLabelProcessor(TimeseriesLabelProcessor):
    """Real-valued label per time step, see :class:`TimeseriesLabelProcessor`.

    Examples:
        >>> from datetime import datetime, timedelta
        >>> from pyhealth.processors import TimeseriesRegressionLabelProcessor
        >>> t0 = datetime(2020, 1, 1)
        >>> times = [t0 + timedelta(hours=h) for h in range(3)]
        >>> TimeseriesRegressionLabelProcessor().process((times, [2.5, float("nan"), 0.5]))
        (tensor([2.5000, 0.0000, 0.5000]), tensor([1., 0., 1.]))
    """

