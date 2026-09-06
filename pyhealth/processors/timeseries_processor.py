from datetime import datetime, timedelta
from collections.abc import Callable, Iterable
from typing import Any, List, Literal, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from . import register_processor
from .base_processor import FeatureProcessor


def _validate_normalization_input(
    timestamps: List[datetime],
    values: np.ndarray,
    sampling_rate: timedelta,
    n_features: int | None,
) -> None:
    """Validate a time series before fitting or applying normalization."""
    if sampling_rate <= timedelta(0):
        raise ValueError("sampling_rate must be positive for normalization.")
    if len(timestamps) == 0:
        raise ValueError("Timestamps list is empty.")
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Normalization requires values with shape (T, F), F > 0.")
    if len(timestamps) != len(values):
        raise ValueError("Timestamps and values must have the same length.")
    if any(left > right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Timestamps must be in nondecreasing order.")
    if n_features is not None and values.shape[1] != n_features:
        raise ValueError("Feature count does not match the fitted training data.")
    if np.isinf(values).any():
        raise ValueError("Infinite values are not supported for normalization.")


def _fit_standardization(
    samples: Iterable[dict[str, Any]],
    field: str,
    preprocess: Callable[[Any], np.ndarray],
) -> tuple[int, list[float], list[float]]:
    """Accumulate population statistics one processed training sample at a time."""
    scaler = StandardScaler()
    fitted = False
    for sample in samples:
        if field not in sample or sample[field] is None:
            continue
        values = preprocess(sample[field])
        if not np.isfinite(values).all():
            raise ValueError("Resampled and imputed training values must be finite.")
        scaler.partial_fit(values)
        fitted = True
    if not fitted:
        raise ValueError(f"No training samples contain usable data for {field!r}.")
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError(
            "Training values produced non-finite normalization statistics."
        )
    # Persist explicit numbers: BaseDataset fingerprints vars(processor) as JSON.
    return scaler.n_features_in_, scaler.mean_.tolist(), scaler.scale_.tolist()


def _standardize(
    values: np.ndarray,
    mean: list[float] | None,
    scale: list[float] | None,
) -> np.ndarray:
    """Apply fixed training statistics without changing processor state."""
    if mean is None or scale is None:
        raise RuntimeError("Call fit() on training samples before normalization.")
    if not np.isfinite(values).all():
        raise ValueError("Resampled and imputed values must be finite.")
    return (values - np.asarray(mean)) / np.asarray(scale)


@register_processor("timeseries")
class TimeseriesProcessor(FeatureProcessor):
    """
    Feature processor for irregular time series with missing values.

    Input:
        - timestamps: List[datetime]
        - values: np.ndarray of shape (T, F)

    Processing:
        1. Uniform sampling at fixed intervals.
        2. Imputation for missing values.
        3. Optional z-score normalization using training-set statistics.

    Output:
        - torch.Tensor of shape (S, F), where S is the number of sampled time steps.

    Args:
        sampling_rate: Uniform sampling interval; defaults to one hour.
        impute_strategy: ``"forward_fill"`` (default) or ``"zero"``.
        normalize_strategy: ``None`` (default) or ``"standard"`` for z-scores.
            With normalization enabled, call ``fit()`` on training samples only.
            Statistics include all resampled, imputed time steps with equal
            weight. Constant features use scale 1. Validation/test samples must
            reuse the fitted processor rather than refitting it.

    Examples:
        >>> times = [datetime(2026, 1, 1), datetime(2026, 1, 1, 1)]
        >>> values = np.array([[10.0], [30.0]])
        >>> processor = TimeseriesProcessor(normalize_strategy="standard")
        >>> processor.fit([{"vitals": (times, values)}], "vitals")
        >>> processor.process((times, values)).tolist()
        [[-1.0], [1.0]]
    """

    def __init__(
        self,
        sampling_rate: timedelta = timedelta(hours=1),
        impute_strategy: str = "forward_fill",
        normalize_strategy: Literal["standard"] | None = None,
    ):
        if normalize_strategy not in (None, "standard"):
            raise ValueError("normalize_strategy must be None or 'standard'.")
        if normalize_strategy is not None and sampling_rate <= timedelta(0):
            raise ValueError("sampling_rate must be positive for normalization.")
        # Configurable sampling rate and imputation method
        self.sampling_rate = sampling_rate
        self.impute_strategy = impute_strategy
        self.n_features = None
        self.normalize_strategy = normalize_strategy
        self._normalization_mean: list[float] | None = None
        self._normalization_scale: list[float] | None = None

    def fit(self, samples: Any, field: str) -> None:
        """Infer feature count and optionally fit training-only z-score statistics.

        Normalization uses the population standard deviation (ddof=0) across
        resampled, imputed time steps. Each fit replaces previous statistics;
        a failed fit leaves normalization unfitted. Missing/None fields are
        skipped, but an empty training collection cannot fit normalization.

        Args:
            samples: Iterable of sample dictionaries.
            field: The field name to extract from samples.
        """
        self._normalization_mean = None
        self._normalization_scale = None
        if getattr(self, "normalize_strategy", None) == "standard":
            self.n_features = None
            (
                self.n_features,
                self._normalization_mean,
                self._normalization_scale,
            ) = _fit_standardization(samples, field, self._resample_and_impute)
            return

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

    def process(self, value: Tuple[List[datetime], np.ndarray]) -> torch.Tensor:
        """Resample, impute, and optionally apply fixed training statistics.

        Raises:
            RuntimeError: If normalization is enabled but fitting has not succeeded.
            ValueError: If the series is invalid for normalization.
        """
        sampled_values = self._resample_and_impute(value)
        if getattr(self, "normalize_strategy", None) == "standard":
            sampled_values = _standardize(
                sampled_values, self._normalization_mean, self._normalization_scale
            )
        elif self.n_features is None:
            self.n_features = sampled_values.shape[1]
        return torch.tensor(sampled_values, dtype=torch.float)

    def _resample_and_impute(
        self, value: Tuple[List[datetime], np.ndarray]
    ) -> np.ndarray:
        """Produce the same unnormalized grid for fitting and processing."""
        timestamps, values = value

        if len(timestamps) == 0:
            raise ValueError("Timestamps list is empty.")

        if getattr(self, "normalize_strategy", None) == "standard":
            values = np.asarray(values, dtype=np.float64)
            _validate_normalization_input(
                timestamps, values, self.sampling_rate, self.n_features
            )
        else:
            values = np.asarray(values)
        num_features = values.shape[1]

        # Step 1: Uniform sampling
        start_time = timestamps[0]
        end_time = timestamps[-1]
        total_steps = int((end_time - start_time) / self.sampling_rate) + 1

        sampled_values = np.full((total_steps, num_features), np.nan)

        # Map original timestamps to indices in the sampled grid
        for t, v in zip(timestamps, values):
            idx = int((t - start_time) / self.sampling_rate)
            if 0 <= idx < total_steps:
                sampled_values[idx] = v

        # Step 2: Imputation
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

        return sampled_values

    def size(self):
        # Size equals number of features, unknown until first process
        return self.n_features

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
        strategy = getattr(self, "normalize_strategy", None)
        normalization = f", normalize_strategy={strategy!r}" if strategy else ""
        return (
            f"TimeSeriesProcessor(sampling_rate={self.sampling_rate}, "
            f"impute_strategy='{self.impute_strategy}'{normalization})"
        )
