from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("timeseries")
class TimeseriesProcessor(FeatureProcessor):
    """
    Feature processor for irregular time series with missing values.

    Converts raw (timestamps, values) tuples into uniformly sampled, imputed,
    and optionally normalized tensors ready for model consumption.

    Input:
        - timestamps: List[datetime] — observation times (sorted).
        - values: np.ndarray of shape (T, F) — T observations of F features.

    Processing pipeline:
        1. Uniform resampling at sampling_rate intervals.
        2. Imputation of missing grid cells (forward-fill or zero).
        3. Normalization using training-set statistics (if enabled).

    Output:
        torch.Tensor of shape (S, F) where S is the number of resampled
        time steps.

    Args:
        sampling_rate: Interval between resampled time steps. Defaults to 1 hour.
        impute_strategy: How to fill missing values after resampling.
            "forward_fill" carries the last value forward (default).
            "zero" replaces NaNs with 0.0.
        normalize_strategy: How to scale feature values.
            None: no normalization (default).
            "standard": z-score normalization.
            "minmax": scales to [0, 1].

    Example:
        from datetime import datetime, timedelta
        import numpy as np
        from pyhealth.processors.timeseries_processor import TimeseriesProcessor

        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )

        # Fit on training data to learn per-feature statistics
        train_samples = [
            {"vitals": (timestamps_1, values_1)},
            {"vitals": (timestamps_2, values_2)},
        ]
        proc.fit(train_samples, field="vitals")

        # Process individual samples
        tensor = proc.process((timestamps, values))
    """

    # Valid options for normalize_strategy, this is checked in __init__
    _VALID_NORMALIZE_STRATEGIES = {None, "standard", "minmax"}

    def __init__(
        self,
        sampling_rate: timedelta = timedelta(hours=1),
        impute_strategy: str = "forward_fill",
        normalize_strategy: Optional[str] = None,
    ):
        # ── Validate arguments ────────────────────────────────────────────
        if impute_strategy not in ("forward_fill", "zero"):
            raise ValueError(
                f"Unsupported imputation strategy: '{impute_strategy}'. "
                f"Choose 'forward_fill' or 'zero'."
            )
        if normalize_strategy not in self._VALID_NORMALIZE_STRATEGIES:
            raise ValueError(
                f"Unsupported normalization strategy: '{normalize_strategy}'. "
                f"Choose from {self._VALID_NORMALIZE_STRATEGIES}."
            )

        self.sampling_rate = sampling_rate
        self.impute_strategy = impute_strategy
        self.normalize_strategy = normalize_strategy

        # Set during fit()
        self.n_features: Optional[int] = None

        # Normalization statistics — populated by fit() when a strategy is set.
        # Each is an ndarray of shape (F,) where F = number of features.
        self.mean_: Optional[np.ndarray] = None   # used by "standard"
        self.std_: Optional[np.ndarray] = None    # used by "standard"
        self.min_: Optional[np.ndarray] = None    # used by "minmax"
        self.max_: Optional[np.ndarray] = None    # used by "minmax"

    # ── fit() ─────────────────────────────────────────────────────────────

    def fit(self, samples: Any, field: str) -> None:
        """Fit the processor on training samples.

        This method is called once on the training split. It performs two jobs:

        1. Determine n_features - the number of features (columns) in
           the time series. This is needed by downstream components that
           allocate weight matrices sized by feature count.

        2. Compute normalization statistics (if normalize_strategy is
           set) - iterates over all training samples, resamples and imputes
           each one (using the same logic as process()), and accumulates
           per-feature statistics:

           - "standard": computes the global mean and standard deviation.
           - "minmax": computes the global min and max.

           These statistics are stored on the processor instance and applied
           during every subsequent call to process().

        Why we resample and impute before computing statistics:
            The raw data is irregularly sampled and may contain gaps. If we
            computed statistics on raw values, we would be measuring the
            distribution of observed values, which differs from the
            distribution of model-input values (which include imputed slots).
            By running the full resampling and imputation pipeline first, the
            statistics match what the model will actually see.

        Args:
            samples: Iterable of sample dictionaries. Each dict should have a
                key matching field whose value is a (timestamps, values) tuple.
            field: The key in each sample dict that holds the time series data.
        """
        # ── Step 1: Determine n_features from the first valid sample ──────
        first_values = None
        for sample in samples:
            if field in sample and sample[field] is not None:
                _, values = sample[field]
                values = np.asarray(values)
                if values.ndim == 2:
                    self.n_features = values.shape[1]
                elif values.ndim == 1:
                    self.n_features = 1
                first_values = values
                break

        # If no normalization requested, we are done (backward-compatible path)
        if self.normalize_strategy is None:
            return

        if first_values is None:
            return  # no valid samples at all

        # ── Step 2: Collect all resampled+imputed training values ──────────
        # We process each sample through the resampling and imputation pipeline
        # (but NOT normalization, since we are computing stats now) and collect
        # all resulting values.  We then compute global statistics from the
        # concatenated result.
        all_values = []
        for sample in samples:
            if field not in sample or sample[field] is None:
                continue
            timestamps, raw_values = sample[field]
            raw_values = np.asarray(raw_values)
            if raw_values.ndim == 1:
                raw_values = raw_values[:, None]
            if len(timestamps) == 0:
                continue

            # Resample + impute (same logic as process(), without normalization)
            resampled = self._resample_and_impute(timestamps, raw_values)
            all_values.append(resampled)

        if len(all_values) == 0:
            return

        # Concatenate across all patients and time steps → shape (N_total, F)
        all_values = np.concatenate(all_values, axis=0)

        # ── Step 3: Compute per-feature statistics ────────────────────────
        if self.normalize_strategy == "standard":
            self.mean_ = all_values.mean(axis=0)   # shape (F,)
            self.std_ = all_values.std(axis=0)      # shape (F,)

            # Guard against zero-variance features.  If a feature is constant
            # across the entire training set (std=0), normalizing it would
            # cause division-by-zero.  We set std to 1.0 for those features,
            # which effectively leaves them unscaled (they are already
            # "normalized" since they have no variance).
            zero_var_mask = self.std_ == 0.0
            if zero_var_mask.any():
                self.std_[zero_var_mask] = 1.0

        elif self.normalize_strategy == "minmax":
            self.min_ = all_values.min(axis=0)   # shape (F,)
            self.max_ = all_values.max(axis=0)   # shape (F,)

            # Guard against zero-range features (max == min).  Same rationale
            # as zero-variance above: set range to 1.0 to avoid division-by-zero.
            zero_range_mask = (self.max_ - self.min_) == 0.0
            if zero_range_mask.any():
                self.max_[zero_range_mask] = self.min_[zero_range_mask] + 1.0

    # ── process() ─────────────────────────────────────────────────────────

    def process(self, value: Tuple[List[datetime], np.ndarray]) -> torch.Tensor:
        """Process a single time series sample.

        Applies the full pipeline: resample → impute → normalize (if enabled).

        Args:
            value: A tuple of ``(timestamps, values)``.

        Returns:
            ``torch.Tensor`` of shape ``(S, F)`` — the processed time series.

        Raises:
            ValueError: If ``timestamps`` is empty.
        """
        timestamps, values = value

        if len(timestamps) == 0:
            raise ValueError("Timestamps list is empty.")

        values = np.asarray(values)
        if values.ndim == 1:
            values = values[:, None]

        # Steps 1–2: Resample to uniform grid and impute missing values
        sampled_values = self._resample_and_impute(timestamps, values)

        # Step 3: Normalize (if statistics were learned during fit)
        sampled_values = self._normalize(sampled_values)

        if self.n_features is None:
            self.n_features = sampled_values.shape[1]

        return torch.tensor(sampled_values, dtype=torch.float)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _resample_and_impute(
        self,
        timestamps: List[datetime],
        values: np.ndarray,
    ) -> np.ndarray:
        """Resample to a uniform grid and impute missing values.

        This is the shared core of both ``fit()`` and ``process()``, extracted
        to avoid code duplication and ensure identical behavior.

        Args:
            timestamps: Sorted list of observation times.
            values: Array of shape ``(T, F)``.

        Returns:
            ``np.ndarray`` of shape ``(S, F)`` — resampled and imputed values.
        """
        num_features = values.shape[1]
        start_time = timestamps[0]
        end_time = timestamps[-1]
        total_steps = int((end_time - start_time) / self.sampling_rate) + 1

        sampled_values = np.full((total_steps, num_features), np.nan)

        # Map original observations onto the uniform grid
        for t, v in zip(timestamps, values):
            idx = int((t - start_time) / self.sampling_rate)
            if 0 <= idx < total_steps:
                sampled_values[idx] = v

        # Impute missing grid cells
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

        return sampled_values

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Apply normalization to resampled values using learned statistics.

        If no normalization strategy is set, or if statistics have not been
        learned (e.g. ``fit()`` was not called), returns values unchanged.

        Args:
            values: Array of shape ``(S, F)``.

        Returns:
            Normalized array of the same shape.
        """
        if self.normalize_strategy == "standard" and self.mean_ is not None:
            return (values - self.mean_) / self.std_
        elif self.normalize_strategy == "minmax" and self.min_ is not None:
            return (values - self.min_) / (self.max_ - self.min_)
        return values

    # ── Serialization ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save learned normalization statistics to disk.

        Uses NumPy's ``.npz`` format for efficient, portable storage.
        Only saves arrays that have been computed (depends on the strategy).

        Args:
            path: File path (should end in ``.npz``).
        """
        data = {}
        if self.mean_ is not None:
            data["mean"] = self.mean_
        if self.std_ is not None:
            data["std"] = self.std_
        if self.min_ is not None:
            data["min"] = self.min_
        if self.max_ is not None:
            data["max"] = self.max_
        if self.n_features is not None:
            data["n_features"] = np.array([self.n_features])
        if data:
            np.savez(path, **data)

    def load(self, path: str) -> None:
        """Load normalization statistics from disk.

        Args:
            path: File path to a ``.npz`` file saved by :meth:`save`.
        """
        if not path.endswith(".npz"):
            path = path + ".npz"
        loaded = np.load(path)
        if "mean" in loaded:
            self.mean_ = loaded["mean"]
        if "std" in loaded:
            self.std_ = loaded["std"]
        if "min" in loaded:
            self.min_ = loaded["min"]
        if "max" in loaded:
            self.max_ = loaded["max"]
        if "n_features" in loaded:
            self.n_features = int(loaded["n_features"][0])

    # ── Metadata ──────────────────────────────────────────────────────────

    def size(self):
        """Number of features per time step.  Known after ``fit()`` or first ``process()``."""
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
        return (
            f"TimeseriesProcessor("
            f"sampling_rate={self.sampling_rate}, "
            f"impute_strategy='{self.impute_strategy}', "
            f"normalize_strategy={self.normalize_strategy!r})"
        )
