"""TemporalTimeseriesProcessor — a TemporalFeatureProcessor that wraps the
existing TimeseriesProcessor but **preserves** timestamps in the output dict
instead of discarding them after resampling.

Why: The original TimeseriesProcessor returns a plain ``Tensor (S, F)`` with
no timestamps — the temporal information is silently consumed during uniform
resampling.  This wrapper exposes those resampled timestamps so that
``UnifiedMultimodalEmbeddingModel`` can sort and align events across modalities
on a shared timeline.
"""

from datetime import datetime, timedelta
from typing import Any, Iterable, Dict, List, Literal, Tuple

import numpy as np
import torch

from . import register_processor
from .base_processor import ModalityType, TemporalFeatureProcessor
from .timeseries_processor import (
    _fit_standardization,
    _standardize,
    _validate_normalization_input,
)


@register_processor("temporal_timeseries")
class TemporalTimeseriesProcessor(TemporalFeatureProcessor):
    """Temporal-aware wrapper around the classic TimeseriesProcessor.

    Identical processing to ``TimeseriesProcessor`` (uniform resampling +
    forward-fill imputation), but returns a **dict** ``{"value": Tensor,
    "time": Tensor}`` instead of a bare tensor, making it compatible with
    ``UnifiedMultimodalEmbeddingModel``.

    Input tuple format:
        ``(timestamps: List[datetime], values: np.ndarray[T, F])``

    Output dict:
        ``{"value": FloatTensor (S, F), "time": FloatTensor (S,)}``
        — ``S`` is determined by ``sampling_rate`` and the observation window.
        — ``time`` contains hours elapsed from the first observation.

    Args:
        sampling_rate: Uniform re-sampling interval.  Defaults to 1 hour.
        impute_strategy: Currently only ``"forward_fill"`` is supported.
        normalize_strategy: ``None`` (default) or ``"standard"``. Z-scores use
            training-only statistics after resampling and imputation, weighting
            each time step equally. Constant features use scale 1. Only the
            ``"value"`` tensor is normalized; timestamps are unchanged.

    Example::

        proc = TemporalTimeseriesProcessor(sampling_rate=timedelta(hours=2))
        from datetime import datetime, timedelta
        ts  = [datetime(2023,1,1,0), datetime(2023,1,1,4), datetime(2023,1,1,8)]
        val = np.array([[120.0, 80.0], [115.0, 78.0], [118.0, 82.0]])
        out = proc.process_temporal((ts, val))
        # out["value"].shape  → (5, 2)   ← 5 two-hour steps over 8 h
        # out["time"].shape   → (5,)     ← [0., 2., 4., 6., 8.] hours

    Examples:
        >>> times = [datetime(2026, 1, 1), datetime(2026, 1, 1, 1)]
        >>> values = np.array([[10.0], [30.0]])
        >>> processor = TemporalTimeseriesProcessor(normalize_strategy="standard")
        >>> processor.fit([{"vitals": (times, values)}], "vitals")
        >>> output = processor.process((times, values))
        >>> output["value"].tolist()
        [[-1.0], [1.0]]
        >>> output["time"].tolist()
        [0.0, 1.0]
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
        self.sampling_rate = sampling_rate
        self.impute_strategy = impute_strategy
        self.n_features: int | None = None
        self.normalize_strategy = normalize_strategy
        self._normalization_mean: list[float] | None = None
        self._normalization_scale: list[float] | None = None

    # ── FeatureProcessor interface ─────────────────────────────────────────

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """Infer feature count and optionally fit training-only z-score statistics.

        Statistics use population standard deviation (ddof=0) after filling.
        Each fit replaces previous statistics, and a failed fit leaves
        normalization unfitted. Missing/None fields are skipped; normalization
        requires at least one usable training sample.
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

        for sample in samples:
            if field in sample and sample[field] is not None:
                _, values = sample[field]
                arr = np.asarray(values)
                if arr.ndim == 2:
                    self.n_features = arr.shape[1]
                elif arr.ndim == 1:
                    self.n_features = 1
                break

    def process(self, value: Tuple[List[datetime], np.ndarray]) -> dict:
        """Process and return a dict compatible with TemporalFeatureProcessor.

        Args:
            value: ``(timestamps, values)`` where timestamps is a list of
                ``datetime`` objects and values is a ``np.ndarray`` of shape
                ``(T, F)`` or ``(T,)``.

        Returns:
            ``{"value": FloatTensor (S, F), "time": FloatTensor (S,)}``

        Raises:
            RuntimeError: If normalization is enabled but fitting has not succeeded.
            ValueError: If the series is invalid for normalization.
        """
        sampled_values = self._resample_and_impute(value)
        if getattr(self, "normalize_strategy", None) == "standard":
            sampled_values = _standardize(
                sampled_values, self._normalization_mean, self._normalization_scale
            )

        hours_per_step = self.sampling_rate.total_seconds() / 3600.0
        time_hours = np.array(
            [i * hours_per_step for i in range(len(sampled_values))], dtype=np.float32
        )
        return {
            "value": torch.tensor(sampled_values, dtype=torch.float32),
            "time": torch.tensor(time_hours, dtype=torch.float32),
        }

    def _resample_and_impute(
        self, value: Tuple[List[datetime], np.ndarray]
    ) -> np.ndarray:
        """Produce the same unnormalized grid for fitting and processing."""
        timestamps, values = value

        if len(timestamps) == 0:
            raise ValueError("Timestamps list is empty.")

        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]  # (T,) → (T, 1)

        if getattr(self, "normalize_strategy", None) == "standard":
            _validate_normalization_input(
                timestamps, values, self.sampling_rate, self.n_features
            )

        num_features = values.shape[1]
        start_time = timestamps[0]
        end_time = timestamps[-1]
        total_steps = int((end_time - start_time) / self.sampling_rate) + 1

        sampled_values = np.full((total_steps, num_features), np.nan)

        for t, v in zip(timestamps, values):
            idx = int((t - start_time) / self.sampling_rate)
            if 0 <= idx < total_steps:
                sampled_values[idx] = v

        # Forward-fill imputation
        for f in range(num_features):
            last = 0.0
            for i in range(total_steps):
                if not np.isnan(sampled_values[i, f]):
                    last = sampled_values[i, f]
                else:
                    sampled_values[i, f] = last

        return sampled_values

    # process_temporal delegates to process (already returns dict)
    def process_temporal(self, value) -> dict:
        return self.process(value)

    def is_token(self) -> bool:
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value", "time")

    def dim(self) -> tuple[int, ...]:
        return (2, 1)

    def spatial(self) -> tuple[bool, ...]:
        return (True, False)

    # ── TemporalFeatureProcessor interface ────────────────────────────────

    def modality(self) -> ModalityType:
        """Continuous vitals / lab timeseries → NUMERIC modality."""
        return ModalityType.NUMERIC

    def value_dim(self) -> int:
        """Number of features per time-step (used with nn.Linear).
        Must be called after fit()."""
        return self.n_features if self.n_features is not None else 1

    def size(self) -> int | None:
        """Alias for value_dim() — mirrors TimeseriesProcessor API."""
        return self.n_features

    def __repr__(self) -> str:
        strategy = getattr(self, "normalize_strategy", None)
        normalization = f", normalize_strategy={strategy!r}" if strategy else ""
        return (
            f"TemporalTimeseriesProcessor("
            f"sampling_rate={self.sampling_rate}, "
            f"n_features={self.n_features}{normalization})"
        )
