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
from typing import Any, Iterable, Dict, List, Tuple

import numpy as np
import torch

from . import register_processor
from .base_processor import ModalityType, TemporalFeatureProcessor


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

    Example::

        proc = TemporalTimeseriesProcessor(sampling_rate=timedelta(hours=2))
        from datetime import datetime, timedelta
        ts  = [datetime(2023,1,1,0), datetime(2023,1,1,4), datetime(2023,1,1,8)]
        val = np.array([[120.0, 80.0], [115.0, 78.0], [118.0, 82.0]])
        out = proc.process_temporal((ts, val))
        # out["value"].shape  → (5, 2)   ← 5 two-hour steps over 8 h
        # out["time"].shape   → (5,)     ← [0., 2., 4., 6., 8.] hours
    """

    def __init__(
        self,
        sampling_rate: timedelta = timedelta(hours=1),
        impute_strategy: str = "forward_fill",
    ):
        self.sampling_rate = sampling_rate
        self.impute_strategy = impute_strategy
        self.n_features: int | None = None

    # ── FeatureProcessor interface ─────────────────────────────────────────

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """Infer feature dimension from the first valid sample."""
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
        """
        timestamps, values = value

        if len(timestamps) == 0:
            raise ValueError("Timestamps list is empty.")

        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]  # (T,) → (T, 1)

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

        # Build time tensor (hours from first observation)
        hours_per_step = self.sampling_rate.total_seconds() / 3600.0
        time_hours = np.array(
            [i * hours_per_step for i in range(total_steps)], dtype=np.float32
        )

        return {
            "value": torch.tensor(sampled_values, dtype=torch.float32),
            "time":  torch.tensor(time_hours, dtype=torch.float32),
        }

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
        return (
            f"TemporalTimeseriesProcessor("
            f"sampling_rate={self.sampling_rate}, "
            f"n_features={self.n_features})"
        )
