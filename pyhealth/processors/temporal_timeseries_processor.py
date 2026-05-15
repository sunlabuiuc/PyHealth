"""TemporalTimeseriesProcessor — a TemporalFeatureProcessor that wraps the
resampling/imputation/normalization logic from :class:`TimeseriesProcessor`
but **preserves timestamps** in the output dict instead of discarding them.

Why this exists
---------------
The original ``TimeseriesProcessor`` returns a plain ``Tensor (S, F)`` —
temporal information is consumed during uniform resampling and then lost.
This processor exposes those resampled timestamps so that
``UnifiedMultimodalEmbeddingModel`` can sort and align events across
modalities on a shared timeline.

Normalization support
---------------------
This processor mirrors the normalization support added to
``TimeseriesProcessor``.  The same strategies (``"standard"``, ``"minmax"``,
``None``) are available and behave identically:

* Statistics are computed during ``fit()`` from the training split only.
* During ``process()``, the learned statistics are applied to the ``"value"``
  tensor.  The ``"time"`` tensor (hours elapsed) is never normalized — it is
  an alignment signal, not a clinical feature.

See :mod:`pyhealth.processors.timeseries_processor` for a detailed explanation
of why normalization matters and how edge cases are handled.
"""
from datetime import datetime, timedelta
from typing import Any, Iterable, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import register_processor
from .base_processor import ModalityType, TemporalFeatureProcessor


@register_processor("temporal_timeseries")
class TemporalTimeseriesProcessor(TemporalFeatureProcessor):
    """Temporal-aware time series processor with optional normalization.

    Identical processing to ``TimeseriesProcessor`` (uniform resampling +
    imputation + optional normalization), but returns a **dict**
    ``{"value": Tensor, "time": Tensor}`` instead of a bare tensor, making it
    compatible with ``UnifiedMultimodalEmbeddingModel``.

    Input tuple format:
        ``(timestamps: List[datetime], values: np.ndarray[T, F])``

    Output dict:
        ``{"value": FloatTensor (S, F), "time": FloatTensor (S,)}``
        — ``S`` is determined by ``sampling_rate`` and the observation window.
        — ``time`` contains hours elapsed from the first observation.
        — ``value`` is optionally normalized using training-set statistics.

    Args:
        sampling_rate: Uniform re-sampling interval.  Defaults to 1 hour.
        impute_strategy: ``"forward_fill"`` (default) or ``"zero"``.
        normalize_strategy: ``None`` (default), ``"standard"``, or ``"minmax"``.
            See :class:`TimeseriesProcessor` for details.

    Example::

        proc = TemporalTimeseriesProcessor(
            sampling_rate=timedelta(hours=2),
            normalize_strategy="standard",
        )
        proc.fit(train_samples, field="vitals")

        out = proc.process((timestamps, values))
        # out["value"].shape  → (S, F)   ← normalized if strategy is set
        # out["time"].shape   → (S,)     ← hours elapsed, never normalized
    """

    _VALID_NORMALIZE_STRATEGIES = {None, "standard", "minmax"}

    def __init__(
        self,
        sampling_rate: timedelta = timedelta(hours=1),
        impute_strategy: str = "forward_fill",
        normalize_strategy: Optional[str] = None,
    ):
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
        self.n_features: int | None = None

        # Normalization statistics — populated by fit()
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    # ── FeatureProcessor interface ─────────────────────────────────────────

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """Fit the processor: learn n_features and normalization statistics.

        Mirrors the logic of :meth:`TimeseriesProcessor.fit` — iterates over
        all training samples, resamples+imputes each, and computes per-feature
        statistics for the chosen normalization strategy.

        Args:
            samples: Iterable of sample dictionaries.
            field: Key in each sample dict holding ``(timestamps, values)``.
        """
        # Step 1: Determine n_features
        for sample in samples:
            if field in sample and sample[field] is not None:
                _, values = sample[field]
                arr = np.asarray(values)
                if arr.ndim == 2:
                    self.n_features = arr.shape[1]
                elif arr.ndim == 1:
                    self.n_features = 1
                break

        if self.normalize_strategy is None:
            return

        # Step 2: Collect all resampled+imputed values
        all_values = []
        for sample in samples:
            if field not in sample or sample[field] is None:
                continue
            timestamps, raw_values = sample[field]
            raw_values = np.asarray(raw_values, dtype=float)
            if raw_values.ndim == 1:
                raw_values = raw_values[:, None]
            if len(timestamps) == 0:
                continue
            resampled = self._resample_and_impute(timestamps, raw_values)
            all_values.append(resampled)

        if len(all_values) == 0:
            return

        all_values = np.concatenate(all_values, axis=0)

        # Step 3: Compute statistics
        if self.normalize_strategy == "standard":
            self.mean_ = all_values.mean(axis=0)
            self.std_ = all_values.std(axis=0)
            zero_var = self.std_ == 0.0
            if zero_var.any():
                self.std_[zero_var] = 1.0
        elif self.normalize_strategy == "minmax":
            self.min_ = all_values.min(axis=0)
            self.max_ = all_values.max(axis=0)
            zero_range = (self.max_ - self.min_) == 0.0
            if zero_range.any():
                self.max_[zero_range] = self.min_[zero_range] + 1.0

    def process(self, value: Tuple[List[datetime], np.ndarray]) -> dict:
        """Process a single sample and return a dict with value and time tensors.

        Args:
            value: ``(timestamps, values)`` tuple.

        Returns:
            ``{"value": FloatTensor (S, F), "time": FloatTensor (S,)}``

        Raises:
            ValueError: If ``timestamps`` is empty.
        """
        timestamps, values = value

        if len(timestamps) == 0:
            raise ValueError("Timestamps list is empty.")

        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]

        num_features = values.shape[1]
        start_time = timestamps[0]
        end_time = timestamps[-1]
        total_steps = int((end_time - start_time) / self.sampling_rate) + 1

        # Resample + impute
        sampled_values = self._resample_and_impute(timestamps, values)

        # Normalize (value tensor only — time tensor is not a clinical feature)
        sampled_values = self._normalize(sampled_values)

        if self.n_features is None:
            self.n_features = num_features

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

    # ── Internal helpers ──────────────────────────────────────────────────

    def _resample_and_impute(
        self,
        timestamps: List[datetime],
        values: np.ndarray,
    ) -> np.ndarray:
        """Resample to uniform grid and impute missing values.

        Shared between ``fit()`` and ``process()`` to guarantee identical
        preprocessing.

        Args:
            timestamps: Sorted observation times.
            values: Array of shape ``(T, F)``.

        Returns:
            ``np.ndarray`` of shape ``(S, F)``.
        """
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
        if self.impute_strategy == "forward_fill":
            for f in range(num_features):
                last = 0.0
                for i in range(total_steps):
                    if not np.isnan(sampled_values[i, f]):
                        last = sampled_values[i, f]
                    else:
                        sampled_values[i, f] = last
        elif self.impute_strategy == "zero":
            sampled_values = np.nan_to_num(sampled_values, nan=0.0)

        return sampled_values

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Apply normalization using statistics from fit().

        Returns values unchanged if no strategy is set or fit() was not called.
        """
        if self.normalize_strategy == "standard" and self.mean_ is not None:
            return (values - self.mean_) / self.std_
        elif self.normalize_strategy == "minmax" and self.min_ is not None:
            return (values - self.min_) / (self.max_ - self.min_)
        return values

    # ── Serialization ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save normalization statistics to disk as ``.npz``."""
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
        """Load normalization statistics from a ``.npz`` file."""
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
            f"impute_strategy='{self.impute_strategy}', "
            f"normalize_strategy={self.normalize_strategy!r})"
        )
