from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

from . import register_processor
from .base_processor import FeatureProcessor


@register_processor("tpc_timeseries")
class TPCTimeseriesProcessor(FeatureProcessor):
    """Hourly TPC time-series tensor from irregular MIMIC observations.

    Consumes the ``ts`` payload from ``RemainingLengthOfStayTPC_MIMIC4`` and emits
    shape ``(T, F, 2)``: scaled values plus decay weights. Robust scaling
    uses per-feature 5th/95th percentiles learned in :meth:`fit`.

    The payload dict contains ``prefill_start``, ``icu_start``, ``pred_start``,
    ``pred_end``, ordered ``feature_itemids``, and ``long_df`` with keys
    ``timestamp``, ``itemid``, ``value``, and ``source`` (``chartevents`` or
    ``labevents``).

    Args:
        sampling_rate: Resampling interval; must be one hour (paper default).
        decay_base: Base ``b`` for ``decay = b ** hours_since_last_observation``.
        clip_min: Lower clip after scaling.
        clip_max: Upper clip after scaling.

    Returns:
        From :meth:`process`, a ``torch.float32`` tensor ``(T, F, 2)``. Channel 0 is
        forward-filled scaled values (0 before first observation). Channel 1 is the
        decay trace (1 at a fresh sample, ``decay_base**j`` after ``j`` hours of
        silence, 0 if never observed).

    Examples:
        >>> from datetime import datetime, timedelta
        >>> processor = TPCTimeseriesProcessor()
        >>> prefill = datetime(2020, 1, 1, 0)
        >>> payload = {
        ...     "prefill_start": prefill,
        ...     "icu_start":     prefill,
        ...     "pred_start":    prefill + timedelta(hours=5),
        ...     "pred_end":      prefill + timedelta(hours=10),
        ...     "feature_itemids": ["A", "B"],
        ...     "long_df": {
        ...         "timestamp": [prefill],
        ...         "itemid":    ["A"],
        ...         "value":     [80.0],
        ...         "source":    ["chartevents"],
        ...     }
        ... }
        >>> processor.fit([{"ts": payload}], "ts")
        >>> out = processor.process(payload)
        >>> out.shape
        torch.Size([5, 2, 2])
    """

    def __init__(
        self,
        sampling_rate: timedelta = timedelta(hours=1),
        decay_base: float = 0.75,
        clip_min: float = -4.0,
        clip_max: float = 4.0,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.decay_base = float(decay_base)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        # Feature-dependent robust scaling parameters, keyed by itemid.
        self._p5: Dict[str, float] = {}
        self._p95: Dict[str, float] = {}
        self._feature_itemids: List[str] = []

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """Collect values per itemid and store 5th/95th percentile bounds."""
        values_by_item: Dict[str, List[float]] = {}
        feature_itemids: List[str] | None = None

        for sample in samples:
            if field not in sample or sample[field] is None:
                continue
            payload = sample[field]
            if not isinstance(payload, dict):
                continue
            if feature_itemids is None:
                feature_itemids = [str(x) for x in payload.get("feature_itemids", [])]
            long_df = payload.get("long_df") or {}
            itemids = long_df.get("itemid", [])
            vals = long_df.get("value", [])
            for itemid, v in zip(itemids, vals):
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                key = str(itemid)
                values_by_item.setdefault(key, []).append(fv)

        self._feature_itemids = feature_itemids or sorted(values_by_item.keys())
        for itemid in self._feature_itemids:
            arr = np.asarray(values_by_item.get(itemid, []), dtype=float)
            if arr.size == 0:
                self._p5[itemid] = 0.0
                self._p95[itemid] = 1.0
                continue
            self._p5[itemid] = float(np.nanpercentile(arr, 5))
            self._p95[itemid] = float(np.nanpercentile(arr, 95))

    def _scale(self, itemid: str, x: float) -> float:
        """Scale ``x`` with stored percentiles and clip to ``[clip_min, clip_max]``."""
        p5 = self._p5.get(itemid, 0.0)
        p95 = self._p95.get(itemid, 1.0)
        if p95 == p5:
            return 0.0
        scaled = 2.0 * (x - p5) / (p95 - p5) - 1.0
        return float(np.clip(scaled, self.clip_min, self.clip_max))

    def process(self, value: Dict[str, Any]) -> torch.Tensor:
        """Build the hourly forward-filled tensor for one ICU stay."""
        prefill_start: datetime = value["prefill_start"]
        pred_start: datetime = value["pred_start"]
        pred_end: datetime = value["pred_end"]
        feature_itemids: Sequence[str] = value["feature_itemids"]
        long_df = value["long_df"]

        step_hours = int(self.sampling_rate.total_seconds() // 3600)
        if step_hours != 1:
            raise ValueError(
                "TPCTimeseriesProcessor currently supports 1-hour sampling only."
            )

        total_steps = int((pred_end - prefill_start).total_seconds() // 3600)
        if total_steps <= 0:
            raise ValueError("Invalid time window for TPC time series.")

        start_idx = int((pred_start - prefill_start).total_seconds() // 3600)
        pred_steps = int((pred_end - pred_start).total_seconds() // 3600)
        if pred_steps <= 0:
            raise ValueError("Invalid prediction window for TPC time series.")

        n_feat = len(feature_itemids)
        sampled = np.full((total_steps, n_feat), np.nan, dtype=float)
        observed = np.zeros((total_steps, n_feat), dtype=bool)

        col_index = {str(itemid): j for j, itemid in enumerate(feature_itemids)}

        ts_list = long_df.get("timestamp", [])
        item_list = long_df.get("itemid", [])
        val_list = long_df.get("value", [])
        for ts, itemid, v in zip(ts_list, item_list, val_list):
            if ts is None or itemid is None or v is None:
                continue
            itemid = str(itemid)
            if itemid not in col_index:
                continue
            try:
                t: datetime = ts
                idx = int((t - prefill_start).total_seconds() // 3600)
                if idx < 0 or idx >= total_steps:
                    continue
                fv = self._scale(itemid, float(v))
            except Exception:
                continue
            j = col_index[itemid]
            sampled[idx, j] = fv
            observed[idx, j] = True

        values_ff = np.zeros((total_steps, n_feat), dtype=float)
        decay = np.zeros((total_steps, n_feat), dtype=float)
        for j in range(n_feat):
            last_value = 0.0
            last_seen: int | None = None
            for t in range(total_steps):
                if observed[t, j] and not np.isnan(sampled[t, j]):
                    last_value = float(sampled[t, j])
                    last_seen = t
                    values_ff[t, j] = last_value
                    decay[t, j] = 1.0
                else:
                    values_ff[t, j] = last_value
                    if last_seen is None:
                        decay[t, j] = 0.0
                    else:
                        dt = t - last_seen
                        decay[t, j] = float(self.decay_base**dt)

        values_ff = values_ff[start_idx : start_idx + pred_steps]
        decay = decay[start_idx : start_idx + pred_steps]

        out = np.stack([values_ff, decay], axis=-1)
        return torch.tensor(out, dtype=torch.float32)

    def size(self) -> int:
        """Number of time-series features (length of ``feature_itemids``)."""
        return len(self._feature_itemids)

    def is_token(self) -> bool:
        """Continuous values, not discrete tokens."""
        return False

    def schema(self) -> tuple[str, ...]:
        """Schema tag for the value channel (decay rides alongside)."""
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        """Three-dimensional output: time, feature, channel."""
        return (3,)

    def spatial(self) -> tuple[bool, ...]:
        """Only the leading time axis is spatial."""
        return (True, False, False)
