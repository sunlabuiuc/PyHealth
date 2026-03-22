from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np

from .base_task import BaseTask


class HourlyLOSEICU(BaseTask):
    """Hourly remaining length-of-stay regression task for eICU.

    Time-series preprocessing:
      1. Bucket into hourly bins
      2. Keep most recent measurement per hour
      3. Forward-fill missing values
      4. Compute decay = hours since last real observation

    Static preprocessing:
      - Numeric: float values
      - Categorical: stable ID -> vocab index -> one-hot

    Output:
      time_series: [t, num_features * 3] (value, mask, decay)
      static: [num_static_dims]
      target_los_hours: regression target
    """

    task_name: str = "HourlyLOSEICU"

    def __init__(
        self,
        time_series_tables: Optional[List[str]] = None,
        time_series_features: Optional[Dict[str, List[str]]] = None,
        numeric_static_features: Optional[List[str]] = None,
        categorical_static_features: Optional[List[str]] = None,
        min_history_hours: int = 5,
        max_hours: int = 48,
    ):
        self.time_series_tables = time_series_tables or ["lab"]
        self.time_series_features = time_series_features or {
            "lab": ["-basos"],
        }

        self.numeric_static_features = numeric_static_features or []
        self.categorical_static_features = categorical_static_features or []

        self.min_history_hours = min_history_hours
        self.max_hours = max_hours

        self.static_vocab: Dict[str, Dict[Any, int]] = {
            feature: {} for feature in self.categorical_static_features
        }

        self.input_schema = {
            "time_series": ("tensor", {}),
            "static": ("tensor", {}),
        }

        self.output_schema = {
            "target_los_hours": "regression",
        }

    def pre_filter(self, global_event_df):
        return global_event_df

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            x = float(value)
            if math.isnan(x) or math.isinf(x):
                return None
            return x
        except Exception:
            return None

    def _get_total_hours(self, attr: Dict[str, Any]) -> Optional[float]:
        minutes = self._safe_float(attr.get("unitdischargeoffset"))
        if minutes is None:
            return None
        return minutes / 60.0

    def _build_feature_index(self):
        names = []
        for table in self.time_series_tables:
            names.extend(self.time_series_features.get(table, []))
        return names, {n: i for i, n in enumerate(names)}

    def _register_category(self, feature, category):
        vocab = self.static_vocab.setdefault(feature, {})
        if category not in vocab:
            vocab[category] = len(vocab)
        return vocab[category]

    def _encode_static(self, attr):
        numeric = []
        categorical = []

        for f in self.numeric_static_features:
            v = self._safe_float(attr.get(f))
            numeric.append(0.0 if v is None else v)

        for f in self.categorical_static_features:
            raw = attr.get(f)
            cid = "__MISSING__" if raw is None else raw
            idx = self._register_category(f, cid)
            size = len(self.static_vocab[f])

            one_hot = [0.0] * size
            one_hot[idx] = 1.0
            categorical.extend(one_hot)

        return numeric + categorical

    def _extract_hourly(self, patient, usable_hours, feature_index, num_features):
        latest_vals = [[None] * num_features for _ in range(usable_hours)]
        latest_time = [[-float("inf")] * num_features for _ in range(usable_hours)]

        try:
            events = patient.get_events("lab")
        except Exception:
            events = []

        for e in events:
            attr = e.attr_dict

            name = attr.get("labname")
            if name not in feature_index:
                continue

            val = self._safe_float(attr.get("labresult"))
            offset = self._safe_float(attr.get("labresultoffset"))

            if val is None or offset is None:
                continue

            hr = int(offset / 60.0)
            if hr < 0 or hr >= usable_hours:
                continue

            fi = feature_index[name]

            if offset >= latest_time[hr][fi]:
                latest_time[hr][fi] = offset
                latest_vals[hr][fi] = val

        filled = []
        mask = []
        decay = []

        last_val = [0.0] * num_features
        last_seen = [None] * num_features

        for h in range(usable_hours):
            f_row, m_row, d_row = [], [], []

            for f in range(num_features):
                v = latest_vals[h][f]

                if v is not None:
                    last_val[f] = v
                    last_seen[f] = h
                    f_row.append(v)
                    m_row.append(1.0)
                    d_row.append(0.0)
                else:
                    f_row.append(last_val[f])
                    m_row.append(0.0)

                    if last_seen[f] is None:
                        d_row.append(float(h + 1))
                    else:
                        d_row.append(float(h - last_seen[f]))

            filled.append(f_row)
            mask.append(m_row)
            decay.append(d_row)

        return filled, mask, decay

    def _combine(self, filled, mask, decay):
        combined = []
        for h in range(len(filled)):
            row = []
            for f in range(len(filled[h])):
                row.append(filled[h][f])
                row.append(mask[h][f])
                row.append(decay[h][f])
            combined.append(row)
        return combined

    def __call__(self, patient):
        samples = []

        try:
            events = patient.get_events("patient")
        except Exception:
            return samples

        if not events:
            return samples

        attr = events[0].attr_dict

        total_hours = self._get_total_hours(attr)
        if total_hours is None or total_hours < self.min_history_hours:
            return samples

        usable_hours = int(min(total_hours, self.max_hours))
        if usable_hours < self.min_history_hours:
            return samples

        names, index = self._build_feature_index()
        if not names:
            return samples

        static_vec = self._encode_static(attr)

        filled, mask, decay = self._extract_hourly(
            patient, usable_hours, index, len(names)
        )

        ts = self._combine(filled, mask, decay)

        feature_names = []
        for n in names:
            feature_names.extend([f"{n}_val", f"{n}_mask", f"{n}_decay"])

        for t in range(self.min_history_hours, usable_hours + 1):
            remaining = max(total_hours - t, 0.0)

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": attr.get("patientunitstayid"),
                    "time_series": ts[:t],
                    "static": static_vec,
                    "target_los_hours": float(remaining),
                    "feature_names": feature_names,
                    "history_hours": t,

                    # raw categorical statics for later train-only vocab fitting
                    "categorical_static_raw": {
                    "gender": str(attr.get("gender")) if attr.get("gender") is not None else "__MISSING__",
                    "ethnicity": str(attr.get("ethnicity")) if attr.get("ethnicity") is not None else "__MISSING__",
                    },
                }
            )

        return samples
