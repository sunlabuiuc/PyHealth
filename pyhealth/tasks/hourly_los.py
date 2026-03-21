from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_task import BaseTask


class HourlyLOSEICU(BaseTask):
    """Hourly remaining length-of-stay regression task for eICU.

    Each Patient object in this PyHealth branch behaves like a stay-level
    container with event partitions. We use:
      - patient.get_events("patient") for stay metadata
      - patient.get_events("lab") for lab measurements

    LoS target is remaining ICU length of stay in hours, derived from
    unitdischargeoffset (minutes).
    """

    task_name: str = "HourlyLOSEICU"

    def __init__(
        self,
        time_series_tables: Optional[List[str]] = None,
        time_series_features: Optional[Dict[str, List[str]]] = None,
        static_features: Optional[List[str]] = None,
        min_history_hours: int = 5,
        max_hours: int = 48,
    ):
        self.time_series_tables = time_series_tables or ["lab"]
        self.time_series_features = time_series_features or {
            "lab": ["-basos"],
        }
        self.static_features = static_features or []
        self.min_history_hours = min_history_hours
        self.max_hours = max_hours

        self.input_schema: Dict[str, Tuple[str, Dict[str, Any]]] = {
            "time_series": ("tensor", {}),
            "static": ("tensor", {}),
        }
        self.output_schema: Dict[str, str] = {
            "target_los_hours": "regression",
        }

    def pre_filter(self, global_event_df):
        return global_event_df

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        # Patient/stay metadata
        try:
            patient_events = patient.get_events("patient")
        except Exception:
            return samples

        if not patient_events:
            return samples

        p_event = patient_events[0]
        p_attr = p_event.attr_dict

        try:
            total_hours = float(p_attr["unitdischargeoffset"]) / 60.0
        except Exception:
            return samples

        if total_hours < self.min_history_hours:
            return samples

        usable_hours = int(min(total_hours, self.max_hours))
        if usable_hours < self.min_history_hours:
            return samples

        # Features
        flattened_feature_names: List[str] = []
        for table in self.time_series_tables:
            flattened_feature_names.extend(self.time_series_features.get(table, []))

        num_features = len(flattened_feature_names)
        if num_features == 0:
            return samples

        feature_index = {name: i for i, name in enumerate(flattened_feature_names)}

        # Hourly bins: [hour][feature] -> list of values
        hour_bins: List[List[List[float]]] = [
            [[] for _ in range(num_features)] for _ in range(usable_hours)
        ]

        # Process lab events
        if "lab" in self.time_series_tables:
            try:
                lab_events = patient.get_events("lab")
            except Exception:
                lab_events = []

            for event in lab_events:
                attr = event.attr_dict

                name = attr.get("labname")
                if name not in feature_index:
                    continue

                try:
                    value = float(attr.get("labresult"))
                    offset_hours = float(attr.get("labresultoffset")) / 60.0
                except Exception:
                    continue

                hour_idx = int(offset_hours)
                if hour_idx < 0 or hour_idx >= usable_hours:
                    continue

                idx = feature_index[name]
                hour_bins[hour_idx][idx].append(value)

        # Aggregate to dense hourly matrix
        hourly_matrix: List[List[float]] = []
        for h in range(usable_hours):
            row: List[float] = []
            for f in range(num_features):
                vals = hour_bins[h][f]
                row.append(float(np.mean(vals)) if vals else 0.0)
            hourly_matrix.append(row)

        # Generate one causal sample per hour
        for t in range(self.min_history_hours, usable_hours + 1):
            remaining = max(total_hours - t, 0.0)

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": p_attr.get("patientunitstayid"),
                    "feature_names": flattened_feature_names,
                    "time_series": hourly_matrix[:t],
                    "static": [],
                    "target_los_hours": float(remaining),
                    "history_hours": t,
                }
            )

        return samples
