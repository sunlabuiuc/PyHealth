from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
from datetime import datetime

import torch

from .base_task import BaseTask


class HourlyLOSEICU(BaseTask):
    """Hourly remaining length-of-stay regression task for ICU datasets.

    Supports both:
      - eICU-style data
      - MIMIC-IV-style data

    For each ICU stay:
      1. Bucket measurements into hourly bins
      2. Keep the most recent measurement within each hour
      3. Forward-fill missing values
      4. Compute decay = hours since last real observation

    Outputs:
      - time_series: [T, 3F] as [value, mask, decay] per feature
      - static
      - target_los_hours
      - target_los_sequence
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
        self.time_series_features = time_series_features or {"lab": ["-basos"]}
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

    def _safe_datetime(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value

        s = str(value).strip()
        if not s:
            return None

        s = s.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s)
        except Exception:
            pass

        fmts = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d",
        ]
        for fmt in fmts:
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        return None

    def _norm_name(self, x: Any) -> str:
        if x is None:
            return ""
        return str(x).strip().lower()

    def _register_category(self, feature, category):
        vocab = self.static_vocab.setdefault(feature, {})
        if category not in vocab:
            vocab[category] = len(vocab)
        return vocab[category]

    def _encode_static(self, attr: Dict[str, Any]) -> List[float]:
        numeric = []
        categorical = []

        for f in self.categorical_static_features:
            raw = attr.get(f)

            # normalize category
            if raw is None:
                cid = "__MISSING__"
            else:
                cid = str(raw)

            # ensure stable vocab
            vocab = self.static_vocab.setdefault(f, {})

            if cid not in vocab:
                vocab[cid] = len(vocab)

            # FIXED SIZE: lock to max observed size
            max_size = 10  # small, safe cap for gender/race/etc.

            one_hot = [0.0] * max_size
            idx = vocab[cid]

            if idx < max_size:
                one_hot[idx] = 1.0

            categorical.extend(one_hot)

        return numeric + categorical

    def _build_feature_index(self):
        names = []
        for table in self.time_series_tables:
            names.extend(self.time_series_features.get(table, []))

        normalized_names = [self._norm_name(n) for n in names]
        return normalized_names, {n: i for i, n in enumerate(normalized_names)}

    def _combine_value_mask_decay(self, filled, mask, decay):
        combined = []
        for h in range(len(filled)):
            row = []
            for f in range(len(filled[h])):
                row.append(filled[h][f])
                row.append(mask[h][f])
                row.append(decay[h][f])
            combined.append(row)
        return combined

    def _build_feature_names(self, normalized_names: List[str]) -> List[str]:
        feature_names = []
        for n in normalized_names:
            feature_names.extend([f"{n}_val", f"{n}_mask", f"{n}_decay"])
        return feature_names

    def _make_hourly_tensor(
        self,
        observations: List[tuple[int, int, float, float]],
        usable_hours: int,
        num_features: int,
    ):
        """
        observations: list of (hour_index, feature_index, value, precise_offset_hours)
        """
        latest_vals = [[None] * num_features for _ in range(usable_hours)]
        latest_time = [[-float("inf")] * num_features for _ in range(usable_hours)]

        for hr, fi, val, precise_offset in observations:
            if hr < 0 or hr >= usable_hours:
                continue
            if precise_offset >= latest_time[hr][fi]:
                latest_time[hr][fi] = precise_offset
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

        return self._combine_value_mask_decay(filled, mask, decay)

    def _make_samples_for_stay(
        self,
        patient,
        visit_id: Any,
        total_hours: float,
        static_attr: Dict[str, Any],
        observations: List[tuple[int, int, float, float]],
        normalized_feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        samples = []

        if total_hours is None or total_hours < self.min_history_hours:
            return samples

        usable_hours = int(min(total_hours, self.max_hours))
        if usable_hours < self.min_history_hours:
            return samples

        static_vec = self._encode_static(static_attr)
        num_features = len(normalized_feature_names)
        ts = self._make_hourly_tensor(observations, usable_hours, num_features)
        feature_names = self._build_feature_names(normalized_feature_names)

        raw_cats = {}
        for f in self.categorical_static_features:
            raw_cats[f] = (
                str(static_attr.get(f))
                if static_attr.get(f) is not None
                else "__MISSING__"
            )

        for t in range(self.min_history_hours, usable_hours + 1):
            remaining = max(total_hours - t, 0.0)

            target_los_sequence = [
                float(max(total_hours - hour_idx, 0.0))
                for hour_idx in range(1, t + 1)
            ]

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": visit_id,
                    "time_series": ts[:t],
                    "static": static_vec,
                    "target_los_hours": float(remaining),
                    "target_los_sequence": torch.tensor(
                        target_los_sequence, dtype=torch.float32
                    ),
                    "feature_names": feature_names,
                    "history_hours": t,
                    "categorical_static_raw": raw_cats,
                }
            )

        return samples

    def _build_eicu_samples(self, patient) -> List[Dict[str, Any]]:
        samples = []

        try:
            patient_events = patient.get_events("patient")
        except Exception:
            patient_events = []

        if not patient_events:
            return samples

        anchor_attr = patient_events[0].attr_dict
        total_minutes = self._safe_float(anchor_attr.get("unitdischargeoffset"))
        if total_minutes is None:
            return samples

        total_hours = total_minutes / 60.0
        if total_hours < self.min_history_hours:
            return samples

        usable_hours = int(min(total_hours, self.max_hours))
        if usable_hours < self.min_history_hours:
            return samples

        normalized_names, feature_index = self._build_feature_index()
        if not normalized_names:
            return samples

        observations = []

        for table in self.time_series_tables:
            try:
                events = patient.get_events(table)
            except Exception:
                events = []

            for e in events:
                attr = e.attr_dict

                name = self._norm_name(attr.get("labname"))
                if name not in feature_index:
                    continue

                value = self._safe_float(attr.get("labresult"))
                minutes = self._safe_float(attr.get("labresultoffset"))
                if value is None or minutes is None:
                    continue

                offset_hours = minutes / 60.0
                hr = int(offset_hours)
                fi = feature_index[name]
                observations.append((hr, fi, value, offset_hours))

        visit_id = (
            anchor_attr.get("patientunitstayid")
            or anchor_attr.get("visit_id")
            or patient.patient_id
        )

        static_attr = dict(anchor_attr)

        samples.extend(
            self._make_samples_for_stay(
                patient=patient,
                visit_id=visit_id,
                total_hours=total_hours,
                static_attr=static_attr,
                observations=observations,
                normalized_feature_names=normalized_names,
            )
        )

        return samples

    def _build_mimic_samples(self, patient) -> List[Dict[str, Any]]:
        samples = []

        try:
            patient_rows = patient.get_events("patients")
        except Exception:
            patient_rows = []

        try:
            admission_rows = patient.get_events("admissions")
        except Exception:
            admission_rows = []

        try:
            icu_rows = patient.get_events("icustays")
        except Exception:
            icu_rows = []

        try:
            lab_rows = patient.get_events("labevents")
        except Exception:
            lab_rows = []

        if not icu_rows:
            return samples

        patient_static = patient_rows[0].attr_dict if patient_rows else {}

        admissions_by_hadm = {}
        for a in admission_rows:
            hadm_id = a.attr_dict.get("hadm_id")
            if hadm_id is not None and hadm_id not in admissions_by_hadm:
                admissions_by_hadm[hadm_id] = a

        normalized_names, feature_index = self._build_feature_index()
        if not normalized_names:
            return samples

        for icu_event in icu_rows:
            icu_attr = dict(icu_event.attr_dict)
            hadm_id = icu_attr.get("hadm_id")
            stay_id = icu_attr.get("stay_id")

            intime = getattr(icu_event, "timestamp", None)
            outtime = self._safe_datetime(icu_attr.get("outtime"))

            if intime is None or outtime is None:
                continue

            total_hours = (outtime - intime).total_seconds() / 3600.0
            if total_hours < self.min_history_hours:
                continue

            static_attr = dict(patient_static)
            if hadm_id in admissions_by_hadm:
                static_attr.update(admissions_by_hadm[hadm_id].attr_dict)
            static_attr.update(icu_attr)

            observations = []

            for lab_event in lab_rows:
                lab_attr = lab_event.attr_dict

                lab_hadm_id = lab_attr.get("hadm_id")
                if hadm_id is not None and lab_hadm_id is not None and str(lab_hadm_id) != str(hadm_id):
                    continue

                name = self._norm_name(lab_attr.get("label"))
                if name not in feature_index:
                    continue

                value = self._safe_float(lab_attr.get("valuenum"))
                if value is None:
                    value = self._safe_float(lab_attr.get("value"))
                if value is None:
                    continue

                event_time = getattr(lab_event, "timestamp", None)
                if event_time is None:
                    event_time = self._safe_datetime(lab_attr.get("storetime"))
                if event_time is None:
                    continue

                if event_time < intime or event_time > outtime:
                    continue

                offset_hours = (event_time - intime).total_seconds() / 3600.0
                hr = int(offset_hours)
                fi = feature_index[name]

                observations.append((hr, fi, value, offset_hours))

            samples.extend(
                self._make_samples_for_stay(
                    patient=patient,
                    visit_id=stay_id or hadm_id or patient.patient_id,
                    total_hours=total_hours,
                    static_attr=static_attr,
                    observations=observations,
                    normalized_feature_names=normalized_names,
                )
            )

        return samples

    def __call__(self, patient):
        # eICU path
        try:
            if patient.get_events("patient"):
                return self._build_eicu_samples(patient)
        except Exception:
            pass

        # MIMIC-IV path
        try:
            if patient.get_events("icustays"):
                return self._build_mimic_samples(patient)
        except Exception:
            pass

        return []