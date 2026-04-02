from __future__ import annotations

from typing import Any, Dict, List, Optional
import math
from datetime import datetime, timedelta

import torch

from .base_task import BaseTask


class HourlyLOSEICU(BaseTask):
    """Hourly remaining length-of-stay regression task for ICU datasets.

    This task builds hourly time-series samples for ICU remaining length-of-stay
    prediction from either eICU-style or MIMIC-IV-style patient records.

    For each ICU stay, the task:
        1. Extracts observations from up to `pre_icu_hours` before ICU admission
           through ICU discharge.
        2. Buckets measurements into hourly bins.
        3. Keeps the most recent measurement within each hour.
        4. Forward-fills missing values.
        5. Computes decay features based on hours since last real observation.
        6. Drops pre-ICU rows after forward-fill so the final sequence is ICU-only.
        7. Produces autoregressive hourly samples for remaining LoS prediction.

    Output fields include:
        - ``time_series``: shape [T, 3F] with per-feature [value, mask, decay]
        - ``static``: numeric and one-hot encoded static features
        - ``target_los_hours``: scalar remaining LoS target at the current hour
        - ``target_los_sequence``: sequence of remaining LoS values through history

    Notes:
        - This implementation is intended to align with the paper's preprocessing
          logic of including pre-ICU data during forward-fill, then removing
          pre-ICU rows before model input.
        - Categorical static features are one-hot encoded with a small capped
          vocabulary per field.
    """

    task_name: str = "HourlyLOSEICU"

    def __init__(
        self,
        diagnosis_tables: Optional[List[str]] = None,
        include_diagnoses: bool = False,
        diagnosis_time_limit_hours: int = 5,
        time_series_tables: Optional[List[str]] = None,
        time_series_features: Optional[Dict[str, List[str]]] = None,
        numeric_static_features: Optional[List[str]] = None,
        categorical_static_features: Optional[List[str]] = None,
        min_history_hours: int = 5,
        max_hours: int = 48,
        pre_icu_hours: int = 24,
    ):
        """Initialize the hourly ICU LoS task.

        Args:
            diagnosis_tables: Diagnosis-related tables to inspect for diagnosis
                extraction, primarily for eICU.
            include_diagnoses: Whether to extract and attach diagnosis strings.
            diagnosis_time_limit_hours: Maximum diagnosis time from ICU admission
                to include, in hours.
            time_series_tables: Tables that contain time-series observations.
            time_series_features: Mapping from table name to the list of feature
                names to extract from that table.
            numeric_static_features: Static numeric features to encode directly.
            categorical_static_features: Static categorical features to one-hot
                encode using task-local vocabularies.
            min_history_hours: Minimum history length required before a sample is
                emitted.
            max_hours: Maximum ICU hours to keep from a stay.
            pre_icu_hours: Number of hours before ICU admission to include during
                extraction and forward-fill before cropping back to ICU-only rows.
        """
        self.diagnosis_tables = diagnosis_tables or []
        self.include_diagnoses = include_diagnoses
        self.diagnosis_time_limit_hours = diagnosis_time_limit_hours
        self.time_series_tables = time_series_tables or ["lab"]
        self.time_series_features = time_series_features or {"lab": ["-basos"]}
        self.numeric_static_features = numeric_static_features or []
        self.categorical_static_features = categorical_static_features or []
        self.min_history_hours = min_history_hours
        self.max_hours = max_hours
        self.pre_icu_hours = pre_icu_hours

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
        """Optionally filter the global event dataframe before task processing.

        Args:
            global_event_df: The global event dataframe prepared by the dataset.

        Returns:
            The input dataframe unchanged.
        """
        return global_event_df

    def _split_hierarchical_diagnosis(self, raw: Any) -> List[str]:
        """Split a hierarchical diagnosis string into cumulative prefix tokens.

        Example:
            "a | b | c" -> ["a", "a | b", "a | b | c"]

        Args:
            raw: Raw diagnosis string or other value.

        Returns:
            A list of cumulative hierarchical diagnosis prefixes.
        """
        if raw is None:
            return []

        s = str(raw).strip()
        if not s:
            return []

        parts = [p.strip().lower() for p in s.split("|") if p.strip()]
        if not parts:
            return []

        prefixes = []
        current = []
        for part in parts:
            current.append(part)
            prefixes.append(" | ".join(current))
        return prefixes

    def _extract_eicu_diagnoses(self, patient) -> List[str]:
        """Extract hierarchical diagnosis tokens for an eICU patient.

        Only diagnoses within `diagnosis_time_limit_hours` are included.

        Args:
            patient: PyHealth patient object.

        Returns:
            A sorted list of unique diagnosis tokens.
        """
        if not self.include_diagnoses:
            return []

        diagnosis_tokens = set()

        for table in self.diagnosis_tables:
            try:
                events = patient.get_events(table)
            except Exception:
                events = []

            for e in events:
                attr = e.attr_dict

                offset_minutes = None
                for offset_key in [
                    "pasthistoryoffset",
                    "admitdxenteredoffset",
                    "diagnosisoffset",
                ]:
                    offset_minutes = self._safe_float(attr.get(offset_key))
                    if offset_minutes is not None:
                        break

                if offset_minutes is None:
                    continue

                offset_hours = offset_minutes / 60.0
                if offset_hours > self.diagnosis_time_limit_hours:
                    continue

                raw_candidates = [
                    attr.get("pasthistorypath"),
                    attr.get("admitdxpath"),
                    attr.get("diagnosisstring"),
                    attr.get("diagnosispath"),
                ]

                raw_value = None
                for cand in raw_candidates:
                    if cand is not None and str(cand).strip():
                        raw_value = cand
                        break

                if raw_value is None:
                    continue

                for tok in self._split_hierarchical_diagnosis(raw_value):
                    diagnosis_tokens.add(tok)

        return sorted(diagnosis_tokens)

    def _safe_float(self, value: Any) -> Optional[float]:
        """Convert a value to a finite float if possible.

        Args:
            value: Input value.

        Returns:
            A finite float, or None if conversion fails or the value is NaN/inf.
        """
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
        """Convert a value to a datetime if possible.

        Supports ISO strings and several common datetime formats.

        Args:
            value: Input value.

        Returns:
            A datetime object, or None if parsing fails.
        """
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
        """Normalize a feature name to a lowercase stripped string.

        Args:
            x: Raw feature name.

        Returns:
            Normalized feature name string.
        """
        if x is None:
            return ""
        return str(x).strip().lower()

    def _register_category(self, feature, category):
        """Register a categorical value in the static vocabulary.

        Args:
            feature: Static feature name.
            category: Category value.

        Returns:
            The integer index assigned to the category.
        """
        vocab = self.static_vocab.setdefault(feature, {})
        if category not in vocab:
            vocab[category] = len(vocab)
        return vocab[category]

    def _encode_static(self, attr: Dict[str, Any]) -> List[float]:
        """Encode static numeric and categorical features into a flat vector.

        Numeric features are inserted directly after safe float conversion.
        Categorical features are one-hot encoded with a small capped vocabulary.

        Args:
            attr: Source attribute dictionary.

        Returns:
            Encoded static feature vector.
        """
        numeric = []
        categorical = []

        for f in self.numeric_static_features:
            val = self._safe_float(attr.get(f))
            if val is None:
                val = 0.0
            numeric.append(float(val))

        for f in self.categorical_static_features:
            raw = attr.get(f)

            if raw is None:
                cid = "__MISSING__"
            else:
                cid = str(raw)

            vocab = self.static_vocab.setdefault(f, {})

            if cid not in vocab:
                vocab[cid] = len(vocab)

            max_size = 10
            one_hot = [0.0] * max_size
            idx = vocab[cid]

            if idx < max_size:
                one_hot[idx] = 1.0

            categorical.extend(one_hot)

        return numeric + categorical

    def _build_feature_index(self):
        """Build the normalized time-series feature list and index mapping.

        Returns:
            A tuple of:
                - normalized feature names in deterministic order
                - mapping from feature name to feature index

        Raises:
            ValueError: If no time-series features are defined.
        """
        names = []
        for table in self.time_series_tables:
            names.extend(self.time_series_features.get(table, []))

        normalized_names = [self._norm_name(n) for n in names]

        if len(normalized_names) == 0:
            raise ValueError("No time-series features defined.")

        if len(set(normalized_names)) != len(normalized_names):
            seen = set()
            deduped = []
            for n in normalized_names:
                if n not in seen:
                    seen.add(n)
                    deduped.append(n)
            normalized_names = deduped

        normalized_names = list(normalized_names)

        print("\n[DEBUG] Feature list (normalized):")
        for i, n in enumerate(normalized_names):
            print(f"  {i}: {n}")
        print(f"[DEBUG] Total features: {len(normalized_names)}\n")

        return normalized_names, {n: i for i, n in enumerate(normalized_names)}

    def _combine_value_mask_decay(self, filled, mask, decay):
        """Interleave filled values, masks, and decay into [value, mask, decay].

        Args:
            filled: Forward-filled values, shape [T, F].
            mask: Observation mask, shape [T, F].
            decay: Decay features, shape [T, F].

        Returns:
            Combined feature matrix of shape [T, 3F].
        """
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
        """Build output feature names for value, mask, and decay channels.

        Args:
            normalized_names: Base time-series feature names.

        Returns:
            Expanded feature names in [name_val, name_mask, name_decay] order.
        """
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
        """Build an hourly [value, mask, decay] tensor from bucketed observations.

        Each observation is a tuple:
            (hour_index, feature_index, value, precise_offset_hours)

        For each hour and feature, the most recent measurement within that hour
        is retained. Missing values are forward-filled. Mask indicates whether
        the value was observed in that exact hour. Decay follows 0.75^j where j
        is the number of hours since the last real observation.

        Args:
            observations: Bucketed observations.
            usable_hours: Number of hours in the timeline being built.
            num_features: Number of time-series features.

        Returns:
            Combined [T, 3F] tensor as a Python list of rows.
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
                    d_row.append(1.0)
                else:
                    f_row.append(last_val[f])
                    m_row.append(0.0)

                    if last_seen[f] is None:
                        d_row.append(0.0)
                    else:
                        j = h - last_seen[f]
                        d_row.append(float(0.75 ** j))

            filled.append(f_row)
            mask.append(m_row)
            decay.append(d_row)

        return self._combine_value_mask_decay(filled, mask, decay)

    def _make_cropped_hourly_tensor(
        self,
        observations: List[tuple[int, int, float, float]],
        total_hours: float,
        num_features: int,
    ):
        """Build an extended timeline tensor and crop it back to ICU-only rows.

        The timeline begins `pre_icu_hours` before ICU admission, allowing
        pre-ICU observations to participate in hourly bucketing and forward-fill.
        After the extended tensor is built, the leading pre-ICU rows are removed.

        Args:
            observations: Observations indexed on an extended timeline whose
                zero point is ICU admission minus `self.pre_icu_hours`.
            total_hours: ICU stay length in hours.
            num_features: Number of time-series features.

        Returns:
            Cropped ICU-only [T, 3F] tensor as a Python list of rows.
        """
        usable_hours = int(min(total_hours, self.max_hours))
        extended_hours = self.pre_icu_hours + usable_hours

        full_ts = self._make_hourly_tensor(
            observations=observations,
            usable_hours=extended_hours,
            num_features=num_features,
        )

        return full_ts[self.pre_icu_hours : self.pre_icu_hours + usable_hours]

    def _make_samples_for_stay(
        self,
        patient,
        visit_id: Any,
        total_hours: float,
        static_attr: Dict[str, Any],
        observations: List[tuple[int, int, float, float]],
        normalized_feature_names: List[str],
        diagnosis_raw: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Create autoregressive hourly samples for a single ICU stay.

        Args:
            patient: PyHealth patient object.
            visit_id: Stay or visit identifier.
            total_hours: Total ICU stay length in hours.
            static_attr: Static attributes to encode.
            observations: Extended-timeline observations.
            normalized_feature_names: Base feature names.
            diagnosis_raw: Optional diagnosis tokens.

        Returns:
            A list of task samples, one for each prediction hour from
            `min_history_hours` through `usable_hours`.
        """
        samples = []

        if total_hours is None or total_hours < self.min_history_hours:
            return samples

        usable_hours = int(min(total_hours, self.max_hours))
        if usable_hours < self.min_history_hours:
            return samples

        static_vec = self._encode_static(static_attr)
        num_features = len(normalized_feature_names)
        ts = self._make_cropped_hourly_tensor(
            observations=observations,
            total_hours=total_hours,
            num_features=num_features,
        )
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
                    "diagnosis_raw": "|".join(diagnosis_raw) if diagnosis_raw else "",
                }
            )

        return samples

    def _build_eicu_samples(self, patient) -> List[Dict[str, Any]]:
        """Build hourly LoS samples for an eICU patient.

        This method extracts time-series observations using eICU offset fields,
        shifts them onto an extended timeline that includes pre-ICU hours,
        and creates ICU-only samples after cropping.

        Args:
            patient: PyHealth patient object.

        Returns:
            List of generated task samples.
        """
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

        extra_features = ["time in the icu", "time of day"]
        for f in extra_features:
            if f not in normalized_names:
                normalized_names.append(f)

        feature_index = {name: i for i, name in enumerate(normalized_names)}

        observed_features = set()
        observations = []

        for table in self.time_series_tables:
            try:
                events = patient.get_events(table)
            except Exception:
                events = []

            schema = self._get_eicu_table_schema(table)
            if schema is None:
                continue

            name_key, value_key, offset_key = schema

            for e in events:
                attr = e.attr_dict

                if name_key is None and value_key is None:
                    minutes = self._safe_float(attr.get(offset_key))
                    if minutes is None:
                        continue

                    offset_hours = minutes / 60.0
                    extended_hour = int(offset_hours) + self.pre_icu_hours

                    for raw_name, raw_val in attr.items():
                        norm_name = self._norm_name(raw_name)
                        if norm_name not in feature_index:
                            continue

                        value = self._safe_float(raw_val)
                        if value is None:
                            continue

                        observed_features.add(norm_name)
                        fi = feature_index[norm_name]
                        observations.append((extended_hour, fi, value, offset_hours))

                    ti_idx = feature_index.get("time in the icu")
                    if ti_idx is not None:
                        observations.append(
                            (extended_hour, ti_idx, offset_hours, offset_hours)
                        )

                    time_of_day = None
                    admit_time_str = anchor_attr.get("hospitaladmittime24")
                    if admit_time_str:
                        try:
                            h, m, s = map(int, admit_time_str.split(":"))
                            base_hour = h + m / 60.0 + s / 3600.0
                            time_of_day = (base_hour + offset_hours) % 24
                        except Exception:
                            pass

                    tod_idx = feature_index.get("time of day")
                    if tod_idx is not None and time_of_day is not None:
                        observations.append(
                            (extended_hour, tod_idx, time_of_day, offset_hours)
                        )

                    continue

                name = self._norm_name(attr.get(name_key))
                if name not in feature_index:
                    continue

                value = self._safe_float(attr.get(value_key))
                minutes = self._safe_float(attr.get(offset_key))
                if value is None or minutes is None:
                    continue

                observed_features.add(name)

                offset_hours = minutes / 60.0
                extended_hour = int(offset_hours) + self.pre_icu_hours
                fi = feature_index[name]
                observations.append((extended_hour, fi, value, offset_hours))

                ti_idx = feature_index.get("time in the icu")
                if ti_idx is not None:
                    observations.append(
                        (extended_hour, ti_idx, offset_hours, offset_hours)
                    )

                time_of_day = None
                admit_time_str = anchor_attr.get("hospitaladmittime24")
                if admit_time_str:
                    try:
                        h, m, s = map(int, admit_time_str.split(":"))
                        base_hour = h + m / 60.0 + s / 3600.0
                        time_of_day = (base_hour + offset_hours) % 24
                    except Exception:
                        pass

                tod_idx = feature_index.get("time of day")
                if tod_idx is not None and time_of_day is not None:
                    observations.append(
                        (extended_hour, tod_idx, time_of_day, offset_hours)
                    )

        visit_id = (
            anchor_attr.get("patientunitstayid")
            or anchor_attr.get("visit_id")
            or patient.patient_id
        )

        static_attr = dict(anchor_attr)
        diagnosis_raw = self._extract_eicu_diagnoses(patient)

        samples.extend(
            self._make_samples_for_stay(
                patient=patient,
                visit_id=visit_id,
                total_hours=total_hours,
                static_attr=static_attr,
                observations=observations,
                normalized_feature_names=normalized_names,
                diagnosis_raw=diagnosis_raw,
            )
        )

        if observed_features:
            print(f"[DEBUG] Observed features for patient: {sorted(observed_features)}")

        return samples

    def _get_eicu_table_schema(self, table: str):
        """Return the schema tuple for supported eICU time-series tables.

        Args:
            table: Table name.

        Returns:
            A tuple of (name_key, value_key, offset_key), or None if unsupported.
        """
        table = str(table).strip().lower()

        schema = {
            "lab": ("labname", "labresult", "labresultoffset"),
            "respiratorycharting": (
                "respchartvaluelabel",
                "respchartvalue",
                "respchartoffset",
            ),
            "nursecharting": (
                "nursingchartcelltypevallabel",
                "nursingchartvalue",
                "nursingchartoffset",
            ),
            "vitalperiodic": (None, None, "observationoffset"),
            "vitalaperiodic": (None, None, "observationoffset"),
        }

        return schema.get(table)

    def _get_mimic_table_schema(self, table: str):
        """Return the schema tuple for supported MIMIC-IV time-series tables.

        Args:
            table: Table name.

        Returns:
            A tuple of (name_key, value_key, time_key), or None if unsupported.
        """
        table = str(table).strip().lower()

        schema = {
            "labevents": ("label", None, "timestamp"),
            "chartevents": ("label", None, "timestamp"),
        }

        return schema.get(table)

    def _build_mimic_samples(self, patient) -> List[Dict[str, Any]]:
        """Build hourly LoS samples for a MIMIC-IV patient.

        This method extracts time-series observations from up to
        `pre_icu_hours` before ICU admission through ICU discharge, shifts them
        onto an extended timeline, and creates ICU-only samples after cropping.

        Args:
            patient: PyHealth patient object.

        Returns:
            List of generated task samples.
        """
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

        if not icu_rows:
            return samples

        patient_static = patient_rows[0].attr_dict if patient_rows else {}

        admissions_by_hadm = {}
        for a in admission_rows:
            hadm_id = a.attr_dict.get("hadm_id")
            if hadm_id is not None and hadm_id not in admissions_by_hadm:
                admissions_by_hadm[hadm_id] = a

        normalized_names, feature_index = self._build_feature_index()
        observed_features = set()
        if not normalized_names:
            return samples

        extra_features = ["time in the icu", "time of day"]
        for f in extra_features:
            if f not in normalized_names:
                normalized_names.append(f)

        feature_index = {name: i for i, name in enumerate(normalized_names)}

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
            pre_icu_start = intime - timedelta(hours=self.pre_icu_hours)

            for table in self.time_series_tables:
                try:
                    events = patient.get_events(table)
                except Exception:
                    events = []

                schema = self._get_mimic_table_schema(table)
                if schema is None:
                    continue

                name_key, _, _ = schema

                for event in events:
                    attr = event.attr_dict

                    event_hadm_id = attr.get("hadm_id")
                    if (
                        hadm_id is not None
                        and event_hadm_id is not None
                        and str(event_hadm_id) != str(hadm_id)
                    ):
                        continue

                    name = self._norm_name(attr.get(name_key))
                    if name not in feature_index:
                        continue

                    value = self._safe_float(attr.get("valuenum"))
                    if value is None:
                        value = self._safe_float(attr.get("value"))
                    if value is None:
                        continue

                    event_time = getattr(event, "timestamp", None)
                    if event_time is None:
                        event_time = self._safe_datetime(attr.get("storetime"))
                    if event_time is None:
                        event_time = self._safe_datetime(attr.get("charttime"))
                    if event_time is None:
                        continue

                    if event_time < pre_icu_start or event_time > outtime:
                        continue

                    observed_features.add(name)

                    offset_hours = (event_time - intime).total_seconds() / 3600.0
                    extended_hour = int(offset_hours) + self.pre_icu_hours
                    fi = feature_index[name]
                    observations.append((extended_hour, fi, value, offset_hours))

                    ti_idx = feature_index.get("time in the icu")
                    if ti_idx is not None:
                        observations.append(
                            (extended_hour, ti_idx, offset_hours, offset_hours)
                        )

                    tod_idx = feature_index.get("time of day")
                    if tod_idx is not None:
                        time_of_day = (
                            event_time.hour
                            + event_time.minute / 60.0
                            + event_time.second / 3600.0
                        )
                        observations.append(
                            (extended_hour, tod_idx, time_of_day, offset_hours)
                        )

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

        if observed_features:
            print(f"[DEBUG] Observed features for patient: {sorted(observed_features)}")

        return samples

    def __call__(self, patient):
        """Dispatch task construction based on dataset-specific patient contents.

        Args:
            patient: PyHealth patient object.

        Returns:
            A list of generated task samples for the patient.
        """
        try:
            if patient.get_events("patient"):
                return self._build_eicu_samples(patient)
        except Exception as e:
            print("[DEBUG] eICU task error:", repr(e))
            raise

        try:
            if patient.get_events("icustays"):
                return self._build_mimic_samples(patient)
        except Exception as e:
            print("[DEBUG] MIMIC task error:", repr(e))
            raise

        return []
