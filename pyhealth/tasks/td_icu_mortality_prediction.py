"""Temporal Difference ICU Mortality Prediction Task for MIMIC-IV.

This module implements the Semi-Markov state construction and mortality
prediction task described in:

    Frost, T., Li, K., & Harris, S. (2024). Robust Real-Time Mortality
    Prediction in the Intensive Care Unit using Temporal Difference Learning.
    Proceedings of Machine Learning Research, 259, 350-363.

The task constructs patient states from irregularly sampled ICU measurements
using the 5-tuple representation {value, timepoint, feature, delta_value,
delta_time}, suitable for training with both supervised and temporal
difference learning approaches.
"""

from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np
import polars as pl

from pyhealth.tasks.base_task import BaseTask


class TDICUMortalityPredictionMIMIC4(BaseTask):
    """Task for TD-learning-based ICU mortality prediction using MIMIC-IV.

    Constructs Semi-Markov states from irregularly sampled lab measurements.
    Each state consists of a context window of recent measurements encoded as
    5-tuples {value, timepoint, feature_index, delta_value, delta_time}.

    The task generates one sample per measurement event (state marker) per
    ICU admission, with mortality labels at 1, 3, 7, 14, and 28 day horizons.

    Args:
        context_length: Maximum number of measurements per state. Defaults
            to 400.
        input_window_hours: Hours of lookback for building state context.
            Defaults to 168 (7 days).
        prediction_horizon_days: Primary mortality prediction horizon in days.
            Defaults to 28.
        state_sample_rate: Fraction of state markers to sample per admission
            to control dataset size. 1.0 uses all markers. Defaults to 1.0.
        min_measurements: Minimum number of measurements required in the
            context window to generate a sample. Defaults to 3.

    Attributes:
        task_name: Name identifier for this task.
        input_schema: Schema for input data (timeseries of 5-tuples).
        output_schema: Schema for output data (binary mortality label).

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from td_icu_mortality_prediction import TDICUMortalityPredictionMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["labevents"],
        ... )
        >>> task = TDICUMortalityPredictionMIMIC4(
        ...     context_length=400,
        ...     input_window_hours=168,
        ...     prediction_horizon_days=28,
        ... )
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "TDICUMortalityPredictionMIMIC4"
    input_schema: Dict[str, str] = {"measurements": "timeseries"}
    output_schema: Dict[str, str] = {"mortality_28d": "binary"}

    LAB_FEATURE_MAP: ClassVar[Dict[str, str]] = {
        "ALT": "ALT",
        "AST": "AST",
        "Albumin": "Albumin",
        "Alkaline Phosphatase": "ALP",
        "Amylase": "Amylase",
        "Anion Gap": "Anion Gap",
        "Bicarbonate": "HCO3",
        "Bilirubin, Total": "Bilirubin",
        "Calcium, Total": "Calcium",
        "Chloride": "Chloride",
        "Creatinine": "Creatinine",
        "Glucose": "Glucose",
        "Hematocrit": "Haematocrit",
        "Hemoglobin": "Haemoglobin",
        "Lactate": "Lactate",
        "Lipase": "Lipase",
        "Magnesium": "Magnesium",
        "Phosphate": "Phosphate",
        "Platelet Count": "Platelets",
        "Potassium": "Potassium",
        "Sodium": "Sodium",
        "Troponin T": "Troponin - T",
        "Urea Nitrogen": "Urea",
        "White Blood Cells": "WBC",
        "pH": "pH",
        "pCO2": "Blood Gas pCO2",
        "pO2": "Blood Gas pO2",
        "Base Excess": "Base Excess",
        "C-Reactive Protein": "CRP",
        "INR(PT)": "INR",
        "PT": "Prothrombin Time",
        "LDH": "LDH",
    }

    FEATURE_NAMES: ClassVar[List[str]] = sorted(set(LAB_FEATURE_MAP.values()))
    DEMOGRAPHIC_FEATURES: ClassVar[Dict[str, int]] = {
        "age": 0,
        "gender": 1,
        "patientweight": 2,
    }
    FEATURE_INDEX: ClassVar[Dict[str, int]] = {
        name: i + 3 for i, name in enumerate(FEATURE_NAMES)
    }

    def __init__(
        self,
        context_length: int = 400,
        input_window_hours: int = 168,
        prediction_horizon_days: int = 28,
        state_sample_rate: float = 1.0,
        min_measurements: int = 3,
    ):
        self.context_length = context_length
        self.input_window_hours = input_window_hours
        self.prediction_horizon_days = prediction_horizon_days
        self.state_sample_rate = state_sample_rate
        self.min_measurements = min_measurements

    def _parse_timestamp(self, ts: Any) -> Optional[datetime]:
        """Parse a timestamp string or datetime into a datetime object."""
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(ts, fmt)
                except ValueError:
                    continue
        return None

    def _compute_mortality_labels(
        self,
        state_time: datetime,
        death_time: Optional[datetime],
    ) -> Dict[str, int]:
        """Compute mortality labels at multiple horizons from a state time."""
        labels = {}
        for days in [1, 3, 7, 14, 28]:
            if death_time is not None:
                time_to_death = (death_time - state_time).total_seconds() / 3600
                labels[f"mortality_{days}d"] = int(
                    time_to_death <= days * 24 and time_to_death >= 0
                )
            else:
                labels[f"mortality_{days}d"] = 0
        return labels

    def _build_state_tuples(
        self,
        measurements: List[Dict[str, Any]],
        state_time: datetime,
        age: float,
        gender: int,
        weight: float,
    ) -> Dict[str, List[float]]:
        """Build the 5-tuple representation for a state.

        Each measurement within the lookback window is encoded as:
            - value: the measurement value
            - timepoint: hours before the state marker time
            - feature: integer index of the feature
            - delta_value: change from previous measurement of same feature
            - delta_time: hours since previous measurement of same feature

        Demographics (age, gender, weight) are prepended with timepoint=0.

        Args:
            measurements: List of measurement dicts with keys
                {feature, value, timestamp}.
            state_time: The current state marker timestamp.
            age: Patient age in years.
            gender: Patient gender (0=Male, 1=Female).
            weight: Patient weight in kg.

        Returns:
            Dict with keys: values, timepoints, features, delta_values,
            delta_times. Each is a list of length <= context_length.
        """
        window_start = state_time - timedelta(hours=self.input_window_hours)

        window_measurements = []
        for m in measurements:
            ts = m["timestamp"]
            if window_start <= ts <= state_time:
                hours_before = (state_time - ts).total_seconds() / 3600
                window_measurements.append(
                    {
                        "value": m["value"],
                        "timepoint": hours_before,
                        "feature": m["feature"],
                        "feature_idx": self.FEATURE_INDEX.get(m["feature"], -1),
                    }
                )

        window_measurements.sort(key=lambda x: (x["feature"], x["timepoint"]))

        prev_by_feature: Dict[str, Dict[str, float]] = {}
        for m in window_measurements:
            feat = m["feature"]
            if feat in prev_by_feature:
                m["delta_value"] = m["value"] - prev_by_feature[feat]["value"]
                m["delta_time"] = m["timepoint"] - prev_by_feature[feat]["timepoint"]
            else:
                m["delta_value"] = float("nan")
                m["delta_time"] = float("nan")
            prev_by_feature[feat] = {
                "value": m["value"],
                "timepoint": m["timepoint"],
            }

        window_measurements.sort(key=lambda x: x["timepoint"])

        values = [float(age), float(gender), float(weight)]
        timepoints = [0.0, 0.0, 0.0]
        features = [
            self.DEMOGRAPHIC_FEATURES["age"],
            self.DEMOGRAPHIC_FEATURES["gender"],
            self.DEMOGRAPHIC_FEATURES["patientweight"],
        ]
        delta_values = [float("nan"), float("nan"), float("nan")]
        delta_times = [float("nan"), float("nan"), float("nan")]

        remaining = self.context_length - 3
        for m in window_measurements[:remaining]:
            values.append(m["value"])
            timepoints.append(m["timepoint"])
            features.append(m["feature_idx"])
            delta_values.append(m["delta_value"])
            delta_times.append(m["delta_time"])

        return {
            "values": values,
            "timepoints": timepoints,
            "features": features,
            "delta_values": delta_values,
            "delta_times": delta_times,
        }

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient into TD-ICU mortality prediction samples.

        For each ICU admission, every lab measurement event serves as a state
        marker. At each state marker, we look back up to input_window_hours
        and construct a context window of 5-tuples.

        Args:
            patient: A PyHealth Patient object with events from MIMIC-IV.

        Returns:
            List of sample dicts, each containing:
                - patient_id: Patient identifier
                - admission_id: Hospital admission identifier
                - measurements: Tuple of (timestamps, feature_matrix) where
                    feature_matrix columns are
                    [value, timepoint, feature_idx, delta_value, delta_time]
                - mortality_28d: Binary mortality label at primary horizon
                - mortality_1d through mortality_14d: Additional horizon labels
        """
        samples = []

        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []
        demographics = demographics[0]
        anchor_age = int(demographics.anchor_age)
        if anchor_age < 18:
            return []
        gender = 1 if demographics.gender == "F" else 0

        death_time = None
        if hasattr(demographics, "dod") and demographics.dod is not None:
            death_time = self._parse_timestamp(demographics.dod)

        admissions = patient.get_events(event_type="admissions")

        for admission in admissions:
            admit_time = admission.timestamp
            dischtime = self._parse_timestamp(admission.dischtime)
            if dischtime is None:
                continue

            hadm_id = admission.hadm_id

            admission_death_time = None
            if int(admission.hospital_expire_flag) == 1 and death_time is not None:
                admission_death_time = death_time
            elif int(admission.hospital_expire_flag) == 1 and dischtime is not None:
                admission_death_time = dischtime

            labevents_df = patient.get_events(
                event_type="labevents",
                start=admit_time,
                end=dischtime,
                return_df=True,
            )

            if labevents_df.height == 0:
                continue

            known_labels = list(self.LAB_FEATURE_MAP.keys())
            if "labevents/label" in labevents_df.columns:
                labevents_df = labevents_df.filter(
                    pl.col("labevents/label").is_in(known_labels)
                )
            else:
                continue

            if labevents_df.height == 0:
                continue

            valuenum_col = "labevents/valuenum"
            if valuenum_col not in labevents_df.columns:
                continue

            labevents_df = labevents_df.filter(pl.col(valuenum_col).is_not_null())

            if labevents_df.height == 0:
                continue

            measurements = []
            for row in labevents_df.iter_rows(named=True):
                ts = row.get("timestamp")
                if ts is None:
                    continue
                ts = self._parse_timestamp(ts) if isinstance(ts, str) else ts
                if ts is None:
                    continue

                lab_label = row.get("labevents/label", "")
                feature_name = self.LAB_FEATURE_MAP.get(lab_label)
                if feature_name is None:
                    continue

                try:
                    value = float(row[valuenum_col])
                except (TypeError, ValueError):
                    continue

                measurements.append(
                    {
                        "timestamp": ts,
                        "feature": feature_name,
                        "value": value,
                    }
                )

            if len(measurements) < self.min_measurements:
                continue

            measurements.sort(key=lambda x: x["timestamp"])

            weight = 86.0 if gender == 0 else 74.0

            try:
                age = admit_time.year - int(demographics.anchor_year) + anchor_age
            except (TypeError, ValueError):
                age = anchor_age

            state_times = sorted(set(m["timestamp"] for m in measurements))

            if self.state_sample_rate < 1.0:
                n_keep = max(1, int(len(state_times) * self.state_sample_rate))
                rng = np.random.default_rng(hash(patient.patient_id) & 0xFFFFFFFF)
                indices = rng.choice(len(state_times), size=n_keep, replace=False)
                state_times = [state_times[i] for i in sorted(indices)]

            for state_time in state_times:
                state = self._build_state_tuples(
                    measurements, state_time, age, gender, weight
                )

                n_meas = len(state["values"]) - 3  
                if n_meas < self.min_measurements:
                    continue

                mortality_labels = self._compute_mortality_labels(
                    state_time, admission_death_time
                )

                n = len(state["values"])
                feature_matrix = np.column_stack(
                    [
                        np.array(state["values"], dtype=np.float32),
                        np.array(state["timepoints"], dtype=np.float32),
                        np.array(state["features"], dtype=np.float32),
                        np.array(state["delta_values"], dtype=np.float32),
                        np.array(state["delta_times"], dtype=np.float32),
                    ]
                )

                timestamps = [state_time] * n

                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "admission_id": hadm_id,
                        "measurements": (timestamps, feature_matrix),
                        "mortality_1d": mortality_labels["mortality_1d"],
                        "mortality_3d": mortality_labels["mortality_3d"],
                        "mortality_7d": mortality_labels["mortality_7d"],
                        "mortality_14d": mortality_labels["mortality_14d"],
                        "mortality_28d": mortality_labels["mortality_28d"],
                    }
                )

        return samples
