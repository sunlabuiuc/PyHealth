"""Tests for TDICUMortalityPredictionMIMIC4 task.

Uses synthetic patient data (no real MIMIC data required).
Tests complete in milliseconds. All data is generated in-memory
using MagicMock objects; no temporary files or directories are
created, so no cleanup is required.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np

from pyhealth.tasks.td_icu_mortality_prediction import TDICUMortalityPredictionMIMIC4


def make_event(attrs):
    """Create a mock Event object from a dict of attributes."""
    event = MagicMock()
    for k, v in attrs.items():
        setattr(event, k, v)
    return event


def make_lab_row(timestamp, label, valuenum):
    """Create a dict representing a labevents row."""
    return {
        "timestamp": timestamp,
        "labevents/label": label,
        "labevents/valuenum": valuenum,
        "labevents/itemid": "00000",
    }


def make_labevents_df(rows):
    """Create a mock polars DataFrame from lab rows."""
    import polars as pl

    if not rows:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "labevents/label": pl.Utf8,
                "labevents/valuenum": pl.Float64,
                "labevents/itemid": pl.Utf8,
            }
        )
    return pl.DataFrame(rows)


def make_patient(
    patient_id="P001",
    age=65,
    gender="M",
    anchor_year=2015,
    dod=None,
    admissions=None,
    lab_rows=None,
):
    """Create a mock PyHealth Patient with configurable events.

    Args:
        patient_id: Patient identifier.
        age: Anchor age.
        gender: "M" or "F".
        anchor_year: Anchor year for age calculation.
        dod: Date of death string or None.
        admissions: List of admission dicts, each with keys:
            hadm_id, admittime, dischtime, hospital_expire_flag.
        lab_rows: List of lab row dicts for make_lab_row.
    """
    patient = MagicMock()
    patient.patient_id = patient_id

    demographics = make_event(
        {
            "anchor_age": age,
            "gender": gender,
            "anchor_year": anchor_year,
            "dod": dod,
        }
    )

    if admissions is None:
        admissions = [
            {
                "hadm_id": "H001",
                "admittime": datetime(2020, 1, 1, 0, 0),
                "dischtime": "2020-01-10 00:00:00",
                "hospital_expire_flag": 0,
            }
        ]

    admission_events = []
    for adm in admissions:
        admission_events.append(
            make_event(
                {
                    "timestamp": adm["admittime"],
                    "hadm_id": adm["hadm_id"],
                    "dischtime": adm["dischtime"],
                    "hospital_expire_flag": adm["hospital_expire_flag"],
                }
            )
        )

    if lab_rows is None:
        lab_rows = []

    lab_df = make_labevents_df(lab_rows)

    def get_events_side_effect(
        event_type=None, start=None, end=None, return_df=False, filters=None
    ):
        if event_type == "patients":
            return [demographics]
        elif event_type == "admissions":
            return admission_events
        elif event_type == "labevents":
            if return_df:
                return lab_df
            return []
        return []

    patient.get_events = MagicMock(side_effect=get_events_side_effect)
    return patient


class TestTDICUMortalityTaskInit(unittest.TestCase):
    """Test task initialization and configuration."""

    def test_default_init(self):
        task = TDICUMortalityPredictionMIMIC4()
        self.assertEqual(task.context_length, 400)
        self.assertEqual(task.input_window_hours, 168)
        self.assertEqual(task.prediction_horizon_days, 28)
        self.assertEqual(task.state_sample_rate, 1.0)
        self.assertEqual(task.min_measurements, 3)

    def test_custom_init(self):
        task = TDICUMortalityPredictionMIMIC4(
            context_length=200,
            input_window_hours=72,
            prediction_horizon_days=7,
            state_sample_rate=0.5,
            min_measurements=5,
        )
        self.assertEqual(task.context_length, 200)
        self.assertEqual(task.input_window_hours, 72)
        self.assertEqual(task.prediction_horizon_days, 7)

    def test_schema(self):
        task = TDICUMortalityPredictionMIMIC4()
        self.assertIn("measurements", task.input_schema)
        self.assertIn("mortality_28d", task.output_schema)
        self.assertEqual(task.task_name, "TDICUMortalityPredictionMIMIC4")

    def test_feature_index_has_all_features(self):
        task = TDICUMortalityPredictionMIMIC4()
        for feat_name in task.FEATURE_NAMES:
            self.assertIn(feat_name, task.FEATURE_INDEX)
        # Demographics should have indices 0, 1, 2
        self.assertEqual(task.DEMOGRAPHIC_FEATURES["age"], 0)
        self.assertEqual(task.DEMOGRAPHIC_FEATURES["gender"], 1)
        self.assertEqual(task.DEMOGRAPHIC_FEATURES["patientweight"], 2)


class TestTDICUMortalityTaskSampling(unittest.TestCase):
    """Test sample generation with synthetic patients."""

    def _make_lab_rows(self, base_time, n=10, features=None):
        """Generate n lab measurement rows spread over time."""
        if features is None:
            features = ["Albumin", "Creatinine", "Glucose", "Sodium"]
        rows = []
        for i in range(n):
            ts = base_time + timedelta(hours=i * 2)
            feat = features[i % len(features)]
            rows.append(make_lab_row(ts, feat, 3.5 + i * 0.1))
        return rows

    def test_basic_sample_generation(self):
        """Test that samples are generated for a normal patient."""
        task = TDICUMortalityPredictionMIMIC4(min_measurements=1)
        base_time = datetime(2020, 1, 1, 6, 0)
        lab_rows = self._make_lab_rows(base_time, n=10)

        patient = make_patient(lab_rows=lab_rows)
        samples = task(patient)

        self.assertGreater(len(samples), 0)
        for s in samples:
            self.assertIn("patient_id", s)
            self.assertIn("admission_id", s)
            self.assertIn("measurements", s)
            self.assertIn("mortality_28d", s)
            self.assertIn("mortality_1d", s)
            self.assertIn(s["mortality_28d"], [0, 1])

    def test_sample_measurement_structure(self):
        """Test the 5-tuple structure of measurements."""
        task = TDICUMortalityPredictionMIMIC4(min_measurements=1)
        base_time = datetime(2020, 1, 1, 6, 0)
        lab_rows = self._make_lab_rows(base_time, n=6)

        patient = make_patient(lab_rows=lab_rows)
        samples = task(patient)

        self.assertGreater(len(samples), 0)
        timestamps, feature_matrix = samples[0]["measurements"]
        # feature_matrix should have 5 columns:
        # [value, timepoint, feature_idx, delta_value, delta_time]
        self.assertEqual(feature_matrix.shape[1], 5)
        # First 3 rows are demographics (age, gender, weight)
        self.assertEqual(feature_matrix[0, 2], 0.0)  # age feature idx
        self.assertEqual(feature_matrix[1, 2], 1.0)  # gender feature idx
        self.assertEqual(feature_matrix[2, 2], 2.0)  # weight feature idx

    def test_context_length_limit(self):
        """Test that samples respect the context_length limit."""
        task = TDICUMortalityPredictionMIMIC4(context_length=10, min_measurements=1)
        base_time = datetime(2020, 1, 1, 6, 0)
        lab_rows = self._make_lab_rows(base_time, n=50)

        patient = make_patient(lab_rows=lab_rows)
        samples = task(patient)

        for s in samples:
            _, feature_matrix = s["measurements"]
            self.assertLessEqual(feature_matrix.shape[0], 10)

    def test_underage_patient_excluded(self):
        """Patients under 18 should return no samples."""
        task = TDICUMortalityPredictionMIMIC4(min_measurements=1)
        base_time = datetime(2020, 1, 1, 6, 0)
        lab_rows = self._make_lab_rows(base_time, n=5)

        patient = make_patient(age=15, lab_rows=lab_rows)
        samples = task(patient)
        self.assertEqual(len(samples), 0)

    def test_no_lab_events(self):
        """Patient with no lab events should return no samples."""
        task = TDICUMortalityPredictionMIMIC4()
        patient = make_patient(lab_rows=[])
        samples = task(patient)
        self.assertEqual(len(samples), 0)

    def test_insufficient_measurements(self):
        """Patient with fewer measurements than min_measurements."""
        task = TDICUMortalityPredictionMIMIC4(min_measurements=10)
        base_time = datetime(2020, 1, 1, 6, 0)
        lab_rows = self._make_lab_rows(base_time, n=2)

        patient = make_patient(lab_rows=lab_rows)
        samples = task(patient)
        self.assertEqual(len(samples), 0)

    def test_mortality_label_deceased_patient(self):
        """Patient who dies should have mortality=1 for appropriate horizons."""
        task = TDICUMortalityPredictionMIMIC4(min_measurements=1)
        admit_time = datetime(2020, 1, 1, 0, 0)
        death_time = admit_time + timedelta(days=2)

        lab_rows = [
            make_lab_row(admit_time + timedelta(hours=1), "Albumin", 3.5),
            make_lab_row(admit_time + timedelta(hours=2), "Creatinine", 1.2),
            make_lab_row(admit_time + timedelta(hours=3), "Glucose", 110.0),
        ]

        patient = make_patient(
            dod="2020-01-03 00:00:00",
            admissions=[
                {
                    "hadm_id": "H001",
                    "admittime": admit_time,
                    "dischtime": "2020-01-03 00:00:00",
                    "hospital_expire_flag": 1,
                }
            ],
            lab_rows=lab_rows,
        )
        samples = task(patient)

        self.assertGreater(len(samples), 0)
        # First measurement is at hour 1, death at day 2 (48h away)
        # So 3d, 7d, 14d, 28d should be 1; 1d should be 0
        first_sample = samples[0]
        self.assertEqual(first_sample["mortality_3d"], 1)
        self.assertEqual(first_sample["mortality_7d"], 1)
        self.assertEqual(first_sample["mortality_28d"], 1)

    def test_surviving_patient_labels(self):
        """Surviving patient should have all mortality labels = 0."""
        task = TDICUMortalityPredictionMIMIC4(min_measurements=1)
        base_time = datetime(2020, 1, 1, 6, 0)
        lab_rows = self._make_lab_rows(base_time, n=5)

        patient = make_patient(lab_rows=lab_rows)
        samples = task(patient)

        for s in samples:
            self.assertEqual(s["mortality_1d"], 0)
            self.assertEqual(s["mortality_3d"], 0)
            self.assertEqual(s["mortality_7d"], 0)
            self.assertEqual(s["mortality_14d"], 0)
            self.assertEqual(s["mortality_28d"], 0)

    def test_state_sample_rate(self):
        """Subsampling state markers should reduce number of samples."""
        base_time = datetime(2020, 1, 1, 6, 0)
        lab_rows = self._make_lab_rows(base_time, n=20)

        task_full = TDICUMortalityPredictionMIMIC4(
            state_sample_rate=1.0, min_measurements=1
        )
        task_half = TDICUMortalityPredictionMIMIC4(
            state_sample_rate=0.5, min_measurements=1
        )

        patient_full = make_patient(lab_rows=lab_rows)
        patient_half = make_patient(lab_rows=lab_rows)

        samples_full = task_full(patient_full)
        samples_half = task_half(patient_half)

        self.assertGreater(len(samples_full), len(samples_half))

    def test_delta_computation(self):
        """Test that delta values are computed correctly for same feature."""
        task = TDICUMortalityPredictionMIMIC4(context_length=400, min_measurements=1)
        base_time = datetime(2020, 1, 5, 0, 0)

        # Two Albumin measurements, 4 hours apart
        lab_rows = [
            make_lab_row(base_time, "Albumin", 3.0),
            make_lab_row(base_time + timedelta(hours=4), "Albumin", 3.5),
            make_lab_row(base_time + timedelta(hours=8), "Creatinine", 1.0),
        ]

        patient = make_patient(lab_rows=lab_rows)
        samples = task(patient)

        # Get the last state marker (at hour 8)
        last_sample = samples[-1]
        _, matrix = last_sample["measurements"]

        # After demographics (3 rows), lab measurements follow
        # Find the second Albumin entry (should have delta_value)
        lab_rows_data = matrix[3:]  # skip demographics
        albumin_idx = task.FEATURE_INDEX["Albumin"]
        albumin_rows = [r for r in lab_rows_data if r[2] == float(albumin_idx)]
        # One of the albumin measurements should have a non-NaN delta_value
        if len(albumin_rows) >= 2:
            # The delta is computed in sorted order (by feature, then time)
            # Magnitude should be 0.5 (3.5 - 3.0)
            self.assertAlmostEqual(abs(albumin_rows[1][3]), 0.5, places=3)

    def test_gender_encoding(self):
        """Test that gender is correctly encoded (M=0, F=1)."""
        task = TDICUMortalityPredictionMIMIC4(min_measurements=1)
        base_time = datetime(2020, 1, 1, 6, 0)
        lab_rows = self._make_lab_rows(base_time, n=5)

        patient_m = make_patient(gender="M", lab_rows=lab_rows)
        patient_f = make_patient(gender="F", lab_rows=lab_rows)

        samples_m = task(patient_m)
        samples_f = task(patient_f)

        _, matrix_m = samples_m[0]["measurements"]
        _, matrix_f = samples_f[0]["measurements"]

        # Gender is second demographic (index 1), value column is 0
        self.assertEqual(matrix_m[1, 0], 0.0)  # Male
        self.assertEqual(matrix_f[1, 0], 1.0)  # Female

    def test_weight_defaults(self):
        """Test default weight assignment by gender."""
        task = TDICUMortalityPredictionMIMIC4(min_measurements=1)
        base_time = datetime(2020, 1, 1, 6, 0)
        lab_rows = self._make_lab_rows(base_time, n=5)

        patient_m = make_patient(gender="M", lab_rows=lab_rows)
        patient_f = make_patient(gender="F", lab_rows=lab_rows)

        samples_m = task(patient_m)
        samples_f = task(patient_f)

        _, matrix_m = samples_m[0]["measurements"]
        _, matrix_f = samples_f[0]["measurements"]

        # Weight is third demographic (index 2), value column is 0
        self.assertEqual(matrix_m[2, 0], 86.0)  # Male default
        self.assertEqual(matrix_f[2, 0], 74.0)  # Female default


class TestTDICUMortalityTaskEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_multiple_admissions(self):
        """Patient with multiple admissions should generate samples for each."""
        task = TDICUMortalityPredictionMIMIC4(min_measurements=1)
        admit1 = datetime(2020, 1, 1, 0, 0)
        admit2 = datetime(2020, 6, 1, 0, 0)

        admissions = [
            {
                "hadm_id": "H001",
                "admittime": admit1,
                "dischtime": "2020-01-10 00:00:00",
                "hospital_expire_flag": 0,
            },
            {
                "hadm_id": "H002",
                "admittime": admit2,
                "dischtime": "2020-06-10 00:00:00",
                "hospital_expire_flag": 0,
            },
        ]

        lab_rows = [
            make_lab_row(admit1 + timedelta(hours=1), "Albumin", 3.5),
            make_lab_row(admit1 + timedelta(hours=2), "Creatinine", 1.2),
            make_lab_row(admit1 + timedelta(hours=3), "Glucose", 100.0),
            make_lab_row(admit2 + timedelta(hours=1), "Albumin", 3.0),
            make_lab_row(admit2 + timedelta(hours=2), "Sodium", 140.0),
            make_lab_row(admit2 + timedelta(hours=3), "Potassium", 4.0),
        ]

        patient = make_patient(admissions=admissions, lab_rows=lab_rows)
        samples = task(patient)

        admission_ids = set(s["admission_id"] for s in samples)
        self.assertEqual(admission_ids, {"H001", "H002"})

    def test_input_window_boundary(self):
        """Measurements outside the input window should be excluded."""
        task = TDICUMortalityPredictionMIMIC4(input_window_hours=24, min_measurements=1)
        state_time = datetime(2020, 1, 5, 12, 0)

        lab_rows = [
            # This is >24h before the last measurement, should be excluded
            # from that state's context
            make_lab_row(state_time - timedelta(hours=30), "Albumin", 3.5),
            # This is within 24h
            make_lab_row(state_time - timedelta(hours=12), "Creatinine", 1.2),
            make_lab_row(state_time, "Glucose", 100.0),
        ]

        patient = make_patient(lab_rows=lab_rows)
        samples = task(patient)

        # The last state marker's context should not include the first
        # measurement
        last_sample = [s for s in samples][-1]
        _, matrix = last_sample["measurements"]
        # 3 demographics + at most 2 lab measurements (not 3)
        self.assertLessEqual(matrix.shape[0], 5)

    def test_no_demographics_returns_empty(self):
        """Patient with no demographics event returns empty."""
        task = TDICUMortalityPredictionMIMIC4()
        patient = MagicMock()
        patient.patient_id = "P999"
        patient.get_events = MagicMock(return_value=[])
        samples = task(patient)
        self.assertEqual(len(samples), 0)


if __name__ == "__main__":
    unittest.main()
