"""Tests for MIMIC3TLSDataset and InHospitalMortalityTLS task.

Contributors:
    Akshad Pai (NetID: avpai2), Matthew Ruth (NetID: mrruth2)

Paper:
    On the Importance of Step-wise Embeddings for Heterogeneous Clinical
    Time-Series (Kuznetsova et al., JMLR 2023)

Paper link:
    https://jmlr.org/papers/v24/22-0850.html

Description:
    Unit tests for TLS dataset constants/groupings and for IHM TLS task
    ``__call__`` on synthetic Patient events (no real MIMIC or HDF5).

Uses synthetic data only. No real MIMIC-III or HDF5 data required.
"""

import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset
from pyhealth.datasets.mimic3_tls import MIMIC3TLSDataset
from pyhealth.tasks import InHospitalMortalityTLS


def _feature_vector(fill: float) -> list:
    """Length-42 feature vector with a single diagnostic value."""
    return [fill] + [0.0] * (len(MIMIC3TLSDataset.FEATURE_NAMES) - 1)


def _tls_timeseries_row(
    patient_id: str,
    hour_offset: int,
    ihm_label,
    feature_vec: list,
) -> dict:
    """One global-event row matching BaseDataset TLS table layout."""
    base = datetime(2000, 1, 1)
    row = {
        "patient_id": patient_id,
        "event_type": "timeseries",
        "timestamp": base + timedelta(hours=hour_offset),
        "timeseries/stay_id": patient_id,
        "timeseries/ihm_label": ihm_label,
    }
    for i, name in enumerate(MIMIC3TLSDataset.FEATURE_NAMES):
        row[f"timeseries/{name}"] = feature_vec[i]
    return row


def _patient_from_tls_rows(rows: list) -> Patient:
    return Patient(patient_id=rows[0]["patient_id"], data_source=pl.DataFrame(rows))


class TestMIMIC3TLSConstants(unittest.TestCase):
    """Test that dataset class constants are properly defined."""

    def test_feature_names_length(self):
        """FEATURE_NAMES should have exactly 42 entries."""
        self.assertEqual(len(MIMIC3TLSDataset.FEATURE_NAMES), 42)

    def test_feature_names_unique(self):
        """All feature names should be unique."""
        names = MIMIC3TLSDataset.FEATURE_NAMES
        self.assertEqual(len(names), len(set(names)))

    def test_organ_groups_cover_all_features(self):
        """ORGAN_GROUPS should cover all 42 feature indices exactly."""
        all_idx = set()
        for indices in MIMIC3TLSDataset.ORGAN_GROUPS.values():
            all_idx.update(indices)
        self.assertEqual(all_idx, set(range(42)))

    def test_organ_groups_no_overlap(self):
        """ORGAN_GROUPS should have no overlapping indices."""
        seen = set()
        for indices in MIMIC3TLSDataset.ORGAN_GROUPS.values():
            overlap = seen & set(indices)
            self.assertEqual(len(overlap), 0, f"Overlap: {overlap}")
            seen.update(indices)

    def test_type_groups_cover_all_features(self):
        """TYPE_GROUPS should cover all 42 feature indices exactly."""
        all_idx = set()
        for indices in MIMIC3TLSDataset.TYPE_GROUPS.values():
            all_idx.update(indices)
        self.assertEqual(all_idx, set(range(42)))

    def test_type_groups_no_overlap(self):
        """TYPE_GROUPS should have no overlapping indices."""
        seen = set()
        for indices in MIMIC3TLSDataset.TYPE_GROUPS.values():
            overlap = seen & set(indices)
            self.assertEqual(len(overlap), 0, f"Overlap: {overlap}")
            seen.update(indices)

    def test_organ_groups_indices_matches_dict(self):
        """ORGAN_GROUPS_INDICES should match ORGAN_GROUPS values."""
        expected = list(MIMIC3TLSDataset.ORGAN_GROUPS.values())
        self.assertEqual(
            MIMIC3TLSDataset.ORGAN_GROUPS_INDICES, expected
        )

    def test_type_groups_indices_matches_dict(self):
        """TYPE_GROUPS_INDICES should match TYPE_GROUPS values."""
        expected = list(MIMIC3TLSDataset.TYPE_GROUPS.values())
        self.assertEqual(
            MIMIC3TLSDataset.TYPE_GROUPS_INDICES, expected
        )

    def test_organ_group_names(self):
        """Check expected organ group names are present."""
        expected_names = {
            "CNS", "circulation", "hematology",
            "pulmonary", "renal", "other",
        }
        self.assertEqual(
            set(MIMIC3TLSDataset.ORGAN_GROUPS.keys()), expected_names
        )

    def test_type_group_names(self):
        """Check expected type group names are present."""
        expected_names = {"lab", "monitored", "observed", "other"}
        self.assertEqual(
            set(MIMIC3TLSDataset.TYPE_GROUPS.keys()), expected_names
        )


class TestInHospitalMortalityTLSWithSyntheticData(unittest.TestCase):
    """Test IHM task using create_sample_dataset with synthetic data."""

    def setUp(self):
        """Create a synthetic dataset mimicking TLS output format."""
        np.random.seed(42)
        self.samples = [
            {
                "patient_id": f"patient-{i}",
                "time_series": np.random.randn(8, 42).tolist(),
                "ihm": i % 2,
            }
            for i in range(3)
        ]
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"time_series": "tensor"},
            output_schema={"ihm": "binary"},
            dataset_name="test_tls",
        )

    def test_dataset_length(self):
        """Dataset should contain all synthetic samples."""
        self.assertEqual(len(self.dataset), 3)

    def test_sample_keys(self):
        """Each sample should have time_series and ihm keys."""
        sample = self.dataset[0]
        self.assertIn("time_series", sample)
        self.assertIn("ihm", sample)

    def test_time_series_shape(self):
        """time_series should be a small (8, 42) tensor."""
        sample = self.dataset[0]
        ts = sample["time_series"]
        if isinstance(ts, tuple):
            ts = ts[0]
        self.assertEqual(ts.shape, (8, 42))

    def test_ihm_label_values(self):
        """ihm labels should be 0 or 1."""
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            label = sample["ihm"]
            if hasattr(label, "item"):
                label = label.item()
            self.assertIn(label, [0, 1, 0.0, 1.0])

    def test_schema_matches(self):
        """Dataset schemas should match task definition."""
        self.assertEqual(
            self.dataset.input_schema, {"time_series": "tensor"}
        )
        self.assertEqual(
            self.dataset.output_schema, {"ihm": "binary"}
        )

    def test_short_observation_window(self):
        """Samples with shorter time series should work."""
        samples_short = [
            {
                "patient_id": f"p-{i}",
                "time_series": np.random.randn(6, 42).tolist(),
                "ihm": i % 2,
            }
            for i in range(2)
        ]
        ds = create_sample_dataset(
            samples=samples_short,
            input_schema={"time_series": "tensor"},
            output_schema={"ihm": "binary"},
            dataset_name="test_short",
        )
        self.assertEqual(len(ds), 2)
        sample = ds[0]
        ts = sample["time_series"]
        if isinstance(ts, tuple):
            ts = ts[0]
        self.assertEqual(ts.shape[0], 6)
        self.assertEqual(ts.shape[1], 42)

    def test_feature_subset(self):
        """Samples with feature subset should have reduced dimensions."""
        subset = [2, 3, 5, 6, 7, 8, 9, 10]  # monitored features
        samples_sub = [
            {
                "patient_id": f"p-{i}",
                "time_series": np.random.randn(8, len(subset)).tolist(),
                "ihm": i % 2,
            }
            for i in range(2)
        ]
        ds = create_sample_dataset(
            samples=samples_sub,
            input_schema={"time_series": "tensor"},
            output_schema={"ihm": "binary"},
            dataset_name="test_subset",
        )
        sample = ds[0]
        ts = sample["time_series"]
        if isinstance(ts, tuple):
            ts = ts[0]
        self.assertEqual(ts.shape[1], len(subset))


class TestInHospitalMortalityTLSCall(unittest.TestCase):
    """Exercise InHospitalMortalityTLS.__call__ on synthetic Patient events."""

    def test_sorts_events_by_timestamp(self):
        rows = [
            _tls_timeseries_row("p1", 2, 0, _feature_vector(30.0)),
            _tls_timeseries_row("p1", 0, 0, _feature_vector(10.0)),
            _tls_timeseries_row("p1", 1, 0, _feature_vector(20.0)),
        ]
        patient = _patient_from_tls_rows(rows)
        task = InHospitalMortalityTLS(observation_hours=48)
        out = task(patient)
        self.assertEqual(len(out), 1)
        ts = out[0]["time_series"]
        self.assertEqual(ts[0][0], 10.0)
        self.assertEqual(ts[1][0], 20.0)
        self.assertEqual(ts[2][0], 30.0)

    def test_truncates_to_observation_hours(self):
        rows = [
            _tls_timeseries_row("p1", h, 1, _feature_vector(float(h)))
            for h in range(6)
        ]
        patient = _patient_from_tls_rows(rows)
        task = InHospitalMortalityTLS(observation_hours=3)
        out = task(patient)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]["time_series"]), 3)
        for i, row in enumerate(out[0]["time_series"]):
            self.assertEqual(row[0], float(i))

    def test_label_from_first_event_after_sort(self):
        rows = [
            _tls_timeseries_row("p1", 1, 0, _feature_vector(1.0)),
            _tls_timeseries_row("p1", 0, 1, _feature_vector(0.0)),
        ]
        patient = _patient_from_tls_rows(rows)
        out = InHospitalMortalityTLS()(patient)
        self.assertEqual(out[0]["ihm"], 1)

    def test_nan_and_non_numeric_features_become_zero(self):
        vec = _feature_vector(5.0)
        vec[3] = float("nan")
        rows = [
            _tls_timeseries_row("p1", 0, 0, vec),
            _tls_timeseries_row("p1", 1, 0, _feature_vector(2.0)),
        ]
        vec2 = _feature_vector(1.0)
        vec2[5] = "not_a_number"
        rows[1] = _tls_timeseries_row("p1", 1, 0, vec2)
        patient = _patient_from_tls_rows(rows)
        out = InHospitalMortalityTLS()(patient)
        row0 = out[0]["time_series"][0]
        self.assertEqual(row0[3], 0.0)
        row1 = out[0]["time_series"][1]
        self.assertEqual(row1[5], 0.0)

    def test_feature_subset_columns(self):
        vec = [float(i) for i in range(42)]
        rows = [
            _tls_timeseries_row("p1", 0, 0, vec),
            _tls_timeseries_row("p1", 1, 0, vec),
        ]
        patient = _patient_from_tls_rows(rows)
        subset = [0, 2, 4]
        task = InHospitalMortalityTLS(feature_subset=subset)
        out = task(patient)
        self.assertEqual(len(out[0]["time_series"][0]), 3)
        self.assertEqual(out[0]["time_series"][0], [0.0, 2.0, 4.0])

    def test_returns_empty_for_no_timeseries_events(self):
        base = datetime(2000, 1, 1)
        row = {
            "patient_id": "p1",
            "event_type": "other",
            "timestamp": base,
            "timeseries/stay_id": "p1",
            "timeseries/ihm_label": 0,
            **{
                f"timeseries/{n}": 0.0
                for n in MIMIC3TLSDataset.FEATURE_NAMES
            },
        }
        patient = Patient(
            patient_id="p1", data_source=pl.DataFrame([row])
        )
        self.assertEqual(InHospitalMortalityTLS()(patient), [])

    def test_returns_empty_for_single_timestep(self):
        rows = [_tls_timeseries_row("p1", 0, 0, _feature_vector(1.0))]
        patient = _patient_from_tls_rows(rows)
        self.assertEqual(InHospitalMortalityTLS()(patient), [])

    def test_returns_empty_for_invalid_ihm_label(self):
        rows = [
            _tls_timeseries_row("p1", 0, 2, _feature_vector(0.0)),
            _tls_timeseries_row("p1", 1, 2, _feature_vector(0.0)),
        ]
        patient = _patient_from_tls_rows(rows)
        self.assertEqual(InHospitalMortalityTLS()(patient), [])

    def test_returns_empty_for_non_parseable_ihm(self):
        rows = [
            _tls_timeseries_row("p1", 0, "no", _feature_vector(0.0)),
            _tls_timeseries_row("p1", 1, "no", _feature_vector(0.0)),
        ]
        patient = _patient_from_tls_rows(rows)
        self.assertEqual(InHospitalMortalityTLS()(patient), [])


if __name__ == "__main__":
    unittest.main()
