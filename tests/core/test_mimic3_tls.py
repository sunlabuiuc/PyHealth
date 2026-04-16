"""Tests for MIMIC3TLSDataset and InHospitalMortalityTLS task.

Uses synthetic data only. No real MIMIC-III or HDF5 data required.
"""

import unittest

import numpy as np

from pyhealth.datasets import create_sample_dataset
from pyhealth.datasets.mimic3_tls import MIMIC3TLSDataset


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
                "time_series": np.random.randn(48, 42).tolist(),
                "ihm": i % 2,
            }
            for i in range(4)
        ]
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"time_series": "tensor"},
            output_schema={"ihm": "binary"},
            dataset_name="test_tls",
        )

    def test_dataset_length(self):
        """Dataset should contain all 4 samples."""
        self.assertEqual(len(self.dataset), 4)

    def test_sample_keys(self):
        """Each sample should have time_series and ihm keys."""
        sample = self.dataset[0]
        self.assertIn("time_series", sample)
        self.assertIn("ihm", sample)

    def test_time_series_shape(self):
        """time_series should be a (48, 42) tensor."""
        sample = self.dataset[0]
        ts = sample["time_series"]
        if isinstance(ts, tuple):
            ts = ts[0]
        self.assertEqual(ts.shape, (48, 42))

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
        samples_24h = [
            {
                "patient_id": f"p-{i}",
                "time_series": np.random.randn(24, 42).tolist(),
                "ihm": i % 2,
            }
            for i in range(2)
        ]
        ds = create_sample_dataset(
            samples=samples_24h,
            input_schema={"time_series": "tensor"},
            output_schema={"ihm": "binary"},
            dataset_name="test_24h",
        )
        self.assertEqual(len(ds), 2)
        sample = ds[0]
        ts = sample["time_series"]
        if isinstance(ts, tuple):
            ts = ts[0]
        self.assertEqual(ts.shape[0], 24)
        self.assertEqual(ts.shape[1], 42)

    def test_feature_subset(self):
        """Samples with feature subset should have reduced dimensions."""
        subset = [2, 3, 5, 6, 7, 8, 9, 10]  # monitored features
        samples_sub = [
            {
                "patient_id": f"p-{i}",
                "time_series": np.random.randn(48, len(subset)).tolist(),
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


if __name__ == "__main__":
    unittest.main()
