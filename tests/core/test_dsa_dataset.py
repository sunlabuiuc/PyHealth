"""Tests for the DSA dataset using a tiny synthetic directory tree."""

import os
import shutil
import tempfile
import unittest

import numpy as np

from pyhealth.datasets.dsa import (
    DSADataset,
    build_dsa_metadata,
    compute_channel_ranges,
    load_dsa_segment,
    split_sensors,
)

SUBJECTS = (1, 2)
ACTIVITIES = (1, 2)
SEGMENTS = (1, 2)


def _write_synthetic_dsa(root: str) -> None:
    """Writes 2 subjects x 2 activities x 2 segments of random data."""
    rng = np.random.default_rng(0)
    for act in ACTIVITIES:
        for sub in SUBJECTS:
            folder = os.path.join(root, f"a{act:02d}", f"p{sub}")
            os.makedirs(folder)
            for seg in SEGMENTS:
                data = rng.normal(size=(125, 45))
                np.savetxt(
                    os.path.join(folder, f"s{seg:02d}.txt"), data, delimiter=","
                )


class TestDSAHelpers(unittest.TestCase):
    """Tests for the metadata builder and array helpers."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        _write_synthetic_dsa(cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    def test_metadata_rows_and_columns(self):
        meta = build_dsa_metadata(self.root)
        self.assertEqual(len(meta), 8)
        self.assertEqual(
            list(meta.columns), ["subject", "activity", "segment", "signal_file"]
        )
        self.assertEqual(set(meta["subject"]), {"p1", "p2"})
        self.assertEqual(set(meta["activity"]), {1, 2})
        self.assertTrue(all(os.path.isfile(p) for p in meta["signal_file"]))

    def test_load_segment_shape_and_dtype(self):
        meta = build_dsa_metadata(self.root)
        segment = load_dsa_segment(meta["signal_file"].iloc[0])
        self.assertEqual(segment.shape, (125, 45))
        self.assertEqual(segment.dtype, np.float32)

    def test_load_segment_rejects_bad_shape(self):
        bad = os.path.join(self.root, "bad.txt")
        np.savetxt(bad, np.zeros((10, 45)), delimiter=",")
        with self.assertRaises(ValueError):
            load_dsa_segment(bad)

    def test_split_sensors_keeps_pairing(self):
        segment = np.arange(125 * 45, dtype=np.float32).reshape(125, 45)
        sensors = split_sensors(segment)
        self.assertEqual(sensors.shape, (5, 125, 9))
        # Sensor k, time t, channel c comes from column 9k + c of row t.
        np.testing.assert_array_equal(sensors[3, 7, 4], segment[7, 3 * 9 + 4])
        # Every sensor view covers the same 125 time steps.
        np.testing.assert_array_equal(sensors[0, :, 0], segment[:, 0])
        np.testing.assert_array_equal(sensors[4, :, 8], segment[:, 44])

    def test_channel_ranges_cover_all_segments(self):
        meta = build_dsa_metadata(self.root)
        mins, maxs = compute_channel_ranges(meta["signal_file"])
        self.assertEqual(mins.shape, (5, 9))
        self.assertTrue(np.all(maxs > mins))
        segment = load_dsa_segment(meta["signal_file"].iloc[0])
        sensors = split_sensors(segment)
        self.assertTrue(np.all(sensors.min(axis=1) >= mins))
        self.assertTrue(np.all(sensors.max(axis=1) <= maxs))

    def test_channel_ranges_empty_raises(self):
        with self.assertRaises(ValueError):
            compute_channel_ranges([])

    def test_missing_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            build_dsa_metadata(os.path.join(self.root, "nope"))

    def test_empty_root_raises(self):
        empty = tempfile.mkdtemp()
        try:
            with self.assertRaises(ValueError):
                build_dsa_metadata(empty)
        finally:
            shutil.rmtree(empty)


class TestDSADataset(unittest.TestCase):
    """Tests for DSADataset itself."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.cache_dir = tempfile.mkdtemp()
        _write_synthetic_dsa(cls.root)
        cls.dataset = DSADataset(root=cls.root, cache_dir=cls.cache_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)
        shutil.rmtree(cls.cache_dir, ignore_errors=True)

    def test_metadata_csv_written(self):
        self.assertTrue(os.path.isfile(os.path.join(self.root, "dsa-pyhealth.csv")))

    def test_patients(self):
        self.assertEqual(sorted(self.dataset.unique_patient_ids), ["p1", "p2"])

    def test_events_per_patient(self):
        patient = self.dataset.get_patient("p1")
        events = patient.get_events(event_type="dsa")
        self.assertEqual(len(events), 4)


if __name__ == "__main__":
    unittest.main()
