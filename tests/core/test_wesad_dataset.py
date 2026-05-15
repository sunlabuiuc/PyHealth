"""Tests for the WESAD dataset class using synthetic data."""

import os
import pickle
import tempfile
import unittest

import numpy as np

from pyhealth.datasets.wesad import WESADDataset, EDA_SAMPLE_RATE


def _make_synthetic_subject(subject_id: str, root: str, n_seconds: int = 300) -> None:
    """Creates a synthetic WESAD pickle file for testing."""
    n_eda = n_seconds * EDA_SAMPLE_RATE          # 4 Hz
    n_chest = n_eda * 175                         # ~700 Hz chest device

    eda = np.random.rand(n_eda, 1).astype(np.float32)

    # Labels: first third baseline (1), second third stress (2), rest amusement (3)
    labels = np.ones(n_chest, dtype=int)
    labels[n_chest // 3: 2 * n_chest // 3] = 2
    labels[2 * n_chest // 3:] = 3

    data = {
        "signal": {"wrist": {"EDA": eda}},
        "label": labels,
    }

    subject_dir = os.path.join(root, subject_id)
    os.makedirs(subject_dir, exist_ok=True)
    with open(os.path.join(subject_dir, f"{subject_id}.pkl"), "wb") as f:
        pickle.dump(data, f)


class TestWESADDataset(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.subjects = ["S2", "S3"]
        for sid in self.subjects:
            _make_synthetic_subject(sid, self.tmp_dir)

    def test_loads_without_error(self):
        dataset = WESADDataset(
            root=self.tmp_dir,
            subjects=self.subjects,
            window_size=60,
            step_size=10,
        )
        self.assertGreater(len(dataset), 0)

    def test_sample_shape(self):
        dataset = WESADDataset(
            root=self.tmp_dir,
            subjects=self.subjects,
            window_size=60,
            step_size=10,
        )
        sample = dataset[0]
        self.assertEqual(sample["eda"].shape, (60,))

    def test_binary_labels_only(self):
        dataset = WESADDataset(
            root=self.tmp_dir,
            subjects=self.subjects,
            label_map={1: 0, 2: 1},
        )
        labels = {s["label"] for s in dataset.samples}
        self.assertTrue(labels.issubset({0, 1}))

    def test_three_class_label_map(self):
        dataset = WESADDataset(
            root=self.tmp_dir,
            subjects=self.subjects,
            label_map={1: 0, 2: 1, 3: 2},
        )
        labels = {s["label"] for s in dataset.samples}
        self.assertTrue(labels.issubset({0, 1, 2}))

    def test_missing_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            WESADDataset(root="/nonexistent/path", subjects=self.subjects)

    def test_subject_id_in_sample(self):
        dataset = WESADDataset(
            root=self.tmp_dir,
            subjects=self.subjects,
        )
        self.assertIn(dataset[0]["subject_id"], self.subjects)

    def test_window_size_respected(self):
        for window_size in [30, 60, 120]:
            dataset = WESADDataset(
                root=self.tmp_dir,
                subjects=self.subjects,
                window_size=window_size,
            )
            self.assertEqual(dataset[0]["eda"].shape[0], window_size)


if __name__ == "__main__":
    unittest.main()