"""Tests for the stress detection task using synthetic data."""

import os
import pickle
import tempfile
import unittest

import numpy as np
import torch

from pyhealth.datasets.wesad import WESADDataset, EDA_SAMPLE_RATE
from pyhealth.tasks.stress_detection import StressDetectionDataset


def _make_synthetic_subject(subject_id: str, root: str, n_seconds: int = 300) -> None:
    n_eda = n_seconds * EDA_SAMPLE_RATE
    n_chest = n_eda * 175
    eda = np.random.rand(n_eda, 1).astype(np.float32)
    labels = np.ones(n_chest, dtype=int)
    labels[n_chest // 3: 2 * n_chest // 3] = 2
    labels[2 * n_chest // 3:] = 3
    data = {"signal": {"wrist": {"EDA": eda}}, "label": labels}
    subject_dir = os.path.join(root, subject_id)
    os.makedirs(subject_dir, exist_ok=True)
    with open(os.path.join(subject_dir, f"{subject_id}.pkl"), "wb") as f:
        pickle.dump(data, f)


class TestStressDetectionDataset(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.subjects = ["S2", "S3", "S4"]
        for sid in self.subjects:
            _make_synthetic_subject(sid, self.tmp_dir)
        raw = WESADDataset(
            root=self.tmp_dir,
            subjects=self.subjects,
            window_size=60,
            step_size=10,
            label_map={1: 0, 2: 1},
        )
        self.task = StressDetectionDataset(raw.samples)

    def test_len(self):
        self.assertGreater(len(self.task), 0)

    def test_getitem_types(self):
        eda, label = self.task[0]
        self.assertIsInstance(eda, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)

    def test_eda_shape(self):
        eda, _ = self.task[0]
        self.assertEqual(eda.shape, (60,))

    def test_label_dtype(self):
        _, label = self.task[0]
        self.assertEqual(label.dtype, torch.long)

    def test_subject_filter(self):
        filtered = StressDetectionDataset(
            self.task.samples, subject_filter=["S2"]
        )
        subjects_in_filtered = {s["subject_id"] for s in filtered.samples}
        self.assertEqual(subjects_in_filtered, {"S2"})

    def test_lnso_split(self):
        train_ds, test_ds = self.task.get_subject_splits(test_subjects=["S2"])
        train_subjects = {s["subject_id"] for s in train_ds.samples}
        test_subjects = {s["subject_id"] for s in test_ds.samples}
        self.assertNotIn("S2", train_subjects)
        self.assertEqual(test_subjects, {"S2"})
        self.assertEqual(len(train_ds) + len(test_ds), len(self.task))

    def test_num_classes_binary(self):
        self.assertEqual(self.task.num_classes, 2)


if __name__ == "__main__":
    unittest.main()