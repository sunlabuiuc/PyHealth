"""Tests for WESADDataset and StressClassificationWESAD.

Uses synthetic pickle files to test dataset loading, metadata
preparation, and task processing without requiring the real
WESAD dataset.
"""

import os
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from pyhealth.datasets.wesad import WESADDataset, WESAD_SUBJECT_IDS
from pyhealth.tasks.stress_classification_wesad import (
    StressClassificationWESAD,
)


def _create_synthetic_subject(root: str, subject_id: str) -> str:
    """Create a minimal synthetic WESAD pickle file for testing.

    Generates 200 samples of wrist EDA at 4 Hz (50 seconds) and
    700*50 = 35000 chest-rate labels to match the ratio.

    Returns:
        Path to the created pickle file.
    """
    subject_dir = os.path.join(root, subject_id)
    os.makedirs(subject_dir, exist_ok=True)

    n_eda_samples = 200  # 50 seconds at 4 Hz
    n_label_samples = 35000  # ~700 Hz chest rate

    eda = np.random.rand(n_eda_samples, 1).astype(np.float64)

    # First 60% baseline (1), next 30% stress (2), last 10% amusement (3)
    labels = np.ones(n_label_samples, dtype=np.int32)
    stress_start = int(n_label_samples * 0.6)
    amusement_start = int(n_label_samples * 0.9)
    labels[stress_start:amusement_start] = 2
    labels[amusement_start:] = 3

    data = {
        "signal": {
            "wrist": {
                "EDA": eda,
                "BVP": np.random.rand(n_eda_samples * 16, 1),
                "ACC": np.random.rand(n_eda_samples * 8, 3),
                "TEMP": np.random.rand(n_eda_samples, 1),
            },
            "chest": {
                "ECG": np.random.rand(n_label_samples, 1),
                "EDA": np.random.rand(n_label_samples, 1),
                "EMG": np.random.rand(n_label_samples, 1),
                "Temp": np.random.rand(n_label_samples, 1),
                "ACC": np.random.rand(n_label_samples, 3),
                "Resp": np.random.rand(n_label_samples, 1),
            },
        },
        "label": labels,
        "subject": subject_id,
    }

    pkl_path = os.path.join(subject_dir, f"{subject_id}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    return pkl_path


class TestWESADMetadata(unittest.TestCase):
    """Test metadata preparation for WESADDataset."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.subject_ids = ["S2", "S3"]
        for sid in self.subject_ids:
            _create_synthetic_subject(self.temp_dir, sid)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_prepare_metadata_creates_csv(self):
        """Metadata CSV is created with correct columns."""
        dataset = WESADDataset(root=self.temp_dir)
        csv_path = os.path.join(self.temp_dir, "wesad-pyhealth.csv")
        self.assertTrue(os.path.exists(csv_path))

        df = pd.read_csv(csv_path)
        self.assertIn("subject_id", df.columns)
        self.assertIn("signal_file", df.columns)
        self.assertIn("sampling_rate", df.columns)
        self.assertEqual(len(df), len(self.subject_ids))

    def test_prepare_metadata_subject_ids(self):
        """Metadata contains the correct subject IDs."""
        dataset = WESADDataset(root=self.temp_dir)
        csv_path = os.path.join(self.temp_dir, "wesad-pyhealth.csv")
        df = pd.read_csv(csv_path)
        self.assertEqual(
            sorted(df["subject_id"].tolist()),
            sorted(self.subject_ids),
        )

    def test_no_subjects_raises_error(self):
        """FileNotFoundError raised when no valid subjects exist."""
        empty_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(FileNotFoundError):
                WESADDataset(root=empty_dir)
        finally:
            shutil.rmtree(empty_dir)


class TestWESADDataset(unittest.TestCase):
    """Test WESADDataset initialization and patient access."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.subject_ids = ["S2", "S3", "S4"]
        for sid in self.subject_ids:
            _create_synthetic_subject(self.temp_dir, sid)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """Dataset initializes with correct name and root."""
        dataset = WESADDataset(root=self.temp_dir)
        self.assertEqual(dataset.dataset_name, "wesad")
        self.assertEqual(dataset.root, self.temp_dir)

    def test_patient_count(self):
        """All synthetic subjects are loaded."""
        dataset = WESADDataset(root=self.temp_dir)
        self.assertEqual(
            len(dataset.unique_patient_ids),
            len(self.subject_ids),
        )

    def test_get_patient(self):
        """Individual patient records are accessible."""
        dataset = WESADDataset(root=self.temp_dir)
        patient = dataset.get_patient("S2")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "S2")

    def test_get_patient_not_found(self):
        """Accessing a nonexistent patient raises an error."""
        dataset = WESADDataset(root=self.temp_dir)
        with self.assertRaises(AssertionError):
            dataset.get_patient("S99")

    def test_stats(self):
        """stats() runs without error."""
        dataset = WESADDataset(root=self.temp_dir)
        dataset.stats()

    def test_default_task(self):
        """default_task returns StressClassificationWESAD."""
        dataset = WESADDataset(root=self.temp_dir)
        task = dataset.default_task
        self.assertIsInstance(task, StressClassificationWESAD)


class TestStressClassificationWESADTask(unittest.TestCase):
    """Test StressClassificationWESAD task processing."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pkl_path = _create_synthetic_subject(self.temp_dir, "S2")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _make_dummy_patient(self):
        """Create a mock patient with one event pointing to our pickle."""
        event = MagicMock()
        event.signal_file = self.pkl_path

        patient = MagicMock()
        patient.patient_id = "S2"
        patient.get_events.return_value = [event]
        return patient

    def test_task_schema(self):
        """Task has correct schemas and name."""
        task = StressClassificationWESAD()
        self.assertEqual(task.task_name, "StressClassification")
        self.assertEqual(task.input_schema, {"signal": "tensor"})
        self.assertEqual(task.output_schema, {"label": "binary"})

    def test_call_produces_samples(self):
        """Task __call__ produces non-empty sample list."""
        task = StressClassificationWESAD(window_size_sec=10.0)
        patient = self._make_dummy_patient()
        samples = task(patient)
        self.assertIsInstance(samples, list)
        self.assertGreater(len(samples), 0)

    def test_sample_keys(self):
        """Each sample has required keys."""
        task = StressClassificationWESAD(window_size_sec=10.0)
        patient = self._make_dummy_patient()
        samples = task(patient)
        for sample in samples:
            self.assertIn("patient_id", sample)
            self.assertIn("signal", sample)
            self.assertIn("label", sample)

    def test_binary_labels(self):
        """Labels are binary (0 or 1)."""
        task = StressClassificationWESAD(window_size_sec=10.0)
        patient = self._make_dummy_patient()
        samples = task(patient)
        labels = {s["label"] for s in samples}
        self.assertTrue(labels.issubset({0, 1}))

    def test_feature_shape_with_features(self):
        """When use_features=True, signal has shape (4,)."""
        task = StressClassificationWESAD(
            window_size_sec=10.0, use_features=True
        )
        patient = self._make_dummy_patient()
        samples = task(patient)
        for sample in samples:
            self.assertEqual(sample["signal"].shape, (4,))

    def test_raw_signal_shape(self):
        """When use_features=False, signal has shape (window_length,)."""
        task = StressClassificationWESAD(
            window_size_sec=10.0, use_features=False
        )
        patient = self._make_dummy_patient()
        samples = task(patient)
        expected_length = int(10.0 * 4)  # 4 Hz * 10 sec = 40
        for sample in samples:
            self.assertEqual(sample["signal"].shape, (expected_length,))

    def test_different_window_sizes(self):
        """Varying window_size_sec changes sample count."""
        patient = self._make_dummy_patient()

        task_10 = StressClassificationWESAD(window_size_sec=10.0)
        task_25 = StressClassificationWESAD(window_size_sec=25.0)

        samples_10 = task_10(patient)
        samples_25 = task_25(patient)

        self.assertGreater(len(samples_10), len(samples_25))

    def test_patient_id_preserved(self):
        """Patient ID is correctly propagated to samples."""
        task = StressClassificationWESAD(window_size_sec=10.0)
        patient = self._make_dummy_patient()
        samples = task(patient)
        for sample in samples:
            self.assertEqual(sample["patient_id"], "S2")


if __name__ == "__main__":
    unittest.main()
