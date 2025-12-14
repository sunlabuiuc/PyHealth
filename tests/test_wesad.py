"""Unit tests for WESADDataset and stress detection tasks.

Run with: pytest test_wesad.py -v

These tests create synthetic WESAD data to verify:
1. Dataset loading works correctly
2. Patient data structure is correct
3. Task functions create valid samples
4. Caching works properly
"""

import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np


def create_synthetic_wesad_data(root_dir: str, num_subjects: int = 3) -> None:
    """Create synthetic WESAD data for testing.
    
    Args:
        root_dir: Directory to create test data in.
        num_subjects: Number of subjects to create.
    """
    duration_seconds = 180  # 3 minutes

    for i, sid in enumerate([2, 3, 4][:num_subjects]):
        subject_folder = os.path.join(root_dir, f"S{sid}")
        os.makedirs(subject_folder, exist_ok=True)

        data = {
            "subject": f"S{sid}",
            "signal": {
                "wrist": {
                    "ACC": np.random.randn(duration_seconds * 32, 3).astype(np.float32),
                    "BVP": np.random.randn(duration_seconds * 64, 1).astype(np.float32),
                    "EDA": np.abs(np.random.randn(duration_seconds * 4, 1)).astype(np.float32),
                    "TEMP": 32 + np.random.randn(duration_seconds * 4, 1).astype(np.float32),
                },
                "chest": {},
            },
            "label": np.concatenate([
                np.ones(60 * 700, dtype=np.int32),      # baseline
                np.full(60 * 700, 2, dtype=np.int32),  # stress
                np.full(60 * 700, 3, dtype=np.int32),  # amusement
            ]),
        }

        with open(os.path.join(subject_folder, f"S{sid}.pkl"), "wb") as f:
            pickle.dump(data, f)


class TestWESADDataset(unittest.TestCase):
    """Test cases for WESADDataset."""

    @classmethod
    def setUpClass(cls):
        """Create test data directory."""
        cls.test_dir = tempfile.mkdtemp()
        create_synthetic_wesad_data(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        shutil.rmtree(cls.test_dir)

    def test_dataset_loading(self):
        """Test that dataset loads correctly."""
        from pyhealth.datasets.wesad import WESADDataset

        dataset = WESADDataset(root=self.test_dir, refresh_cache=True)
        
        self.assertEqual(len(dataset), 3)
        self.assertIn("S2", dataset.patients)
        self.assertIn("S3", dataset.patients)
        self.assertIn("S4", dataset.patients)

    def test_patient_data_structure(self):
        """Test patient data has correct structure."""
        from pyhealth.datasets.wesad import WESADDataset

        dataset = WESADDataset(root=self.test_dir, refresh_cache=True)
        patient = dataset.get_patient("S2")

        # Check keys exist
        self.assertIn("patient_id", patient)
        self.assertIn("signal", patient)
        self.assertIn("label", patient)

        # Check wrist signals exist
        wrist = patient["signal"]["wrist"]
        self.assertIn("ACC", wrist)
        self.assertIn("BVP", wrist)
        self.assertIn("EDA", wrist)
        self.assertIn("TEMP", wrist)

        # Check shapes
        self.assertEqual(wrist["ACC"].shape[1], 3)
        self.assertEqual(wrist["BVP"].shape[1], 1)
        self.assertEqual(wrist["EDA"].shape[1], 1)
        self.assertEqual(wrist["TEMP"].shape[1], 1)

    def test_dev_mode(self):
        """Test dev mode loads fewer subjects."""
        from pyhealth.datasets.wesad import WESADDataset

        dataset = WESADDataset(root=self.test_dir, dev=True, refresh_cache=True)
        self.assertLessEqual(len(dataset), 3)

    def test_get_patient_error(self):
        """Test get_patient raises error for invalid ID."""
        from pyhealth.datasets.wesad import WESADDataset

        dataset = WESADDataset(root=self.test_dir, refresh_cache=True)
        
        with self.assertRaises(KeyError):
            dataset.get_patient("S999")

    def test_iteration(self):
        """Test iteration over dataset."""
        from pyhealth.datasets.wesad import WESADDataset

        dataset = WESADDataset(root=self.test_dir, refresh_cache=True)
        patients = list(dataset)
        
        self.assertEqual(len(patients), 3)

    def test_caching(self):
        """Test caching works correctly."""
        from pyhealth.datasets.wesad import WESADDataset

        # First load
        dataset1 = WESADDataset(root=self.test_dir, refresh_cache=True)
        
        # Second load from cache
        dataset2 = WESADDataset(root=self.test_dir, refresh_cache=False)
        
        self.assertEqual(len(dataset1), len(dataset2))


class TestStressDetectionTask(unittest.TestCase):
    """Test cases for stress detection task functions."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()
        create_synthetic_wesad_data(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_binary_task(self):
        """Test binary stress detection task."""
        from pyhealth.datasets.wesad import WESADDataset
        from pyhealth.tasks.stress_detection_wesad import stress_detection_wesad_binary_fn

        dataset = WESADDataset(root=self.test_dir, refresh_cache=True)
        dataset = dataset.set_task(stress_detection_wesad_binary_fn)

        self.assertIsNotNone(dataset.samples)
        self.assertGreater(len(dataset.samples), 0)

        # Check sample structure
        sample = dataset.samples[0]
        self.assertIn("patient_id", sample)
        self.assertIn("record_id", sample)
        self.assertIn("eda", sample)
        self.assertIn("label", sample)

        # Check label is binary
        self.assertIn(sample["label"], [0, 1])

    def test_multiclass_task(self):
        """Test multiclass stress detection task."""
        from pyhealth.datasets.wesad import WESADDataset
        from pyhealth.tasks.stress_detection_wesad import stress_detection_wesad_multiclass_fn

        dataset = WESADDataset(root=self.test_dir, refresh_cache=True)
        dataset = dataset.set_task(stress_detection_wesad_multiclass_fn)

        self.assertGreater(len(dataset.samples), 0)

        # Check label is in valid range
        sample = dataset.samples[0]
        self.assertIn(sample["label"], [0, 1, 2])

    def test_eda_shape(self):
        """Test EDA signal has correct shape (60s * 4Hz = 240)."""
        from pyhealth.datasets.wesad import WESADDataset
        from pyhealth.tasks.stress_detection_wesad import stress_detection_wesad_binary_fn

        dataset = WESADDataset(root=self.test_dir, refresh_cache=True)
        dataset = dataset.set_task(stress_detection_wesad_binary_fn)

        sample = dataset.samples[0]
        self.assertEqual(sample["eda"].shape, (240,))


if __name__ == "__main__":
    unittest.main()
