import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from pyhealth.datasets import BMDHSDataset


class TestBMDHSDataset(unittest.TestCase):
    """Test BMD-HS dataset with synthetic test data."""

    def setUp(self):
        """Set up test data files and directory structure"""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

        # Create train directory
        (self.root / "train").mkdir()

        self.num_patients = 5

        # Create train.csv with labels and recording filenames
        train_data = {
            'patient_id': [f"P{i:03d}" for i in range(1, self.num_patients + 1)],
            'AS': [1, 0, 1, 0, 0],  # Aortic Stenosis
            'AR': [0, 1, 0, 1, 0],  # Aortic Regurgitation
            'MR': [1, 1, 0, 0, 0],  # Mitral Regurgitation
            'MS': [0, 0, 1, 0, 0],  # Mitral Stenosis
            'N': [0, 0, 0, 0, 1],   # Normal
        }

        # Add recording filenames
        for i in range(1, 9):
            train_data[f'recording_{i}'] = [
                f"P{j:03d}_{i}.wav" for j in range(1, self.num_patients + 1)
            ]

        train_df = pd.DataFrame(train_data)
        train_df.to_csv(self.root / "train.csv", index=False)

        # Create additional_metadata.csv
        metadata_data = {
            'patient_id': [f"P{i:03d}" for i in range(1, self.num_patients + 1)],
            'Age': [45, 52, 38, 61, 29],
            'Gender': ['M', 'F', 'M', 'F', 'M'],
            'Smoker': [1, 0, 1, 0, 0],
            'Lives': ['U', 'R', 'R', 'R', 'U'],  # U=Urban, R=Rural
        }
        metadata_df = pd.DataFrame(metadata_data)
        metadata_df.to_csv(self.root / "additional_metadata.csv", index=False)

    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """Test BMDHSDataset initialization"""
        dataset = BMDHSDataset(root=str(self.root))

        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "bmd_hs")
        self.assertEqual(dataset.root, str(self.root))

    def test_patient_count(self):
        """Test all patients are loaded"""
        dataset = BMDHSDataset(root=str(self.root))
        self.assertEqual(len(dataset.unique_patient_ids), self.num_patients)

    def test_stats_method(self):
        """Test stats method runs without error"""
        dataset = BMDHSDataset(root=str(self.root))
        # Just test that it runs without error
        dataset.stats()

    def test_get_patient(self):
        """Test get_patient method"""
        dataset = BMDHSDataset(root=str(self.root))
        patient = dataset.get_patient('P001')
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, 'P001')

    def test_get_patient_not_found(self):
        """Test that patient not found throws error."""
        dataset = BMDHSDataset(root=str(self.root))
        with self.assertRaises(AssertionError):
            dataset.get_patient('P999')

    def test_available_tables(self):
        """Test available tables"""
        dataset = BMDHSDataset(root=str(self.root))
        expected_tables = ["diagnoses", "recordings", "metadata"]
        self.assertEqual(sorted(dataset.tables), sorted(expected_tables))


if __name__ == "__main__":
    unittest.main()