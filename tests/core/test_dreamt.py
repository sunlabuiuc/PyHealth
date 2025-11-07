import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from pyhealth.datasets import DREAMTDataset

class TestDREAMTDatasetNewerVersions(unittest.TestCase):
    """Test DREAMT dataset containing 64Hz and 100Hz folders with local test data."""
    
    def setUp(self):
        """Set up participant info csv and 64Hz 100Hz files"""
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

        (self.root / "data_64Hz").mkdir()
        (self.root / "data_100Hz").mkdir()

        self.num_patients = 5
        patient_data = {
            'SID': [f"S{i:03d}" for i in range(1, self.num_patients + 1)],
            'AGE': np.random.uniform(25, 65, self.num_patients),
            'GENDER': np.random.choice(['M', 'F'], self.num_patients),
            'BMI': np.random.randint(20, 50, self.num_patients),
            'OAHI': np.random.randint(0, 50, self.num_patients),
            'AHI': np.random.randint(0, 50, self.num_patients),
            'Mean_SaO2': [f"{val}%" for val in np.random.randint(85, 99, self.num_patients)],
            'Arousal Index': np.random.randint(1, 100, self.num_patients),
            'MEDICAL_HISTORY': ['Medical History'] * self.num_patients,
            'Sleep_Disorders': ['Sleep Disorder'] * self.num_patients,
        }

        patient_data_df = pd.DataFrame(patient_data)
        patient_data_df.to_csv(self.root / "participant_info.csv", index=False)
        self._create_files()

    def _create_files(self):
        """Create 64Hz and 100Hz files"""
        for i in range(1, self.num_patients + 1):
            sid = f"S{i:03d}"
           
            partial_data = {
            'TIMESTAMP': [np.random.uniform(0, 100)],
            'BVP': [np.random.uniform(1, 10)],
            'HR': [np.random.randint(15, 100)],
            'EDA': [np.random.uniform(0, 1)],
            'TEMP': [np.random.uniform(20, 30)],
            'ACC_X': [np.random.uniform(1, 50)],
            'ACC_Y': [np.random.uniform(1, 50)],
            'ACC_Z': [np.random.uniform(1, 50)],
            'IBI': [np.random.uniform(0.6, 1.2)],
            'Sleep_Stage': [np.random.choice(['W', 'N1', 'N2', 'N3', 'R'])],
            }

            pd.DataFrame(partial_data).to_csv(self.root / "data_64Hz" / f"{sid}_whole_df.csv")
            pd.DataFrame(partial_data).to_csv(self.root / "data_100Hz" / f"{sid}_PSG_df.csv")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        """Test DREAMTDataset initialization"""
        dataset = DREAMTDataset(root=str(self.root))
        
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "dreamt_sleep")
        self.assertEqual(dataset.root, str(self.root))

    def test_metadata_file_created(self):
        """Test dreamt-metadata.csv created"""
        dataset = DREAMTDataset(root=str(self.root))
        metadata_file = self.root / "dreamt-metadata.csv"
        self.assertTrue(metadata_file.exists())

    def test_patient_count(self):
        """Test all patients are added"""
        dataset = DREAMTDataset(root=str(self.root))
        self.assertEqual(len(dataset.unique_patient_ids), self.num_patients)
    
    def test_stats_method(self):
        """Test stats method"""
        dataset = DREAMTDataset(root=str(self.root))
        dataset.stats()

    def test_get_patient(self):
        """Test get_patient method"""
        dataset = DREAMTDataset(root=str(self.root))
        patient = dataset.get_patient('S001')
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, 'S001')
    
    def test_get_patient_not_found(self):
        """Test that patient not found throws error."""
        dataset = DREAMTDataset(root=str(self.root))
        with self.assertRaises(AssertionError):
            dataset.get_patient('S222')

if __name__ == "__main__":
    unittest.main()