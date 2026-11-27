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
        self.rng = np.random.default_rng(42)
        patient_data = {
            "SID": [f"S{i:03d}" for i in range(1, self.num_patients + 1)],
            "AGE": self.rng.uniform(25, 65, self.num_patients),
            "GENDER": self.rng.choice(["M", "F"], self.num_patients),
            "BMI": self.rng.integers(20, 50, self.num_patients),
            "OAHI": self.rng.integers(0, 50, self.num_patients),
            "AHI": self.rng.integers(0, 50, self.num_patients),
            "Mean_SaO2": [
                f"{val}%" for val in self.rng.integers(85, 99, self.num_patients)
            ],
            "Arousal Index": self.rng.integers(1, 100, self.num_patients),
            "MEDICAL_HISTORY": ["Medical History"] * self.num_patients,
            "Sleep_Disorders": ["Sleep Disorder"] * self.num_patients,
        }

        patient_data_df = pd.DataFrame(patient_data)
        patient_data_df.to_csv(self.root / "participant_info.csv", index=False)
        self._create_files()

    def _create_files(self):
        """Create 64Hz and 100Hz files"""
        for i in range(1, self.num_patients + 1):
            sid = f"S{i:03d}"

            partial_data = {
                "TIMESTAMP": [self.rng.uniform(0, 100)],
                "BVP": [self.rng.uniform(1, 10)],
                "HR": [self.rng.integers(15, 100)],
                "EDA": [self.rng.uniform(0, 1)],
                "TEMP": [self.rng.uniform(20, 30)],
                "ACC_X": [self.rng.uniform(1, 50)],
                "ACC_Y": [self.rng.uniform(1, 50)],
                "ACC_Z": [self.rng.uniform(1, 50)],
                "IBI": [self.rng.uniform(0.6, 1.2)],
                "Sleep_Stage": [self.rng.choice(["W", "N1", "N2", "N3", "R"])],
            }

            pd.DataFrame(partial_data).to_csv(
                self.root / "data_64Hz" / f"{sid}_whole_df.csv"
            )
            pd.DataFrame(partial_data).to_csv(
                self.root / "data_100Hz" / f"{sid}_PSG_df.csv"
            )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        fixture = cls(methodName="test_patient_count")
        fixture.setUp()
        cls.addClassCleanup(fixture.tearDown)
        cls.shared_dataset = DREAMTDataset(
            root=str(fixture.root),
            cache_dir=str(fixture.root / "shared_cache"),
        )
        cls.shared_num_patients = fixture.num_patients

    def test_dataset_initialization(self):
        """Test DREAMTDataset initialization"""
        dataset = DREAMTDataset(root=str(self.root), cache_dir=str(self.root / "cache"))

        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "dreamt_sleep")
        self.assertEqual(dataset.root, str(self.root))

    def test_metadata_file_created(self):
        """Test dreamt-metadata.csv created"""
        DREAMTDataset(root=str(self.root), cache_dir=str(self.root / "cache"))
        metadata_file = self.root / "dreamt-metadata.csv"
        self.assertTrue(metadata_file.exists())

    def test_patient_count(self):
        """Test all patients are added"""
        dataset = self.shared_dataset
        self.assertEqual(len(dataset.unique_patient_ids), self.num_patients)

    def test_stats_method(self):
        """Test stats method"""
        dataset = self.shared_dataset
        dataset.stats()

    def test_get_patient(self):
        """Test get_patient method"""
        dataset = self.shared_dataset
        patient = dataset.get_patient("S001")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "S001")

    def test_get_patient_not_found(self):
        """Test that patient not found throws error."""
        dataset = self.shared_dataset
        with self.assertRaises(AssertionError):
            dataset.get_patient("S222")

    def test_custom_cache_dir(self):
        """Test that a caller-provided cache directory is used."""
        cache_root = self.root / "controlled_cache"
        dataset = DREAMTDataset(root=str(self.root), cache_dir=str(cache_root))
        self.assertTrue(str(dataset.cache_dir).startswith(str(cache_root)))


if __name__ == "__main__":
    unittest.main()
