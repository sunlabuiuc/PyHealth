"""
Authors: Asmita Chihnara (asmitac2) and Tithi Sreemany (tithis2)

Test Suite: ICU Mortality Prediction Dataset Tests
Description:
    This test suite verifies the functionality of the ICUMortalityDataset class,
    which implements the PhysioNet Challenge 2012 dataset for ICU mortality prediction.
    The tests cover:
    - Dataset initialization and configuration
    - Patient data loading and processing
    - Statistics calculation
    - Development mode functionality
    - Empty dataset handling
    - Global event DataFrame creation
"""

import os
import unittest
import tempfile
import shutil
import pandas as pd
from pathlib import Path
import logging

from pyhealth.datasets import ICUMortalityDataset

logger = logging.getLogger(__name__)


class TestICUMortalityDataset(unittest.TestCase):
    """Test suite for the ICUMortalityDataset class.

    This test suite verifies the functionality of the ICUMortalityDataset class,
    including:
        - Dataset initialization
        - Patient data loading
        - Statistics calculation
        - Dev mode functionality
        - Empty dataset handling
        - Global event DataFrame creation

    The tests use the PhysioNet Challenge 2012 dataset, specifically the set-a
    portion of the data. The dataset is loaded only once in setUpClass to improve
    test performance.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test environment.

        This method:
        1. Verifies the dataset directory exists
        2. Loads the dataset in both normal and dev modes
        3. Makes the dataset instances available to all test methods
        """
        cls.dataset_root = "temp_data/challenge-2012"
        if not os.path.exists(cls.dataset_root):
            raise FileNotFoundError(f"Dataset directory not found: {cls.dataset_root}")

        # Load dataset once for all tests
        cls.dataset = ICUMortalityDataset(root=cls.dataset_root)
        cls.dev_dataset = ICUMortalityDataset(root=cls.dataset_root, dev=True)

    def setUp(self):
        """No need to create dataset instance for each test anymore."""
        pass

    def test_dataset_initialization(self):
        """Test if the dataset is initialized correctly.

        This test verifies:
            - The dataset instance is created
            - The root directory is set correctly
            - The dataset name is set correctly
        """
        self.assertIsNotNone(self.dataset)
        self.assertEqual(self.dataset.root, self.dataset_root)
        self.assertEqual(self.dataset.dataset_name, "icumortality")

    def test_patient_loading(self):
        """Test if patient data is loaded correctly.

        This test verifies:
            - Patients are loaded into the dataset
            - Patient data contains required fields (age, gender, mortality, measurements)
            - Measurements are loaded for each patient
        """
        # Check if we have patients loaded
        self.assertGreater(len(self.dataset.patients), 0)

        # Check if patient data is loaded correctly for the first patient
        first_patient_id = next(iter(self.dataset.patients))
        patient = self.dataset.patients[first_patient_id]

        # Check required fields
        self.assertIn("age", patient)
        self.assertIn("gender", patient)
        self.assertIn("mortality", patient)
        self.assertIn("measurements", patient)

        # Check measurements
        measurements = patient["measurements"]
        self.assertGreater(len(measurements), 0)

    def test_statistics(self):
        """Test if dataset statistics are calculated correctly.

        This test verifies:
            - Statistics are calculated for non-empty datasets
            - All required statistics are present
            - Statistics are non-negative
        """
        stats = self.dataset.stat()

        # Check that statistics are calculated
        self.assertGreater(stats["# patients"], 0)
        self.assertGreater(stats["# measurements"], 0)
        self.assertIn("mortality rate", stats)

    def test_dev_mode(self):
        """Test if dev mode works correctly.

        This test verifies:
            - Dev mode limits the dataset to 100 patients
            - Statistics are calculated correctly in dev mode
        """
        stats = self.dev_dataset.stat()
        self.assertEqual(
            stats["# patients"], 100
        )  # Dev mode should limit to 100 patients

    def test_empty_dataset(self):
        """Test handling of empty dataset.

        This test:
        1. Creates a temporary directory with empty set-a and outcomes files
        2. Initializes a dataset with the empty directory
        3. Verifies that statistics are zero for an empty dataset
        4. Cleans up the temporary directory
        """
        empty_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(empty_dir, "set-a"))
        pd.DataFrame(columns=["RecordID", "In-hospital_death"]).to_csv(
            os.path.join(empty_dir, "Outcomes-a.txt"), index=False
        )

        empty_dataset = ICUMortalityDataset(root=empty_dir)
        stats = empty_dataset.stat()

        self.assertEqual(stats["# patients"], 0)
        self.assertEqual(stats["# measurements"], 0)
        self.assertEqual(stats["# deaths"], 0)
        self.assertEqual(stats["mortality rate"], "0.00%")

        shutil.rmtree(empty_dir)

    def test_global_event_df(self):
        """Test if the global event DataFrame is created correctly.

        This test verifies:
            - The DataFrame is created with the correct columns
            - The DataFrame contains measurements
            - The event type is set correctly
            - The DataFrame has the expected shape
        """
        # Collect the DataFrame
        df = self.dataset.load_data().collect()
        logger.info(f"Collected dataframe with shape: {df.shape}")

        # Check DataFrame structure
        self.assertIn("patient_id", df.columns)
        self.assertIn("event_type", df.columns)
        self.assertIn("timestamp", df.columns)
        self.assertIn("code", df.columns)
        self.assertIn("value", df.columns)

        # Check DataFrame content
        self.assertGreater(len(df), 0)  # Should have measurements
        self.assertEqual(df["event_type"][0], "measurement")


if __name__ == "__main__":
    unittest.main()
