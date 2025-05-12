import os
import unittest
import requests
import pandas as pd

from pyhealth.datasets.seizure_dataset import SeizureDataset


class TestSeizureDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Download CSV file before running tests."""
        cls.data_dir = "./temp_data"
        cls.csv_filename = "clinical_information.csv"
        cls.file_path = os.path.join(cls.data_dir, cls.csv_filename)

        os.makedirs(cls.data_dir, exist_ok=True)

        if not os.path.exists(cls.file_path):
            print("Downloading clinical_information.csv...")
            url = "https://zenodo.org/record/2547147/files/clinical_information.csv?download=1"
            response = requests.get(url)
            with open(cls.file_path, "wb") as f:
                f.write(response.content)
            print("Download complete.")

    def setUp(self):
        """Initialize the SeizureDataset instance."""
        self.dataset = SeizureDataset(root=self.data_dir)

    def test_dataset_loads_correctly(self):
        """Check that the full dataset loads and is not empty."""
        self.assertIsInstance(self.dataset.all_data, pd.DataFrame)
        self.assertGreater(len(self.dataset.all_data), 0, "The dataset should not be empty.")

    def test_seizure_patients_filtered(self):
        """Check that seizure patients are filtered correctly."""
        seizure_df = self.dataset.seizure_patients
        self.assertIsInstance(seizure_df, pd.DataFrame)
        self.assertGreater(len(seizure_df), 0, "There should be some patients with seizures.")
        self.assertIn("Primary Localisation", seizure_df.columns)

    def test_seizure_patient_ids_are_valid(self):
        """Check that patient IDs are non-empty strings."""
        for pid in self.dataset.seizure_patient_ids:
            self.assertIsInstance(pid, str)
            self.assertTrue(pid.strip(), "Patient ID should not be empty.")

if __name__ == "__main__":
    unittest.main()
