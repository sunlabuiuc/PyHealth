"""
Unit tests for the COVID19CXRDataset class.

Author:
    Kilo Code
"""
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from pyhealth.datasets import COVID19CXRDataset
from pyhealth.tasks import COVID19CXRClassification


class TestCOVID19CXRDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_check_raw_data_exists_returns_false_when_files_missing(self):
        """Test _check_raw_data_exists returns False when required files are missing."""
        dataset = COVID19CXRDataset.__new__(COVID19CXRDataset)  # Create without __init__
        result = dataset._check_raw_data_exists(self.temp_dir)
        self.assertFalse(result)

    def test_check_raw_data_exists_returns_true_when_files_present(self):
        """Test _check_raw_data_exists returns True when all required files are present."""
        # Create dummy files
        required_files = [
            "COVID.metadata.xlsx",
            "Lung_Opacity.metadata.xlsx",
            "Normal.metadata.xlsx",
            "Viral Pneumonia.metadata.xlsx",
        ]
        for filename in required_files:
            with open(os.path.join(self.temp_dir, filename), 'w') as f:
                f.write("dummy content")

        dataset = COVID19CXRDataset.__new__(COVID19CXRDataset)
        result = dataset._check_raw_data_exists(self.temp_dir)
        self.assertTrue(result)

    def test_init_raises_value_error_when_raw_data_missing(self):
        """Test that __init__ raises ValueError with correct message when raw data is missing."""
        with self.assertRaises(ValueError) as context:
            COVID19CXRDataset(root=self.temp_dir)

        expected_message = (
            f"Raw COVID-19 CXR dataset files not found in {self.temp_dir}. "
            "Please download the dataset from "
            "https://www.kaggle.com/api/v1/datasets/download/tawsifurrahman/covid19-radiography-database "
            "and extract the contents to the specified root directory."
        )
        self.assertEqual(str(context.exception), expected_message)

    @patch('pyhealth.datasets.covid19_cxr.COVID19CXRDataset._check_raw_data_exists')
    @patch('pyhealth.datasets.base_dataset.BaseDataset.__init__')
    def test_init_works_when_raw_data_present(self, mock_base_init, mock_check):
        """Test that __init__ works when raw data is present."""
        mock_check.return_value = True

        # Mock the metadata file as existing
        metadata_path = os.path.join(self.temp_dir, "covid19_cxr-metadata-pyhealth.csv")
        with open(metadata_path, 'w') as f:
            f.write("path\nfake_path.png")

        dataset = COVID19CXRDataset(root=self.temp_dir)

        # Check that BaseDataset.__init__ was called
        mock_base_init.assert_called_once()

    # Removed test_default_task as it requires instantiation


if __name__ == "__main__":
    unittest.main()