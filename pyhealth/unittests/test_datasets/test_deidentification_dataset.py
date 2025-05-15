import os
import json
import unittest
from unittest.mock import patch
from pyhealth.datasets.deidentification_dataset import DeidentificationDataset
from pyhealth.tasks.deidentification_task import DeIdentificationTask


class TestDeidentificationDataset(unittest.TestCase):
    """Test suite for DeidentificationDataset class."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment with mock data and temporary files."""
        cls.mock_data = [
            {
                "document_id": "ex_ds_1",
                "text": "This is a sample discharge summary. Diagnosis: Hypertension.",
                "patient_name": "John Doe",
                "dob": "1990-01-01",
                "age": 30,
                "sex": "Male",
                "service": "Cardiology",
                "chief_complaint": "Chest Pain",
                "diagnosis": "Hypertension",
                "treatment": "Medication",
                "follow_up_plan": "Check-up in 2 weeks",
                "discharge_date": "2024-04-01",
                "attending_physician": "Dr. Smith"
            }
        ]
        
        # Temporary file creation to mock the data
        cls.mock_file_path = 'mock_discharge_summaries.json'  # Use a valid file path for your OS
        with open(cls.mock_file_path, 'w') as f:
            json.dump(cls.mock_data, f)

        dataset_config = {
            'table_name': 'discharge_summaries',
            'file_path': cls.mock_file_path,
            'patient_id': 'document_id',
            'timestamp': None,
            'attributes': [
                'document_id', 'text', 'patient_name', 'dob', 'age', 'sex', 'service', 
                'chief_complaint', 'diagnosis', 'treatment', 'follow_up_plan', 'discharge_date', 'attending_physician'
            ]
        }
        
        cls.dataset = DeidentificationDataset(dataset_config)
        cls.task = DeIdentificationTask(cls.dataset)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files after tests."""
        if os.path.exists(cls.mock_file_path):
            os.remove(cls.mock_file_path)

    def test_dataset_loading(self):
        """Test that the dataset is loaded correctly with expected columns."""
        data = self.dataset.data
        self.assertGreater(len(data), 0, "Dataset is empty.")
        for attr in self.dataset.config['attributes']:
            self.assertIn(attr, data.columns, f"Missing attribute: {attr}")
        self.assertIn(self.dataset.patient_id, data.columns, "Patient ID column is missing.")

    def test_missing_file(self):
        """Test handling of missing file scenario."""
        dataset_config = {
            'table_name': 'discharge_summaries',
            'file_path': 'data/non_existent_file.json',
            'patient_id': 'document_id',
            'timestamp': None,
            'attributes': ['document_id', 'text']
        }
        with self.assertRaises(FileNotFoundError):
            DeidentificationDataset(dataset_config)

    def test_invalid_json_format(self):
        """Test handling of invalid JSON format in the file."""
        invalid_json = "{ this is not a valid json }"
        invalid_file_path = 'invalid_json.json'
        with open(invalid_file_path, 'w') as f:
            f.write(invalid_json)
        
        dataset_config = {
            'table_name': 'discharge_summaries',
            'file_path': invalid_file_path,
            'patient_id': 'document_id',
            'timestamp': None,
            'attributes': ['document_id', 'text']
        }
        
        with self.assertRaises(ValueError):
            DeidentificationDataset(dataset_config)

        os.remove(invalid_file_path)

    def test_invalid_data_format(self):
        """Test handling of invalid data format (not a list of records)."""
        invalid_data = {"document_id": "ex_ds_1", "text": "Invalid data format"}
        invalid_file_path = 'invalid_data_format.json'
        with open(invalid_file_path, 'w') as f:
            json.dump(invalid_data, f)
        
        dataset_config = {
            'table_name': 'discharge_summaries',
            'file_path': invalid_file_path,
            'patient_id': 'document_id',
            'timestamp': None,
            'attributes': ['document_id', 'text']
        }
        
        with self.assertRaises(ValueError):
            DeidentificationDataset(dataset_config)

        os.remove(invalid_file_path)

    def test_missing_columns(self):
        """Test handling of missing required columns (e.g., 'patient_id' or 'text')."""
        invalid_data = [{"document_id": "ex_ds_1", "name": "John Doe"}]  # Missing 'text' column
        invalid_file_path = 'missing_columns.json'
        with open(invalid_file_path, 'w') as f:
            json.dump(invalid_data, f)
        
        dataset_config = {
            'table_name': 'discharge_summaries',
            'file_path': invalid_file_path,
            'patient_id': 'document_id',
            'timestamp': None,
            'attributes': ['document_id', 'text', 'name']
        }
        
        with self.assertRaises(ValueError):
            DeidentificationDataset(dataset_config)

        os.remove(invalid_file_path)

    def test_preprocess_data(self):
        """Test the text preprocessing functionality."""
        self.assertNotIn('processed_text', self.dataset.data.columns)
        self.task.pre_process_data()
        self.assertIn('processed_text', self.dataset.data.columns)
        original = self.dataset.data['text'].iloc[0]
        processed = self.dataset.data['processed_text'].iloc[0]
        self.assertNotEqual(original, processed)

    def test_get_patient_data(self):
        """Test retrieving data for a specific patient."""
        data = self.dataset.get_patient_data("ex_ds_1")
        self.assertGreater(len(data), 0)
        self.assertEqual(data['document_id'].iloc[0], "ex_ds_1")

    def test_invalid_patient_id(self):
        """Test retrieving data for an invalid patient ID."""
        data = self.dataset.get_patient_data("non_existent_id")
        self.assertEqual(len(data), 0)


if __name__ == '__main__':
    unittest.main()
