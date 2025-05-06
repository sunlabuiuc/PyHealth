import os
import json
import unittest
import pandas as pd
from pyhealth.datasets.deidentification_dataset import DeidentificationDataset
from pyhealth.tasks.deidentification_task import DeIdentificationTask


class TestDeIdentificationTask(unittest.TestCase):
    """Test suite for the DeIdentificationTask class."""

    @classmethod
    def setUpClass(cls):
        """Set up the test environment (mock data and file)."""
        # Sample mock data
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
            },
            {
                "document_id": "ex_ds_2",
                "text": "Another summary with no diagnosis mentioned.",
                "patient_name": "Jane Doe",
                "dob": "1985-05-15",
                "age": 39,
                "sex": "Female",
                "service": "Internal Medicine",
                "chief_complaint": "Abdominal Pain",
                "diagnosis": "",
                "treatment": "Pain relief",
                "follow_up_plan": "Follow-up in 1 week",
                "discharge_date": "2024-04-10",
                "attending_physician": "Dr. Lee"
            }
        ]

        # Create temporary mock JSON file
        cls.mock_file_path = '/tmp/mock_discharge_summaries.json'
        with open(cls.mock_file_path, 'w') as f:
            json.dump(cls.mock_data, f)

        dataset_config = {
            'table_name': 'discharge_summaries',
            'file_path': cls.mock_file_path,
            'patient_id': 'document_id',
            'timestamp': 'discharge_date',
            'attributes': ['document_id', 'text', 'patient_name', 'dob', 'age', 'sex']
        }

        cls.dataset = DeidentificationDataset(dataset_config)
        cls.task = DeIdentificationTask(cls.dataset)

    @classmethod
    def tearDownClass(cls):
        """Clean up resources (remove the mock file)."""
        if os.path.exists(cls.mock_file_path):
            os.remove(cls.mock_file_path)

    def test_pre_process_data(self):
        """Test preprocessing of the text column."""
        original_text = self.dataset.data['text'].iloc[0]
        self.task.pre_process_data()
        processed_text = self.dataset.data['processed_text'].iloc[0]
        self.assertNotEqual(original_text, processed_text)

    def test_extract_diagnosis(self):
        """Test extraction of diagnosis from the discharge summaries."""
        self.task.extract_diagnosis()
        diagnosis_1 = self.dataset.data['diagnosis_extracted'].iloc[0]
        diagnosis_2 = self.dataset.data['diagnosis_extracted'].iloc[1]

        self.assertEqual(diagnosis_1, 'Diagnosis Found')
        self.assertEqual(diagnosis_2, 'Diagnosis not found')

    def test_get_task_info(self):
        """Test retrieval of task metadata."""
        info = self.task.get_task_info()
        self.assertEqual(info['dataset_name'], 'discharge_summaries')
        self.assertEqual(info['task_type'], 'deidentification')

    def test_single_example_transformation(self):
        """Test the transformation of a single example."""
        example = self.dataset.data.iloc[0]
        transformed = self.task(example)
        self.assertIn('input', transformed)
        self.assertIn('label', transformed)

    def test_evaluate(self):
        """Test evaluation method after diagnosis extraction."""
        self.task.extract_diagnosis()  # Ensure the diagnosis_extracted column exists
        test_data = self.dataset.data.sample(frac=1.0, random_state=42)
        evaluation = self.task.evaluate(test_data)
        self.assertIn('accuracy', evaluation)
        self.assertEqual(evaluation['total'], len(test_data))

    def test_training_loop(self):
        """Test a dummy training loop to ensure no errors are raised."""
        self.task.train(self.dataset.data, epochs=1)  # No assertion; just shouldn't raise an error


if __name__ == '__main__':
    unittest.main()
