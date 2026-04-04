"""
Unit tests for the DSADataset and ActivityClassification classes.

Author:
    Ran You
"""
# CR rayou: delete this
import torch
if not hasattr(torch, 'uint16'):
    torch.uint16 = torch.int16 # This tricks litdata into thinking uint16 exists


import unittest
from pathlib import Path

from pyhealth.datasets import DSADataset
from pyhealth.tasks import ActivityClassification


class TestDSADataset(unittest.TestCase):
    """Test cases for DSADataset."""

    @classmethod
    def setUpClass(cls):
        """Set up test resources path."""
        cls.test_resources = Path(__file__).parent.parent.parent / "test-resources" / "dsa"

    def test_dataset_initialization(self):
        """Test that the dataset initializes correctly."""
        dataset = DSADataset(root=str(self.test_resources))
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "dsa")

    def test_stats(self):
        """Test that stats() runs without error."""
        dataset = DSADataset(root=str(self.test_resources))
        dataset.stats()

    def test_num_patients(self):
        """Test the number of unique patient IDs."""
        dataset = DSADataset(root=str(self.test_resources))
        # Should have 8 unique patient IDs (1-8)
        self.assertEqual(len(dataset.unique_patient_ids), 8)

    def test_get_patient(self):
        """Test retrieving a patient."""
        dataset = DSADataset(root=str(self.test_resources))
        patient = dataset.get_patient("5")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "5")

    def test_get_events(self):
        """Test getting motion sensor data from a patient."""
        dataset = DSADataset(root=str(self.test_resources))
        patient = dataset.get_patient("4")
        events = patient.get_events(event_type="activities")
        # patient 5 has 2 samples in test data
        self.assertEqual(len(events), 3)

    def test_event_attributes(self):
        """Test that event attributes are correctly loaded."""
        dataset = DSADataset(root=str(self.test_resources))
        patient = dataset.get_patient("5")
        events = patient.get_events(event_type="activities")
        event = events[0]

        # Check that attributes exist
        self.assertIn("activity", event)
        self.assertIn("segment", event)
        self.assertIn("sensor", event)

    def test_default_task(self):
        """Test that the default task is ActivityClassification."""
        dataset = DSADataset(root=str(self.test_resources))
        self.assertIsInstance(dataset.default_task, ActivityClassification)

    def test_set_task(self):
        """Test setting and running the activity classification task."""
        dataset = DSADataset(root=str(self.test_resources))
        task = ActivityClassification()
        samples = dataset.set_task(task)

        # Should have samples for activities.
        self.assertGreater(len(samples), 0)

    def test_task_output_format(self):
        """Test that task output has the correct format."""
        dataset = DSADataset(root=str(self.test_resources))
        task = ActivityClassification()
        samples = dataset.set_task(task)

        if len(samples) > 0:
            sample = samples[0]
            self.assertIn("patient_id", sample)
            self.assertIn("T", sample)
            self.assertIn("LA", sample)
            self.assertIn("RA", sample)
            self.assertIn("LL", sample)
            self.assertIn("RL", sample)
            self.assertIn("label", sample)


class TestActivityClassification(unittest.TestCase):
    """Test cases for ActivityClassification task."""

    def test_task_attributes(self):
        """Test task class attributes."""
        task = ActivityClassification()
        self.assertEqual(task.task_name, "ActivityClassification")
        self.assertIn("T", task.input_schema)
        self.assertIn("LA", task.input_schema)
        self.assertIn("RA", task.input_schema)
        self.assertIn("LL", task.input_schema)
        self.assertIn("RL", task.input_schema)
        self.assertIn("label", task.output_schema)

    def test_input_schema(self):
        """Test input schema definition."""
        task = ActivityClassification()
        self.assertEqual(task.input_schema["T"], "sequence")
        self.assertEqual(task.input_schema["RA"], "sequence")
        self.assertEqual(task.input_schema["LA"], "sequence")
        self.assertEqual(task.input_schema["RL"], "sequence")
        self.assertEqual(task.input_schema["LL"], "sequence")

    def test_output_schema(self):
        """Test output schema definition."""
        task = ActivityClassification()
        self.assertEqual(task.output_schema["label"], "text")


if __name__ == "__main__":
    unittest.main()
