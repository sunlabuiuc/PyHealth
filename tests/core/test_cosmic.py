"""
Unit tests for the COSMICDataset and MutationPathogenicityPrediction classes.

Author:
    REDACTED_AUTHOR
"""
import os
import shutil
import unittest
from pathlib import Path

from pyhealth.datasets import COSMICDataset
from pyhealth.tasks import MutationPathogenicityPrediction


class TestCOSMICDataset(unittest.TestCase):
    """Test cases for COSMICDataset."""

    @classmethod
    def setUpClass(cls):
        """Set up test resources path."""
        cls.test_resources = Path(__file__).parent.parent.parent / "test-resources" / "cosmic"

    def test_dataset_initialization(self):
        """Test that the dataset initializes correctly."""
        dataset = COSMICDataset(root=str(self.test_resources))
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "cosmic")

    def test_stats(self):
        """Test that stats() runs without error."""
        dataset = COSMICDataset(root=str(self.test_resources))
        dataset.stats()

    def test_num_patients(self):
        """Test the number of unique patient IDs (samples)."""
        dataset = COSMICDataset(root=str(self.test_resources))
        # Should have 6 unique sample IDs (TCGA-001 through TCGA-006)
        self.assertEqual(len(dataset.unique_patient_ids), 6)

    def test_get_patient(self):
        """Test retrieving a patient/sample record."""
        dataset = COSMICDataset(root=str(self.test_resources))
        patient = dataset.get_patient("TCGA-001")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "TCGA-001")

    def test_get_events(self):
        """Test getting mutation events from a sample."""
        dataset = COSMICDataset(root=str(self.test_resources))
        patient = dataset.get_patient("TCGA-001")
        events = patient.get_events(event_type="mutations")
        # TCGA-001 has 2 mutations in test data
        self.assertEqual(len(events), 2)

    def test_event_attributes(self):
        """Test that event attributes are correctly loaded."""
        dataset = COSMICDataset(root=str(self.test_resources))
        patient = dataset.get_patient("TCGA-001")
        events = patient.get_events(event_type="mutations")
        event = events[0]

        # Check that attributes exist
        self.assertIn("gene_name", event)
        self.assertIn("fathmm_prediction", event)
        self.assertIn("primary_site", event)

    def test_default_task(self):
        """Test that the default task is MutationPathogenicityPrediction."""
        dataset = COSMICDataset(root=str(self.test_resources))
        self.assertIsInstance(dataset.default_task, MutationPathogenicityPrediction)

    def test_set_task(self):
        """Test setting and running the pathogenicity prediction task."""
        dataset = COSMICDataset(root=str(self.test_resources))
        task = MutationPathogenicityPrediction()
        samples = dataset.set_task(task)

        # Should have samples for mutations with FATHMM predictions
        self.assertGreater(len(samples), 0)

    def test_task_output_format(self):
        """Test that task output has the correct format."""
        dataset = COSMICDataset(root=str(self.test_resources))
        task = MutationPathogenicityPrediction()
        samples = dataset.set_task(task)

        if len(samples) > 0:
            sample = samples[0]
            self.assertIn("patient_id", sample)
            self.assertIn("gene_name", sample)
            self.assertIn("fathmm_prediction", sample)
            self.assertIn(sample["fathmm_prediction"], [0, 1])

    def test_pathogenic_vs_neutral_labels(self):
        """Test that FATHMM predictions are correctly converted to binary."""
        dataset = COSMICDataset(root=str(self.test_resources))
        task = MutationPathogenicityPrediction()
        samples = dataset.set_task(task)

        # Count pathogenic and neutral
        pathogenic_count = sum(1 for s in samples if s["fathmm_prediction"] == 1)
        neutral_count = sum(1 for s in samples if s["fathmm_prediction"] == 0)

        # Both should be present in test data
        self.assertGreater(pathogenic_count, 0)
        self.assertGreater(neutral_count, 0)


class TestMutationPathogenicityPrediction(unittest.TestCase):
    """Test cases for MutationPathogenicityPrediction task."""

    def test_task_attributes(self):
        """Test task class attributes."""
        task = MutationPathogenicityPrediction()
        self.assertEqual(task.task_name, "MutationPathogenicityPrediction")
        self.assertIn("gene_name", task.input_schema)
        self.assertIn("fathmm_prediction", task.output_schema)

    def test_input_schema(self):
        """Test input schema definition."""
        task = MutationPathogenicityPrediction()
        self.assertEqual(task.input_schema["gene_name"], "text")
        self.assertEqual(task.input_schema["mutation_description"], "text")
        self.assertEqual(task.input_schema["primary_site"], "text")

    def test_output_schema(self):
        """Test output schema definition."""
        task = MutationPathogenicityPrediction()
        self.assertEqual(task.output_schema["fathmm_prediction"], "binary")


if __name__ == "__main__":
    unittest.main()
