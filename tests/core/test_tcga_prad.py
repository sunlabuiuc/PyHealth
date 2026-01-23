"""
Unit tests for the TCGAPRADDataset and CancerSurvivalPrediction classes.

Author:
    REDACTED_AUTHOR
"""
import os
import shutil
import unittest
from pathlib import Path

from pyhealth.datasets import TCGAPRADDataset
from pyhealth.tasks import CancerSurvivalPrediction, CancerMutationBurden


class TestTCGAPRADDataset(unittest.TestCase):
    """Test cases for TCGAPRADDataset."""

    @classmethod
    def setUpClass(cls):
        """Set up test resources path."""
        cls.test_resources = Path(__file__).parent.parent.parent / "test-resources" / "tcga_prad"

    def test_dataset_initialization(self):
        """Test that the dataset initializes correctly."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "tcga_prad")

    def test_stats(self):
        """Test that stats() runs without error."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        dataset.stats()

    def test_num_patients(self):
        """Test the number of unique patient IDs."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        # Should have 5 unique patients
        self.assertEqual(len(dataset.unique_patient_ids), 5)

    def test_get_patient(self):
        """Test retrieving a patient record."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        patient = dataset.get_patient("TCGA-2A-A8VL")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "TCGA-2A-A8VL")

    def test_get_mutation_events(self):
        """Test getting mutation events from a patient."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        patient = dataset.get_patient("TCGA-2A-A8VL")
        events = patient.get_events(event_type="mutations")
        # TCGA-2A-A8VL has 3 mutations in test data
        self.assertEqual(len(events), 3)

    def test_get_clinical_events(self):
        """Test getting clinical events from a patient."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        patient = dataset.get_patient("TCGA-2A-A8VL")
        events = patient.get_events(event_type="clinical")
        # Each patient has 1 clinical record
        self.assertEqual(len(events), 1)

    def test_mutation_attributes(self):
        """Test that mutation attributes are correctly loaded."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        patient = dataset.get_patient("TCGA-2A-A8VL")
        events = patient.get_events(event_type="mutations")
        event = events[0]

        # Check that attributes exist
        self.assertIn("hugo_symbol", event)
        self.assertIn("variant_classification", event)
        self.assertIn("variant_type", event)

    def test_clinical_attributes(self):
        """Test that clinical attributes are correctly loaded."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        patient = dataset.get_patient("TCGA-2A-A8VL")
        events = patient.get_events(event_type="clinical")
        event = events[0]

        # Check that attributes exist
        self.assertIn("age_at_diagnosis", event)
        self.assertIn("gleason_score", event)
        self.assertIn("vital_status", event)

    def test_default_task(self):
        """Test that the default task is CancerSurvivalPrediction."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        self.assertIsInstance(dataset.default_task, CancerSurvivalPrediction)

    def test_set_task_survival(self):
        """Test setting and running the survival prediction task."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        task = CancerSurvivalPrediction()
        samples = dataset.set_task(task)

        # Should have samples for patients with clinical data
        self.assertGreater(len(samples), 0)

    def test_task_output_format(self):
        """Test that task output has the correct format."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        task = CancerSurvivalPrediction()
        samples = dataset.set_task(task)

        if len(samples) > 0:
            sample = samples[0]
            self.assertIn("patient_id", sample)
            self.assertIn("mutations", sample)
            self.assertIn("vital_status", sample)
            self.assertIn(sample["vital_status"], [0, 1])
            # After processing, mutations is converted to tensor by SequenceProcessor
            self.assertTrue(hasattr(sample["mutations"], '__len__'))

    def test_vital_status_labels(self):
        """Test that vital status is correctly converted to binary."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        task = CancerSurvivalPrediction()
        samples = dataset.set_task(task)

        # Count alive and dead
        dead_count = sum(1 for s in samples if s["vital_status"] == 1)
        alive_count = sum(1 for s in samples if s["vital_status"] == 0)

        # Both should be present in test data
        self.assertGreater(dead_count, 0)
        self.assertGreater(alive_count, 0)


class TestCancerSurvivalPrediction(unittest.TestCase):
    """Test cases for CancerSurvivalPrediction task."""

    def test_task_attributes(self):
        """Test task class attributes."""
        task = CancerSurvivalPrediction()
        self.assertEqual(task.task_name, "CancerSurvivalPrediction")
        self.assertIn("mutations", task.input_schema)
        self.assertIn("vital_status", task.output_schema)

    def test_input_schema(self):
        """Test input schema definition."""
        task = CancerSurvivalPrediction()
        self.assertEqual(task.input_schema["mutations"], "sequence")
        self.assertEqual(task.input_schema["age_at_diagnosis"], "tensor")
        self.assertEqual(task.input_schema["gleason_score"], "tensor")

    def test_output_schema(self):
        """Test output schema definition."""
        task = CancerSurvivalPrediction()
        self.assertEqual(task.output_schema["vital_status"], "binary")


class TestCancerMutationBurden(unittest.TestCase):
    """Test cases for CancerMutationBurden task."""

    @classmethod
    def setUpClass(cls):
        """Set up test resources path."""
        cls.test_resources = Path(__file__).parent.parent.parent / "test-resources" / "tcga_prad"

    def test_task_attributes(self):
        """Test task class attributes."""
        task = CancerMutationBurden()
        self.assertEqual(task.task_name, "CancerMutationBurden")
        self.assertIn("mutations", task.input_schema)
        self.assertIn("high_tmb", task.output_schema)

    def test_set_task(self):
        """Test setting and running the mutation burden task."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        task = CancerMutationBurden()
        samples = dataset.set_task(task)

        # Should have samples
        self.assertGreater(len(samples), 0)

    def test_output_format(self):
        """Test that task output has the correct format."""
        dataset = TCGAPRADDataset(root=str(self.test_resources))
        task = CancerMutationBurden()
        samples = dataset.set_task(task)

        if len(samples) > 0:
            sample = samples[0]
            self.assertIn("patient_id", sample)
            self.assertIn("mutations", sample)
            self.assertIn("high_tmb", sample)
            # high_tmb is converted to int by BinaryLabelProcessor
            self.assertIn(int(sample["high_tmb"]), [0, 1])


if __name__ == "__main__":
    unittest.main()
