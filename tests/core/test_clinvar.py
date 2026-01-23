"""
Unit tests for the ClinVarDataset and VariantClassificationClinVar classes.

Author:
    REDACTED_AUTHOR
"""
import os
import shutil
import unittest
from pathlib import Path

from pyhealth.datasets import ClinVarDataset
from pyhealth.tasks import VariantClassificationClinVar


class TestClinVarDataset(unittest.TestCase):
    """Test cases for ClinVarDataset."""

    @classmethod
    def setUpClass(cls):
        """Set up test resources path."""
        cls.test_resources = Path(__file__).parent.parent.parent / "test-resources" / "clinvar"

    def test_dataset_initialization(self):
        """Test that the dataset initializes correctly."""
        dataset = ClinVarDataset(root=str(self.test_resources))
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "clinvar")

    def test_stats(self):
        """Test that stats() runs without error."""
        dataset = ClinVarDataset(root=str(self.test_resources))
        dataset.stats()

    def test_num_patients(self):
        """Test the number of unique patient IDs (variants)."""
        dataset = ClinVarDataset(root=str(self.test_resources))
        # Each row is a separate "patient" since patient_id is null
        self.assertEqual(len(dataset.unique_patient_ids), 12)

    def test_get_patient(self):
        """Test retrieving a patient/variant record."""
        dataset = ClinVarDataset(root=str(self.test_resources))
        patient_id = dataset.unique_patient_ids[0]
        patient = dataset.get_patient(patient_id)
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, patient_id)

    def test_get_events(self):
        """Test getting events from a patient."""
        dataset = ClinVarDataset(root=str(self.test_resources))
        patient_id = dataset.unique_patient_ids[0]
        patient = dataset.get_patient(patient_id)
        events = patient.get_events(event_type="variants")
        self.assertEqual(len(events), 1)

    def test_event_attributes(self):
        """Test that event attributes are correctly loaded."""
        dataset = ClinVarDataset(root=str(self.test_resources))
        patient_id = dataset.unique_patient_ids[0]
        patient = dataset.get_patient(patient_id)
        events = patient.get_events(event_type="variants")
        event = events[0]

        # Check that attributes exist
        self.assertIn("gene_symbol", event)
        self.assertIn("clinical_significance", event)
        self.assertIn("chromosome", event)

    def test_default_task(self):
        """Test that the default task is VariantClassificationClinVar."""
        dataset = ClinVarDataset(root=str(self.test_resources))
        self.assertIsInstance(dataset.default_task, VariantClassificationClinVar)

    def test_set_task(self):
        """Test setting and running the variant classification task."""
        dataset = ClinVarDataset(root=str(self.test_resources))
        task = VariantClassificationClinVar()
        samples = dataset.set_task(task)

        # Should have samples for variants with valid clinical significance
        self.assertGreater(len(samples), 0)

    def test_task_output_format(self):
        """Test that task output has the correct format."""
        dataset = ClinVarDataset(root=str(self.test_resources))
        task = VariantClassificationClinVar()
        samples = dataset.set_task(task)

        if len(samples) > 0:
            sample = samples[0]
            self.assertIn("patient_id", sample)
            self.assertIn("gene_symbol", sample)
            self.assertIn("clinical_significance", sample)


class TestVariantClassificationClinVar(unittest.TestCase):
    """Test cases for VariantClassificationClinVar task."""

    def test_task_attributes(self):
        """Test task class attributes."""
        task = VariantClassificationClinVar()
        self.assertEqual(task.task_name, "VariantClassificationClinVar")
        self.assertIn("gene_symbol", task.input_schema)
        self.assertIn("clinical_significance", task.output_schema)

    def test_input_schema(self):
        """Test input schema definition."""
        task = VariantClassificationClinVar()
        self.assertEqual(task.input_schema["gene_symbol"], "text")
        self.assertEqual(task.input_schema["variant_type"], "text")
        self.assertEqual(task.input_schema["chromosome"], "text")

    def test_output_schema(self):
        """Test output schema definition."""
        task = VariantClassificationClinVar()
        self.assertEqual(task.output_schema["clinical_significance"], "multiclass")


if __name__ == "__main__":
    unittest.main()
