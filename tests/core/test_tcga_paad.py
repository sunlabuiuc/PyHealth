"""
Unit tests for the TCGAPAADDataset, mirroring PRAD tests style.
"""
import unittest
from pathlib import Path

from pyhealth.datasets import TCGAPAADDataset
from pyhealth.tasks import CancerSurvivalPrediction


class TestTCGAPAADDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_resources = (
            Path(__file__).parent.parent.parent / "test-resources" / "tcga_paad"
        )

    def test_dataset_initialization(self):
        dataset = TCGAPAADDataset(root=str(self.test_resources))
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "tcga_paad")

    def test_stats(self):
        dataset = TCGAPAADDataset(root=str(self.test_resources))
        dataset.stats()

    def test_get_patient(self):
        dataset = TCGAPAADDataset(root=str(self.test_resources))
        patient = dataset.get_patient("TCGA-AB-1234")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "TCGA-AB-1234")

    def test_get_mutation_events(self):
        dataset = TCGAPAADDataset(root=str(self.test_resources))
        patient = dataset.get_patient("TCGA-AB-1234")
        events = patient.get_events(event_type="mutations")
        self.assertGreaterEqual(len(events), 1)

    def test_get_clinical_events(self):
        dataset = TCGAPAADDataset(root=str(self.test_resources))
        patient = dataset.get_patient("TCGA-AB-1234")
        events = patient.get_events(event_type="clinical")
        self.assertEqual(len(events), 1)

    def test_default_task(self):
        dataset = TCGAPAADDataset(root=str(self.test_resources))
        self.assertIsInstance(dataset.default_task, CancerSurvivalPrediction)

    def test_set_task_survival(self):
        dataset = TCGAPAADDataset(root=str(self.test_resources))
        task = CancerSurvivalPrediction()
        samples = dataset.set_task(task)
        self.assertGreater(len(samples), 0)


if __name__ == "__main__":
    unittest.main()
