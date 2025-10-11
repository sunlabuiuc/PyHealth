import unittest
from pyhealth.tasks.allergy_conflict_task import AllergyConflictDetectionTask
from pyhealth.tasks.allergy_conflict_task import MedAllergyConflictDataset


class TestAllergyConflictDetectionTask(unittest.TestCase):
    def setUp(self):
        # Load a small version of the dataset for testing
        self.dataset = MedAllergyConflictDataset(
            root="/Users/royal/Documents/Pyhealth_testing/mimic-iv-note-deidentified-free-text-clinical-notes-2"
        )
        self.task = AllergyConflictDetectionTask(self.dataset)

    def test_conflict_labels(self):
        # Check that get_label returns 0 or 1 for real patient data
        patient_ids = self.dataset.get_all_patient_ids()[:5]  # limit to 5 for speed
        for pid in patient_ids:
            patient = self.dataset.get_patient_by_id(pid)
            label = self.task.get_label(patient)
            self.assertIn(label, [0, 1], f"Label for {pid} was {label}")

    def test_export_conflicts(self):
        # Run the export and check that file is created and non-empty
        self.task.export_conflicts("test_conflict_output.csv")
        with open("test_conflict_output.csv") as f:
            content = f.read()
        self.assertTrue("hadm_id" in content)
        self.assertTrue(len(content.strip()) > 0)


if __name__ == "__main__":
    unittest.main()
