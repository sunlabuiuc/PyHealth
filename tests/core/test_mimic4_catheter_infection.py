import unittest
from pathlib import Path
import math

import polars as pl

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.catheter_infection import (
    CatheterAssociatedInfectionPredictionMIMIC4,
    CatheterAssociatedInfectionPredictionStageNetMIMIC4,
)


class TestMIMIC4CatheterInfectionPrediction(unittest.TestCase):
    """Dataset-backed tests using synthetic rows in mimic4demo CSV files."""

    def setUp(self):
        test_dir = Path(__file__).parent.parent.parent
        self.demo_dataset_path = str(
            test_dir / "test-resources" / "core" / "mimic4demo"
        )
        tables = ["diagnoses_icd", "procedures_icd", "labevents"]
        self.dataset = MIMIC4Dataset(
            ehr_root=self.demo_dataset_path,
            ehr_tables=tables,
        )

    @staticmethod
    def _to_int(value):
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)

    def test_helper_code_matching(self):
        task = CatheterAssociatedInfectionPredictionMIMIC4()

        self.assertTrue(task._is_catheter_code("Y84.6", 10))
        self.assertTrue(task._is_catheter_code("0T9B70Z", "10"))
        self.assertTrue(task._is_catheter_code("996.31", 9))
        self.assertTrue(task._is_catheter_code("37.22", 9))
        self.assertFalse(task._is_catheter_code("Y84.6", 9))

        self.assertTrue(task._is_infection_code("T83.511A", 10))
        self.assertTrue(task._is_infection_code("T83518D", "10"))
        self.assertTrue(task._is_infection_code("996.64", 9))
        self.assertFalse(task._is_infection_code("T83.511A", 9))
        self.assertFalse(task._is_infection_code("N39.0", 10))

    def test_synthetic_patient_outcomes(self):
        task = CatheterAssociatedInfectionPredictionMIMIC4()
        sample_dataset = self.dataset.set_task(task)

        labels_by_patient = {
            self._to_int(sample["patient_id"]): self._to_int(sample["label"])
            for sample in sample_dataset
        }

        # Positive case: catheter first, later infection admission.
        self.assertIn(91001, labels_by_patient)
        self.assertEqual(labels_by_patient[91001], 1)

        # Negative case: catheter first, no later infection.
        self.assertIn(91002, labels_by_patient)
        self.assertEqual(labels_by_patient[91002], 0)

        # Excluded case: infection before catheter evidence.
        self.assertNotIn(91003, labels_by_patient)

    def test_synthetic_patient_outcomes_stagenet_variant(self):
        task = CatheterAssociatedInfectionPredictionStageNetMIMIC4()
        sample_dataset = self.dataset.set_task(task)

        labels_by_patient = {
            self._to_int(sample["patient_id"]): self._to_int(sample["label"])
            for sample in sample_dataset
        }

        self.assertIn(91001, labels_by_patient)
        self.assertEqual(labels_by_patient[91001], 1)

        self.assertIn(91002, labels_by_patient)
        self.assertEqual(labels_by_patient[91002], 0)

        self.assertNotIn(91003, labels_by_patient)

    def test_missing_defaults(self):
        task = CatheterAssociatedInfectionPredictionMIMIC4()

        self.assertEqual(task._ensure_nonempty_sequence([]), ["<missing>"])

        empty_lab_df = pl.DataFrame()
        lab_vector = task._build_lab_vector(empty_lab_df)
        self.assertEqual(len(lab_vector), len(task.LAB_CATEGORY_ORDER))
        self.assertTrue(all(v == 0.0 for v in lab_vector))

    def test_no_nan_labs_in_nested_outputs(self):
        task = CatheterAssociatedInfectionPredictionMIMIC4()
        sample_dataset = self.dataset.set_task(task)

        for sample in sample_dataset:
            for visit_labs in sample["labs"]:
                self.assertFalse(any(math.isnan(v) for v in visit_labs))


if __name__ == "__main__":
    unittest.main()
