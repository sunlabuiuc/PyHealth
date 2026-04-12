from pathlib import Path
import tempfile
import unittest

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import DrugRecommendationMIMIC3


class TestDrugRecommendationMIMIC3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_dir = tempfile.TemporaryDirectory()

        dataset = MIMIC3Dataset(
            root=str(
                Path(__file__).parent.parent.parent
                / "test-resources"
                / "core"
                / "mimic3demo"
            ),
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=cls.cache_dir.name,
        )

        cls.samples = dataset.set_task(DrugRecommendationMIMIC3())

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()

    def test_task_schema(self):
        self.assertIn("task_name", vars(DrugRecommendationMIMIC3))
        self.assertIn("input_schema", vars(DrugRecommendationMIMIC3))
        self.assertIn("output_schema", vars(DrugRecommendationMIMIC3))

        self.assertEqual(
            "DrugRecommendationMIMIC3",
            DrugRecommendationMIMIC3.task_name,
        )
        self.assertIn("conditions", DrugRecommendationMIMIC3.input_schema)
        self.assertIn("procedures", DrugRecommendationMIMIC3.input_schema)
        self.assertIn("drugs_hist", DrugRecommendationMIMIC3.input_schema)

        for key in ("conditions", "procedures", "drugs_hist"):
            self.assertEqual(
                DrugRecommendationMIMIC3.input_schema[key],
                "nested_sequence",
            )

        self.assertIn("drugs", DrugRecommendationMIMIC3.output_schema)
        self.assertEqual(
            DrugRecommendationMIMIC3.output_schema["drugs"],
            "multilabel",
        )

    def test_sample_schema(self):
        for sample in self.samples:
            self.assertIn("patient_id", sample)
            self.assertIn("visit_id", sample)
            self.assertIn("conditions", sample)
            self.assertIn("procedures", sample)
            self.assertIn("drugs_hist", sample)
            self.assertIn("drugs", sample)

    def test_conditions_are_nested(self):
        """Conditions should be a 2-D tensor (visits x codes)."""
        for sample in self.samples:
            cond = sample["conditions"]
            self.assertEqual(
                cond.dim(),
                2,
                "conditions should be a 2-D tensor (nested_sequence)",
            )

    def test_single_visit_patients_excluded(self):
        """Patient 10006 has only 1 visit (142345).

        Drug recommendation requires at least 2 visits.
        """
        patients = [s["patient_id"] for s in self.samples]
        visits = [s["visit_id"] for s in self.samples]

        self.assertNotIn("10006", patients)
        self.assertNotIn("142345", visits)

    def test_visit_without_procedures_excluded(self):
        """Patient 41795: visit 118192 has no procedures.

        Visits missing any of conditions, procedures, or drugs
        are excluded by the task.
        """
        visits = [s["visit_id"] for s in self.samples]
        self.assertNotIn("118192", visits)

    def test_multi_visit_patient_produces_samples(self):
        """Patient 10088 has 3 visits, all with diag+proc+rx.

        Should produce samples for this patient.
        """
        patients = [s["patient_id"] for s in self.samples]
        self.assertIn("10088", patients)


if __name__ == "__main__":
    unittest.main()
