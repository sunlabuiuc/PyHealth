from pathlib import Path
import tempfile
import unittest

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import MortalityPredictionMIMIC3


class TestMortalityPredictionMIMIC3(unittest.TestCase):
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

        cls.samples = dataset.set_task(MortalityPredictionMIMIC3())

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()

    def test_task_schema(self):
        self.assertIn("task_name", vars(MortalityPredictionMIMIC3))
        self.assertIn("input_schema", vars(MortalityPredictionMIMIC3))
        self.assertIn("output_schema", vars(MortalityPredictionMIMIC3))

        self.assertEqual(
            "MortalityPredictionMIMIC3",
            MortalityPredictionMIMIC3.task_name,
        )
        self.assertIn("conditions", MortalityPredictionMIMIC3.input_schema)
        self.assertIn("procedures", MortalityPredictionMIMIC3.input_schema)
        self.assertIn("drugs", MortalityPredictionMIMIC3.input_schema)
        self.assertIn("mortality", MortalityPredictionMIMIC3.output_schema)

    def test_sample_schema(self):
        for sample in self.samples:
            self.assertIn("patient_id", sample)
            self.assertIn("hadm_id", sample)
            self.assertIn("conditions", sample)
            self.assertIn("procedures", sample)
            self.assertIn("drugs", sample)
            self.assertIn("mortality", sample)

    def test_mortality_label_is_binary(self):
        for sample in self.samples:
            label = int(sample["mortality"].item())
            self.assertIn(label, [0, 1])

    def test_mortality_label_from_next_visit(self):
        """Patient 10059: visit 142582 (expire=0) then 122098 (expire=1).

        Mortality label is derived from the NEXT visit's expire flag,
        so visit 142582 should have mortality=1.
        """
        labels = [
            int(s["mortality"].item())
            for s in self.samples
            if s["hadm_id"] == "142582"
        ]

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0], 1)

    def test_surviving_next_visit(self):
        """Patient 10119: visit 157466 (expire=0) then 165436 (expire=0).

        Next visit also survived, so mortality=0.
        """
        labels = [
            int(s["mortality"].item())
            for s in self.samples
            if s["hadm_id"] == "157466"
        ]

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0], 0)

    def test_last_visit_excluded(self):
        """Patient 10059: last visit 122098 should not appear.

        The task drops the last visit because there is no next visit
        to derive the mortality label from.
        """
        visits = [s["hadm_id"] for s in self.samples]

        self.assertIn("142582", visits)
        self.assertNotIn("122098", visits)

    def test_single_visit_patients_excluded(self):
        """Patient 10006 has only 1 visit (142345).

        Patients with a single visit cannot produce mortality samples.
        """
        patients = [s["patient_id"] for s in self.samples]
        visits = [s["hadm_id"] for s in self.samples]

        self.assertNotIn("10006", patients)
        self.assertNotIn("142345", visits)

    def test_visit_without_procedures_excluded(self):
        """Patient 10117: visit 187023 has no procedures or prescriptions.

        Visits missing any of conditions, procedures, or drugs are
        excluded by the task.
        """
        visits = [s["hadm_id"] for s in self.samples]
        self.assertNotIn("187023", visits)


if __name__ == "__main__":
    unittest.main()
