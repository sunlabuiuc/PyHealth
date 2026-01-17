from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

from pyhealth.datasets import OMOPDataset
from pyhealth.tasks import ReadmissionPredictionOMOP


class TestReadmissionPredictionMIMIC3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_dir = tempfile.TemporaryDirectory()

        dataset = OMOPDataset(
            root=str(Path(__file__).parent.parent.parent / "test-resources" / "omop"),
            tables=["person", "visit_occurrence", "condition_occurrence", "procedure_occurrence", "drug_exposure"],
            cache_dir=cls.cache_dir.name,
        )

        # cls.samples15days = dataset.set_task(ReadmissionPredictionOMOP(window=timedelta(days=15)))
        # cls.samples5days = dataset.set_task(ReadmissionPredictionOMOP(window=timedelta(days=5)))
        # cls.sampleswithminors = dataset.set_task(ReadmissionPredictionOMOP(exclude_minors=False))

    @classmethod
    def tearDownClass(cls):
        pass
        # cls.samples15days.close()
        # cls.samples5days.close()
        # cls.sampleswithminors.close()

    def test_task_schema(self):
        self.assertIn("task_name", vars(ReadmissionPredictionOMOP))
        self.assertIn("input_schema", vars(ReadmissionPredictionOMOP))
        self.assertIn("output_schema", vars(ReadmissionPredictionOMOP))

        self.assertEqual("ReadmissionPredictionOMOP", ReadmissionPredictionOMOP.task_name)
        self.assertIn("conditions", ReadmissionPredictionOMOP.input_schema)
        self.assertIn("procedures", ReadmissionPredictionOMOP.input_schema)
        self.assertIn("drugs", ReadmissionPredictionOMOP.input_schema)
        self.assertIn("readmission", ReadmissionPredictionOMOP.output_schema)

    def test_task_defaults(self):
        task = ReadmissionPredictionOMOP()

        self.assertEqual(task.window, timedelta(days=15))
        self.assertTrue(task.exclude_minors)

    # def test_sample_schema(self):
    #     for sample in self.samples15days:
    #         self.assertIn("visit_id", sample)
    #         self.assertIn("patient_id", sample)
    #         self.assertIn("conditions", sample)
    #         self.assertIn("procedures", sample)
    #         self.assertIn("drugs", sample)
    #         self.assertIn("readmission", sample)

    # def test_time_window(self):
    #     """
    #     Patient ID: 10059
    #     Visit ID: 142582
    #     Discharged: 2150-08-13 11:33:00
    #     Next admission: 2150-08-22 17:33:00
    #     """
    #     readmitted5days = [ s["readmission"] for s in self.samples5days if s["visit_id"] == "142582" ]
    #     readmitted15days = [ s["readmission"] for s in self.samples15days if s["visit_id"] == "142582" ]

    #     self.assertEqual(len(readmitted5days), 1)
    #     self.assertEqual(len(readmitted15days), 1)

    #     self.assertFalse(bool(readmitted5days[0].item()))
    #     self.assertTrue(bool(readmitted15days[0].item()))

    # def test_last_admission_is_excluded(self):
    #     '''
    #     Patient ID: 10059
    #     Visit IDs: [142582, 122098]
    #     '''
    #     visits = [ s["visit_id"] for s in self.samples15days ]

    #     self.assertIn("142582", visits)
    #     self.assertNotIn("122098", visits)

    # def test_patient_with_only_one_visit_is_excluded(self):
    #     '''
    #     Patient ID: 10006
    #     Visit IDs: [142345]
    #     Has diagnoses: True
    #     Has prescriptions: True
    #     Has procedures: True
    #     '''
    #     patients = [ s["patient_id"] for s in self.samples15days ]
    #     visits = [ s["visit_id"] for s in self.samples15days ]

    #     self.assertNotIn("10006", patients)
    #     self.assertNotIn("142345", visits)

    # def test_admissions_without_procedures_are_excluded(self):
    #     '''
    #     Test case visits selected using `test-resources/core/mimic3demo/query.py`

    #     The current test dataset does not include applicable admissions with no diagnoses or prescriptions.
    #     The logic is the same for all three cases, so just testing one is okay.
    #     '''
    #     visits = [ s["visit_id"] for s in self.samples15days ]

    #     self.assertNotIn("118192", visits)
    #     self.assertNotIn("174863", visits)
    #     self.assertNotIn("180391", visits)

    # def test_admissions_of_minors_are_excluded(self):
    #     """
    #     Patient ID: 10088
    #     Visit IDs: [169938, 168233, 149044]

    #     Test case visits selected using `test-resources/core/mimic3demo/query.py`
    #     """
    #     visits = [ s["visit_id"] for s in self.samples15days ]

    #     self.assertNotIn("169938", visits)
    #     self.assertIn("168233", visits)
    #     self.assertNotIn("149044", visits)

    #     visits = [ s["visit_id"] for s in self.sampleswithminors ]

    #     self.assertIn("169938", visits)
    #     self.assertIn("168233", visits)
    #     self.assertNotIn("149044", visits)


if __name__ == "__main__":
    unittest.main()
