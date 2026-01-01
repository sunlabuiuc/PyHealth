from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import ReadmissionPredictionMIMIC3


class TestReadmissionPredictionMIMIC3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_dir0 = tempfile.TemporaryDirectory()
        cls.cache_dir1 = tempfile.TemporaryDirectory()
        cls.cache_dir2 = tempfile.TemporaryDirectory()
        cls.cache_dir3 = tempfile.TemporaryDirectory()

        dataset = MIMIC3Dataset(
            root=str(Path(__file__).parent.parent.parent / "test-resources" / "core" / "mimic3demo"),
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=cls.cache_dir0.name
        )

        cls.samples15days = dataset.set_task(
            task=ReadmissionPredictionMIMIC3(window=timedelta(days=15)),
            cache_dir=cls.cache_dir1.name
        )

        cls.samples5days = dataset.set_task(
            task=ReadmissionPredictionMIMIC3(window=timedelta(days=5)),
            cache_dir=cls.cache_dir2.name
        )

        cls.sampleswithminors = dataset.set_task(
            task=ReadmissionPredictionMIMIC3(exclude_minors=False),
            cache_dir=cls.cache_dir3.name
        )

    @classmethod
    def tearDownClass(cls):
        cls.samples15days.close()
        cls.samples5days.close()
        cls.sampleswithminors.close()

        cls.cache_dir1.cleanup()
        cls.cache_dir2.cleanup()
        cls.cache_dir3.cleanup()

        # Deleting the dataset cache tmp dir causes ResourceWarnings
        # These are caused by unclosed files due to lazy loading and can be safely ignored in tests
        import warnings
        warnings.filterwarnings("ignore", category=ResourceWarning)

        cls.cache_dir0.cleanup()

    def test_task_schema(self):
        self.assertIn("task_name", vars(ReadmissionPredictionMIMIC3))
        self.assertIn("input_schema", vars(ReadmissionPredictionMIMIC3))
        self.assertIn("output_schema", vars(ReadmissionPredictionMIMIC3))

        self.assertEqual("ReadmissionPredictionMIMIC3", ReadmissionPredictionMIMIC3.task_name)
        self.assertIn("conditions", ReadmissionPredictionMIMIC3.input_schema)
        self.assertIn("procedures", ReadmissionPredictionMIMIC3.input_schema)
        self.assertIn("drugs", ReadmissionPredictionMIMIC3.input_schema)
        self.assertIn("readmission", ReadmissionPredictionMIMIC3.output_schema)

    def test_task_defaults(self):
        task = ReadmissionPredictionMIMIC3()

        self.assertEqual(task.window, timedelta(days=15))
        self.assertTrue(task.exclude_minors)

    def test_sample_schema(self):
        for sample in self.samples15days:
            self.assertIn("visit_id", sample)
            self.assertIn("patient_id", sample)
            self.assertIn("conditions", sample)
            self.assertIn("procedures", sample)
            self.assertIn("drugs", sample)
            self.assertIn("readmission", sample)

    def test_time_window(self):
        """
        Patient ID: 10059
        Visit ID: 142582
        Discharged: 2150-08-13 11:33:00
        Next admission: 2150-08-22 17:33:00
        """
        readmitted5days = [ s["readmission"] for s in self.samples5days if s["visit_id"] == "142582" ]
        readmitted15days = [ s["readmission"] for s in self.samples15days if s["visit_id"] == "142582" ]

        self.assertEqual(len(readmitted5days), 1)
        self.assertEqual(len(readmitted15days), 1)

        self.assertFalse(bool(readmitted5days[0].item()))
        self.assertTrue(bool(readmitted15days[0].item()))

    def test_last_admission_is_excluded(self):
        '''
        Patient ID: 10059
        Visit IDs: [142582, 122098]
        '''
        visits = [ s["visit_id"] for s in self.samples15days ]

        self.assertIn("142582", visits)
        self.assertNotIn("122098", visits)

    def test_patient_with_only_one_visit_is_excluded(self):
        '''
        Patient ID: 10006
        Visit IDs: [142345]
        Has diagnoses: True
        Has prescriptions: True
        Has procedures: True
        '''
        patients = [ s["patient_id"] for s in self.samples15days ]
        visits = [ s["visit_id"] for s in self.samples15days ]

        self.assertNotIn("10006", patients)
        self.assertNotIn("142345", visits)

    def test_admissions_without_procedures_are_excluded(self):
        '''
        Test case visits selected using `test-resources/core/mimic3demo/query.py`

        The current test dataset does not include applicable admissions with no diagnoses or prescriptions.
        The logic is the same for all three cases, so just testing one is okay.
        '''
        visits = [ s["visit_id"] for s in self.samples15days ]

        self.assertNotIn("118192", visits)
        self.assertNotIn("174863", visits)
        self.assertNotIn("180391", visits)

    def test_admissions_of_minors_are_excluded(self):
        """
        Patient ID: 10088
        Visit IDs: [169938, 168233, 149044]

        Test case visits selected using `test-resources/core/mimic3demo/query.py`
        """
        visits = [ s["visit_id"] for s in self.samples15days ]

        self.assertNotIn("169938", visits)
        self.assertIn("168233", visits)
        self.assertNotIn("149044", visits)

        visits = [ s["visit_id"] for s in self.sampleswithminors ]

        self.assertIn("169938", visits)
        self.assertIn("168233", visits)
        self.assertNotIn("149044", visits)


if __name__ == "__main__":
    unittest.main()
