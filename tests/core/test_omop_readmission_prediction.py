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

        cls.samples15days = dataset.set_task(ReadmissionPredictionOMOP(window=timedelta(days=15)))
        cls.samples5days = dataset.set_task(ReadmissionPredictionOMOP(window=timedelta(days=5)))
        cls.sampleswithminors = dataset.set_task(ReadmissionPredictionOMOP(exclude_minors=False))

    @classmethod
    def tearDownClass(cls):
        cls.samples15days.close()
        cls.samples5days.close()
        cls.sampleswithminors.close()

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

    def test_sample_schema(self):
        for sample in self.samples15days:
            self.assertIn("visit_id", sample)
            self.assertIn("patient_id", sample)
            self.assertIn("conditions", sample)
            self.assertIn("procedures", sample)
            self.assertIn("drugs", sample)
            self.assertIn("readmission", sample)

    def test_time_window(self):
        readmitted5days = [ s["readmission"] for s in self.samples5days if s["visit_id"] == "2" ]
        readmitted15days = [ s["readmission"] for s in self.samples15days if s["visit_id"] == "2" ]

        self.assertEqual(len(readmitted5days), 1)
        self.assertEqual(len(readmitted15days), 1)

        self.assertFalse(bool(readmitted5days[0].item()))
        self.assertTrue (bool(readmitted15days[0].item()))

        readmitted5days = [ s["readmission"] for s in self.samples5days if s["visit_id"] == "3" ]
        readmitted15days = [ s["readmission"] for s in self.samples15days if s["visit_id"] == "3" ]

        self.assertEqual(len(readmitted5days), 1)
        self.assertEqual(len(readmitted15days), 1)

        self.assertTrue(bool(readmitted5days[0].item()))
        self.assertTrue(bool(readmitted15days[0].item()))

    def test_last_admission_is_excluded(self):
        self.assertNotIn("4", [ s["visit_id"] for s in self.samples15days ])
        self.assertNotIn("4", [ s["visit_id"] for s in self.samples5days ])
        self.assertNotIn("4", [ s["visit_id"] for s in self.sampleswithminors ])

    def test_patient_with_only_one_visit_is_excluded(self):
        patients = [ s["patient_id"] for s in self.samples15days ]
        visits = [ s["visit_id"] for s in self.samples15days ]

        self.assertNotIn("2", patients)
        self.assertNotIn("5", visits)

        patients = [ s["patient_id"] for s in self.samples5days ]
        visits = [ s["visit_id"] for s in self.samples5days ]

        self.assertNotIn("2", patients)
        self.assertNotIn("5", visits)

        patients = [ s["patient_id"] for s in self.sampleswithminors ]
        visits = [ s["visit_id"] for s in self.sampleswithminors ]

        self.assertNotIn("2", patients)
        self.assertNotIn("5", visits)

    def test_admissions_without_drugs_are_excluded(self):
        conditions_and_procedures_but_no_drugs = {
            "-7891579270619134632",
            "-4740650079733515392",
            "-5667932875078290562",
            "-7351550472011464089",
            "-6400688276878690493",
            "535331186584817863",
            "-6990694520804042573",
        }

        visits = set([ s["visit_id"] for s in self.samples15days ])
        visits.update([ s["visit_id"] for s in self.samples5days ])
        visits.update([ s["visit_id"] for s in self.sampleswithminors ])

        self.assertTrue(visits.isdisjoint(conditions_and_procedures_but_no_drugs))

    def test_admissions_of_minors_are_excluded(self):
        self.assertNotIn("1", [ s["visit_id"] for s in self.samples15days ])
        self.assertNotIn("1", [ s["visit_id"] for s in self.samples5days ])
        self.assertIn   ("1", [ s["visit_id"] for s in self.sampleswithminors ])


if __name__ == "__main__":
    unittest.main()
