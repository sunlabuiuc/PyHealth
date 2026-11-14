from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import polars as pl

from pyhealth.data import Patient
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import ReadmissionPredictionMIMIC4


class TestReadmissionPredictionMIMIC4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_dir = tempfile.TemporaryDirectory()

        dataset = MIMIC4Dataset(
            ehr_root=str(Path(__file__).parent.parent.parent / "test-resources" / "core" / "mimic4demo"),
            ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=cls.cache_dir.name,
        )

        cls.samples15days = dataset.set_task(ReadmissionPredictionMIMIC4(window=timedelta(days=15)))
        cls.samples5days = dataset.set_task(ReadmissionPredictionMIMIC4(window=timedelta(days=5)))
        cls.samples5daymingap = dataset.set_task(
            ReadmissionPredictionMIMIC4(
                window=timedelta(days=15),
                min_gap=timedelta(days=5),
            )
        )
        cls.sampleswithminors = dataset.set_task(ReadmissionPredictionMIMIC4(exclude_minors=False))

    @classmethod
    def tearDownClass(cls):
        cls.samples15days.close()
        cls.samples5days.close()
        cls.samples5daymingap.close()
        cls.sampleswithminors.close()

    def test_task_schema(self):
        self.assertIn("task_name", vars(ReadmissionPredictionMIMIC4))
        self.assertIn("input_schema", vars(ReadmissionPredictionMIMIC4))
        self.assertIn("output_schema", vars(ReadmissionPredictionMIMIC4))

        self.assertEqual("ReadmissionPredictionMIMIC4", ReadmissionPredictionMIMIC4.task_name)
        self.assertIn("conditions", ReadmissionPredictionMIMIC4.input_schema)
        self.assertIn("procedures", ReadmissionPredictionMIMIC4.input_schema)
        self.assertIn("drugs", ReadmissionPredictionMIMIC4.input_schema)
        self.assertIn("readmission", ReadmissionPredictionMIMIC4.output_schema)

    def test_task_defaults(self):
        task = ReadmissionPredictionMIMIC4()

        self.assertEqual(task.window, timedelta(days=15))
        self.assertTrue(task.exclude_minors)
        self.assertIsNone(task.min_gap)

    def test_positional_arguments_backward_compatible(self):
        task = ReadmissionPredictionMIMIC4(timedelta(days=30), False)

        self.assertEqual(task.window, timedelta(days=30))
        self.assertFalse(task.exclude_minors)
        self.assertIsNone(task.min_gap)

    def test_sample_schema(self):
        for sample in self.samples15days:
            self.assertIn("visit_id", sample)
            self.assertIn("patient_id", sample)
            self.assertIn("conditions", sample)
            self.assertIn("procedures", sample)
            self.assertIn("drugs", sample)
            self.assertIn("readmission", sample)

    def test_time_window(self):
        readmitted5days = [ s["readmission"] for s in self.samples5days if s["visit_id"] == "3" ]
        readmitted15days = [ s["readmission"] for s in self.samples15days if s["visit_id"] == "3" ]

        self.assertEqual(len(readmitted5days), 1)
        self.assertEqual(len(readmitted15days), 1)

        self.assertFalse(bool(readmitted5days[0].item()))
        self.assertTrue (bool(readmitted15days[0].item()))

        readmitted5days = [ s["readmission"] for s in self.samples5days if s["visit_id"] == "4" ]
        readmitted15days = [ s["readmission"] for s in self.samples15days if s["visit_id"] == "4" ]

        self.assertEqual(len(readmitted5days), 1)
        self.assertEqual(len(readmitted15days), 1)

        self.assertTrue(bool(readmitted5days[0].item()))
        self.assertTrue(bool(readmitted15days[0].item()))

    def test_min_gap(self):
        # Patient 2: visit 3, then 4 about 15 days later, then 5 and 6 each
        # within 5 days of the previous discharge. With min_gap=5 days, 4, 5
        # and 6 form one stay, which is the last, so only visit 3 gives a
        # sample; its next admission (4) is within the 15-day window.
        readmitted3 = [
            s["readmission"]
            for s in self.samples5daymingap
            if s["visit_id"] == "3"
        ]
        self.assertEqual(len(readmitted3), 1)
        self.assertTrue(bool(readmitted3[0].item()))
        self.assertNotIn("4", [s["visit_id"] for s in self.samples5daymingap])

    def test_last_admission_is_excluded(self):
        self.assertNotIn("2", [ s["visit_id"] for s in self.samples15days ])
        self.assertNotIn("2", [ s["visit_id"] for s in self.samples5days ])
        self.assertNotIn("2", [ s["visit_id"] for s in self.sampleswithminors ])

        self.assertNotIn("6", [ s["visit_id"] for s in self.samples15days ])
        self.assertNotIn("6", [ s["visit_id"] for s in self.samples5days ])
        self.assertNotIn("6", [ s["visit_id"] for s in self.sampleswithminors ])

    def test_patient_with_only_one_visit_is_excluded(self):
        self.assertNotIn("3", [ s["patient_id"] for s in self.samples15days ])
        self.assertNotIn("7", [ s["visit_id"] for s in self.samples15days ])

        self.assertNotIn("3", [ s["patient_id"] for s in self.samples5days ])
        self.assertNotIn("7", [ s["visit_id"] for s in self.samples5days ])

        self.assertNotIn("3", [ s["patient_id"] for s in self.sampleswithminors ])
        self.assertNotIn("7", [ s["visit_id"] for s in self.sampleswithminors ])

    def test_admissions_without_drugs_are_excluded(self):
        self.assertNotIn("5", [ s["visit_id"] for s in self.samples15days ])
        self.assertNotIn("5", [ s["visit_id"] for s in self.samples5days ])
        self.assertNotIn("5", [ s["visit_id"] for s in self.sampleswithminors ])

    def test_admissions_of_minors_are_excluded(self):
        self.assertNotIn("1", [ s["visit_id"] for s in self.samples15days ])
        self.assertNotIn("1", [ s["visit_id"] for s in self.samples5days ])
        self.assertIn   ("1", [ s["visit_id"] for s in self.sampleswithminors ])


def _patient(admissions):
    """A patient with the given ``(hadm_id, admit, discharge)`` admissions.

    Each admission has one diagnosis, procedure and prescription coded by
    its hadm_id, so a stay's codes show which admissions it merged.
    """
    rows = [{"event_type": "patients", "timestamp": datetime(2150, 1, 1),
             "patients/anchor_age": "60"}]
    for hadm, admit, discharge in admissions:
        rows.append({"event_type": "admissions", "timestamp": admit,
                     "admissions/hadm_id": hadm,
                     "admissions/dischtime": discharge.strftime("%Y-%m-%d %H:%M:%S")})
        for table, code_col, code in (
            ("diagnoses_icd", "icd_code", f"D{hadm}"),
            ("procedures_icd", "icd_code", f"P{hadm}"),
            ("prescriptions", "ndc", f"N{hadm}"),
        ):
            row = {"event_type": table, "timestamp": admit,
                   f"{table}/hadm_id": hadm, f"{table}/{code_col}": code}
            if table != "prescriptions":
                row[f"{table}/icd_version"] = "10"
            rows.append(row)
    return Patient("p", pl.DataFrame(rows))


class TestReadmissionTransfers(unittest.TestCase):
    """A -> B (a transfer 2 hours after A's discharge) -> C (8 days later)."""

    T = datetime(2150, 1, 1)

    def setUp(self):
        t = self.T
        self.patient = _patient([
            ("A", t, t + timedelta(days=1)),
            ("B", t + timedelta(days=1, hours=2), t + timedelta(days=3)),
            ("C", t + timedelta(days=11), t + timedelta(days=12)),
        ])

    def test_transfer_is_merged_into_its_stay(self):
        task = ReadmissionPredictionMIMIC4(window=timedelta(days=15),
                                           min_gap=timedelta(hours=6))
        samples = task(self.patient)
        self.assertEqual([s["visit_id"] for s in samples], ["A"])  # no sample for B
        self.assertEqual(samples[0]["readmission"], 1)  # C is 8 days after B
        self.assertEqual(samples[0]["conditions"], ["10_DA", "10_DB"])
        self.assertEqual(samples[0]["drugs"], ["NA", "NB"])

    def test_merged_stay_without_readmission(self):
        t = self.T
        patient = _patient([
            ("A", t, t + timedelta(days=1)),
            ("B", t + timedelta(days=1, hours=2), t + timedelta(days=3)),
            ("C", t + timedelta(days=60), t + timedelta(days=61)),
        ])
        task = ReadmissionPredictionMIMIC4(window=timedelta(days=15),
                                           min_gap=timedelta(hours=6))
        samples = task(patient)
        self.assertEqual([(s["visit_id"], s["readmission"]) for s in samples],
                         [("A", 0)])

    def test_without_min_gap_a_transfer_is_a_readmission(self):
        samples = ReadmissionPredictionMIMIC4(window=timedelta(days=15))(self.patient)
        self.assertEqual([(s["visit_id"], s["readmission"]) for s in samples],
                         [("A", 1), ("B", 1)])


if __name__ == "__main__":
    unittest.main()
