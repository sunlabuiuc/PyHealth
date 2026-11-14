"""Tests for the next-visit diagnosis prediction tasks.

Contributor: Alon Levy (NetID: alonl2, alonl2@illinois.edu)
Paper: "Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction"
    (Yu et al., CHIL 2025), https://arxiv.org/abs/2502.10689

All patients are small synthetic in-memory records; no MIMIC data is read.
"""

import unittest
from datetime import UTC, datetime

import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import (
    NextVisitDiagnosisPredictionMIMIC3,
    NextVisitDiagnosisPredictionMIMIC4,
)
from pyhealth.tasks.next_visit_diagnosis_prediction import truncate_icd9

# (hadm_id, admittime, [codes]); for MIMIC-IV codes are (code, icd_version).
Visit = tuple[str, datetime, list]


def make_patient(patient_id: str, visits: list[Visit], mimic4: bool = False) -> Patient:
    """Builds a synthetic Patient with admissions and diagnoses_icd events."""
    rows: list[dict[str, object | None]] = []
    for hadm_id, admittime, codes in visits:
        rows.append(
            {
                "event_type": "admissions",
                "timestamp": admittime,
                "admissions/hadm_id": hadm_id,
            }
        )
        for code in codes:
            row = {
                "event_type": "diagnoses_icd",
                "timestamp": admittime,
                "diagnoses_icd/hadm_id": hadm_id,
            }
            if mimic4:
                row["diagnoses_icd/icd_code"] = code[0]
                row["diagnoses_icd/icd_version"] = code[1]
            else:
                row["diagnoses_icd/icd9_code"] = code
            rows.append(row)
    return Patient(patient_id=patient_id, data_source=pl.DataFrame(rows))


T1 = datetime(2100, 1, 1, tzinfo=UTC)
T2 = datetime(2100, 1, 11, tzinfo=UTC)
T3 = datetime(2100, 3, 1, tzinfo=UTC)


class TestNextVisitDiagnosisPredictionMIMIC3(unittest.TestCase):
    def setUp(self):
        self.patient = make_patient(
            "p1",
            [
                ("h1", T1, ["4280", "25000"]),
                ("h2", T2, ["4019", "4280"]),
                ("h3", T3, ["5849", "E8809"]),
            ],
        )

    def test_schema(self):
        task = NextVisitDiagnosisPredictionMIMIC3()
        self.assertEqual(task.input_schema, {"conditions": "nested_sequence"})
        self.assertEqual(task.output_schema, {"next_conditions": "multilabel"})

    def test_last_visit_is_label(self):
        samples = NextVisitDiagnosisPredictionMIMIC3()(self.patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "p1")
        self.assertEqual(sample["conditions"], [["4280", "25000"], ["4019", "4280"]])
        self.assertEqual(sample["next_conditions"], ["5849", "E8809"])

    def test_single_visit_patient_dropped(self):
        patient = make_patient("p2", [("h1", T1, ["4280"])])
        self.assertEqual(NextVisitDiagnosisPredictionMIMIC3()(patient), [])

    def test_unsorted_admissions_are_sorted(self):
        patient = make_patient(
            "p3", [("h3", T3, ["3"]), ("h1", T1, ["1"]), ("h2", T2, ["2"])]
        )
        sample = NextVisitDiagnosisPredictionMIMIC3()(patient)[0]
        self.assertEqual(sample["conditions"], [["1"], ["2"]])
        self.assertEqual(sample["next_conditions"], ["3"])

    def test_float_and_int_hadm_ids_match(self):
        patient = make_patient("p7", [(100, T1, ["1"]), (200, T2, ["2"])])
        patient.data_source = patient.data_source.with_columns(
            pl.col("diagnoses_icd/hadm_id").cast(pl.Float64)
        )
        patient = Patient("p7", patient.data_source)
        sample = NextVisitDiagnosisPredictionMIMIC3()(patient)[0]
        self.assertEqual(sample["conditions"], [["1"]])
        self.assertEqual(sample["next_conditions"], ["2"])

    def test_admission_without_time_is_skipped(self):
        patient = make_patient(
            "p8", [("h1", T1, ["1"]), ("h2", None, ["2"]), ("h3", T3, ["3"])]
        )
        sample = NextVisitDiagnosisPredictionMIMIC3()(patient)[0]
        self.assertEqual(sample["conditions"], [["1"]])
        self.assertEqual(sample["next_conditions"], ["3"])

    def test_visit_without_codes_is_skipped(self):
        patient = make_patient(
            "p4", [("h1", T1, ["1"]), ("h2", T2, []), ("h3", T3, ["3"])]
        )
        sample = NextVisitDiagnosisPredictionMIMIC3()(patient)[0]
        self.assertEqual(sample["conditions"], [["1"]])
        self.assertEqual(sample["next_conditions"], ["3"])

    def test_empty_codes_only_patient_dropped(self):
        patient = make_patient("p5", [("h1", T1, ["1"]), ("h2", T2, [])])
        self.assertEqual(NextVisitDiagnosisPredictionMIMIC3()(patient), [])

    def test_duplicate_codes_removed(self):
        patient = make_patient("p6", [("h1", T1, ["1", "1", "2"]), ("h2", T2, ["3"])])
        sample = NextVisitDiagnosisPredictionMIMIC3()(patient)[0]
        self.assertEqual(sample["conditions"], [["1", "2"]])

    def test_max_visits_keeps_most_recent(self):
        sample = NextVisitDiagnosisPredictionMIMIC3(max_visits=1)(self.patient)[0]
        self.assertEqual(sample["conditions"], [["4019", "4280"]])

    def test_min_visits(self):
        task = NextVisitDiagnosisPredictionMIMIC3(min_visits=4)
        self.assertEqual(task(self.patient), [])

    def test_three_digit_codes(self):
        task = NextVisitDiagnosisPredictionMIMIC3(code_level="3digit")
        sample = task(self.patient)[0]
        self.assertEqual(sample["conditions"], [["428", "250"], ["401", "428"]])
        self.assertEqual(sample["next_conditions"], ["584", "E880"])

    def test_all_prefixes(self):
        task = NextVisitDiagnosisPredictionMIMIC3(sample_mode="all_prefixes")
        samples = task(self.patient)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["conditions"], [["4280", "25000"]])
        self.assertEqual(samples[0]["next_conditions"], ["4019", "4280"])
        self.assertEqual(len(samples[1]["conditions"]), 2)

    def test_visit_gaps(self):
        task = NextVisitDiagnosisPredictionMIMIC3(include_visit_gaps=True)
        self.assertEqual(task.input_schema["visit_gaps"], "tensor")
        sample = task(self.patient)[0]
        self.assertEqual(sample["visit_gaps"], [0.0, 10.0])
        # The optional feature must not leak into other instances.
        self.assertNotIn(
            "visit_gaps", NextVisitDiagnosisPredictionMIMIC3().input_schema
        )

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            NextVisitDiagnosisPredictionMIMIC3(min_visits=1)
        with self.assertRaises(ValueError):
            NextVisitDiagnosisPredictionMIMIC3(max_visits=0)
        with self.assertRaises(ValueError):
            NextVisitDiagnosisPredictionMIMIC3(code_level="chapter")
        with self.assertRaises(ValueError):
            NextVisitDiagnosisPredictionMIMIC3(sample_mode="random")

    def test_truncate_icd9(self):
        self.assertEqual(truncate_icd9("4280"), "428")
        self.assertEqual(truncate_icd9("V5861"), "V58")
        self.assertEqual(truncate_icd9("E8809"), "E880")
        self.assertEqual(truncate_icd9("428.0"), "428")


class TestNextVisitDiagnosisPredictionMIMIC4(unittest.TestCase):
    def test_icd10_rows_dropped(self):
        patient = make_patient(
            "p1",
            [
                ("h1", T1, [("4280", "9"), ("I509", "10")]),
                ("h2", T2, [("I10", "10")]),  # ICD-10 only: visit dropped
                ("h3", T3, [("5849", "9")]),
            ],
            mimic4=True,
        )
        samples = NextVisitDiagnosisPredictionMIMIC4()(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["conditions"], [["4280"]])
        self.assertEqual(samples[0]["next_conditions"], ["5849"])

    def test_icd10_only_patient_dropped(self):
        patient = make_patient(
            "p2",
            [("h1", T1, [("I10", "10")]), ("h2", T2, [("I509", "10")])],
            mimic4=True,
        )
        self.assertEqual(NextVisitDiagnosisPredictionMIMIC4()(patient), [])


if __name__ == "__main__":
    unittest.main()
