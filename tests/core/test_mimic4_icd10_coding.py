import unittest
from datetime import datetime

import polars as pl

from pyhealth.data.data import Patient
from pyhealth.processors import SequenceProcessor, MultiLabelProcessor
from pyhealth.tasks.medical_coding import MIMIC4ICD10Coding, _tokenize_clinical_text


def _make_patient(patient_id, rows):
    """Build a Patient from a list of row dicts.

    Each row must have at least event_type and timestamp. Extra keys
    become columns prefixed with event_type/ (how Patient.get_events
    reconstructs Event.attr_dict).
    """
    records = []
    for row in rows:
        et = row["event_type"]
        ts = row.get("timestamp", datetime(2024, 1, 1))
        rec = {"patient_id": patient_id, "event_type": et, "timestamp": ts}
        for k, v in row.items():
            if k not in ("event_type", "timestamp"):
                rec[f"{et}/{k}"] = v
        records.append(rec)
    df = pl.DataFrame(records)
    return Patient(patient_id=patient_id, data_source=df)


class TestMIMIC4ICD10Coding(unittest.TestCase):
    """Tests for the MIMIC4ICD10Coding task."""

    def test_task_schema_types(self):
        """Schema uses SequenceProcessor for text and MultiLabelProcessor for codes."""
        task = MIMIC4ICD10Coding()
        self.assertIs(task.input_schema["text"], SequenceProcessor)
        self.assertIs(task.output_schema["icd_codes"], MultiLabelProcessor)

    def test_basic_extraction(self):
        """One admission with ICD-10 codes and a discharge note produces one sample."""
        task = MIMIC4ICD10Coding()
        patient = _make_patient("p1", [
            {"event_type": "admissions", "hadm_id": "100"},
            {"event_type": "diagnoses_icd", "hadm_id": "100",
             "icd_code": "E11.321", "icd_version": "10"},
            {"event_type": "diagnoses_icd", "hadm_id": "100",
             "icd_code": "I10", "icd_version": "10"},
            {"event_type": "discharge", "hadm_id": "100",
             "text": "Patient discharged with diabetes and hypertension."},
        ])
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["patient_id"], "p1")
        self.assertIsInstance(samples[0]["text"], list)
        self.assertIn("diabetes", samples[0]["text"])
        self.assertEqual(sorted(samples[0]["icd_codes"]), ["E11.321", "I10"])

    def test_filters_out_icd9(self):
        """ICD-9 codes in the same table are excluded from ICD-10 task output."""
        task = MIMIC4ICD10Coding()
        patient = _make_patient("p2", [
            {"event_type": "admissions", "hadm_id": "200"},
            {"event_type": "diagnoses_icd", "hadm_id": "200",
             "icd_code": "E11.321", "icd_version": "10"},
            {"event_type": "diagnoses_icd", "hadm_id": "200",
             "icd_code": "25000", "icd_version": "9"},
            {"event_type": "discharge", "hadm_id": "200",
             "text": "Discharge summary text here."},
        ])
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["icd_codes"], ["E11.321"])

    def test_no_notes_skips_admission(self):
        """Admission without discharge notes produces no sample."""
        task = MIMIC4ICD10Coding()
        patient = _make_patient("p3", [
            {"event_type": "admissions", "hadm_id": "300"},
            {"event_type": "diagnoses_icd", "hadm_id": "300",
             "icd_code": "I10", "icd_version": "10"},
            # discharge note on a different admission so the column exists
            {"event_type": "discharge", "hadm_id": "999",
             "text": "unrelated"},
        ])
        samples = task(patient)
        self.assertEqual(len(samples), 0)

    def test_no_codes_skips_admission(self):
        """Admission with notes but no ICD codes produces no sample."""
        task = MIMIC4ICD10Coding()
        patient = _make_patient("p4", [
            {"event_type": "admissions", "hadm_id": "400"},
            {"event_type": "discharge", "hadm_id": "400",
             "text": "Patient seen and discharged."},
            # diagnosis on a different admission so the column exists
            {"event_type": "diagnoses_icd", "hadm_id": "999",
             "icd_code": "X00", "icd_version": "10"},
        ])
        samples = task(patient)
        self.assertEqual(len(samples), 0)

    def test_multiple_admissions(self):
        """Each admission produces its own sample with correct codes."""
        task = MIMIC4ICD10Coding()
        patient = _make_patient("p5", [
            {"event_type": "admissions", "hadm_id": "500"},
            {"event_type": "admissions", "hadm_id": "501"},
            {"event_type": "diagnoses_icd", "hadm_id": "500",
             "icd_code": "J44.1", "icd_version": "10"},
            {"event_type": "diagnoses_icd", "hadm_id": "501",
             "icd_code": "K21.0", "icd_version": "10"},
            {"event_type": "discharge", "hadm_id": "500",
             "text": "First admission discharge."},
            {"event_type": "discharge", "hadm_id": "501",
             "text": "Second admission discharge."},
        ])
        samples = task(patient)
        self.assertEqual(len(samples), 2)
        codes_by_admission = {s["icd_codes"][0] for s in samples}
        self.assertEqual(codes_by_admission, {"J44.1", "K21.0"})

    def test_duplicate_codes_deduplicated(self):
        """Duplicate ICD codes for the same admission are deduplicated."""
        task = MIMIC4ICD10Coding()
        patient = _make_patient("p6", [
            {"event_type": "admissions", "hadm_id": "600"},
            {"event_type": "diagnoses_icd", "hadm_id": "600",
             "icd_code": "I10", "icd_version": "10"},
            {"event_type": "diagnoses_icd", "hadm_id": "600",
             "icd_code": "I10", "icd_version": "10"},
            {"event_type": "discharge", "hadm_id": "600",
             "text": "Discharge note."},
        ])
        samples = task(patient)
        self.assertEqual(len(samples[0]["icd_codes"]), 1)

    def test_text_is_lowercased(self):
        """Output tokens are lowercased."""
        task = MIMIC4ICD10Coding()
        patient = _make_patient("p7", [
            {"event_type": "admissions", "hadm_id": "700"},
            {"event_type": "diagnoses_icd", "hadm_id": "700",
             "icd_code": "E11.321", "icd_version": "10"},
            {"event_type": "discharge", "hadm_id": "700",
             "text": "Patient Has DIABETES."},
        ])
        samples = task(patient)
        for token in samples[0]["text"]:
            self.assertEqual(token, token.lower())


class TestTokenizer(unittest.TestCase):
    """Tests for _tokenize_clinical_text."""

    def test_lowercases(self):
        self.assertEqual(_tokenize_clinical_text("Hello World"), ["hello", "world"])

    def test_truncates(self):
        long_text = " ".join(f"word{i}" for i in range(5000))
        tokens = _tokenize_clinical_text(long_text)
        self.assertEqual(len(tokens), 4000)

    def test_empty_string(self):
        self.assertEqual(_tokenize_clinical_text(""), [])


if __name__ == "__main__":
    unittest.main()
