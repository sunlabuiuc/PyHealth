"""
Unit tests for the QueryFocusedEHRTask class.

Tests use synthetic clinical records to verify temporal filtering,
sentence extraction, diagnosis labels, and history-window behavior.

No access to MIMIC-III is required.
"""

import unittest
from datetime import datetime

import polars as pl
import csv
import gzip
import tempfile
from pathlib import Path
import gc

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.data import Patient
from pyhealth.tasks.query_focused_ehr import (
    QueryFocusedEHRTask,
    split_clinical_sentences,
)


class TestQueryFocusedEHRTask(unittest.TestCase):
    """Tests for query-focused EHR sample construction."""

    def setUp(self):
        self.diagnoses = {
            "41519": "Pulmonary embolism",
            "486": "Pneumonia",
        }

        self.t1 = datetime(2020, 1, 1)
        self.t2 = datetime(2020, 4, 10)
        self.t3 = datetime(2020, 5, 1)

    def make_patient(self, rows):
        return Patient(
            patient_id="patient_1",
            data_source=pl.DataFrame(
                rows,
                infer_schema_length=None,
            ),
        )

    def admission(self, visit_id, timestamp):
        return {
            "event_type": "admissions",
            "timestamp": timestamp,
            "admissions/hadm_id": str(visit_id),
        }

    def note(
        self,
        visit_id,
        timestamp,
        text,
        storetime=None,
        category=None,
    ):
        row = {
            "event_type": "noteevents",
            "timestamp": timestamp,
            "noteevents/hadm_id": str(visit_id),
            "noteevents/text": text,
        }

        if storetime is not None:
            row["noteevents/storetime"] = storetime

        if category is not None:
            row["noteevents/category"] = category

        return row

    def diagnosis(self, visit_id, timestamp, code):
        return {
            "event_type": "diagnoses_icd",
            "timestamp": timestamp,
            "diagnoses_icd/hadm_id": str(visit_id),
            "diagnoses_icd/icd9_code": str(code),
        }

    def test_sentence_extraction(self):
        text = (
            "History of DVT. Reports chest pain.\n"
            "Shortness of breath."
        )

        sentences = split_clinical_sentences(text)

        self.assertEqual(
            sentences,
            [
                "History of DVT.",
                "Reports chest pain.",
                "Shortness of breath.",
            ],
        )

    def test_future_diagnosis_and_history_cutoff(self):
        patient = self.make_patient([
            self.admission("100", self.t1),
            self.note(
                "100",
                self.t1,
                "History of DVT. Reports shortness of breath.",
            ),
            self.admission("200", self.t3),
            self.note(
                "200",
                self.t3,
                "FUTURE NOTE MUST BE EXCLUDED.",
            ),
            self.diagnosis("200", self.t3, "41519"),
        ])

        task = QueryFocusedEHRTask(self.diagnoses)
        samples = task(patient)

        self.assertEqual(len(samples), 2)

        labels = {
            sample["diagnosis_code"]: sample["label"]
            for sample in samples
        }

        self.assertEqual(labels["41519"], 1)
        self.assertEqual(labels["486"], 0)

        for sample in samples:
            self.assertEqual(
                sample["sentences"],
                [
                    "History of DVT.",
                    "Reports shortness of breath.",
                ],
            )

            self.assertNotIn(
                "FUTURE NOTE MUST BE EXCLUDED.",
                sample["sentences"],
            )


    def test_grouped_diagnosis_positive(self):
        """Any member ICD code should produce a positive group label."""

        patient = self.make_patient([
            self.admission("100", self.t1),
            self.note(
                "100",
                self.t1,
                "History of DVT.",
            ),
            self.admission("200", self.t3),
            self.diagnosis("200", self.t3, "41512"),
        ])

        task = QueryFocusedEHRTask(
            diagnosis_groups={
                "pulmonary_embolism": {
                    "description": "Pulmonary embolism",
                    "codes": ["41511", "41512", "41519"],
                },
                "pneumonia": {
                    "description": "Pneumonia",
                    "codes": ["486"],
                },
            }
        )

        samples = task(patient)

        labels = {
            sample["diagnosis_code"]: sample["label"]
            for sample in samples
        }

        self.assertEqual(
            labels,
            {
                "pulmonary_embolism": 1,
                "pneumonia": 0,
            },
        )


    def test_negative_query_selection(self):
        """Only never-observed diagnoses should be negative."""

        patient = self.make_patient([
            self.admission("100", self.t1),
            self.note(
                "100",
                self.t1,
                "Previous clinical history.",
            ),
            self.diagnosis("100", self.t1, "486"),
            self.admission("200", self.t2),
            self.note(
                "200",
                self.t2,
                "Additional clinical history.",
            ),
            self.diagnosis("200", self.t2, "41519"),
            self.admission("300", self.t3),
            self.diagnosis("300", self.t3, "4280"),
        ])

        task = QueryFocusedEHRTask(
            diagnosis_groups={
                "pulmonary_embolism": {
                    "description": "Pulmonary embolism",
                    "codes": ["41519"],
                },
                "pneumonia": {
                    "description": "Pneumonia",
                    "codes": ["486"],
                },
                "heart_failure": {
                    "description": "Heart failure",
                    "codes": ["4280"],
                },
                "sepsis": {
                    "description": "Sepsis",
                    "codes": ["0389"],
                },
            }
        )

        samples = task(patient)

        by_visit = {}

        for sample in samples:
            by_visit.setdefault(
                sample["visit_id"], {}
            )[sample["diagnosis_code"]] = sample["label"]

        # Admission 200:
        # Pulmonary embolism is positive.
        # Sepsis is never observed and therefore negative.
        # Pneumonia and heart failure are ambiguous.
        self.assertEqual(
            by_visit["200"],
            {
                "pulmonary_embolism": 1,
                "sepsis": 0,
            },
        )

        # Admission 300:
        # Heart failure is positive.
        # Sepsis remains negative.
        # Pulmonary embolism and pneumonia are ambiguous.
        self.assertEqual(
            by_visit["300"],
            {
                "heart_failure": 1,
                "sepsis": 0,
            },
        )

    def test_icd9_normalization(self):
        """Dotted and undotted ICD-9 codes must match."""

        patient = self.make_patient([
            self.admission("100", self.t1),
            self.note(
                "100",
                self.t1,
                "Previous clinical history.",
            ),
            self.admission("200", self.t3),
            self.diagnosis("200", self.t3, "415.19"),
        ])

        task = QueryFocusedEHRTask(
            diagnosis_groups={
                "pulmonary_embolism": {
                    "description": "Pulmonary embolism",
                    "codes": ["41519"],
                },
            }
        )

        samples = task(patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["label"], 1)


    def test_invalid_diagnosis_groups(self):
        """Reject invalid grouped-diagnosis configurations."""

        invalid_groups = [
            {},
            {
                "pulmonary_embolism": {
                    "description": "",
                    "codes": ["41519"],
                }
            },
            {
                "pulmonary_embolism": {
                    "description": "Pulmonary embolism",
                    "codes": [],
                }
            },
            {
                "pulmonary_embolism": {
                    "description": "Pulmonary embolism",
                    "codes": ["41519", ""],
                }
            },
        ]

        for groups in invalid_groups:
            with self.subTest(groups=groups):
                with self.assertRaises(ValueError):
                    QueryFocusedEHRTask(
                        diagnosis_groups=groups
                    )

        with self.assertRaises(ValueError):
            QueryFocusedEHRTask(
                diagnosis_descriptions={
                    "486": "Pneumonia"
                },
                diagnosis_groups={
                    "pneumonia": {
                        "description": "Pneumonia",
                        "codes": ["486"],
                    }
                },
            )


    def test_single_admission(self):
        patient = self.make_patient([
            self.admission("100", self.t1),
            self.note(
                "100",
                self.t1,
                "Patient reports chest pain.",
            ),
            self.diagnosis("100", self.t1, "41519"),
        ])

        task = QueryFocusedEHRTask(self.diagnoses)

        self.assertEqual(task(patient), [])

    def test_history_windows(self):
        patient = self.make_patient([
            self.admission("100", self.t1),
            self.note("100", self.t1, "Old symptom."),
            self.admission("200", self.t2),
            self.note("200", self.t2, "Recent symptom."),
            self.admission("300", self.t3),
            self.diagnosis("300", self.t3, "41519"),
        ])

        expectations = {
            30: ["Recent symptom."],
            90: ["Recent symptom."],
            365: ["Old symptom.", "Recent symptom."],
            None: ["Old symptom.", "Recent symptom."],
        }

        for days, expected in expectations.items():
            with self.subTest(history_days=days):
                task = QueryFocusedEHRTask(
                    self.diagnoses,
                    history_days=days,
                )

                samples = task(patient)

                target_samples = [
                    sample for sample in samples
                    if sample["visit_id"] == "300"
                ]

                self.assertEqual(len(target_samples), 2)

                for sample in target_samples:
                    self.assertEqual(
                        sample["sentences"],
                        expected,
                    )

    def test_multiple_admissions(self):
        """Generate positives and exclude ambiguous negatives."""

        patient = self.make_patient([
            self.admission("100", self.t1),
            self.note("100", self.t1, "Initial symptom."),
            self.admission("200", self.t2),
            self.note("200", self.t2, "Second symptom."),
            self.diagnosis("200", self.t2, "486"),
            self.admission("300", self.t3),
            self.diagnosis("300", self.t3, "41519"),
        ])

        task = QueryFocusedEHRTask(self.diagnoses)
        samples = task(patient)

        # Both diagnosis groups occur somewhere in the
        # patient record, so neither can be a negative.
        self.assertEqual(len(samples), 2)

        labels = {
            (sample["visit_id"], sample["diagnosis_code"]):
            sample["label"]
            for sample in samples
        }

        self.assertEqual(
            labels,
            {
                ("200", "486"): 1,
                ("300", "41519"): 1,
            },
        )

    def test_missing_timestamps(self):
        patient = self.make_patient([
            self.admission("100", self.t1),
            self.note(
                "100",
                self.t1,
                "Valid historical note.",
            ),
            self.note(
                "100",
                None,
                "MISSING TIMESTAMP NOTE.",
            ),
            self.admission("200", self.t3),
            self.diagnosis("200", self.t3, "41519"),
        ])

        task = QueryFocusedEHRTask(self.diagnoses)
        samples = task(patient)

        self.assertEqual(len(samples), 2)

        for sample in samples:
            self.assertEqual(
                sample["sentences"],
                ["Valid historical note."],
            )

    def test_late_storetime(self):
        late_time = datetime(2020, 5, 2)

        patient = self.make_patient([
            self.admission("100", self.t1),
            self.note(
                "100",
                self.t1,
                "Valid earlier note.",
                storetime=self.t1,
            ),
            self.note(
                "100",
                self.t1,
                "LATE STORED NOTE.",
                storetime=late_time,
            ),
            self.admission("200", self.t3),
            self.diagnosis("200", self.t3, "41519"),
        ])

        task = QueryFocusedEHRTask(self.diagnoses)
        samples = task(patient)

        self.assertEqual(len(samples), 2)

        for sample in samples:
            self.assertEqual(
                sample["sentences"],
                ["Valid earlier note."],
            )

    def test_max_sentences(self):
        patient = self.make_patient([
            self.admission("100", self.t1),
            self.note(
                "100",
                self.t1,
                "First sentence. Second sentence. "
                "Third sentence. Fourth sentence.",
            ),
            self.admission("200", self.t3),
            self.diagnosis("200", self.t3, "41519"),
        ])

        task = QueryFocusedEHRTask(
            self.diagnoses,
            max_sentences=2,
        )

        samples = task(patient)

        self.assertEqual(len(samples), 2)

        for sample in samples:
            self.assertEqual(
                sample["sentences"],
                [
                    "Third sentence.",
                    "Fourth sentence.",
                ],
            )

    def test_note_category_filter(self):
        patient = self.make_patient([
            self.admission("100", self.t1),
            self.note(
                "100",
                self.t1,
                "Radiology observation.",
                category="Radiology",
            ),
            self.note(
                "100",
                self.t1,
                "Nursing observation.",
                category="Nursing",
            ),
            self.admission("200", self.t3),
            self.diagnosis("200", self.t3, "41519"),
        ])

        task = QueryFocusedEHRTask(
            self.diagnoses,
            note_categories=["Radiology"],
        )

        samples = task(patient)

        self.assertEqual(len(samples), 2)

        for sample in samples:
            self.assertEqual(
                sample["sentences"],
                ["Radiology observation."],
            )

    def test_minimum_sentences(self):
        patient = self.make_patient([
            self.admission("100", self.t1),
            self.note(
                "100",
                self.t1,
                "Single sentence.",
            ),
            self.admission("200", self.t3),
            self.diagnosis("200", self.t3, "41519"),
        ])

        task = QueryFocusedEHRTask(
            self.diagnoses,
            min_sentences=2,
        )

        self.assertEqual(task(patient), [])

    def test_invalid_configuration(self):
        """Reject invalid task parameters and unsupported strategies."""

        invalid_configs = [
            {"diagnosis_descriptions": {}},
            {
                "diagnosis_descriptions": self.diagnoses,
                "history_days": 0,
            },
            {
                "diagnosis_descriptions": self.diagnoses,
                "history_days": -1,
            },
            {
                "diagnosis_descriptions": self.diagnoses,
                "max_sentences": 0,
            },
            {
                "diagnosis_descriptions": self.diagnoses,
                "min_sentences": 0,
            },
            {
                "diagnosis_descriptions": self.diagnoses,
                "max_sentences": 2,
                "min_sentences": 3,
            },
            {
                "diagnosis_descriptions": self.diagnoses,
                "negative_sampling": "unsupported_mode",
            },
        ]

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    QueryFocusedEHRTask(**config)


    def test_datetime_parser(self):
        """Parse valid timestamps and reject invalid values."""
        from pyhealth.tasks.query_focused_ehr import (
            parse_clinical_datetime,
        )

        self.assertEqual(
            parse_clinical_datetime("2020-03-02 10:00:00"),
            datetime(2020, 3, 2, 10, 0),
        )

        self.assertEqual(
            parse_clinical_datetime("2020-03-02T10:00:00"),
            datetime(2020, 3, 2, 10, 0),
        )

        self.assertIsNone(
            parse_clinical_datetime("invalid-timestamp")
        )

    def test_negative_sampling_modes(self):
        """Verify label behavior for both negative selection modes."""

        patient = self.make_patient([
            self.admission("100", self.t1),
            self.note("100", self.t1, "Earlier history."),
            self.diagnosis("100", self.t1, "486"),
            self.admission("200", self.t3),
            self.diagnosis("200", self.t3, "41519"),
        ])

        expected = {
            "never_observed": {
                "41519": 1,
            },
            "target_absent": {
                "41519": 1,
                "486": 0,
            },
        }

        for mode, expected_labels in expected.items():
            with self.subTest(mode=mode):
                task = QueryFocusedEHRTask(
                    diagnosis_descriptions=self.diagnoses,
                    negative_sampling=mode,
                )

                samples = task(patient)

                labels = {
                    sample["diagnosis_code"]: sample["label"]
                    for sample in samples
                }

                self.assertEqual(labels, expected_labels)


class TestSetTask(unittest.TestCase):
    """Integration tests using synthetic MIMIC-III CSV tables.

    Tests the real MIMIC3Dataset -> set_task -> SampleDataset
    pipeline without downloading restricted clinical records.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls.tmp.name) / "mimic3"
        cls.root.mkdir()

        # All five tables are expected by the MIMIC3Dataset
        # configuration when using diagnoses and notes.
        tables = {
            "PATIENTS": (
                [
                    "SUBJECT_ID", "GENDER", "DOB", "DOD",
                    "DOD_HOSP", "DOD_SSN", "EXPIRE_FLAG",
                ],
                [
                    {
                        "SUBJECT_ID": "1",
                        "GENDER": "F",
                        "DOB": "1970-01-01 00:00:00",
                        "EXPIRE_FLAG": "0",
                    },
                ],
            ),
            "ADMISSIONS": (
                [
                    "SUBJECT_ID", "HADM_ID", "ADMITTIME",
                    "DISCHTIME", "ADMISSION_TYPE",
                    "ADMISSION_LOCATION", "INSURANCE",
                    "LANGUAGE", "RELIGION", "MARITAL_STATUS",
                    "ETHNICITY", "EDREGTIME", "EDOUTTIME",
                    "DEATHTIME", "DIAGNOSIS",
                    "DISCHARGE_LOCATION", "HOSPITAL_EXPIRE_FLAG",
                ],
                [
                    {
                        "SUBJECT_ID": "1",
                        "HADM_ID": "100",
                        "ADMITTIME": "2020-01-01 08:00:00",
                        "DISCHTIME": "2020-01-05 12:00:00",
                        "ADMISSION_TYPE": "EMERGENCY",
                        "DIAGNOSIS": "Chest pain",
                        "HOSPITAL_EXPIRE_FLAG": "0",
                    },
                    {
                        "SUBJECT_ID": "1",
                        "HADM_ID": "200",
                        "ADMITTIME": "2020-03-01 08:00:00",
                        "DISCHTIME": "2020-03-05 12:00:00",
                        "ADMISSION_TYPE": "EMERGENCY",
                        "DIAGNOSIS": "Pulmonary embolism",
                        "HOSPITAL_EXPIRE_FLAG": "0",
                    },
                ],
            ),
            "ICUSTAYS": (
                [
                    "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID",
                    "INTIME", "OUTTIME", "FIRST_CAREUNIT",
                    "LAST_CAREUNIT", "DBSOURCE",
                ],
                [
                    {
                        "SUBJECT_ID": "1",
                        "HADM_ID": "100",
                        "ICUSTAY_ID": "1000",
                        "INTIME": "2020-01-01 10:00:00",
                        "OUTTIME": "2020-01-04 10:00:00",
                        "FIRST_CAREUNIT": "MICU",
                        "LAST_CAREUNIT": "MICU",
                        "DBSOURCE": "metavision",
                    },
                ],
            ),
            "NOTEEVENTS": (
                [
                    "ROW_ID", "SUBJECT_ID", "HADM_ID",
                    "CHARTDATE", "CHARTTIME", "STORETIME",
                    "CATEGORY", "DESCRIPTION", "TEXT",
                ],
                [
                    {
                        "ROW_ID": "1",
                        "SUBJECT_ID": "1",
                        "HADM_ID": "100",
                        "CHARTDATE": "2020-01-02",
                        "CHARTTIME": "2020-01-02 09:00:00",
                        "STORETIME": "2020-01-02 10:00:00",
                        "CATEGORY": "Physician",
                        "DESCRIPTION": "Progress note",
                        "TEXT": (
                            "History of DVT. "
                            "Patient reports chest pain."
                        ),
                    },
                    {
                        "ROW_ID": "2",
                        "SUBJECT_ID": "1",
                        "HADM_ID": "200",
                        "CHARTDATE": "2020-03-02",
                        "CHARTTIME": "2020-03-02 09:00:00",
                        "STORETIME": "2020-03-02 10:00:00",
                        "CATEGORY": "Physician",
                        "DESCRIPTION": "Progress note",
                        "TEXT": "FUTURE NOTE MUST BE EXCLUDED.",
                    },
                ],
            ),
            "DIAGNOSES_ICD": (
                [
                    "ROW_ID", "SUBJECT_ID", "HADM_ID",
                    "SEQ_NUM", "ICD9_CODE",
                ],
                [
                    {
                        "ROW_ID": "1",
                        "SUBJECT_ID": "1",
                        "HADM_ID": "200",
                        "SEQ_NUM": "1",
                        "ICD9_CODE": "41519",
                    },
                ],
            ),
        }

        for table_name, (columns, rows) in tables.items():
            path = cls.root / f"{table_name}.csv.gz"

            with gzip.open(
                path, "wt", encoding="utf-8", newline=""
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=columns,
                )
                writer.writeheader()
                writer.writerows(rows)

        cls.dataset = MIMIC3Dataset(
            root=str(cls.root),
            tables=["noteevents", "diagnoses_icd"],
            cache_dir=Path(cls.tmp.name) / "cache",
            num_workers=1,
        )

        cls.task = QueryFocusedEHRTask(
            diagnosis_groups={
                "pulmonary_embolism": {
                    "description": "Pulmonary embolism",
                    "codes": ["41511", "41512", "41519"],
                },
                "pneumonia": {
                    "description": "Pneumonia",
                    "codes": ["486"],
                },
            }
        )

        cls.samples = cls.dataset.set_task(
            cls.task,
            num_workers=1,
        )

    @classmethod
    def tearDownClass(cls):
        """Release PyHealth resources before deleting test files."""
        if hasattr(cls, "samples"):
            cls.samples.close()
            del cls.samples

        if hasattr(cls, "dataset"):
            del cls.dataset

        if hasattr(cls, "task"):
            del cls.task

        gc.collect()

        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def test_samples_are_created(self):
        self.assertEqual(len(self.samples), 2)

    def test_diagnosis_labels(self):
        labels = {
            sample["query"]: int(sample["label"].item())
            for sample in self.samples
        }

        self.assertEqual(
            labels,
            {
                "Pulmonary embolism": 1,
                "Pneumonia": 0,
            },
        )

    def test_historical_sentences(self):
        for sample in self.samples:
            self.assertEqual(
                sample["sentences"],
                [
                    "History of DVT.",
                    "Patient reports chest pain.",
                ],
            )

            self.assertNotIn(
                "FUTURE NOTE MUST BE EXCLUDED.",
                sample["sentences"],
            )

    def test_patient_identity(self):
        for sample in self.samples:
            self.assertEqual(
                str(sample["patient_id"]),
                "1",
            )
            self.assertEqual(
                str(sample["visit_id"]),
                "200",
            )

if __name__ == "__main__":
    unittest.main()
