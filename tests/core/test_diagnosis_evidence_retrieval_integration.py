"""End-to-end tests for MIMIC-III diagnosis evidence retrieval.

All clinical records are synthetic. These tests require no protected
MIMIC-III data, external model downloads or GPU resources.
"""

import csv
import gc
import gzip
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import DiagnosisEvidenceRetrievalMIMIC3
from pyhealth.tasks._diagnosis_evidence_utils import digest

REPORT_TEXT = "FINAL REPORT: Findings consistent with stroke."

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "pyhealth"
    / "datasets"
    / "configs"
    / "mimic3.yaml"
)


def write_table(root, filename, columns, rows):
    """Write a synthetic MIMIC-III compressed CSV table."""
    with gzip.open(
        root / filename,
        "wt",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def create_synthetic_mimic3(root):
    """Create the minimum MIMIC-III tables required by the task."""
    write_table(
        root,
        "PATIENTS.csv.gz",
        [
            "row_id",
            "subject_id",
            "gender",
            "dob",
            "dod",
            "dod_hosp",
            "dod_ssn",
            "expire_flag",
        ],
        [
            [1, 100, "F", "2090-01-01 00:00:00", "", "", "", 0],
            [2, 101, "M", "2085-01-01 00:00:00", "", "", "", 0],
        ],
    )

    write_table(
        root,
        "ADMISSIONS.csv.gz",
        [
            "row_id",
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "religion",
            "marital_status",
            "ethnicity",
            "edregtime",
            "edouttime",
            "diagnosis",
            "hospital_expire_flag",
            "has_chartevents_data",
        ],
        [
            [
                i,
                subject,
                hadm,
                "2150-01-02 08:00:00",
                "2150-01-10 12:00:00",
                "",
                "EMERGENCY",
                "EMERGENCY ROOM ADMIT",
                "HOME",
                "Medicare",
                "ENGL",
                "NOT SPECIFIED",
                "SINGLE",
                "UNKNOWN",
                "2150-01-02 08:00:00",
                "2150-01-02 09:00:00",
                "STROKE",
                0,
                1,
            ]
            for i, subject, hadm in [
                (1, 100, 200),
                (2, 101, 201),
            ]
        ],
    )

    write_table(
        root,
        "ICUSTAYS.csv.gz",
        [
            "row_id",
            "subject_id",
            "hadm_id",
            "icustay_id",
            "dbsource",
            "first_careunit",
            "last_careunit",
            "first_wardid",
            "last_wardid",
            "intime",
            "outtime",
            "los",
        ],
        [
            [
                i,
                subject,
                hadm,
                icu,
                "metavision",
                "MICU",
                "MICU",
                50,
                50,
                "2150-01-02 09:00:00",
                "2150-01-04 09:00:00",
                2.0,
            ]
            for i, subject, hadm, icu in [
                (1, 100, 200, 300),
                (2, 101, 201, 301),
            ]
        ],
    )

    write_table(
        root,
        "NOTEEVENTS.csv.gz",
        [
            "row_id",
            "subject_id",
            "hadm_id",
            "chartdate",
            "charttime",
            "storetime",
            "category",
            "description",
            "cgid",
            "iserror",
            "text",
        ],
        [
            # Patient 100: eligible linked historical note.
            [
                101,
                100,
                200,
                "2150-01-01",
                "2150-01-01 09:00:00",
                "2150-01-01 10:00:00",
                "Nursing",
                "Synthetic",
                1,
                "",
                "History of hypertension.",
            ],
            # Eligible admission-less, date-only historical note.
            [
                102,
                100,
                "",
                "2150-01-01",
                "",
                "2150-01-01 11:00:00",
                "Nursing",
                "Synthetic",
                1,
                "",
                "History of smoking.",
            ],
            # Genuine midnight timestamp, not an imputed time.
            [
                103,
                100,
                200,
                "2150-01-02",
                "2150-01-02 00:00:00",
                "2150-01-02 01:00:00",
                "Nursing",
                "Synthetic",
                1,
                "",
                "Patient reports headache.",
            ],
            # Charted before imaging but stored afterward.
            [
                104,
                100,
                200,
                "2150-01-02",
                "2150-01-02 10:00:00",
                "2150-01-02 13:00:00",
                "Nursing",
                "Synthetic",
                1,
                "",
                "Late stored clinical note.",
            ],
            # Reviewed index radiology report.
            [
                105,
                100,
                200,
                "2150-01-02",
                "2150-01-02 12:00:00",
                "2150-01-02 12:00:00",
                "Radiology",
                "CT HEAD",
                1,
                "",
                REPORT_TEXT,
            ],
            # Same-day date-only note: temporally ambiguous.
            [
                106,
                100,
                200,
                "2150-01-02",
                "",
                "2150-01-02 09:00:00",
                "Nursing",
                "Synthetic",
                1,
                "",
                "Ambiguous same-day clinical note.",
            ],
            # Patient 101: one eligible historical note.
            [
                201,
                101,
                201,
                "2150-01-01",
                "2150-01-01 09:00:00",
                "2150-01-01 10:00:00",
                "Nursing",
                "Synthetic",
                1,
                "",
                "History of atrial fibrillation.",
            ],
            # Patient 101: reviewed index report.
            [
                205,
                101,
                201,
                "2150-01-02",
                "2150-01-02 12:00:00",
                "2150-01-02 12:00:00",
                "Radiology",
                "CT HEAD",
                1,
                "",
                REPORT_TEXT,
            ],
        ],
    )


def create_longitudinal_config(target):
    """Generate an opt-in configuration retaining unlinked notes."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for join in config["tables"]["noteevents"]["join"]:
        if join["on"] == "hadm_id":
            join["how"] = "left"

    with target.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def make_task(**kwargs):
    """Create a retrieval task for two reviewed synthetic diagnoses."""
    queries = [
        {
            "patient_id": "100",
            "index_note_id": "105",
            "diagnosis": "stroke",
            "report_hash": digest(REPORT_TEXT),
        },
        {
            "patient_id": "101",
            "index_note_id": "205",
            "diagnosis": "stroke",
            "report_hash": digest(REPORT_TEXT),
        },
    ]

    settings = {
        "min_prior_notes": 1,
        "max_notes": 40,
        "strict_availability": True,
        "unknown_availability": "exclude",
        "deduplication": "exact",
        "history_order": "recent",
        "evidence_type": "risk",
    }

    settings.update(kwargs)

    return DiagnosisEvidenceRetrievalMIMIC3(
        reviewed_queries=queries,
        **settings,
    )


class TestDiagnosisEvidenceRetrievalIntegration(unittest.TestCase):
    """Validate real dataset loading and processed sample storage."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=(os.name == "nt"))
        cls.addClassCleanup(cls.tmp.cleanup)

        base = Path(cls.tmp.name)

        cls.root = base / "mimic3"
        cls.root.mkdir()

        cls.cache = base / "cache"

        create_synthetic_mimic3(cls.root)

        cls.config_path = base / "longitudinal.yaml"
        create_longitudinal_config(cls.config_path)

        cls.standard = MIMIC3Dataset(
            root=str(cls.root),
            tables=["noteevents"],
            cache_dir=cls.cache,
            num_workers=1,
        )

        cls.longitudinal = MIMIC3Dataset(
            root=str(cls.root),
            tables=["noteevents"],
            config_path=str(cls.config_path),
            cache_dir=cls.cache,
            num_workers=1,
        )

    @classmethod
    def tearDownClass(cls):
        """Release dataset references before temporary cache cleanup."""
        cls.standard = None
        cls.longitudinal = None

        gc.collect()

    def test_dataset_cache_isolation(self):
        """Different join policies must use separate event caches."""
        self.assertNotEqual(
            self.standard.cache_dir,
            self.longitudinal.cache_dir,
        )

        self.assertEqual(
            self.standard.config.tables["noteevents"].join[0].how,
            "inner",
        )

        self.assertEqual(
            self.longitudinal.config.tables["noteevents"].join[0].how,
            "left",
        )

    def test_unlinked_note_retention(self):
        """The opt-in configuration must retain admission-less notes."""
        standard_patient = self.standard.get_patient("100")
        longitudinal_patient = self.longitudinal.get_patient("100")

        standard_ids = {
            str(event.row_id) for event in standard_patient.get_events("noteevents")
        }

        longitudinal_ids = {
            str(event.row_id) for event in longitudinal_patient.get_events("noteevents")
        }

        self.assertNotIn("102", standard_ids)
        self.assertIn("102", longitudinal_ids)

        self.assertEqual(
            longitudinal_ids - standard_ids,
            {"102"},
        )

    def test_charttime_provenance(self):
        """Distinguish imputed midnight from recorded midnight."""
        patient = self.longitudinal.get_patient("100")

        events = {
            str(event.row_id): event for event in patient.get_events("noteevents")
        }

        self.assertEqual(events["102"].charttime_missing, "true")
        self.assertEqual(events["103"].charttime_missing, "false")
        self.assertEqual(events["106"].charttime_missing, "true")

        self.assertEqual(events["103"].timestamp.hour, 0)
        self.assertEqual(events["106"].timestamp.hour, 0)

        self.assertEqual(
            events["103"].timestamp,
            events["106"].timestamp,
        )

    def test_retrieval_from_loaded_patient(self):
        """Apply temporal filtering to events loaded from real CSVs."""
        task = make_task()

        patient = self.longitudinal.get_patient("100")
        samples = task(patient)

        self.assertEqual(len(samples), 1)

        sample = samples[0]

        self.assertEqual(
            set(sample["note_ids"]),
            {"101", "102", "103"},
        )

        self.assertNotIn("104", sample["note_ids"])
        self.assertNotIn("105", sample["note_ids"])
        self.assertNotIn("106", sample["note_ids"])

        self.assertEqual(sample["prior_note_count"], 3)

        # Note 102 must retain its original missing chart time.
        metadata = {note["row_id"]: note for note in sample["note_metadata"]}

        self.assertTrue(metadata["102"]["date_only"])
        self.assertIsNone(metadata["102"]["charttime"])

        # Genuine midnight must not be treated as date-only.
        self.assertFalse(metadata["103"]["date_only"])

    def test_standard_and_longitudinal_retrieval(self):
        """Verify the practical impact of retaining unlinked notes."""
        task = make_task()

        standard_sample = task(self.standard.get_patient("100"))[0]

        longitudinal_sample = task(self.longitudinal.get_patient("100"))[0]

        self.assertEqual(
            set(standard_sample["note_ids"]),
            {"101", "103"},
        )

        self.assertEqual(
            set(longitudinal_sample["note_ids"]),
            {"101", "102", "103"},
        )

    def test_set_task_round_trip(self):
        """Verify complete sample construction and disk storage."""
        task = make_task()

        # Direct task results serve as the reference.
        expected = {}

        for patient_id in ("100", "101"):
            patient = self.longitudinal.get_patient(patient_id)
            expected[patient_id] = task(patient)[0]

        # Native PyHealth processing pipeline.
        samples = self.longitudinal.set_task(
            task,
            num_workers=1,
        )

        self.assertEqual(len(samples), 2)
        self.assertEqual(task.output_schema, {})

        observed = {
            str(samples[i]["patient_id"]): samples[i] for i in range(len(samples))
        }

        self.assertEqual(set(observed), {"100", "101"})

        for patient_id, reference in expected.items():
            with self.subTest(patient_id=patient_id):
                sample = observed[patient_id]

                for field in task.input_schema:
                    self.assertEqual(
                        sample[field],
                        reference[field],
                    )

        # Confirm variable-length histories survive disk storage.
        self.assertEqual(
            len(observed["100"]["historical_notes"]),
            3,
        )

        self.assertEqual(
            len(observed["101"]["historical_notes"]),
            1,
        )

        # Read from the existing task cache again.
        cached = self.longitudinal.set_task(
            task,
            num_workers=1,
        )

        self.assertEqual(len(cached), 2)

        for i in range(len(cached)):
            sample = cached[i]
            patient_id = str(sample["patient_id"])

            self.assertEqual(
                sample["historical_notes"],
                expected[patient_id]["historical_notes"],
            )

        # Explicitly release LitData resources.
        samples.close()
        cached.close()

        del samples
        del cached
        gc.collect()

    def test_task_cache_isolation(self):
        """Different task parameters must not share sample caches."""
        risk_task = make_task(evidence_type="risk")
        sign_task = make_task(evidence_type="sign")
        limited_task = make_task(max_notes=1)

        cache_paths = {
            self.longitudinal._task_cache_dir(risk_task),
            self.longitudinal._task_cache_dir(sign_task),
            self.longitudinal._task_cache_dir(limited_task),
        }

        self.assertEqual(len(cache_paths), 3)


if __name__ == "__main__":
    unittest.main()
