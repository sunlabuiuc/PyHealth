
"""Synthetic tests for diagnosis-focused evidence retrieval.

These tests reuse the temporal eligibility and deduplication
methodology from the original research implementation.

No protected MIMIC-III data or model downloads are required.
"""

import unittest
from datetime import datetime

import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import DiagnosisEvidenceRetrievalMIMIC3
from pyhealth.tasks._diagnosis_evidence_utils import (
    eligible_history,
    deduplicate_history,
    choose_history,
    make_samples,
    digest,
)


REPORT = {
    "report_id": "100",
    "subject_id": "1",
    "hadm_id": "20",
    "report_time": "2100-01-02 12:00:00",
    "diagnosis": "stroke",
}


def make_note(row_id="1", **kwargs):
    """Create a synthetic historical clinical note."""
    note = {
        "row_id": row_id,
        "subject_id": "1",
        "hadm_id": None,
        "charttime": "2100-01-01 09:00:00",
        "chartdate": "2100-01-01",
        "storetime": "2100-01-01 10:00:00",
        "text": "History of hypertension.",
        "iserror": None,
        "category": "Nursing",
        "description": "Synthetic note",
    }

    note.update(kwargs)
    return note


def make_index_report(**kwargs):
    """Create the reviewed index radiology report."""
    report = make_note(
        row_id="100",
        hadm_id="20",
        charttime=REPORT["report_time"],
        chartdate="2100-01-02",
        storetime=REPORT["report_time"],
        text="FINAL REPORT: Findings consistent with stroke.",
        category="Radiology",
        description="CT HEAD",
    )

    report.update(kwargs)
    return report


def make_patient(
    notes,
    admission_type="EMERGENCY",
    edregtime="2100-01-02 08:00:00",
    patient_id="1",
):
    """Build a real PyHealth Patient from synthetic events."""
    rows = [
        {
            "patient_id": patient_id,
            "event_type": "admissions",
            "timestamp": datetime(2100, 1, 2, 8),
            "admissions/hadm_id": "20",
            "admissions/admission_type": admission_type,
            "admissions/edregtime": edregtime,
        }
    ]

    for note in notes:
        charttime_missing = not note.get("charttime")

        timestamp = datetime.fromisoformat(
            note["chartdate"]
            if charttime_missing
            else note["charttime"]
        )

        row = {
            "patient_id": patient_id,
            "event_type": "noteevents",
            "timestamp": timestamp,
            "noteevents/charttime_missing": charttime_missing,
        }

        for key in (
            "row_id",
            "hadm_id",
            "chartdate",
            "storetime",
            "iserror",
            "text",
            "category",
            "description",
        ):
            row[f"noteevents/{key}"] = note.get(key)

        rows.append(row)

    return Patient(
        patient_id=patient_id,
        data_source=pl.DataFrame(
            rows,
            infer_schema_length=None,
        ),
    )


def make_task(queries=None, **kwargs):
    """Create a task with small synthetic cohort thresholds."""
    settings = {
        "min_prior_notes": 1,
        "strict_availability": True,
        "unknown_availability": "exclude",
        "deduplication": "exact",
    }

    settings.update(kwargs)

    return DiagnosisEvidenceRetrievalMIMIC3(
        reviewed_queries=[REPORT] if queries is None else queries,
        **settings,
    )


def note_ids(notes):
    """Return source IDs for selected notes."""
    return [note["row_id"] for note in notes]


class TestDiagnosisEvidenceUtilities(unittest.TestCase):
    """Regression tests adapted from the original research code."""

    def test_temporal_boundaries_and_errors(self):
        notes = [
            make_note("1"),
            make_note(
                "2",
                charttime=None,
                chartdate="2100-01-02",
            ),
            make_note(
                "3",
                charttime=None,
                chartdate="2100-01-01",
            ),
            make_note("4", charttime=REPORT["report_time"]),
            make_note("5", storetime="2100-01-03"),
            make_note("6", iserror="1"),
            make_note("7", text=""),
            make_note("8", subject_id="2"),
            make_note("9", storetime=REPORT["report_time"]),
            make_note("100"),
            make_note("10", text=None),
        ]

        result = eligible_history(notes, REPORT)

        self.assertEqual(
            set(note_ids(result)),
            {"1", "3"},
        )

    def test_relaxed_storage_time_filter(self):
        notes = [
            make_note("1"),
            make_note("2", storetime="2100-01-03"),
        ]

        result = eligible_history(
            notes,
            REPORT,
            strict_availability=False,
        )

        self.assertEqual(note_ids(result), ["1", "2"])

    def test_unknown_availability_policies(self):
        notes = [
            make_note("1"),
            make_note("2", storetime=None),
        ]

        for policy, expected in (
            ("exclude", ["1"]),
            ("retain", ["1", "2"]),
        ):
            with self.subTest(policy=policy):
                result = eligible_history(
                    notes,
                    REPORT,
                    unknown_availability=policy,
                )

                self.assertEqual(note_ids(result), expected)

                if policy == "retain":
                    self.assertTrue(
                        result[-1]["availability_unknown"]
                    )

    def test_identical_and_conflicting_note_ids(self):
        note = make_note()

        result = eligible_history([note, note], REPORT)

        self.assertEqual(len(result), 1)

        conflicting = dict(note, text="Conflicting text.")

        with self.assertRaisesRegex(
            ValueError,
            "Conflicting duplicate",
        ):
            eligible_history([note, conflicting], REPORT)

    def test_exact_deduplication(self):
        notes = [
            make_note("1"),
            make_note(
                "2",
                charttime="2100-01-01 11:00:00",
                text="History  of hypertension.",
            ),
            make_note("3", text="Patient reports headache."),
        ]

        history = eligible_history(notes, REPORT)
        result, audit = deduplicate_history(history, "exact")

        self.assertEqual(note_ids(result), ["1", "3"])
        self.assertEqual(audit["exact_duplicates_removed"], 1)
        self.assertEqual(result[0]["duplicate_note_ids"], ["2"])

    def test_no_text_deduplication(self):
        history = eligible_history(
            [make_note("1"), make_note("2")],
            REPORT,
        )

        result, audit = deduplicate_history(history, "none")

        self.assertEqual(len(result), 2)
        self.assertEqual(audit["exact_duplicates_removed"], 0)

    def test_known_availability_preferred_for_duplicates(self):
        notes = [
            make_note("1", storetime=None),
            make_note(
                "2",
                charttime="2100-01-01 11:00:00",
            ),
        ]

        history = eligible_history(
            notes,
            REPORT,
            unknown_availability="retain",
        )

        result, _ = deduplicate_history(history, "exact")

        self.assertEqual(note_ids(result), ["2"])

    def test_report_version_deduplication(self):
        notes = [
            make_note(
                "1",
                category="Radiology",
                description="CT HEAD",
                text="Clip #123 PFI REPORT: Preliminary finding.",
            ),
            make_note(
                "2",
                category="Radiology",
                description="CT HEAD",
                text="Clip #123 FINAL REPORT: Final finding.",
            ),
        ]

        history = eligible_history(notes, REPORT)

        result, audit = deduplicate_history(
            history,
            "report_versions",
        )

        self.assertEqual(note_ids(result), ["2"])
        self.assertEqual(audit["report_versions_removed"], 1)
        self.assertEqual(result[0]["superseded_note_ids"], ["1"])

    def test_history_selection(self):
        notes = [
            make_note(
                str(i),
                charttime=f"2100-01-01 0{i}:00:00",
                text=f"Distinct clinical note {i}.",
            )
            for i in range(1, 5)
        ]

        history = eligible_history(notes, REPORT)

        for order, expected in (
            ("recent", ["3", "4"]),
            ("earliest", ["1", "2"]),
        ):
            with self.subTest(order=order):
                selected = choose_history(history, 2, order)

                self.assertEqual(note_ids(selected), expected)

        self.assertEqual(len(choose_history(history)), 4)

    def test_unique_patient_diagnosis(self):
        reviews = [
            REPORT,
            dict(REPORT, report_id="101", diagnosis="Stroke"),
        ]

        samples = make_samples(
            reviews,
            [make_note()],
            min_prior_notes=1,
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["report_id"], "100")

    def test_minimum_notes_after_deduplication(self):
        samples = make_samples(
            [REPORT],
            [make_note("1"), make_note("2")],
            min_prior_notes=2,
            deduplication="exact",
        )

        self.assertEqual(samples, [])


class TestDiagnosisEvidenceRetrievalMIMIC3(unittest.TestCase):
    """Test native Patient-to-sample conversion."""

    def setUp(self):
        self.notes = [
            make_note("1"),
            make_note(
                "2",
                charttime="2100-01-01 11:00:00",
                text="Patient reports headache.",
            ),
            make_index_report(),
        ]

        self.patient = make_patient(self.notes)

    def test_native_patient_sample(self):
        task = make_task(min_prior_notes=2)
        samples = task(self.patient)

        self.assertEqual(len(samples), 1)

        sample = samples[0]

        self.assertEqual(sample["patient_id"], "1")
        self.assertEqual(sample["diagnosis"], "stroke")
        self.assertEqual(sample["note_ids"], ["1", "2"])
        self.assertEqual(sample["prior_note_count"], 2)
        self.assertEqual(sample["index_note_id"], "100")
        self.assertEqual(task.output_schema, {})

        self.assertEqual(
            sample["historical_notes"],
            [
                "History of hypertension.",
                "Patient reports headache.",
            ],
        )

        self.assertEqual(
            len(sample["note_metadata"]),
            len(sample["historical_notes"]),
        )

        self.assertNotIn("report_text", sample)

    def test_minimum_and_maximum_notes(self):
        self.assertEqual(
            make_task(min_prior_notes=3)(self.patient),
            [],
        )

        sample = make_task(max_notes=1)(self.patient)[0]

        self.assertEqual(sample["note_ids"], ["2"])
        self.assertEqual(sample["prior_note_count"], 2)

    def test_date_only_note_provenance(self):
        notes = [
            make_note("1", charttime=None),
            make_note(
                "2",
                charttime=None,
                chartdate="2100-01-02",
                text="Same-day note.",
            ),
            make_index_report(),
        ]

        sample = make_task()(make_patient(notes))[0]

        self.assertEqual(sample["note_ids"], ["1"])
        self.assertTrue(
            sample["note_metadata"][0]["date_only"]
        )
        self.assertIsNone(
            sample["note_metadata"][0]["charttime"]
        )

    def test_deduplication_before_history_limit(self):
        notes = [
            make_note("1", text="Earlier distinct history."),
            make_note(
                "2",
                charttime="2100-01-01 10:00:00",
                text="Repeated history.",
            ),
            make_note(
                "3",
                charttime="2100-01-01 11:00:00",
                text="Repeated history.",
            ),
            make_note(
                "4",
                text="Unknown availability.",
                storetime=None,
            ),
            make_index_report(),
        ]

        sample = make_task(
            min_prior_notes=2,
            max_notes=2,
        )(make_patient(notes))[0]

        self.assertEqual(sample["note_ids"], ["1", "2"])

        self.assertEqual(
            sample["history_audit"]["exact_duplicates_removed"],
            1,
        )

        self.assertEqual(
            sample["history_audit"]["unknown_availability_excluded"],
            1,
        )

    def test_admission_eligibility(self):
        for admission_type in ("EMERGENCY", "URGENT"):
            with self.subTest(admission_type=admission_type):
                patient = make_patient(
                    self.notes,
                    admission_type=admission_type,
                )

                self.assertEqual(len(make_task()(patient)), 1)

        for admission_type in ("ELECTIVE", "NEWBORN"):
            with self.subTest(admission_type=admission_type):
                patient = make_patient(
                    self.notes,
                    admission_type=admission_type,
                )

                self.assertEqual(make_task()(patient), [])

    def test_ed_window(self):
        self.assertEqual(
            make_task(window_hours=4)(self.patient)[0]["diagnosis"],
            "stroke",
        )

        self.assertEqual(
            make_task(window_hours=2)(self.patient),
            [],
        )

        patient = make_patient(
            self.notes,
            edregtime="2100-01-02 13:00:00",
        )

        self.assertEqual(make_task()(patient), [])

    def test_missing_ed_registration(self):
        patient = make_patient(
            self.notes,
            edregtime=None,
        )

        self.assertEqual(make_task()(patient), [])

    def test_evidence_types(self):
        for evidence_type in ("risk", "sign"):
            with self.subTest(evidence_type=evidence_type):
                task = make_task(evidence_type=evidence_type)
                sample = task(self.patient)[0]

                self.assertEqual(
                    sample["evidence_type"],
                    evidence_type,
                )

    def test_correct_report_hash(self):
        query = dict(
            REPORT,
            report_hash=digest(self.notes[-1]["text"]),
        )

        samples = make_task([query])(self.patient)

        self.assertEqual(len(samples), 1)

    def test_incorrect_report_hash(self):
        query = dict(
            REPORT,
            report_hash=digest("Modified report."),
        )

        with self.assertRaisesRegex(
            ValueError,
            "text hash mismatch",
        ):
            make_task([query])(self.patient)

    def test_missing_report_with_hash(self):
        query = dict(
            REPORT,
            report_hash=digest(self.notes[-1]["text"]),
        )

        patient = make_patient(self.notes[:-1])

        with self.assertRaisesRegex(
            ValueError,
            "target report is missing",
        ):
            make_task([query])(patient)

    def test_missing_report_without_hash(self):
        patient = make_patient(self.notes[:-1])

        self.assertEqual(make_task()(patient), [])

    def test_report_timestamp_mismatch(self):
        query = dict(
            REPORT,
            report_time="2100-01-02 11:00:00",
        )

        with self.assertRaisesRegex(
            ValueError,
            "timestamp mismatch",
        ):
            make_task([query])(self.patient)

    def test_report_admission_mismatch(self):
        query = dict(REPORT, hadm_id="999")

        with self.assertRaisesRegex(
            ValueError,
            "admission ID mismatch",
        ):
            make_task([query])(self.patient)

    def test_patient_without_query(self):
        self.assertEqual(make_task([])(self.patient), [])

    def test_invalid_configuration(self):
        cases = [
            {"min_prior_notes": 0},
            {"window_hours": 0},
            {"min_prior_notes": 3, "max_notes": 2},
            {"deduplication": "invalid"},
            {"history_order": "invalid"},
            {"unknown_availability": "invalid"},
            {"evidence_type": "invalid"},
        ]

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    make_task(**kwargs)


if __name__ == "__main__":
    unittest.main()
