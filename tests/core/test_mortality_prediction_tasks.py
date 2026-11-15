"""Mortality tasks run on the MIMIC-IV demo and on hand-built patients."""

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import (
    MortalityPredictionMIMIC4,
    MultimodalMortalityPredictionMIMIC3,
)

DEMO = Path(__file__).parents[2] / "test-resources/core/mimic4demo"


class FakePatient:
    """Events by type, filtered on attributes like ``Patient.get_events``."""

    def __init__(self, events, patient_id="p1"):
        self.patient_id = patient_id
        self.events = events

    def get_events(self, event_type=None, start=None, end=None, filters=None):
        events = list(self.events.get(event_type, []))
        for attr, _, value in filters or []:
            events = [e for e in events if getattr(e, attr) == value]
        if start is not None:
            events = [e for e in events if e.timestamp >= start]
        if end is not None:
            events = [e for e in events if e.timestamp <= end]
        return events


def event(timestamp=None, **attributes):
    return SimpleNamespace(timestamp=timestamp, **attributes)


class TestMortalityPredictionMIMIC4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = tempfile.TemporaryDirectory()
        dataset = MIMIC4Dataset(
            ehr_root=str(DEMO),
            ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=cls.cache.name,
        )
        cls.patients = list(dataset.iter_patients())

    @classmethod
    def tearDownClass(cls):
        cls.cache.cleanup()

    def test_samples_on_the_demo(self):
        task = MortalityPredictionMIMIC4()
        samples = [s for p in self.patients for s in task(p)]
        self.assertGreater(len(samples), 0)
        for sample in samples:
            self.assertIn(sample["mortality"], (0, 1))
            for key in ("conditions", "procedures", "drugs"):
                self.assertTrue(sample[key])
                self.assertTrue(all(isinstance(c, str) and c for c in sample[key]))
        # one sample per admission that has a next admission, at most
        for patient in self.patients:
            n_admissions = len(patient.get_events(event_type="admissions"))
            self.assertLessEqual(len(task(patient)), max(n_admissions - 1, 0))

    def test_label_comes_from_the_next_admission(self):
        t0 = datetime(2150, 1, 1)  # noqa: DTZ001
        admissions = [
            event(
                t0 + timedelta(days=30 * i),
                hadm_id=f"h{i}",
                dischtime=(t0 + timedelta(days=30 * i + 2)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                hospital_expire_flag=flag,
            )
            for i, flag in enumerate(["0", "0", "1"])
        ]
        codes = {
            table: [
                event(a.timestamp + timedelta(hours=1), icd_code="X", ndc="N")
                for a in admissions
            ]
            for table in ("diagnoses_icd", "procedures_icd", "prescriptions")
        }
        patient = FakePatient(
            {"patients": [event(t0, anchor_age=60)], "admissions": admissions, **codes}
        )
        samples = MortalityPredictionMIMIC4()(patient)
        self.assertEqual([s["visit_id"] for s in samples], ["h0", "h1"])
        self.assertEqual([s["mortality"] for s in samples], [0, 1])

    def test_skips_minors_and_single_admissions(self):
        t0 = datetime(2150, 1, 1)  # noqa: DTZ001
        admission = event(
            t0, hadm_id="h0", dischtime="2150-01-02 00:00:00", hospital_expire_flag=0
        )
        task = MortalityPredictionMIMIC4()
        minor = FakePatient(
            {"patients": [event(t0, anchor_age=12)], "admissions": [admission] * 2}
        )
        self.assertEqual(task(minor), [])
        single = FakePatient(
            {"patients": [event(t0, anchor_age=40)], "admissions": [admission]}
        )
        self.assertEqual(task(single), [])
        self.assertEqual(task(FakePatient({})), [])


class TestMultimodalMortalityPredictionMIMIC3(unittest.TestCase):
    def test_codes_notes_and_label_per_visit(self):
        admissions = [
            event(hadm_id="h0", hospital_expire_flag=0),
            event(hadm_id="h1", hospital_expire_flag="1"),
            event(hadm_id="h2", hospital_expire_flag=None),
        ]
        patient = FakePatient(
            {
                "admissions": admissions,
                "diagnoses_icd": [event(hadm_id="h0", icd9_code="4019")],
                "procedures_icd": [event(hadm_id="h1", icd9_code="3893")],
                "prescriptions": [
                    event(hadm_id="h0", ndc="0001"),
                    event(hadm_id="h0", ndc=""),
                ],
                "noteevents": [
                    event(hadm_id="h0", text="first. "),
                    event(hadm_id="h0", text="second."),
                ],
            }
        )
        samples = MultimodalMortalityPredictionMIMIC3()(patient)
        self.assertEqual([s["hadm_id"] for s in samples], ["h0", "h1"])
        # the label is the next visit's flag; an unknown flag counts as 0
        self.assertEqual([s["mortality"] for s in samples], [1, 0])
        first, second = samples
        self.assertEqual(first["conditions"], ["4019"])
        self.assertEqual(first["drugs"], ["0001"])
        self.assertEqual(first["clinical_notes"], "first. second.")
        self.assertEqual(second["procedures"], ["3893"])
        self.assertEqual(second["clinical_notes"], "")

    def test_one_visit_gives_no_sample(self):
        patient = FakePatient({"admissions": [event(hadm_id="h0")]})
        self.assertEqual(MultimodalMortalityPredictionMIMIC3()(patient), [])


if __name__ == "__main__":
    unittest.main()
