import unittest
from datetime import datetime, timedelta

import polars as pl

from pyhealth.tasks.dka import T1DDKAPredictionMIMIC4


class DummyAdmission:
    def __init__(self, hadm_id: str, timestamp: datetime):
        self.hadm_id = hadm_id
        self.timestamp = timestamp
        self.dischtime = None


class DummyDiagnosis:
    def __init__(self, icd_code: str, icd_version: int | str, timestamp: datetime, hadm_id: str):
        self.icd_code = icd_code
        self.icd_version = icd_version
        self.timestamp = timestamp
        self.hadm_id = hadm_id


class DummyPatient:
    def __init__(self, patient_id: str, admissions: list[DummyAdmission], diagnoses: list[DummyDiagnosis]):
        self.patient_id = patient_id
        self._admissions = admissions
        self._diagnoses = diagnoses
        self._procedures: list = []

    def get_events(self, event_type: str, filters=None, return_df: bool = False):
        if event_type == "admissions":
            events = self._admissions
        elif event_type == "diagnoses_icd":
            events = self._diagnoses
        elif event_type == "procedures_icd":
            events = self._procedures
        elif event_type == "labevents" and return_df:
            return pl.DataFrame(schema={"labevents/itemid": pl.Utf8, "labevents/valuenum": pl.Float64})
        else:
            events = []

        if filters:
            for field, op, value in filters:
                if op == "==":
                    events = [e for e in events if getattr(e, field, None) == value]
        return events


def _build_patient(t0: datetime, admission_days: list[int], diag_specs: list[tuple[str, int | str, int, str]]) -> DummyPatient:
    admissions = [DummyAdmission(f"a{i+1}", t0 + timedelta(days=day)) for i, day in enumerate(admission_days)]
    base_diag = DummyDiagnosis("E10.10", 10, t0, "t1dm")
    diagnoses = [base_diag] + [
        DummyDiagnosis(code, version, t0 + timedelta(days=day), hadm_id)
        for code, version, day, hadm_id in diag_specs
    ]
    return DummyPatient("patient", admissions, diagnoses)


class TestT1DDKAPredictionWindow(unittest.TestCase):
    def setUp(self):
        self.window_days = 90
        self.task = T1DDKAPredictionMIMIC4(dka_window_days=self.window_days)
        self.t0 = datetime(2020, 1, 1)

    def _assert_within_window(self, sample, expected_visits: int):
        times, visits = sample["icd_codes"]
        self.assertEqual(len(visits), expected_visits)
        self.assertEqual(len(times), expected_visits)
        total_hours = sum(times)
        self.assertLessEqual(total_hours, self.window_days * 24)

    def test_sequences_do_not_exceed_window(self):
        patient = _build_patient(
            self.t0,
            admission_days=[10, 70, 80],
            diag_specs=[
                ("E109", 10, 10, "a1"),
                ("E109", 10, 70, "a2"),
                ("E1011", 10, 95, "a3"),
            ],
        )
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self._assert_within_window(sample, expected_visits=2)
        self.assertEqual(sample["label"], 0)

        patient_long_history = _build_patient(
            self.t0,
            admission_days=[5, 40, 100],
            diag_specs=[
                ("E109", 10, 5, "a1"),
                ("E109", 10, 40, "a2"),
                ("E109", 10, 100, "a3"),
            ],
        )
        samples_long = self.task(patient_long_history)
        self.assertEqual(len(samples_long), 1)
        sample_long = samples_long[0]
        self._assert_within_window(sample_long, expected_visits=2)
        self.assertEqual(sample_long["label"], 0)


if __name__ == "__main__":
    unittest.main()
