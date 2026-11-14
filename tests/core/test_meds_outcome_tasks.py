"""Tests for ICUAdmissionMEDS, HospitalReadmissionMEDS and
ProlongedLengthOfStayMEDS.

Most tests apply a task directly to an in-memory ``Patient``, which keeps them
fast and exposes the raw string codes the leakage tests must inspect. One test
runs all three tasks through ``MEDSDataset.set_task`` on synthetic Parquet.
All data here is synthetic.
"""

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from pyhealth.data import Patient
from pyhealth.datasets import MEDSDataset
from pyhealth.tasks import (
    HospitalReadmissionMEDS,
    ICUAdmissionMEDS,
    ProlongedLengthOfStayMEDS,
)

T0 = datetime(2024, 1, 1, 8, 0, 0)  # noqa: DTZ001 (MEDS times are tz-naive)

ADMIT = "HOSPITAL_ADMISSION//EW EMER.//ER"
HOME = "HOSPITAL_DISCHARGE//HOME"
DIED = "HOSPITAL_DISCHARGE//DIED"
ICU_IN = "ICU_ADMISSION//MICU"
ICU_OUT = "ICU_DISCHARGE//MICU"
DEATH = "MEDS_DEATH"
LAB = "LAB//51221//%"
DX = "DIAGNOSIS//ICD//10//I10"

_CONFIG = """version: "1.0"
tables:
  meds:
    file_path: "data"
    patient_id: "subject_id"
    timestamp: "time"
    attributes:
      - "code"
      - "numeric_value"
      - "hadm_id"
"""


def _hours(days: float) -> float:
    return days * 24.0


def _at(offset_hours: float | None) -> datetime | None:
    if offset_hours is None:
        return None
    return T0 + timedelta(hours=offset_hours)


def _patient(rows, hadm_dtype=pl.Int64) -> Patient:
    """Rows are ``(offset_hours_or_None, code, hadm_id_or_None)``."""
    frame = pl.DataFrame(
        {
            "patient_id": ["1"] * len(rows),
            "event_type": ["meds"] * len(rows),
            "timestamp": pl.Series(
                [_at(r[0]) for r in rows], dtype=pl.Datetime("ms")
            ),
            "meds/code": pl.Series([r[1] for r in rows], dtype=pl.String),
            "meds/hadm_id": pl.Series([r[2] for r in rows], dtype=hadm_dtype),
        }
    )
    return Patient(patient_id="1", data_source=frame)


def _stay(hadm, start, days, discharge_code=HOME, between=()):
    """Admission, optional events and discharge, ``start`` hours after T0."""
    rows = [(start, ADMIT, hadm)]
    rows.extend(between)
    rows.append((start + _hours(days), discharge_code, hadm))
    return rows


class TestICUAdmissionMEDS(unittest.TestCase):
    def test_icu_stay_is_positive(self):
        rows = _stay(1, 0, 3, between=[(5, ICU_IN, 1), (20, ICU_OUT, 1)])
        samples = ICUAdmissionMEDS()(_patient(rows))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["icu_admission"], 1)
        self.assertEqual(samples[0]["hadm_id"], 1)

    def test_ward_stay_is_negative(self):
        rows = _stay(1, 0, 3, between=[(5, LAB, 1)])
        self.assertEqual(ICUAdmissionMEDS()(_patient(rows))[0]["icu_admission"], 0)

    def test_icu_event_without_hadm_inside_stay_is_positive(self):
        rows = _stay(1, 0, 3, between=[(5, ICU_IN, None)])
        self.assertEqual(ICUAdmissionMEDS()(_patient(rows))[0]["icu_admission"], 1)

    def test_icu_after_discharge_belongs_to_next_stay(self):
        rows = _stay(1, 0, 3) + _stay(2, 200, 3, between=[(205, ICU_IN, 2)])
        samples = ICUAdmissionMEDS()(_patient(rows))
        labels = {s["hadm_id"]: s["icu_admission"] for s in samples}
        self.assertEqual(labels, {1: 0, 2: 1})

    def test_death_outside_icu_follows_death_as_positive(self):
        rows = _stay(1, 0, 3, discharge_code=DIED)
        patient = _patient(rows)
        self.assertEqual(ICUAdmissionMEDS()(patient)[0]["icu_admission"], 1)
        icu_only = ICUAdmissionMEDS(death_as_positive=False)
        self.assertEqual(icu_only(patient)[0]["icu_admission"], 0)

    def test_meds_death_inside_stay_without_died_code_is_positive(self):
        rows = _stay(1, 0, 3, between=[(70, DEATH, None)])
        self.assertEqual(ICUAdmissionMEDS()(_patient(rows))[0]["icu_admission"], 1)

    def test_death_after_discharge_is_negative(self):
        rows = _stay(1, 0, 3) + [(80, DEATH, None)]
        self.assertEqual(ICUAdmissionMEDS()(_patient(rows))[0]["icu_admission"], 0)

    def test_input_stops_at_admission(self):
        rows = (
            [(None, "GENDER//F", None), (-100, LAB, None)]
            + _stay(
                1,
                0,
                3,
                discharge_code=DIED,
                between=[(0, ICU_IN, 1), (0, LAB + "//same-time", 1), (5, LAB, 1)],
            )
            + [(73, DEATH, None)]
        )
        codes = ICUAdmissionMEDS()(_patient(rows))[0]["codes"]
        self.assertEqual(codes, ["GENDER//F", LAB, ADMIT])

    def test_icu_entry_stamped_before_admission_stays_out_of_input(self):
        rows = [(-1, ICU_IN, 1)] + _stay(1, 0, 3)
        sample = ICUAdmissionMEDS()(_patient(rows))[0]
        self.assertEqual(sample["icu_admission"], 1)
        self.assertNotIn(ICU_IN, sample["codes"])

    def test_date_resolution_events_of_this_stay_stay_out_of_input(self):
        hcpcs = "HCPCS//G0378"
        rows = (
            _stay(1, 0, 3, between=[(-8, hcpcs, 1)])  # midnight of T0's day
            + [(-1, LAB, 1)]  # timed ED lab of this stay is kept
        )
        for task_cls in (ICUAdmissionMEDS, ProlongedLengthOfStayMEDS):
            with self.subTest(task=task_cls.__name__):
                codes = task_cls()(_patient(rows))[0]["codes"]
                self.assertEqual(codes, [LAB, ADMIT])

    def test_date_resolution_events_of_earlier_stays_remain(self):
        hcpcs = "HCPCS//G0378"
        rows = _stay(1, 16, 3, between=[(40, hcpcs, 1)]) + _stay(2, 200, 3)
        codes = ICUAdmissionMEDS()(_patient(rows))[1]["codes"]
        self.assertIn(hcpcs, codes)

    def test_previous_stay_is_history(self):
        rows = _stay(1, 0, 3, between=[(5, ICU_IN, 1), (9, ICU_OUT, 1)]) + _stay(
            2, 200, 3
        )
        samples = ICUAdmissionMEDS()(_patient(rows))
        self.assertEqual(samples[1]["codes"], [ADMIT, ICU_IN, ICU_OUT, HOME, ADMIT])
        self.assertEqual(samples[1]["icu_admission"], 0)


class TestHospitalReadmissionMEDS(unittest.TestCase):
    def test_admission_within_window_is_positive(self):
        rows = _stay(1, 0, 3) + _stay(2, _hours(3 + 29), 2)
        samples = HospitalReadmissionMEDS()(_patient(rows))
        labels = {s["hadm_id"]: s["readmission"] for s in samples}
        self.assertEqual(labels, {1: 1, 2: 0})

    def test_window_is_inclusive(self):
        rows = _stay(1, 0, 3) + _stay(2, _hours(3 + 30), 2)
        patient = _patient(rows)
        self.assertEqual(HospitalReadmissionMEDS()(patient)[0]["readmission"], 1)
        shorter = HospitalReadmissionMEDS(window_days=29.0)
        self.assertEqual(shorter(patient)[0]["readmission"], 0)

    def test_admission_after_window_is_negative(self):
        rows = _stay(1, 0, 3) + _stay(2, _hours(3 + 31), 2)
        self.assertEqual(
            HospitalReadmissionMEDS()(_patient(rows))[0]["readmission"], 0
        )

    def test_death_within_window_is_positive(self):
        rows = _stay(1, 0, 3) + [(_hours(3 + 10), DEATH, None)]
        self.assertEqual(
            HospitalReadmissionMEDS()(_patient(rows))[0]["readmission"], 1
        )

    def test_in_hospital_death_is_excluded(self):
        died_code = _stay(1, 0, 3, discharge_code=DIED)
        meds_death = _stay(1, 0, 3, between=[(70, DEATH, None)])
        self.assertEqual(HospitalReadmissionMEDS()(_patient(died_code)), [])
        self.assertEqual(HospitalReadmissionMEDS()(_patient(meds_death)), [])

    def test_input_keeps_this_discharge_but_no_outcome(self):
        rows = (
            [(None, "GENDER//F", None)]
            + _stay(1, 0, 3, between=[(5, LAB, 1)])
            + [(_hours(3), DX, 1), (_hours(3), "DRG//HCFA//871", 1)]
            # Stamped at the discharge instant, but not part of this stay.
            + _stay(2, _hours(3), 2)
            + [(_hours(3), LAB, None), (_hours(4), DEATH, None)]
        )
        sample = HospitalReadmissionMEDS()(_patient(rows))[0]
        self.assertEqual(sample["readmission"], 1)
        self.assertEqual(
            sorted(sample["codes"]),
            sorted(["GENDER//F", ADMIT, LAB, HOME, DX, "DRG//HCFA//871"]),
        )
        self.assertEqual(sample["codes"].count(ADMIT), 1)

    def test_same_day_readmission_codes_stay_out_of_the_input(self):
        # Discharge at 14:00, readmission at 18:00 the same day; the next
        # stay's HCPCS code is stamped at midnight of that day, before this
        # discharge, and must not reach the input.
        hcpcs = "HCPCS//G0378"
        rows = (
            _stay(1, 0, 0.25)  # T0 is 08:00, so the discharge is at 14:00
            + _stay(2, 10, 1)
            + [(-8, hcpcs, 2)]  # 00:00 of the same day, stay 2
        )
        sample = HospitalReadmissionMEDS()(_patient(rows))[0]
        self.assertEqual(sample["hadm_id"], 1)
        self.assertEqual(sample["readmission"], 1)
        self.assertNotIn(hcpcs, sample["codes"])
        self.assertEqual(sorted(sample["codes"]), sorted([ADMIT, HOME]))

    def test_full_followup_drops_censored_negatives_only(self):
        censored = _stay(1, 0, 3) + [(_hours(3 + 5), LAB, None)]
        observed = _stay(1, 0, 3) + [(_hours(3 + 40), LAB, None)]
        positive = _stay(1, 0, 3) + _stay(2, _hours(3 + 5), 1)
        task = HospitalReadmissionMEDS(require_full_followup=True)
        self.assertEqual(task(_patient(censored)), [])
        self.assertEqual(task(_patient(observed))[0]["readmission"], 0)
        self.assertEqual(task(_patient(positive))[0]["readmission"], 1)
        self.assertEqual(len(HospitalReadmissionMEDS()(_patient(censored))), 1)


class TestProlongedLengthOfStayMEDS(unittest.TestCase):
    def test_labels_around_inclusive_threshold(self):
        for days, expected in ((10, 1), (11, 1), (9.5, 0)):
            with self.subTest(days=days):
                samples = ProlongedLengthOfStayMEDS()(_patient(_stay(1, 0, days)))
                self.assertEqual(samples[0]["prolonged_los"], expected)

    def test_custom_threshold(self):
        task = ProlongedLengthOfStayMEDS(threshold_days=2.0)
        self.assertEqual(task(_patient(_stay(1, 0, 3)))[0]["prolonged_los"], 1)

    def test_stay_ending_in_death_is_kept(self):
        rows = _stay(1, 0, 12, discharge_code=DIED)
        samples = ProlongedLengthOfStayMEDS()(_patient(rows))
        self.assertEqual(samples[0]["prolonged_los"], 1)

    def test_stay_is_measured_to_meds_death(self):
        rows = _stay(1, 0, 12, between=[(_hours(4), DEATH, None)])
        samples = ProlongedLengthOfStayMEDS()(_patient(rows))
        self.assertEqual(samples[0]["prolonged_los"], 0)

    def test_input_stops_at_admission(self):
        rows = _stay(
            1,
            0,
            12,
            discharge_code=DIED,
            between=[(5, LAB, 1), (6, ICU_IN, 1)],
        ) + [(_hours(12) + 3, DEATH, None)]
        codes = ProlongedLengthOfStayMEDS()(_patient(rows))[0]["codes"]
        self.assertEqual(codes, [ADMIT])


class TestSharedBehavior(unittest.TestCase):
    TASKS = (ICUAdmissionMEDS, HospitalReadmissionMEDS, ProlongedLengthOfStayMEDS)

    def _assert_no_samples(self, rows):
        for task_cls in self.TASKS:
            with self.subTest(task=task_cls.__name__):
                self.assertEqual(task_cls()(_patient(rows)), [])

    def test_no_events(self):
        self._assert_no_samples([])

    def test_no_stays(self):
        self._assert_no_samples([(0, LAB, None), (5, DEATH, None)])

    def test_stay_without_discharge(self):
        self._assert_no_samples([(0, ADMIT, 1), (5, LAB, 1)])

    def test_stay_without_admission(self):
        self._assert_no_samples([(0, LAB, 1), (5, HOME, 1)])

    def test_zero_length_stay(self):
        self._assert_no_samples([(5, ADMIT, 1), (5, HOME, 1)])

    def test_admission_events_without_hadm(self):
        self._assert_no_samples([(0, ADMIT, None), (5, HOME, None)])

    def test_stay_without_other_events_yields_admission_only(self):
        for task_cls in (ICUAdmissionMEDS, ProlongedLengthOfStayMEDS):
            with self.subTest(task=task_cls.__name__):
                samples = task_cls()(_patient(_stay(1, 0, 3)))
                self.assertEqual(samples[0]["codes"], [ADMIT])

    def test_float_hadm_id_stays_integral(self):
        rows = [
            (r[0], r[1], None if r[2] is None else float(r[2]))
            for r in _stay(1, 0, 3) + [(90, LAB, None)]
        ]
        patient = _patient(rows, hadm_dtype=pl.Float64)
        for task_cls in self.TASKS:
            with self.subTest(task=task_cls.__name__):
                hadm_id = task_cls()(patient)[0]["hadm_id"]
                self.assertIsInstance(hadm_id, int)

    def test_outcome_events_never_reach_input(self):
        rows = (
            _stay(1, 0, 3, between=[(5, ICU_IN, 1), (9, ICU_OUT, 1)])
            + _stay(2, 200, 12, discharge_code=DIED, between=[(205, ICU_IN, 2)])
            + [(200 + _hours(12) + 2, DEATH, None)]
        )
        patient = _patient(rows)
        for task_cls in (ICUAdmissionMEDS, ProlongedLengthOfStayMEDS):
            with self.subTest(task=task_cls.__name__):
                first, second = task_cls()(patient)
                self.assertEqual(first["codes"], [ADMIT])
                self.assertEqual(
                    second["codes"], [ADMIT, ICU_IN, ICU_OUT, HOME, ADMIT]
                )
        sample = HospitalReadmissionMEDS()(patient)[0]
        self.assertEqual(sample["hadm_id"], 1)
        self.assertEqual(sample["readmission"], 1)
        self.assertNotIn(ADMIT, sample["codes"][1:])
        self.assertNotIn(DIED, sample["codes"])
        self.assertNotIn(DEATH, sample["codes"])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            HospitalReadmissionMEDS(window_days=0)
        with self.assertRaises(ValueError):
            ProlongedLengthOfStayMEDS(threshold_days=-1)


class TestSetTask(unittest.TestCase):
    """All three tasks through MEDSDataset, on synthetic typed Parquet."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        root = self.temp_dir / "meds"
        (root / "data").mkdir(parents=True)
        config_path = self.temp_dir / "meds_hadm.yaml"
        config_path.write_text(_CONFIG)

        rows = [
            (1, None, "GENDER//F", None),
            (1, 0, ADMIT, 10),
            (1, 2, LAB, 10),
            (1, 5, ICU_IN, 10),
            (1, _hours(11), HOME, 10),
            (1, _hours(20), ADMIT, 11),
            (1, _hours(20) + 3, LAB, 11),
            (1, _hours(22), DIED, 11),
            (2, 0, ADMIT, 20),
            (2, 3, LAB, 20),
            (2, _hours(2), HOME, 20),
            (2, _hours(50), LAB, None),
        ]
        pl.DataFrame(
            {
                "subject_id": pl.Series([r[0] for r in rows], dtype=pl.Int64),
                "time": pl.Series([_at(r[1]) for r in rows], dtype=pl.Datetime("us")),
                "code": pl.Series([r[2] for r in rows], dtype=pl.String),
                "numeric_value": pl.Series([None] * len(rows), dtype=pl.Float32),
                "hadm_id": pl.Series([r[3] for r in rows], dtype=pl.Int64),
            }
        ).write_parquet(root / "data" / "0.parquet")

        self.dataset = MEDSDataset(
            root=str(root),
            config_path=str(config_path),
            cache_dir=self.temp_dir / "cache",
            num_workers=1,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _assert_labels(self, task, key, labels):
        samples = self.dataset.set_task(task, num_workers=1)
        self.assertEqual(len(samples), len(labels))
        self.assertEqual(
            sorted(int(samples[i][key]) for i in range(len(samples))),
            sorted(labels),
        )
        samples.close()

    def test_icu_admission(self):
        self._assert_labels(ICUAdmissionMEDS(), "icu_admission", [1, 1, 0])

    def test_readmission(self):
        self._assert_labels(HospitalReadmissionMEDS(), "readmission", [1, 0])

    def test_prolonged_los(self):
        self._assert_labels(ProlongedLengthOfStayMEDS(), "prolonged_los", [1, 0, 0])

    def test_static_events_survive_the_pipeline(self):
        codes = ICUAdmissionMEDS()(self.dataset.get_patient("1"))[0]["codes"]
        self.assertEqual(codes, ["GENDER//F", ADMIT])


if __name__ == "__main__":
    unittest.main()
