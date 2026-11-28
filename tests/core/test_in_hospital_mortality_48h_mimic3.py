"""Tests for InHospitalMortality48hMIMIC3.

Builds a tiny, fully synthetic MIMIC-III-shaped cohort (2-5 patients across
the patients/admissions/icustays/chartevents/labevents tables) directly as
:class:`~pyhealth.data.Patient` objects, in the same event layout
``MIMIC3Dataset`` produces with its default config: one row per event,
``<table>/<column>`` attribute columns, and attribute values as strings. No
real or demo MIMIC-III data is read, and no dataset (or Dask cluster) is
built, so the whole module runs in well under a second. The task output is
also fed through ``create_sample_dataset`` to check it matches the task's
input/output schemas.
"""

import unittest
from datetime import datetime

import polars as pl

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset
from pyhealth.tasks.in_hospital_mortality_48h_mimic3 import (
    FEATURE_NAMES,
    MISSING,
    VARIABLES,
    InHospitalMortality48hMIMIC3,
    _hadm_key,
)

_IDX = {name: i for i, name in enumerate(VARIABLES)}
_N = len(VARIABLES)

# Timestamp column per table, as in pyhealth/datasets/configs/mimic3.yaml.
_TIMESTAMP = {
    "patients": None,
    "admissions": "admittime",
    "icustays": "intime",
    "chartevents": "charttime",
    "labevents": "charttime",
}


def _mean_of(features, variable):
    return features[_IDX[variable]]


def _std_of(features, variable):
    return features[_N + _IDX[variable]]


def _patients(tables: dict[str, list[dict]]) -> dict[int, Patient]:
    """Turn table rows into one ``Patient`` per ``subject_id``."""
    events: dict[int, list[dict]] = {}
    for table, rows in tables.items():
        time_col = _TIMESTAMP[table]
        for row in rows:
            sid = row["subject_id"]
            stamp = row.get(time_col) if time_col else None
            event = {
                "patient_id": str(sid),
                "event_type": table,
                "timestamp": (
                    datetime.fromisoformat(stamp) if stamp is not None else None
                ),
            }
            for col, value in row.items():
                if col not in ("subject_id", time_col):
                    event[f"{table}/{col}"] = (
                        None if value is None else str(value)
                    )
            events.setdefault(sid, []).append(event)
    columns = sorted({key for rows in events.values() for e in rows for key in e})
    schema = {
        col: pl.Datetime("ms") if col == "timestamp" else pl.Utf8 for col in columns
    }
    return {
        sid: Patient(
            str(sid),
            pl.DataFrame(
                [{col: e.get(col) for col in columns} for e in rows], schema=schema
            ),
        )
        for sid, rows in events.items()
    }


def _run_task(tables: dict[str, list[dict]]) -> list[dict]:
    task = InHospitalMortality48hMIMIC3()
    samples = []
    for patient in _patients(tables).values():
        samples.extend(task(patient))
    return samples


def _chart(itemid, charttime, valuenum=None, value=None):
    return {
        "subject_id": 1, "hadm_id": 100, "icustay_id": 1000, "itemid": itemid,
        "charttime": charttime, "valuenum": valuenum, "value": value,
    }


_DEFAULT_COHORT = {
    "patients": [
        # p1: adult, 50h ICU stay, rich chartevents+labevents, survives.
        {"subject_id": 1, "gender": "F", "dob": "1980-01-01 00:00:00"},
        # p2: too young (age < 18 at admission).
        {"subject_id": 2, "gender": "M", "dob": "2140-01-01 00:00:00"},
        # p3: adult but short ICU stay (< 48h).
        {"subject_id": 3, "gender": "M", "dob": "1970-01-01 00:00:00"},
        # p4: adult, exactly 48h ICU stay, no chartevents/labevents at all,
        # dies in hospital.
        {"subject_id": 4, "gender": "F", "dob": "1960-01-01 00:00:00"},
    ],
    "admissions": [
        {"subject_id": 1, "hadm_id": 100, "admittime": "2150-01-10 00:00:00",
         "dischtime": "2150-01-20 00:00:00", "hospital_expire_flag": 0},
        {"subject_id": 2, "hadm_id": 200, "admittime": "2150-01-10 00:00:00",
         "dischtime": "2150-01-20 00:00:00", "hospital_expire_flag": 0},
        {"subject_id": 3, "hadm_id": 300, "admittime": "2150-01-10 00:00:00",
         "dischtime": "2150-01-12 00:00:00", "hospital_expire_flag": 0},
        {"subject_id": 4, "hadm_id": 400, "admittime": "2150-02-01 00:00:00",
         "dischtime": "2150-02-10 00:00:00", "hospital_expire_flag": 1},
    ],
    "icustays": [
        {"subject_id": 1, "hadm_id": 100, "icustay_id": 1000,
         "intime": "2150-01-10 00:00:00", "outtime": "2150-01-12 02:00:00"},
        {"subject_id": 2, "hadm_id": 200, "icustay_id": 2000,
         "intime": "2150-01-10 00:00:00", "outtime": "2150-01-13 00:00:00"},
        {"subject_id": 3, "hadm_id": 300, "icustay_id": 3000,
         "intime": "2150-01-10 00:00:00", "outtime": "2150-01-10 10:00:00"},
        {"subject_id": 4, "hadm_id": 400, "icustay_id": 4000,
         "intime": "2150-02-01 00:00:00", "outtime": "2150-02-03 00:00:00"},
    ],
    "chartevents": [
        # heart rate (itemid 211): two valid readings + one outlier.
        _chart(211, "2150-01-10 01:00:00", valuenum=80),
        _chart(211, "2150-01-10 02:00:00", valuenum=90),
        _chart(211, "2150-01-10 03:00:00", valuenum=999),  # outlier, dropped
        # capillary refill rate (itemid 3348): one categorical reading.
        _chart(3348, "2150-01-10 01:00:00", value="Normal <3 Secs"),
        # GCS eye opening (itemid 184): two categorical readings.
        _chart(184, "2150-01-10 01:00:00", value="4 Spontaneously"),
        _chart(184, "2150-01-10 05:00:00", value="To Pain"),
        # weight in lb (itemid 3581): one reading, unit-converted.
        _chart(3581, "2150-01-10 01:00:00", valuenum=154),
        # weight already in kg (itemid 3580): must NOT be lb-converted.
        _chart(3580, "2150-01-10 02:00:00", valuenum=70),
        # temperature in F (itemid 678): one reading, unit-converted.
        _chart(678, "2150-01-10 01:00:00", valuenum=98.6),
        # FiO2 charted as a percentage (itemid 3420): rescaled to a fraction.
        _chart(3420, "2150-01-10 01:00:00", valuenum=40),
        # SpO2 charted as a fraction (itemid 646): rescaled to a percentage.
        _chart(646, "2150-01-10 01:00:00", valuenum=0.95),
        # an itemid the task does not use: ignored.
        _chart(999999, "2150-01-10 01:00:00", valuenum=5),
        # outside the 48h window: must not be counted.
        _chart(211, "2150-01-12 01:00:00", valuenum=40),
    ],
    "labevents": [
        # glucose (lab itemid 50931): one reading.
        {"subject_id": 1, "hadm_id": 100, "itemid": 50931,
         "charttime": "2150-01-10 02:00:00", "valuenum": 110,
         "label": "GLUCOSE", "fluid": "BLOOD", "category": "CHEMISTRY"},
    ],
}


class TestInHospitalMortality48hMIMIC3(unittest.TestCase):
    """Applies the task once to the shared synthetic cohort."""

    @classmethod
    def setUpClass(cls):
        task = InHospitalMortality48hMIMIC3()
        cls.by_subject = {}
        for sid, patient in _patients(_DEFAULT_COHORT).items():
            samples = task(patient)
            if samples:
                cls.by_subject[sid] = samples

    def test_feature_names_length(self):
        self.assertEqual(len(FEATURE_NAMES), 34)
        self.assertEqual(len(VARIABLES), 17)

    def test_age_and_los_exclusions(self):
        by_subject = self.by_subject
        # p2 (age < 18) and p3 (LOS < 48h) produce no samples.
        self.assertNotIn(2, by_subject)
        self.assertNotIn(3, by_subject)
        # p1 and p4 (LOS == 48h exactly) are both included.
        self.assertIn(1, by_subject)
        self.assertIn(4, by_subject)

    def test_patient4_all_missing_and_died(self):
        sample = self.by_subject[4][0]
        self.assertEqual(sample["mortality"], 1)
        self.assertEqual(sample["features"], [MISSING] * 34)

    def test_heart_rate_mean_std_excludes_outlier_and_out_of_window(self):
        sample = self.by_subject[1][0]
        # Only the two in-window, in-range readings (80, 90) count.
        self.assertAlmostEqual(_mean_of(sample["features"], "heart_rate"), 85.0)
        self.assertAlmostEqual(
            _std_of(sample["features"], "heart_rate"), 50.0 ** 0.5, places=6
        )

    def test_single_reading_mean_set_std_missing(self):
        features = self.by_subject[1][0]["features"]
        self.assertAlmostEqual(
            _mean_of(features, "capillary_refill_rate"), 0.0
        )  # "Normal <3 Secs" -> 0.0
        self.assertEqual(_std_of(features, "capillary_refill_rate"), MISSING)

    def test_categorical_gcs_mapping_and_stats(self):
        features = self.by_subject[1][0]["features"]
        # "4 Spontaneously" -> 4.0, "To Pain" -> 2.0
        self.assertAlmostEqual(
            _mean_of(features, "glascow_coma_scale_eye_opening"), 3.0
        )
        self.assertAlmostEqual(
            _std_of(features, "glascow_coma_scale_eye_opening"),
            2.0 ** 0.5,
            places=6,
        )

    def test_unit_conversions(self):
        features = self.by_subject[1][0]["features"]
        # Two readings: 154 lb (itemid 3581, converted) and 70 kg (itemid
        # 3580, already kg -- must NOT be converted a second time).
        self.assertAlmostEqual(
            _mean_of(features, "weight"), (154 * 0.45359237 + 70) / 2, places=4
        )
        self.assertAlmostEqual(_mean_of(features, "temperature"), 37.0, places=4)

    def test_weight_already_in_kg_is_not_double_converted(self):
        """Regression test for itemid 3580 ("Present Weight (kg)"), which
        was previously misclassified as pounds and lb-converted."""
        mean = _mean_of(self.by_subject[1][0]["features"], "weight")
        # Treating 70 kg as 70 lb would pull the mean down to about 50.8.
        self.assertGreater(mean, 65.0)

    def test_fio2_percentage_reading_is_rescaled_to_fraction(self):
        features = self.by_subject[1][0]["features"]
        self.assertAlmostEqual(
            _mean_of(features, "fraction_inspired_oxygen"), 0.40, places=4
        )

    def test_oxygen_saturation_fraction_reading_is_rescaled_to_percentage(self):
        features = self.by_subject[1][0]["features"]
        self.assertAlmostEqual(
            _mean_of(features, "oxygen_saturation"), 95.0, places=4
        )

    def test_lab_sourced_value(self):
        features = self.by_subject[1][0]["features"]
        self.assertAlmostEqual(_mean_of(features, "glucose"), 110.0)
        self.assertEqual(_std_of(features, "glucose"), MISSING)

    def test_unmeasured_variable_is_missing(self):
        features = self.by_subject[1][0]["features"]
        for variable in (
            "diastolic_blood_pressure", "systolic_blood_pressure", "ph"
        ):
            self.assertEqual(_mean_of(features, variable), MISSING)
            self.assertEqual(_std_of(features, variable), MISSING)

    def test_samples_fit_the_task_schemas(self):
        task = InHospitalMortality48hMIMIC3()
        samples = [s for rows in self.by_subject.values() for s in rows]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name="test_mortality48h",
        )
        self.assertEqual(len(dataset), 2)
        self.assertEqual(tuple(dataset[0]["features"].shape), (34,))


class TestHadmKey(unittest.TestCase):
    """Unit tests for the hadm_id join-key coercion helper."""

    def test_int_float_and_numeric_string_all_coerce_equal(self):
        self.assertEqual(_hadm_key(100), _hadm_key(100.0))
        self.assertEqual(_hadm_key("100"), _hadm_key(100))
        self.assertEqual(_hadm_key("100.0"), _hadm_key(100))

    def test_none_coerces_to_none_without_raising(self):
        """Regression test: a naive int(float(value)) raises TypeError on
        None, which would crash __call__ for every patient instead of just
        skipping the one record with a missing hadm_id."""
        self.assertIsNone(_hadm_key(None))


class TestHadmIdJoin(unittest.TestCase):
    """Regression tests for joining an ICU stay to its admission."""

    @staticmethod
    def _cohort(icustay_hadm_id, admission_hadm_id="500"):
        admissions = []
        if admission_hadm_id is not None:
            admissions.append({
                "subject_id": 9, "hadm_id": admission_hadm_id,
                "admittime": "2150-01-10 00:00:00",
                "dischtime": "2150-01-20 00:00:00",
                "hospital_expire_flag": 0,
            })
        return {
            "patients": [
                {"subject_id": 9, "gender": "F", "dob": "1980-01-01 00:00:00"}
            ],
            "admissions": admissions,
            "icustays": [{
                "subject_id": 9, "hadm_id": icustay_hadm_id, "icustay_id": 9000,
                "intime": "2150-01-10 00:00:00",
                "outtime": "2150-01-12 02:00:00",
            }],
        }

    def test_mismatched_hadm_id_formats_still_join(self):
        """ADMISSIONS and ICUSTAYS are scanned as separate tables, so on real
        data one can hold "500" and the other "500.0". Without _hadm_key the
        join would silently miss and drop an eligible ICU stay."""
        samples = _run_task(self._cohort(icustay_hadm_id="500.0"))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["mortality"], 0)

    def test_missing_hadm_id_is_skipped_not_raised(self):
        """An icustay row with no hadm_id must be skipped, not crash
        __call__ for the whole patient."""
        cohort = self._cohort(icustay_hadm_id=None, admission_hadm_id=None)
        self.assertEqual(_run_task(cohort), [])


if __name__ == "__main__":
    unittest.main()
