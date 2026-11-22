"""Tests for the MIMIC-Extract binary tasks, on tiny synthetic patients."""

import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import (
    MIMICExtractLengthOfStay,
    MIMICExtractMortalityHospital,
    MIMICExtractMortalityICU,
)

VARIABLES = ["heart rate", "glucose"]
INTIME = datetime(2100, 1, 1)


def _stay(stay_id, los_icu, mort_hosp, mort_icu):
    return {
        "event_type": "patients",
        "timestamp": INTIME,
        "patients/icustay_id": stay_id,
        "patients/los_icu": los_icu,
        "patients/mort_hosp": mort_hosp,
        "patients/mort_icu": mort_icu,
    }


def _hour(stay_id, hour, variable, mean, count=1.0, std=0.0):
    return {
        "event_type": "vitals_labs",
        "timestamp": INTIME + timedelta(hours=hour),
        "vitals_labs/icustay_id": stay_id,
        "vitals_labs/hours_in": hour,
        "vitals_labs/variable": variable,
        "vitals_labs/mean": mean,
        "vitals_labs/count": count,
        "vitals_labs/std": std,
    }


def _patient(patient_id, rows):
    df = pl.DataFrame(rows).with_columns(pl.lit(patient_id).alias("patient_id"))
    return Patient(patient_id=patient_id, data_source=df)


def _features(sample, n_hours=24, aggregates=3):
    """Reshapes a sample's flat features to (hours, variables, aggregates)."""
    return np.array(sample["features"]).reshape(
        n_hours, len(VARIABLES), aggregates
    )


class TestMIMICExtractTasks(unittest.TestCase):
    """Labels, cohort filter, and features of the MIMIC-Extract tasks."""

    def setUp(self):
        # p1: died in hospital, not in ICU; 48 h stay; heart rate at hours
        # 0 and 2, an hour outside the window, and an unknown variable.
        self.p1 = _patient(
            "p1",
            [
                _stay(101, 2.0, 1, 0),
                _hour(101, 0, "heart rate", 80.0),
                _hour(101, 2, "heart rate", 90.0, count=2.0, std=1.5),
                _hour(101, 30, "heart rate", 200.0),
                _hour(101, 1, "unknown variable", 5.0),
            ],
        )
        # p2: 24 h stay, shorter than the default 24 + 6 h.
        self.p2 = _patient(
            "p2", [_stay(201, 1.0, 0, 0), _hour(201, 0, "glucose", 100.0)]
        )
        # p3: two stays (8 days and 4 days), each with its own glucose.
        self.p3 = _patient(
            "p3",
            [
                _stay(301, 8.0, 0, 0),
                _stay(302, 4.0, 1, 1),
                _hour(301, 0, "glucose", 110.0),
                _hour(302, 0, "glucose", 300.0),
            ],
        )

    def test_mortality_hospital_label(self):
        task = MIMICExtractMortalityHospital(variables=VARIABLES)
        (sample,) = task(self.p1)
        self.assertEqual(sample["label"], 1)
        self.assertEqual(sample["visit_id"], "101")
        self.assertEqual(sample["patient_id"], "p1")

    def test_mortality_icu_label(self):
        task = MIMICExtractMortalityICU(variables=VARIABLES)
        self.assertEqual(task(self.p1)[0]["label"], 0)
        labels = {s["visit_id"]: s["label"] for s in task(self.p3)}
        self.assertEqual(labels, {"301": 0, "302": 1})

    def test_length_of_stay_thresholds(self):
        over_3 = MIMICExtractLengthOfStay(variables=VARIABLES, threshold_days=3)
        over_7 = MIMICExtractLengthOfStay(variables=VARIABLES, threshold_days=7)
        self.assertEqual(
            {s["visit_id"]: s["label"] for s in over_3(self.p3)},
            {"301": 1, "302": 1},
        )
        self.assertEqual(
            {s["visit_id"]: s["label"] for s in over_7(self.p3)},
            {"301": 1, "302": 0},
        )
        self.assertEqual(over_3(self.p1)[0]["label"], 0)

    def test_short_stay_excluded(self):
        task = MIMICExtractMortalityHospital(variables=VARIABLES)
        self.assertEqual(task(self.p2), [])

    def test_shorter_window_includes_short_stay(self):
        task = MIMICExtractMortalityHospital(
            variables=VARIABLES, window_hours=12, gap_hours=6
        )
        (sample,) = task(self.p2)
        self.assertEqual(len(sample["features"]), 12 * len(VARIABLES) * 3)

    def test_feature_length(self):
        task = MIMICExtractMortalityHospital(variables=VARIABLES)
        (sample,) = task(self.p1)
        self.assertEqual(len(sample["features"]), 24 * len(VARIABLES) * 3)

    def test_carry_forward_and_fill(self):
        task = MIMICExtractMortalityHospital(variables=VARIABLES, fill_value=-1.0)
        grid = _features(task(self.p1)[0])
        mean, count, std = grid[..., 0], grid[..., 1], grid[..., 2]
        # heart rate: 80 at hour 0, carried to hour 1, 90 from hour 2 on.
        self.assertEqual(list(mean[:4, 0]), [80.0, 80.0, 90.0, 90.0])
        self.assertEqual(mean[23, 0], 90.0)
        # glucose never measured: means and stds filled, counts 0.
        self.assertTrue(np.all(mean[:, 1] == -1.0))
        self.assertTrue(np.all(count[:, 1] == 0.0))
        self.assertEqual(count[2, 0], 2.0)
        self.assertEqual(std[2, 0], 1.5)
        self.assertEqual(std[1, 0], -1.0)  # stds are not carried forward

    def test_hours_after_window_ignored(self):
        task = MIMICExtractMortalityHospital(variables=VARIABLES)
        grid = _features(task(self.p1)[0])
        self.assertNotIn(200.0, grid[..., 0])

    def test_stays_do_not_mix(self):
        task = MIMICExtractMortalityHospital(variables=VARIABLES)
        by_stay = {s["visit_id"]: _features(s) for s in task(self.p3)}
        self.assertEqual(by_stay["301"][0, 1, 0], 110.0)
        self.assertEqual(by_stay["302"][0, 1, 0], 300.0)

    def test_aggregate_subset(self):
        task = MIMICExtractMortalityHospital(
            variables=VARIABLES, aggregates=["mean"]
        )
        (sample,) = task(self.p1)
        self.assertEqual(len(sample["features"]), 24 * len(VARIABLES))
        self.assertEqual(sample["features"][0], 80.0)

    def test_schema_is_binary(self):
        task = MIMICExtractLengthOfStay(variables=VARIABLES)
        self.assertEqual(task.output_schema, {"label": "binary"})
        self.assertEqual(task.input_schema, {"features": "tensor"})

    def test_variable_names_are_case_insensitive(self):
        # MIMIC-Extract's own names ("Heart Rate") match the dataset's
        # lower-case names, and are stored lower case.
        task = MIMICExtractMortalityHospital(variables=["Heart Rate", "GLUCOSE"])
        self.assertEqual(task.variables, ["heart rate", "glucose"])
        grid = _features(task(self.p1)[0])
        self.assertEqual(grid[0, 0, 0], 80.0)  # heart rate mean is populated

    def test_event_names_are_case_insensitive(self):
        patient = _patient(
            "p4", [_stay(401, 2.0, 0, 0), _hour(401, 0, "Heart Rate", 70.0)]
        )
        task = MIMICExtractMortalityHospital(variables=VARIABLES)
        self.assertEqual(_features(task(patient)[0])[0, 0, 0], 70.0)

    def test_duplicate_variable_names_rejected(self):
        with self.assertRaises(ValueError):
            MIMICExtractMortalityHospital(variables=["heart rate", "Heart Rate"])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            MIMICExtractMortalityHospital(variables=[])
        with self.assertRaises(ValueError):
            MIMICExtractMortalityHospital(variables=VARIABLES, window_hours=0)
        with self.assertRaises(ValueError):
            MIMICExtractMortalityHospital(variables=VARIABLES, gap_hours=-1)
        with self.assertRaises(ValueError):
            MIMICExtractMortalityHospital(variables=VARIABLES, aggregates=["max"])
        with self.assertRaises(ValueError):
            MIMICExtractLengthOfStay(variables=VARIABLES, threshold_days=0)


if __name__ == "__main__":
    unittest.main()
