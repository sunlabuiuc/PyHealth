import unittest
from datetime import datetime

import numpy as np
import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import (
    MIMIC3Forecasting,
    mimic3_forecasting_statistics,
)

VARIABLES = (
    "lab:Albumin",
    "input:Albumin 5%",
    "output:Foley",
    "prescription:Aspirin",
)


def _patient(patient_id="1", second_admission=False):
    rows = [
        {
            "patient_id": patient_id,
            "event_type": "patients",
            "timestamp": None,
            "patients/dob": "1980-01-01 00:00:00",
        },
        {
            "patient_id": patient_id,
            "event_type": "admissions",
            "timestamp": datetime(2020, 1, 1),
            "admissions/hadm_id": "10",
            "admissions/dischtime": "2020-01-05 00:00:00",
            "admissions/admission_type": "EMERGENCY",
            "admissions/has_chartevents_data": "1",
        },
        {
            "patient_id": patient_id,
            "event_type": "labevents",
            "timestamp": datetime(2020, 1, 1, 0, 0),
            "labevents/hadm_id": "10",
            "labevents/label": "Albumin",
            "labevents/valuenum": "10",
        },
        {
            "patient_id": patient_id,
            "event_type": "labevents",
            "timestamp": datetime(2020, 1, 1, 0, 10),
            "labevents/hadm_id": "10",
            "labevents/label": "Albumin",
            "labevents/valuenum": "14",
        },
        {
            "patient_id": patient_id,
            "event_type": "inputevents_mv",
            "timestamp": datetime(2020, 1, 1, 1, 0),
            "inputevents_mv/hadm_id": "10",
            "inputevents_mv/label": "Albumin 5%",
            "inputevents_mv/amount": "10",
            "inputevents_mv/endtime": "2020-01-01 02:00:00",
        },
        {
            "patient_id": patient_id,
            "event_type": "outputevents",
            "timestamp": datetime(2020, 1, 2, 12, 0),
            "outputevents/hadm_id": "10",
            "outputevents/label": "Foley",
            "outputevents/value": "100",
        },
        {
            "patient_id": patient_id,
            "event_type": "prescriptions",
            "timestamp": datetime(2020, 1, 2, 12, 30),
            "prescriptions/hadm_id": "10",
            "prescriptions/drug": "Aspirin",
            "prescriptions/dose_val_rx": "1-3",
        },
        {
            "patient_id": patient_id,
            "event_type": "labevents",
            "timestamp": datetime(2020, 1, 2, 13, 0),
            "labevents/hadm_id": "10",
            "labevents/label": "Albumin",
            "labevents/valuenum": "20",
        },
    ]
    if second_admission:
        rows.append(
            {
                "patient_id": patient_id,
                "event_type": "admissions",
                "timestamp": datetime(2020, 2, 1),
                "admissions/hadm_id": "11",
                "admissions/dischtime": "2020-02-05 00:00:00",
                "admissions/admission_type": "EMERGENCY",
                "admissions/has_chartevents_data": "1",
            }
        )
    return Patient(patient_id, pl.from_dicts(rows))


class _Dataset:
    def __init__(self, patients):
        self.patients = patients

    def iter_patients(self):
        yield from self.patients


class TestMIMIC3Forecasting(unittest.TestCase):
    def test_builds_benchmark_windows_and_aggregates_tables(self):
        task = MIMIC3Forecasting(
            variables=VARIABLES,
            min_observations=1,
        )
        (sample,) = task(_patient())
        self.assertEqual(sample["x"].shape, (72, 4))
        self.assertEqual(sample["y"].shape, (3, 4))
        self.assertEqual(sample["x"][0, 0], 12.0)
        self.assertEqual(sample["x"][2, 1], 5.0)
        self.assertEqual(sample["x"][3, 1], 5.0)
        np.testing.assert_allclose(sample["y_time"], [0.75, 73 / 96, 74 / 96])
        self.assertEqual(sample["y"][0, 2], 100.0)
        self.assertEqual(sample["y"][1, 3], 2.0)
        self.assertEqual(sample["y"][2, 0], 20.0)

    def test_normalizes_values_and_drops_outliers(self):
        mean = dict.fromkeys(VARIABLES, 0.0)
        std = dict.fromkeys(VARIABLES, 10.0)
        task = MIMIC3Forecasting(
            mean=mean,
            std=std,
            variables=VARIABLES,
            min_observations=1,
            outlier_sd=5.0,
        )
        (sample,) = task(_patient())
        self.assertAlmostEqual(sample["x"][0, 0], 1.2)
        self.assertTrue(np.isnan(sample["y"][0, 2]))

    def test_benchmark_cohort_excludes_multiple_admissions(self):
        task = MIMIC3Forecasting(variables=VARIABLES, min_observations=1)
        self.assertEqual(task(_patient(second_admission=True)), [])

    def test_statistics_use_aggregated_values(self):
        mean, std = mimic3_forecasting_statistics(
            _Dataset([_patient()]),
            variables=VARIABLES,
            min_observations=1,
        )
        self.assertEqual(mean["lab:Albumin"], 16.0)
        self.assertEqual(std["input:Albumin 5%"], 1.0)
        self.assertEqual(mean["prescription:Aspirin"], 2.0)

    def test_validates_configuration(self):
        with self.assertRaisesRegex(ValueError, "both mean and std"):
            MIMIC3Forecasting(mean={}, variables=VARIABLES)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            MIMIC3Forecasting(variables=["lab:not-a-variable"])
        with self.assertRaisesRegex(ValueError, "positive divisor"):
            MIMIC3Forecasting(variables=VARIABLES, bin_minutes=40)


if __name__ == "__main__":
    unittest.main()
