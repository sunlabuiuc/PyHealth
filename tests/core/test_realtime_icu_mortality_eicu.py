# Authors: Hanyang Chen (hc81), Siqi Song (siqis3)
# Paper: Frost, Li & Harris. Robust Real-Time Mortality Prediction in the Intensive
#   Care Unit using Temporal Difference Learning. ML4H 2024, PMLR 259:350-363.
# Paper link: https://proceedings.mlr.press/v259/frost25a.html
# Description: Tests for RealTimeICUMortalityEICU on synthetic eICU patients only.

import unittest

import numpy as np
import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import RealTimeICUMortalityEICU, RealTimeICUMortalityMIMIC4

TASK = RealTimeICUMortalityEICU()
FID = {name: i + 1 for i, name in enumerate(TASK.feature_names)}


def _stay(stay, hosp, admit=-60, discharge=3000, status="Alive", age="67", weight=80):
    row = {
        "patientunitstayid": stay,
        "patienthealthsystemstayid": hosp,
        "hospitaladmitoffset": admit,
        "hospitaldischargeoffset": discharge,
        "hospitaldischargestatus": status,
        "age": age,
        "gender": "Female",
        "admissionweight": weight,
    }
    return {"event_type": "patient", **{f"patient/{k}": str(v) for k, v in row.items()}}


def _lab(stay, offset, name, value):
    row = {
        "patientunitstayid": stay,
        "labresultoffset": offset,
        "labname": name,
        "labresult": value,
    }
    return {"event_type": "lab", **{f"lab/{k}": str(v) for k, v in row.items()}}


def _patient(*rows, pid="002-1"):
    df = pl.from_dicts(list(rows), infer_schema_length=None)
    df = df.with_columns(
        pl.lit(pid).alias("patient_id"), pl.lit(None, pl.Datetime).alias("timestamp")
    )
    return Patient(patient_id=pid, data_source=df)


def _rows(sample, key="measurements"):
    return np.asarray(sample[key], dtype=float)


class TestRealTimeICUMortalityEICU(unittest.TestCase):
    def test_shares_mimic_vocabulary(self):
        self.assertEqual(TASK.feature_names, RealTimeICUMortalityMIMIC4().feature_names)
        self.assertEqual(len(set(TASK.LAB_NAMES.values())), 32)
        self.assertEqual(TASK.output_schema, RealTimeICUMortalityMIMIC4.output_schema)
        with self.assertRaises(ValueError):
            RealTimeICUMortalityEICU(feature_set="all")

    def test_lab_mapping_and_units(self):
        patient = _patient(
            _stay(1, 10),
            _lab(1, 0, "creatinine", 1.0),
            _lab(1, 0, "CRP", 2.0),
            _lab(1, 0, "ionized calcium", 4.4),
            _lab(1, 0, "MCV", 90.0),  # not a task feature
        )
        (sample,) = TASK(patient)
        rows = _rows(sample)
        values = {int(r[0]): r[2] for r in rows}
        self.assertAlmostEqual(values[FID["Creatinine"]], 88.42, places=4)
        self.assertAlmostEqual(values[FID["C-reactive protein"]], 20.0)
        self.assertAlmostEqual(values[FID["Calcium (ionised)"]], 4.4 * 0.2495)
        self.assertEqual(len(rows), 3 + 3)  # three labs + age/gender/weight
        np.testing.assert_allclose(rows[-3:, 2], [67.0, 1.0, 80.0])

    def test_stays_of_one_hospitalisation_share_a_clock(self):
        # Stay 2 starts 2 days after stay 1 (hospital-clock offsets -60, -2940).
        patient = _patient(
            _stay(1, 10, admit=-60),
            _stay(2, 10, admit=-60 - 2880),
            _lab(1, 0, "lactate", 2.0),
            _lab(2, -1440, "lactate", 3.0),  # 1 day after stay 1's lab
        )
        first, second = TASK(patient)
        self.assertEqual(first["terminal"], 0.0)  # next state 24 h later
        np.testing.assert_allclose(_rows(first, "next_measurements"), _rows(second))
        lactate = _rows(second)[_rows(second)[:, 0] == FID["Lactate"]]
        np.testing.assert_allclose(lactate[:, 1], [1440.0, 0.0])
        np.testing.assert_allclose(lactate[-1, 3:], [1440.0, 1.0])

    def test_hospitalisations_are_separate_timelines(self):
        patient = _patient(
            _stay(1, 10),
            _stay(2, 20),
            _lab(1, 0, "lactate", 2.0),
            _lab(2, 1440, "lactate", 3.0),
        )
        samples = TASK(patient)
        self.assertEqual(len(samples), 2)
        self.assertTrue(all(s["terminal"] == 1.0 for s in samples))
        self.assertTrue(all(len(s["measurements"]) == 4 for s in samples))

    def test_in_hospital_death_labels(self):
        died = _patient(
            _stay(1, 10, admit=-60, discharge=5 * 1440, status="Expired"),
            _lab(1, 0, "sodium", 140),
        )
        self.assertEqual(TASK(died)[0]["mortality_horizons"], [0.0, 0.0, 1.0, 1.0, 1.0])
        alive = _patient(_stay(1, 10, status="Alive"), _lab(1, 0, "sodium", 140))
        self.assertEqual(TASK(alive)[0]["mortality_horizons"], [0.0] * 5)

    def test_age_over_89_and_missing_values(self):
        patient = _patient(
            _stay(1, 10, age="> 89", weight=""),
            _lab(1, 0, "sodium", "n/a"),
            _lab(1, 5, "sodium", 135),
        )
        (sample,) = TASK(patient)
        np.testing.assert_allclose(_rows(sample)[-3:, 2], [90.0, 1.0, 74.0])
        self.assertEqual(TASK(_patient(_stay(1, 10))), [])  # no labs

    def test_pre_filter(self):
        frame = pl.DataFrame(
            {
                "patient_id": ["p"] * 4,
                "event_type": ["patient", "lab", "lab", "medication"],
                "lab/labname": [None, "lactate", "MCV", None],
            }
        ).lazy()
        kept = TASK.pre_filter(frame).collect()
        self.assertEqual(kept["event_type"].to_list(), ["patient", "lab"])


if __name__ == "__main__":
    unittest.main()
