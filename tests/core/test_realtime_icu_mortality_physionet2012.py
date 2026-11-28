# Authors: Hanyang Chen (hc81), Siqi Song (siqis3)
# Paper: Frost, Li & Harris. Robust Real-Time Mortality Prediction in the Intensive
#   Care Unit using Temporal Difference Learning. ML4H 2024, PMLR 259:350-363.
# Paper link: https://proceedings.mlr.press/v259/frost25a.html
# Description: Tests for RealTimeICUMortalityPhysioNet2012 (and the shared
#   include_features option) on synthetic PhysioNet 2012 records only.

import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import (
    RealTimeICUMortalityMIMIC4,
    RealTimeICUMortalityPhysioNet2012,
)

ORIGIN = datetime(2012, 1, 1)
TASK = RealTimeICUMortalityPhysioNet2012()
FID = {name: i + 1 for i, name in enumerate(TASK.feature_names)}


def _info(age=70, gender=0, weight=80, survival=-1):
    row = {"age": age, "gender": gender, "weight": weight, "survival": survival}
    return {
        "event_type": "patients",
        "timestamp": None,
        **{f"patients/{k}": str(v) for k, v in row.items()},
    }


def _obs(hours, parameter, value):
    minutes = int(hours * 60)
    return {
        "event_type": "events",
        "timestamp": ORIGIN + timedelta(minutes=minutes),
        "events/minutes": str(minutes),
        "events/parameter": parameter,
        "events/value": str(value),
    }


def _patient(*rows, pid="132539"):
    df = pl.from_dicts(list(rows), infer_schema_length=None)
    df = df.with_columns(pl.lit(pid).alias("patient_id"))
    return Patient(patient_id=pid, data_source=df)


def _rows(sample, key="measurements"):
    return np.asarray(sample[key], dtype=float)


class TestRealTimeICUMortalityPhysioNet2012(unittest.TestCase):
    def test_vocabulary_extends_mimic(self):
        mimic = RealTimeICUMortalityMIMIC4().feature_names
        self.assertEqual(TASK.feature_names[: len(mimic)], mimic)
        self.assertEqual(TASK.num_features, len(mimic) + 16)
        self.assertEqual(len(set(TASK.LAB_PARAMETERS.values()) - set(mimic)), 0)
        with self.assertRaises(ValueError):
            RealTimeICUMortalityPhysioNet2012(feature_set="drugs")

    def test_labs_units_vitals_and_demographics(self):
        patient = _patient(
            _info(age=70, gender=0, weight=80),
            _obs(0, "Creatinine", 1.0),
            _obs(0, "BUN", 28.0),
            _obs(0, "HR", 90),
            _obs(0.5, "Weight", 82.0),
            _obs(1, "Temp", -1),  # invalid value, dropped
            _obs(1, "HR", 95),
        )
        last = _rows(TASK(patient)[-1])
        values = {int(r[0]): r[2] for r in last[:-3]}
        self.assertAlmostEqual(values[FID["Creatinine"]], 88.42, places=4)
        self.assertAlmostEqual(values[FID["Urea"]], 28.0 * 0.357, places=6)
        self.assertEqual(values[FID["Heart rate"]], 95.0)  # newest reading
        self.assertNotIn(FID["Temperature"], values)
        # gender 0 is female in the challenge data -> 1.0; charted weight wins
        np.testing.assert_allclose(last[-3:, 2], [70.0, 1.0, 82.0])

    def test_labs_feature_set_keeps_shared_ids(self):
        patient = _patient(_info(), _obs(0, "Creatinine", 1.0), _obs(0, "HR", 90))
        (sample,) = RealTimeICUMortalityPhysioNet2012(feature_set="labs")(patient)
        ids = set(_rows(sample)[:, 0].astype(int))
        self.assertNotIn(FID["Heart rate"], ids)
        self.assertTrue(max(ids) <= RealTimeICUMortalityMIMIC4().num_features)

    def test_survival_labels(self):
        died = _patient(_info(survival=5), _obs(0, "Lactate", 2.0))
        self.assertEqual(TASK(died)[0]["mortality_horizons"], [0.0, 0.0, 1.0, 1.0, 1.0])
        alive = _patient(_info(survival=-1), _obs(0, "Lactate", 2.0))
        self.assertEqual(TASK(alive)[0]["mortality_horizons"], [0.0] * 5)

    def test_terminal_beyond_observation_window(self):
        patient = _patient(
            _info(), _obs(0, "HR", 80), _obs(30, "HR", 85), _obs(47, "HR", 90)
        )
        s0, s30, s47 = TASK(patient)
        self.assertEqual(s0["terminal"], 0.0)  # next state at 30 h
        np.testing.assert_allclose(_rows(s0, "next_measurements"), _rows(s30))
        self.assertEqual((s30["terminal"], s47["terminal"]), (1.0, 1.0))

    def test_include_features(self):
        patient = _patient(_info(), _obs(0, "Creatinine", 1.0), _obs(0, "HR", 90))
        task = RealTimeICUMortalityPhysioNet2012(include_features=["Heart rate"])
        (sample,) = task(patient)
        ids = _rows(sample)[:-3, 0].astype(int).tolist()
        self.assertEqual(ids, [FID["Heart rate"]])
        with self.assertRaises(ValueError):
            RealTimeICUMortalityMIMIC4(include_features=["Not a feature"])

    def test_edge_cases_and_pre_filter(self):
        self.assertEqual(TASK(_patient(_info())), [])  # no observations
        self.assertEqual(TASK(_patient(_info(age="nan"), _obs(0, "HR", 80))), [])
        frame = pl.DataFrame(
            {
                "patient_id": ["p"] * 3,
                "event_type": ["patients", "events", "events"],
                "events/parameter": [None, "HR", "ICUType"],
            }
        ).lazy()
        kept = TASK.pre_filter(frame).collect()
        self.assertEqual(kept["event_type"].to_list(), ["patients", "events"])


if __name__ == "__main__":
    unittest.main()
