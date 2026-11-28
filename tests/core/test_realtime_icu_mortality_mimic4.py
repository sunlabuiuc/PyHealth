# Authors: Hanyang Chen (hc81), Siqi Song (siqis3)
# Paper: Frost, Li & Harris. Robust Real-Time Mortality Prediction in the Intensive
#   Care Unit using Temporal Difference Learning. ML4H 2024, PMLR 259:350-363.
# Paper link: https://proceedings.mlr.press/v259/frost25a.html
# Description: Tests for RealTimeICUMortalityMIMIC4 on synthetic patients only.

import math
import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import RealTimeICUMortalityMIMIC4

T0 = datetime(2150, 1, 1, 8, 0, 0)
FMT = "%Y-%m-%d %H:%M:%S"
CREATININE = "220615"  # chartevents, mg/dL -> umol/L (x 88.42)
LACTATE = "225668"  # chartevents, no unit change
NORADRENALINE = "221906"  # inputevents, mg
FENTANYL_PUSH = "225972"  # inputevents, mcg
VANCOMYCIN = "225798"  # inputevents, antibiotic (binary)

TASK = RealTimeICUMortalityMIMIC4()
NAMES = TASK.feature_names
FID = {name: i + 1 for i, name in enumerate(NAMES)}


def _event(event_type, timestamp, **attrs):
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _lab(hours, itemid, value):
    return _event(
        "chartevents",
        T0 + timedelta(hours=hours),
        itemid=itemid,
        valuenum=value,
        valueuom="",
    )


def _drug(hours, itemid, minutes, amount, unit="mg", weight=70):
    start = T0 + timedelta(hours=hours)
    return _event(
        "inputevents",
        start,
        itemid=itemid,
        endtime=(start + timedelta(minutes=minutes)).strftime(FMT),
        amount=amount,
        amountuom=unit,
        patientweight=weight,
    )


def _patient(*events, death_hours=None, dod=None, gender="F", age=60, pid="p1"):
    deathtime = None
    if death_hours is not None:
        deathtime = (T0 + timedelta(hours=death_hours)).strftime(FMT)
    rows = [
        _event(
            "patients",
            None,
            gender=gender,
            anchor_age=age,
            anchor_year=2150,
            dod=dod,
        ),
        _event("admissions", T0, hadm_id=1, deathtime=deathtime),
        # Every table exists in every patient, as in the global event frame.
        _event("chartevents", None, itemid=None, valuenum=None, valueuom=None),
        _event(
            "inputevents",
            None,
            itemid=None,
            endtime=None,
            amount=None,
            amountuom=None,
            patientweight=None,
        ),
        *events,
    ]
    df = pl.from_dicts(rows, infer_schema_length=None)
    df = df.with_columns(pl.lit(pid).alias("patient_id"))
    return Patient(patient_id=pid, data_source=df)


def _rows(sample, key="measurements"):
    return np.asarray(sample[key], dtype=float)


class TestRealTimeICUMortalityMIMIC4(unittest.TestCase):
    def test_schema_and_vocabulary(self):
        self.assertEqual(TASK.task_name, "RealTimeICUMortalityMIMIC4")
        self.assertEqual(set(TASK.input_schema), {"measurements", "next_measurements"})
        self.assertEqual(TASK.output_schema["mortality"], "binary")
        # 32 labs, 34 infused drugs (rate + bolus), 38 binary drugs, 3 demographics
        self.assertEqual(len(set(TASK.LAB_ITEMIDS.values())), 32)
        self.assertEqual(len(set(TASK.DRUG_ITEMIDS.values())), 72)
        self.assertEqual(TASK.num_features, 32 + 34 * 2 + 38 + 3)
        self.assertEqual(len(set(NAMES)), len(NAMES))
        self.assertEqual(NAMES[-3:], ["age", "gender", "weight"])

    def test_one_sample_per_state_marker(self):
        patient = _patient(
            _lab(0, LACTATE, 2.0), _lab(0, CREATININE, 1.0), _lab(5, LACTATE, 3.0)
        )
        samples = TASK(patient)
        self.assertEqual(len(samples), 2)  # two unique measurement times
        for sample in samples:
            self.assertEqual(_rows(sample).shape[1], 5)
            self.assertEqual(sample["patient_id"], "p1")

    def test_tuples_ordering_units_and_deltas(self):
        patient = _patient(
            _lab(0, CREATININE, 1.0), _lab(2, LACTATE, 1.5), _lab(4, CREATININE, 2.0)
        )
        last = _rows(TASK(patient)[-1])
        clinical, demo = last[:-3], last[-3:]
        # oldest -> newest: creatinine(240 min before), lactate(120), creatinine(0)
        np.testing.assert_allclose(clinical[:, 1], [240.0, 120.0, 0.0])
        newest = clinical[-1]
        self.assertEqual(newest[0], FID["Creatinine"])
        self.assertAlmostEqual(newest[2], 2.0 * 88.42, places=4)
        self.assertAlmostEqual(newest[3], 240.0)  # minutes since last creatinine
        self.assertAlmostEqual(newest[4], 88.42, places=4)
        self.assertTrue(math.isnan(clinical[0][3]))  # first in window: no delta
        np.testing.assert_array_equal(
            demo[:, 0], [FID["age"], FID["gender"], FID["weight"]]
        )
        np.testing.assert_allclose(demo[:, 2], [60.0, 1.0, 74.0])  # F default 74 kg

    def test_next_state_and_terminal(self):
        patient = _patient(
            _lab(0, LACTATE, 2.0), _lab(10, LACTATE, 2.0), _lab(30, LACTATE, 2.5)
        )
        s0, s10, s30 = TASK(patient)
        # Next state of t=0 is the first marker in [24h, 48h]: t=30h.
        self.assertEqual(s0["terminal"], 0.0)
        np.testing.assert_allclose(_rows(s0, "next_measurements"), _rows(s30))
        self.assertEqual(s10["terminal"], 1.0)  # [34h, 58h] holds no marker
        self.assertEqual(s30["terminal"], 1.0)
        np.testing.assert_allclose(_rows(s30, "next_measurements"), _rows(s30))

    def test_next_state_delay_is_configurable(self):
        patient = _patient(_lab(0, LACTATE, 2.0), _lab(5, LACTATE, 2.0))
        task = RealTimeICUMortalityMIMIC4(next_state_delay_hours=4)
        first, _ = task(patient)
        self.assertEqual(first["terminal"], 0.0)
        zero_delay = RealTimeICUMortalityMIMIC4(next_state_delay_hours=0)
        _, last = zero_delay(patient)
        self.assertEqual(last["terminal"], 1.0)  # never its own next state

    def test_mortality_labels_from_deathtime(self):
        patient = _patient(_lab(0, LACTATE, 2.0), death_hours=10 * 24)
        (sample,) = TASK(patient)
        self.assertEqual(sample["mortality_horizons"], [0.0, 0.0, 0.0, 1.0, 1.0])
        self.assertEqual(sample["mortality"], 1)
        self.assertEqual(sample["terminal_reward"], 1.0)
        task = RealTimeICUMortalityMIMIC4(label_horizon_days=7)
        self.assertEqual(task(patient)[0]["mortality"], 0)

    def test_mortality_labels_from_dod_and_survivors(self):
        dod = (T0 + timedelta(days=2)).strftime("%Y-%m-%d")
        died = _patient(_lab(0, LACTATE, 2.0), dod=dod)
        self.assertEqual(TASK(died)[0]["mortality_horizons"][1:], [1.0] * 4)
        alive = _patient(_lab(0, LACTATE, 2.0))
        self.assertEqual(TASK(alive)[0]["mortality_horizons"], [0.0] * 5)

    def test_infusions_boluses_and_binary_drugs(self):
        patient = _patient(
            _drug(0, NORADRENALINE, minutes=60, amount=6.0),  # 0.1 mg/min
            _drug(0.5, NORADRENALINE, minutes=60, amount=12.0),  # +0.2 mg/min
            _drug(3, FENTANYL_PUSH, minutes=1, amount=50, unit="mcg"),  # bolus
            _drug(3, VANCOMYCIN, minutes=90, amount=1000, unit="dose"),
            _lab(4, LACTATE, 2.0),
        )
        last = _rows(TASK(patient)[-1])
        rate = last[last[:, 0] == FID["Noradrenaline rate"]]
        # rates at 0h (0.1), 0.5h (0.3), 1h (0.2), 1.5h (0 = stopped)
        np.testing.assert_allclose(sorted(rate[:, 2]), [0.0, 0.1, 0.2, 0.3], atol=1e-9)
        bolus = last[last[:, 0] == FID["Fentanyl bolus"]]
        np.testing.assert_allclose(bolus[:, 2], [0.05])  # 50 mcg -> 0.05 mg
        vanc = last[last[:, 0] == FID["Vancomycin bolus"]]
        np.testing.assert_allclose(vanc[:, 2], [1.0])
        self.assertNotIn(FID.get("Vancomycin rate", -1), last[:, 0])
        np.testing.assert_allclose(last[-1, 2], 70.0)  # charted weight

    def test_current_infusion_is_prioritised(self):
        patient = _patient(
            _drug(0, NORADRENALINE, minutes=600, amount=60.0),
            _lab(1, LACTATE, 2.0),
        )
        at_1h = _rows(TASK(patient)[1])  # markers: 0h, 1h, 10h (infusion stops)
        current = at_1h[(at_1h[:, 0] == FID["Noradrenaline rate"]) & (at_1h[:, 1] == 0)]
        np.testing.assert_allclose(current[:, 2], [0.1])
        np.testing.assert_allclose(current[:, 3:], [[0.0, 0.0]])

    def test_max_measurements_keeps_novel_features(self):
        events = [_lab(h * 0.1, LACTATE, 2.0 + h) for h in range(20)]
        events.append(_lab(0, CREATININE, 1.0))  # oldest, but a novel feature
        task = RealTimeICUMortalityMIMIC4(max_measurements=8)
        last = _rows(task(_patient(*events))[-1])
        self.assertEqual(len(last), 8)
        self.assertIn(FID["Creatinine"], last[:-3, 0])
        lactate_times = last[last[:, 0] == FID["Lactate"], 1]
        np.testing.assert_allclose(lactate_times, [18.0, 12.0, 6.0, 0.0])

    def test_lookback_window(self):
        patient = _patient(_lab(0, CREATININE, 1.0), _lab(10, LACTATE, 2.0))
        task = RealTimeICUMortalityMIMIC4(lookback_hours=5)
        last = _rows(task(patient)[-1])
        self.assertNotIn(FID["Creatinine"], last[:, 0])

    def test_feature_sets(self):
        patient = _patient(
            _drug(0, NORADRENALINE, minutes=60, amount=6.0), _lab(0, LACTATE, 2.0)
        )
        labs = RealTimeICUMortalityMIMIC4(feature_set="labs")(patient)
        ids = set(_rows(labs[-1])[:, 0])
        self.assertNotIn(FID["Noradrenaline rate"], ids)
        drugs = RealTimeICUMortalityMIMIC4(feature_set="drugs")(patient)
        self.assertNotIn(FID["Lactate"], set(_rows(drugs[0])[:, 0]))
        with self.assertRaises(ValueError):
            RealTimeICUMortalityMIMIC4(feature_set="vitals")

    def test_state_subsampling_is_deterministic(self):
        patient = _patient(*[_lab(h, LACTATE, 2.0) for h in range(30)])
        task = RealTimeICUMortalityMIMIC4(state_sample_prob=0.5, seed=3)
        first = [_rows(s) for s in task(patient)]
        second = [_rows(s) for s in task(patient)]
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)  # NaN-aware
        self.assertLess(len(first), 30)
        capped = RealTimeICUMortalityMIMIC4(max_states_per_patient=4)(patient)
        self.assertEqual(len(capped), 4)

    def test_edge_cases(self):
        self.assertEqual(TASK(_patient()), [])  # no measurements
        unknown = _patient(_lab(0, "999999", 1.0))
        self.assertEqual(TASK(unknown), [])
        bad_value = _patient(_lab(0, LACTATE, "n/a"))
        self.assertEqual(TASK(bad_value), [])
        with self.assertRaises(ValueError):
            RealTimeICUMortalityMIMIC4(max_measurements=2)

    def test_pre_filter(self):
        rows = [
            {"patient_id": "p", "event_type": "chartevents", "chartevents/itemid": i}
            for i in (LACTATE, "220045")
        ] + [{"patient_id": "p", "event_type": "labevents", "chartevents/itemid": None}]
        rows.append({"patient_id": "p", "event_type": "patients"})
        frame = pl.DataFrame(rows).lazy()
        kept = TASK.pre_filter(frame).collect()
        self.assertEqual(
            sorted(kept["event_type"].to_list()), ["chartevents", "patients"]
        )


if __name__ == "__main__":
    unittest.main()
