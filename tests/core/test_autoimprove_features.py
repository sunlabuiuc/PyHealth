"""pyhealth.autoimprove.features: operators, definitions, refusals and the future-blind test."""

import math
import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pyhealth.autoimprove.evaluate import fold_parts
from pyhealth.autoimprove.features import EngineeredDataset, FeatureSpace, FeatureSpec
from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset

T = datetime(2020, 12, 31)


def day(n):
    """n days from the prediction time (negative is before)."""
    return T + timedelta(days=n)


def _events():
    rows = []
    for pid in [f"p{i}" for i in range(12)]:  # 12 patients share the SABA rows
        rows += [
            {"patient_id": pid, "event_type": "med_orders", "timestamp": day(d),
             "med_orders/drug_class": cls}
            for d, cls in [(-10, "SABA"), (-40, "SABA"), (-100, "SABA"),
                           (-400, "SABA"), (-20, "controller")]
        ]
    rows += [
        {"patient_id": "p0", "event_type": "labs", "timestamp": day(d),
         "labs/eos": v}
        for d, v in [(-30, "300"), (-200, "500"), (-500, "100")]
    ]
    rows += [
        # an old stay, and one that began before t but ended after it
        {"patient_id": "p0", "event_type": "admissions", "timestamp": day(-50),
         "admissions/los": "5", "admissions/dischtime": str(day(-45))},
        {"patient_id": "p0", "event_type": "admissions", "timestamp": day(-3),
         "admissions/los": "7", "admissions/dischtime": str(day(4))},
        {"patient_id": "p0", "event_type": "admissions", "timestamp": day(10),
         "admissions/los": "2", "admissions/dischtime": str(day(12))},
    ]
    return pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime("us")))


class FakeDataset:
    """The parts of a BaseDataset a FeatureSpace uses."""

    tables = ["med_orders", "labs", "admissions"]

    def __init__(self):
        self.df = _events()

    @property
    def global_event_df(self):
        return self.df.lazy()

    def get_patient(self, pid):
        return Patient(pid, self.df.filter(pl.col("patient_id") == pid))


VOCAB = {"SABA": {"event_type": "med_orders", "attribute": "drug_class",
                  "values": ["SABA"], "noun": "SABA orders"},
         "controller": {"event_type": "med_orders", "attribute": "drug_class",
                        "values": ["controller"], "noun": "controller orders"}}


def _samples(n=2):
    samples = [{"patient_id": f"p{i}", "t": T, "x": [float(i), 1.0], "label": i % 2}
               for i in range(n)]
    return create_sample_dataset(samples, {"x": "tensor"}, {"label": "binary"})


def _space(**kw):
    return FeatureSpace(FakeDataset(), time_key="t", vocabulary=VOCAB,
                        columns={"x": ["age", "bias"]},
                        known_at={"admissions": "dischtime"}, **kw)


def _one(fs, spec, samples=None):
    spec = fs.validate({**spec, "rationale": "test", "direction": "+"})
    return fs.compute([spec], samples or _samples())[0, 0], spec


SABA = {"event_type": "med_orders", "group": "SABA"}


class TestOperators(unittest.TestCase):
    def setUp(self):
        self.fs = _space()

    def test_counts_and_flags(self):
        self.assertEqual(_one(self.fs, {"op": "count", "window": 365, "select": SABA})[0], 3)
        self.assertEqual(_one(self.fs, {"op": "exists", "window": 30, "select": {
            "event_type": "med_orders", "group": "controller"}})[0], 1)
        self.assertEqual(_one(self.fs, {"op": "days_since", "select": SABA})[0], 10)
        value, spec = _one(self.fs, {"op": "ge", "x": 3, "of": {
            "op": "count", "window": 365, "select": SABA}})
        self.assertEqual(value, 1)
        self.assertEqual(spec.name, "ge3__count_365d__med_orders_SABA")
        self.assertEqual(spec.definition, "1 if there were 3 or more SABA orders in "
                         "the 365 days before the prediction time, else 0")

    def test_values(self):
        lab = {"event_type": "labs"}
        stat = {"op": "stat", "select": lab, "attr": "eos", "window": 365}
        for agg, expected in [("max", 500), ("min", 300), ("mean", 400), ("last", 300)]:
            self.assertEqual(_one(self.fs, {**stat, "agg": agg})[0], expected)
        slope = _one(self.fs, {**stat, "agg": "slope"})[0]
        self.assertAlmostEqual(slope, (300 - 500) / 170)
        value, spec = _one(self.fs, {"op": "band", "cuts": [150, 300, 500],
                                     "of": {**stat, "agg": "max"}})
        self.assertEqual(value, 3)
        self.assertIn("0 below 150, 1 from 150 to below 300", spec.definition)
        # windows are [t - w, t): day -30 is inside a 30-day window
        self.assertEqual(_one(self.fs, {**stat, "agg": "max", "window": 30})[0], 300)
        spec = self.fs.validate({**stat, "agg": "max", "rationale": "r"})
        self.assertTrue(math.isnan(self.fs.compute([spec], _samples())[1, 0]))  # no labs

    def test_distinct_trend_blocks_and_combinations(self):
        med = {"event_type": "med_orders"}
        self.assertEqual(_one(self.fs, {"op": "n_distinct", "select": med,
                                        "attr": "drug_class", "window": 365})[0], 2)
        self.assertAlmostEqual(_one(self.fs, {"op": "trend", "select": SABA,
                                              "short": 30, "long": 365})[0],
                               1 - 3 * 30 / 365)
        value, spec = _one(self.fs, {"op": "blocks_with", "select": SABA,
                                     "block": 91, "n": 4})
        self.assertEqual(value, 2)
        self.assertIn("how many of the 4 periods of 91 days", spec.definition)
        adm = {"op": "count", "window": 365, "select": {"event_type": "admissions"}}
        saba = {"op": "count", "window": 365, "select": SABA}
        self.assertEqual(_one(self.fs, {"op": "ratio", "a": adm, "b": saba})[0], 0.75)
        self.assertEqual(_one(self.fs, {"op": "diff", "a": saba, "b": adm})[0], 1)
        self.assertEqual(_one(self.fs, {"op": "any", "a": adm, "b": saba})[0], 1)
        self.assertEqual(_one(self.fs, {"op": "cross", "a": adm, "b": saba})[0], 6)
        column = self.fs.validate({"op": "column", "column": "age", "rationale": "r"})
        self.assertEqual(self.fs.compute([column], _samples(4))[:, 0].tolist(),
                         [0.0, 1.0, 2.0, 3.0])

    def test_where_filters_and_names(self):
        value, spec = _one(self.fs, {"op": "count", "window": 365, "select": {
            "event_type": "med_orders", "where": {"drug_class": ["controller"]}}})
        self.assertEqual(value, 1)
        self.assertEqual(spec.name, "count_365d__med_orders_drug_class_controller")
        self.assertIn("with drug_class controller", spec.definition)


class TestRefusals(unittest.TestCase):
    def setUp(self):
        self.fs = _space()

    def refused(self, spec, text):
        with self.assertRaises(ValueError) as cm:
            self.fs.validate(spec)
        self.assertIn(text, str(cm.exception))

    def test_refusals_carry_a_reason(self):
        ok = {"op": "count", "window": 365, "select": SABA, "rationale": "r"}
        self.refused({**ok, "rationale": ""}, "rationale")
        self.refused({**ok, "window": 100}, "window must be one of")
        self.refused({**ok, "op": "exec"}, "unknown op")
        self.refused({**ok, "select": {"event_type": "notes"}}, "unknown event_type")
        self.refused({**ok, "select": {"event_type": "med_orders", "group": "LABA"}},
                     "unknown group")
        self.refused({**ok, "select": {"event_type": "labs", "group": "SABA"}},
                     "belongs to")
        # labs/eos values occur for one patient only: below the small-cell limit
        self.refused({**ok, "select": {"event_type": "labs", "where": {"eos": ["300"]}}},
                     "at least 11 patients")
        self.refused({**ok, "direction": "sideways"}, "direction")
        self.refused({"op": "ge", "x": 1, "rationale": "r", "of": {
            "op": "ge", "x": 1, "of": {"op": "ge", "x": 1, "of": ok}}}, "nest")
        with self.assertRaises(ValueError):
            self.fs.validate_many([ok] * 9)
        with self.assertRaises(ValueError):
            self.fs.validate_many([ok, ok])

    def test_round_trip(self):
        spec = self.fs.validate({"op": "ge", "x": 3, "of": {
            "op": "count", "window": 365, "select": SABA},
            "rationale": "reliever overuse signals poor control", "direction": "+",
            "citations": ["PMID 1"]})
        again = self.fs.validate(FeatureSpec.from_dict(spec.to_dict()).to_dict())
        self.assertEqual(again, spec)


class TestFutureBlind(unittest.TestCase):
    def test_value_known_after_t_is_caught(self):
        fs = _space()
        leak = fs.validate({"op": "stat", "agg": "max", "attr": "los", "window": 30,
                            "select": {"event_type": "admissions"}, "rationale": "r"})
        safe = fs.validate({"op": "count", "window": 365, "select": SABA,
                            "rationale": "r"})
        self.assertEqual(fs.compute([leak], _samples())[0, 0], 7)  # the leak
        result = fs.future_blind([leak, safe], _samples())
        self.assertIn("known at or after", result[leak.name])
        self.assertIsNone(result[safe.name])

    def test_derived_event_built_from_all_events_is_caught(self):
        def all_events(patient, t):  # wrong: ignores t
            stays = patient.get_events("admissions", return_df=True)
            return [{"event_type": "episodes", "timestamp": t - timedelta(days=1),
                     "n_stays": stays.height}]

        def before_t(patient, t):  # right
            stays = patient.get_events("admissions", end=t, return_df=True)
            stays = stays.filter(pl.col("timestamp") < t)
            return [{"event_type": "episodes", "timestamp": t - timedelta(days=1),
                     "n_stays": stays.height}]

        spec = {"op": "stat", "agg": "max", "attr": "n_stays", "window": 30,
                "select": {"event_type": "episodes"}, "rationale": "r"}
        for hook, leaks in ((all_events, True), (before_t, False)):
            fs = _space(derived_events=hook, derived_types={"episodes": ["n_stays"]})
            valid = fs.validate(spec)
            reason = fs.future_blind([valid], _samples())[valid.name]
            self.assertEqual(reason is not None, leaks)

    def test_check_reports_what_a_reviewer_needs(self):
        rows = _space().check([{"op": "exists", "window": 90, "select": SABA,
                                "rationale": "recent reliever use", "direction": "+"}],
                              _samples(3))
        self.assertEqual(rows[0]["coverage"], "100%")
        self.assertIsNone(rows[0]["future_blind"])
        self.assertIn("SABA order", rows[0]["definition"])


class TestEngineeredDataset(unittest.TestCase):
    def test_per_fold_statistics(self):
        base = _samples(6)
        raw = np.array([[1.0], [2.0], [np.nan], [3.0], [100.0], [4.0]])
        fe = EngineeredDataset(base, raw, ["f"])
        self.assertEqual(fe.input_schema["engineered"], "tensor")
        self.assertEqual(len(fe), 6)
        fit, held, _ = fold_parts(fe, fe, [0, 1, 2, 3], [4, 5])
        expected = fe.fit_stats([0, 1, 2, 3])
        self.assertEqual(fit.stats, expected)
        self.assertEqual(held.stats, expected)  # held out fold: training stats only
        self.assertEqual(expected["median"], [2.0])
        # the missing value is imputed with the fold median, then scaled
        value = float(fit[2]["engineered"][0])
        self.assertAlmostEqual(value, (2.0 - expected["mean"][0]) / expected["sd"][0],
                               places=5)
        self.assertEqual(held.raw[:, 0].tolist(), [100.0, 4.0])


if __name__ == "__main__":
    unittest.main()
