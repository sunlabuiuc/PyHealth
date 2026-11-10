"""pyhealth.metrics.clinical: alerts, net benefit, calibration bins, subgroups."""

import math
import unittest

import numpy as np

from pyhealth.metrics import (
    alert_rate_metrics,
    calibration_bins,
    clinical_summary,
    net_benefit,
    subgroup_metrics,
)


class TestClinicalMetrics(unittest.TestCase):
    def test_alert_rates_by_hand(self):
        y = [1, 0, 1, 0, 0, 0, 0, 0, 1, 0]
        p = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        out = alert_rate_metrics(y, p, rates=(0.1, 0.3), min_count=0)
        self.assertEqual(out["top 10%"]["n_flagged"], 1)
        self.assertAlmostEqual(out["top 10%"]["sensitivity"], 1 / 3)
        self.assertEqual(out["top 10%"]["ppv"], 1.0)
        top30 = out["top 30%"]
        self.assertEqual((top30["n_flagged"], top30["ppv"]), (3, 2 / 3))
        self.assertAlmostEqual(top30["nne"], 1.5)
        # midway between the lowest flagged (0.7) and next (0.6) risks, so no
        # one sample's risk is published
        self.assertAlmostEqual(top30["threshold"], 0.65)
        everyone = alert_rate_metrics(y, p, rates=(1.0,), min_count=0)["top 100%"]
        self.assertEqual(everyone["threshold"], 0.0)
        # too few flagged to publish; withheld by default
        small = alert_rate_metrics(y, p, rates=(0.1,))
        self.assertEqual(small["top 10%"], {"rate": 0.1, "suppressed": True})

    def test_net_benefit_by_hand(self):
        y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        p = [0.9, 0.3, 0.4, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        nb = net_benefit(y, p, thresholds=(0.25,), min_count=0)[0]
        # acting at >= 0.25: TP 2, FP 1, n 10, odds 1/3
        self.assertAlmostEqual(nb["model"], 2 / 10 - 1 / 10 / 3)
        self.assertAlmostEqual(nb["treat_all"], 0.2 - 0.8 / 3)
        # by default TP 2 and FP 1 are too few to publish; treat_all still is
        nb = net_benefit(y, p, thresholds=(0.25,))[0]
        self.assertEqual(sorted(nb), ["suppressed", "threshold", "treat_all"])
        self.assertAlmostEqual(nb["treat_all"], 0.2 - 0.8 / 3)

    def test_net_benefit_withholds_small_true_or_false_positive_counts(self):
        y = np.array([1, 0] * 50)
        p = np.linspace(1, 0, 100)
        # acting on the top 30: 15 TP, 15 FP; the top 20: 10 TP, 10 FP
        rows = net_benefit(y, p, thresholds=(p[29], p[19]))
        self.assertIn("model", rows[0])
        self.assertTrue(rows[1]["suppressed"])
        # the top 30 come from 5 patients: withheld
        rows = net_benefit(y, p, thresholds=(p[29],), groups=np.arange(100) % 5)
        self.assertTrue(rows[0]["suppressed"])

    def test_counts_cannot_be_recovered_from_their_complement(self):
        rng = np.random.default_rng(5)
        y = np.zeros(300, dtype=int)
        y[:8] = 1
        p = rng.uniform(size=300)
        # n and positives are withheld together: n - positives is not derivable
        s = clinical_summary(y, p)
        self.assertEqual((s["n"], s["positives"]), (None, None))
        y = (rng.uniform(size=300) < 0.3).astype(int)
        site = np.array(["a"] * 140 + ["b"] * 150 + ["c"] * 10)
        groups = subgroup_metrics(y, p, {"site": site})["site"]["groups"]
        # c is too small; a (the next smallest) is withheld too, so c's
        # counts are not the total minus a and b
        self.assertEqual((groups["c"]["n"], groups["c"]["positives"]), (None, None))
        self.assertEqual((groups["a"]["n"], groups["a"]["positives"]), (None, None))
        self.assertIn("roc_auc", groups["a"])  # its metrics are still shown
        self.assertEqual(groups["b"]["n"], 150)
        # a group with few non-events hides both counts, not only one
        y2 = np.ones(300, dtype=int)
        y2[:150] = rng.integers(0, 2, 150)
        two = subgroup_metrics(y2, p, {"half": ["x"] * 150 + ["y"] * 150})
        self.assertEqual(two["half"]["groups"]["y"]["n"], None)
        self.assertEqual(two["half"]["groups"]["y"]["positives"], None)

    def test_calibration_bins_merge_small_cells(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=50)
        y = (rng.uniform(size=50) < p).astype(int)
        bins = calibration_bins(y, p, n_bins=10, min_count=11)
        self.assertTrue(all(b["n"] >= 11 for b in bins))
        self.assertEqual(sum(b["n"] for b in bins), 50)
        means = [b["mean_predicted"] for b in bins]
        self.assertEqual(means, sorted(means))
        # with patient ids, a bin needs 11 distinct patients
        patients = np.repeat(np.arange(10), 5)  # 10 patients, 5 samples each
        self.assertEqual(calibration_bins(y, p, 10, 11, groups=patients), [])

    def test_subgroups_suppress_and_gap(self):
        rng = np.random.default_rng(1)
        n = 400
        sex = np.where(np.arange(n) < 300, "F", "M")
        y = rng.integers(0, 2, n)
        p = np.where(sex == "F", 0.2 + 0.6 * y, rng.uniform(size=n))
        small = np.array(["a"] * (n - 5) + ["b"] * 5)
        out = subgroup_metrics(y, p, {"sex": sex, "site": small})
        self.assertEqual(out["sex"]["groups"]["F"]["roc_auc"], 1.0)
        self.assertGreater(out["sex"]["max_gap"]["roc_auc"], 0.3)
        site_b = out["site"]["groups"]["b"]
        self.assertIsNone(site_b["n"])  # 5 samples: suppressed
        self.assertNotIn("roc_auc", site_b)
        self.assertEqual(out["site"]["max_gap"], {})

    def test_alerts_withhold_small_caught_or_false_alarm_counts(self):
        y = np.array([1] * 5 + [0] * 95)  # 5 events: caught counts are small
        p = np.linspace(1, 0, 100)
        out = alert_rate_metrics(y, p, rates=(0.2,), min_count=11)
        self.assertEqual(out["top 20%"], {"rate": 0.2, "suppressed": True})
        y = np.array([1] * 30 + [0] * 70)  # 20 flagged, all events: 0 false alarms
        out = alert_rate_metrics(y, p, rates=(0.2,), min_count=11)
        self.assertTrue(out["top 20%"]["suppressed"])
        y = np.array([1, 0] * 50)  # 20 flagged: 10 caught, 10 false alarms
        self.assertTrue(alert_rate_metrics(y, p, (0.2,), 11)["top 20%"]["suppressed"])
        self.assertEqual(alert_rate_metrics(y, p, (0.3,), 11)["top 30%"]["n_flagged"], 30)
        # with patient ids, 30 flagged samples from 5 patients are withheld
        patients = np.arange(100) % 5
        out = alert_rate_metrics(y, p, (0.3,), 11, groups=patients)
        self.assertTrue(out["top 30%"]["suppressed"])

    def test_calibration_bins_have_enough_events_and_non_events(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(size=400)
        y = (rng.uniform(size=400) < p).astype(int)
        bins = calibration_bins(y, p, n_bins=10, min_count=11)
        self.assertGreater(len(bins), 1)
        for b in bins:
            events = round(b["observed"] * b["n"])
            self.assertGreaterEqual(min(events, b["n"] - events), 11)

    def test_summary_withholds_results_with_few_events(self):
        rng = np.random.default_rng(4)
        y = np.zeros(300, dtype=int)
        y[:8] = 1  # 8 events
        p = rng.uniform(size=300)
        s = clinical_summary(y, p)
        self.assertIsNone(s["n"])
        self.assertIsNone(s["positives"])
        self.assertIsNone(s["prevalence"])
        self.assertEqual((s["alerts"], s["calibration_bins"], s["net_benefit"]),
                         ({}, [], []))
        self.assertIn("brier", s["calibration"])
        y[:40] = 1
        s = clinical_summary(y, p)
        self.assertAlmostEqual(s["prevalence"], 40 / 300)
        self.assertTrue(s["net_benefit"])

    def test_summary_has_every_part(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 300)
        p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.15, 300), 0.01, 0.99)
        s = clinical_summary(y, p, subgroups={"band": rng.choice(["x", "y"], 300)})
        self.assertAlmostEqual(s["prevalence"], y.mean())
        self.assertIn("calibration_slope", s["calibration"])
        self.assertEqual(len(s["net_benefit"]), 3)
        shown = [r["model"] for r in s["net_benefit"] if not r.get("suppressed")]
        self.assertTrue(shown)
        self.assertTrue(all(math.isfinite(m) for m in shown))
        self.assertIn("band", s["subgroups"])


if __name__ == "__main__":
    unittest.main()
