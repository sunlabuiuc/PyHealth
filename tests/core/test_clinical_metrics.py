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
        out = alert_rate_metrics(y, p, rates=(0.1, 0.3))
        self.assertEqual(out["top 10%"]["n_flagged"], 1)
        self.assertAlmostEqual(out["top 10%"]["sensitivity"], 1 / 3)
        self.assertEqual(out["top 10%"]["ppv"], 1.0)
        top30 = out["top 30%"]
        self.assertEqual((top30["n_flagged"], top30["ppv"]), (3, 2 / 3))
        self.assertAlmostEqual(top30["nne"], 1.5)
        self.assertEqual(top30["threshold"], 0.7)
        # too few flagged to publish
        small = alert_rate_metrics(y, p, rates=(0.1,), min_count=11)
        self.assertEqual(small["top 10%"], {"rate": 0.1, "suppressed": True})

    def test_net_benefit_by_hand(self):
        y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        p = [0.9, 0.3, 0.4, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        nb = net_benefit(y, p, thresholds=(0.25,))[0]
        # acting at >= 0.25: TP 2, FP 1, n 10, odds 1/3
        self.assertAlmostEqual(nb["model"], 2 / 10 - 1 / 10 / 3)
        self.assertAlmostEqual(nb["treat_all"], 0.2 - 0.8 / 3)

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

    def test_summary_has_every_part(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 300)
        p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.15, 300), 0.01, 0.99)
        s = clinical_summary(y, p, subgroups={"band": rng.choice(["x", "y"], 300)})
        self.assertAlmostEqual(s["prevalence"], y.mean())
        self.assertIn("calibration_slope", s["calibration"])
        self.assertEqual(len(s["net_benefit"]), 3)
        self.assertTrue(all(math.isfinite(r["model"]) for r in s["net_benefit"]))
        self.assertIn("band", s["subgroups"])


if __name__ == "__main__":
    unittest.main()
