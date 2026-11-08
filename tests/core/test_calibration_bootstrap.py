"""Calibration metrics in binary_metrics_fn, and patient-clustered bootstrap CIs."""

import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from pyhealth.metrics import binary_metrics_fn
from pyhealth.metrics.bootstrap import bootstrap_ci, paired_bootstrap_diff


def _logit(p):
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _calibrated(n=50_000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


class TestCalibrationMetrics(unittest.TestCase):
    def setUp(self):
        self.y, self.p = _calibrated()

    def _metrics(self, p, names):
        return binary_metrics_fn(self.y, p, metrics=names)

    def test_brier_matches_sklearn(self):
        out = self._metrics(self.p, ["brier"])
        self.assertAlmostEqual(out["brier"], brier_score_loss(self.y, self.p))

    def test_oe_ratio_is_observed_over_expected(self):
        out = self._metrics(self.p, ["oe_ratio"])
        self.assertAlmostEqual(out["oe_ratio"], self.y.sum() / self.p.sum())

    def test_calibrated_model(self):
        out = self._metrics(
            self.p, ["oe_ratio", "calibration_slope", "calibration_intercept"]
        )
        self.assertAlmostEqual(out["oe_ratio"], 1.0, delta=0.02)
        self.assertAlmostEqual(out["calibration_slope"], 1.0, delta=0.05)
        self.assertAlmostEqual(out["calibration_intercept"], 0.0, delta=0.05)

    def test_overconfident_model_has_slope_below_one(self):
        overconfident = _sigmoid(2 * _logit(self.p))
        out = self._metrics(overconfident, ["calibration_slope"])
        self.assertAlmostEqual(out["calibration_slope"], 0.5, delta=0.03)

    def test_underpredicting_model_has_positive_intercept(self):
        too_low = _sigmoid(_logit(self.p) - 1.0)
        out = self._metrics(too_low, ["calibration_intercept", "oe_ratio"])
        self.assertAlmostEqual(out["calibration_intercept"], 1.0, delta=0.05)
        self.assertGreater(out["oe_ratio"], 1.0)

    def test_slope_matches_unpenalised_logistic_regression(self):
        y, p = _calibrated(n=3_000, seed=1)
        noisy = _sigmoid(1.3 * _logit(p) + 0.2)
        lr = LogisticRegression(penalty=None, tol=1e-10, max_iter=1000)
        lr.fit(_logit(noisy).reshape(-1, 1), y)
        out = binary_metrics_fn(y, noisy, metrics=["calibration_slope"])
        self.assertAlmostEqual(out["calibration_slope"], lr.coef_[0][0], places=4)


class TestBootstrap(unittest.TestCase):
    def setUp(self):
        self.y, self.p = _calibrated(n=600, seed=2)

    def test_deterministic_and_contains_estimate(self):
        a = bootstrap_ci(self.y, self.p, "roc_auc", n_boot=200, seed=7)
        b = bootstrap_ci(self.y, self.p, "roc_auc", n_boot=200, seed=7)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["estimate"], roc_auc_score(self.y, self.p))
        self.assertLessEqual(a["lower"], a["estimate"])
        self.assertLessEqual(a["estimate"], a["upper"])
        self.assertEqual(a["n_boot"], 200)

    def test_callable_metric(self):
        out = bootstrap_ci(self.y, self.p, roc_auc_score, n_boot=50, seed=0)
        self.assertAlmostEqual(out["estimate"], roc_auc_score(self.y, self.p))

    def test_groups_resample_whole_patients(self):
        # Two samples per patient, each patient with its own probability.
        y = np.repeat(self.y[:200], 2)
        p = np.repeat(self.p[:200], 2)
        groups = np.repeat(np.arange(200), 2)

        def every_patient_whole(y_res, p_res):
            _, counts = np.unique(p_res, return_counts=True)
            if np.any(counts % 2):
                raise AssertionError("a patient's samples were split up")
            return roc_auc_score(y_res, p_res)

        out = bootstrap_ci(y, p, every_patient_whole, groups=groups, n_boot=100, seed=0)
        self.assertEqual(out["n_boot"], 100)

    def test_single_class_resamples_are_skipped_and_counted(self):
        y = np.zeros(40, dtype=int)
        y[0] = 1
        p = np.linspace(0.01, 0.99, 40)
        out = bootstrap_ci(y, p, "roc_auc", n_boot=300, seed=0)
        self.assertGreater(out["n_skipped"], 0)
        self.assertEqual(out["n_boot"] + out["n_skipped"], 300)

    def test_paired_difference_uses_identical_resamples(self):
        same = paired_bootstrap_diff(self.y, self.p, self.p, "roc_auc", n_boot=100, seed=0)
        self.assertEqual((same["estimate"], same["lower"], same["upper"]), (0.0, 0.0, 0.0))

        worse = np.clip(self.p + np.random.default_rng(3).normal(0, 0.3, self.p.size), 0.01, 0.99)
        diff = paired_bootstrap_diff(self.y, self.p, worse, "roc_auc", n_boot=300, seed=0)
        self.assertAlmostEqual(
            diff["estimate"], roc_auc_score(self.y, self.p) - roc_auc_score(self.y, worse)
        )
        self.assertGreater(diff["lower"], 0.0)  # the clean model is clearly better


if __name__ == "__main__":
    unittest.main()
