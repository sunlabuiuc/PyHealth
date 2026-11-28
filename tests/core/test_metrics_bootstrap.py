"""Tests for bootstrap_metrics_ci (multiclass / multilabel CIs).

Synthetic arrays only; small n_boot keeps every test millisecond-fast.
"""

import math
import unittest

import numpy as np

from pyhealth.metrics import (
    binary_metrics_fn,
    bootstrap_metrics_ci,
    multiclass_metrics_fn,
)
from pyhealth.metrics.bootstrap import _resample_indices


def _multiclass_data(n_per_class=5, seed=0):
    """Class indices + noisy one-hot probabilities."""
    rng = np.random.default_rng(seed)
    y_true = np.repeat(np.arange(3), n_per_class)
    y_prob = np.full((y_true.size, 3), 0.2)
    y_prob[np.arange(y_true.size), y_true] = 0.6
    return y_true, y_prob + rng.random((y_true.size, 3)) * 0.1


class TestBootstrapMetricsCI(unittest.TestCase):
    def test_estimate_matches_full_metrics(self):
        y_true, y_prob = _multiclass_data()
        ci = bootstrap_metrics_ci(
            y_true, y_prob, ["f1_macro"], "multiclass", n_boot=20, seed=1
        )
        full = multiclass_metrics_fn(y_true, y_prob, metrics=["f1_macro"])
        self.assertEqual(ci["f1_macro"]["estimate"], full["f1_macro"])

    def test_binary_mode_matches_binary_metrics(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.7, 0.8, 0.4, 0.6, 0.2, 0.9])
        ci = bootstrap_metrics_ci(
            y_true, y_prob, ["roc_auc"], "binary", n_boot=20, seed=2
        )
        full = binary_metrics_fn(y_true, y_prob, metrics=["roc_auc"])
        self.assertEqual(ci["roc_auc"]["estimate"], full["roc_auc"])

    def test_same_seed_is_deterministic(self):
        y_true, y_prob = _multiclass_data()
        first = bootstrap_metrics_ci(
            y_true, y_prob, ["f1_macro"], "multiclass", n_boot=30, seed=5
        )
        second = bootstrap_metrics_ci(
            y_true, y_prob, ["f1_macro"], "multiclass", n_boot=30, seed=5
        )
        self.assertEqual(first, second)

    def test_perfect_classifier_bounds_collapse(self):
        y_true = np.repeat(np.arange(3), 4)
        y_prob = np.eye(3)[y_true]
        ci = bootstrap_metrics_ci(
            y_true, y_prob, ["f1_macro"], "multiclass", n_boot=50, seed=2
        )
        entry = ci["f1_macro"]
        self.assertGreater(entry["n_boot"], 0)
        self.assertEqual(entry["estimate"], 1.0)
        for key in ("mean", "lower", "upper"):
            self.assertAlmostEqual(entry[key], 1.0, places=10)

    def test_multiclass_skips_resamples_missing_a_class(self):
        y_true, y_prob = _multiclass_data(n_per_class=3)
        ci = bootstrap_metrics_ci(
            y_true, y_prob, ["f1_macro"], "multiclass", n_boot=300, seed=7
        )
        entry = ci["f1_macro"]
        self.assertGreater(entry["n_skipped"], 0)
        self.assertEqual(entry["n_boot"] + entry["n_skipped"], 300)

    def test_multilabel_skips_resamples_without_positive_class(self):
        y_true = np.zeros((12, 2))
        y_true[:, 0] = 1.0
        y_true[11, 1] = 1.0  # a single positive for class 1
        y_prob = y_true * 0.4 + 0.3
        ci = bootstrap_metrics_ci(
            y_true, y_prob, ["accuracy"], "multilabel", n_boot=300, seed=3
        )
        entry = ci["accuracy"]
        self.assertGreater(entry["n_skipped"], 0)
        self.assertEqual(entry["n_boot"] + entry["n_skipped"], 300)

    def test_all_positive_class_skips_every_resample(self):
        y_true = np.zeros((8, 2))
        y_true[:, 0] = 1.0  # class 1 never positive: no resample is usable
        y_prob = y_true + 0.1
        ci = bootstrap_metrics_ci(
            y_true, y_prob, ["accuracy"], "multilabel", n_boot=10, seed=4
        )
        entry = ci["accuracy"]
        self.assertEqual(entry["n_boot"], 0)
        self.assertEqual(entry["n_skipped"], 10)
        for key in ("mean", "lower", "upper"):
            self.assertTrue(math.isnan(entry[key]))

    def test_bounds_match_manual_percentiles(self):
        y_true, y_prob = _multiclass_data(n_per_class=4)
        n_boot, seed = 40, 11
        ci = bootstrap_metrics_ci(
            y_true, y_prob, ["f1_macro"], "multiclass",
            n_boot=n_boot, seed=seed, alpha=0.10,
        )
        scores = [
            multiclass_metrics_fn(y_true[idx], y_prob[idx],
                                  metrics=["f1_macro"])["f1_macro"]
            for idx in _resample_indices(len(y_true), None, n_boot, seed)
            if np.unique(y_true[idx]).size == 3
        ]
        lower, upper = np.quantile(scores, [0.05, 0.95])
        self.assertAlmostEqual(ci["f1_macro"]["lower"], lower, places=10)
        self.assertAlmostEqual(ci["f1_macro"]["upper"], upper, places=10)
        self.assertAlmostEqual(ci["f1_macro"]["mean"], np.mean(scores),
                               places=10)

    def test_invalid_mode_raises(self):
        y_true, y_prob = _multiclass_data(n_per_class=2)
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            bootstrap_metrics_ci(y_true, y_prob, ["f1_macro"], "seq")

    def test_invalid_metric_name_raises(self):
        y_true, y_prob = _multiclass_data(n_per_class=2)
        with self.assertRaisesRegex(ValueError, "Unknown metric"):
            bootstrap_metrics_ci(
                y_true, y_prob, ["not_a_metric"], "multiclass"
            )

    def test_multiple_metrics_share_resamples(self):
        y_true, y_prob = _multiclass_data(n_per_class=4)
        ci = bootstrap_metrics_ci(
            y_true, y_prob, ["f1_macro", "accuracy"], "multiclass",
            n_boot=25, seed=9,
        )
        self.assertEqual(
            ci["f1_macro"]["n_boot"] + ci["f1_macro"]["n_skipped"], 25
        )
        self.assertEqual(ci["accuracy"]["n_boot"], ci["f1_macro"]["n_boot"])


if __name__ == "__main__":
    unittest.main()
