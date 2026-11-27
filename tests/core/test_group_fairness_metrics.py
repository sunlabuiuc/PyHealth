"""Tests for pyhealth.metrics.group_fairness."""

import unittest
import warnings

import numpy as np
from sklearn.metrics import log_loss

from pyhealth.metrics.group_fairness import (
    best_f1_threshold,
    group_fairness_metrics_fn,
)


class TestGroupFairnessMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        self.y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.7, 0.9])
        self.groups = np.array(["M"] * 4 + ["F"] * 4)

    def score(self, **kwargs):
        return group_fairness_metrics_fn(
            self.y_true, self.y_prob, self.groups, threshold=0.5, **kwargs
        )

    def test_auc_per_group_worst_and_gap(self):
        s = self.score()
        self.assertAlmostEqual(s["auc"], 0.875)
        self.assertAlmostEqual(s["auc_M"], 0.75)
        self.assertAlmostEqual(s["auc_F"], 1.0)
        self.assertAlmostEqual(s["worst_auc"], 0.75)
        self.assertAlmostEqual(s["auc_gap"], 0.25)

    def test_threshold_metrics(self):
        s = self.score()
        self.assertAlmostEqual(s["acc"], 0.75)
        self.assertAlmostEqual(s["fpr_M"], 0.0)
        self.assertAlmostEqual(s["fnr_M"], 0.5)
        self.assertAlmostEqual(s["fpr_F"], 0.5)
        self.assertAlmostEqual(s["fnr_F"], 0.0)
        self.assertAlmostEqual(s["fpr"], 0.25)
        self.assertAlmostEqual(s["fnr"], 0.25)
        self.assertEqual(s["threshold"], 0.5)

    def test_equalized_odds(self):
        self.assertAlmostEqual(self.score()["eqodd"], 0.5)

    def test_eqodd_needs_two_groups(self):
        groups = np.array(["a", "a", "b", "b", "c", "c", "c", "c"])
        s = group_fairness_metrics_fn(self.y_true, self.y_prob, groups)
        self.assertNotIn("eqodd", s)
        with self.assertRaises(ValueError):
            group_fairness_metrics_fn(
                self.y_true, self.y_prob, groups, metrics=["eqodd"]
            )

    def test_bce_matches_sklearn(self):
        s = self.score(metrics=["bce"])
        self.assertAlmostEqual(s["bce"], log_loss(self.y_true, self.y_prob))
        self.assertAlmostEqual(s["bce_M"], log_loss(self.y_true[:4], self.y_prob[:4]))
        self.assertNotIn("auc", s)

    def test_tpr_at_tnr80(self):
        s = self.score(metrics=["tpr_at_tnr80"])
        self.assertAlmostEqual(s["tpr_at_tnr80_F"], 1.0)
        self.assertAlmostEqual(s["tpr_at_tnr80_M"], 0.5)

    def test_ece(self):
        groups = np.array(["a"] * 4 + ["b"] * 4)
        y_true = np.array([1, 1, 1, 0, 1, 0, 0, 0])
        y_prob = np.array([0.9] * 4 + [0.25] * 4)
        s = group_fairness_metrics_fn(y_true, y_prob, groups, metrics=["ece"])
        self.assertAlmostEqual(s["ece_a"], 0.15)
        self.assertAlmostEqual(s["ece_b"], 0.0)

    def test_missing_groups_raise(self):
        for missing in (None, float("nan")):
            groups = ["M"] * 7 + [missing]
            with self.assertRaises(ValueError):
                group_fairness_metrics_fn(self.y_true, self.y_prob, groups)

    def test_mixed_type_groups(self):
        groups = [1, 1, 1, 1, "x", "x", "x", "x"]
        s = group_fairness_metrics_fn(self.y_true, self.y_prob, groups, threshold=0.5)
        self.assertAlmostEqual(s["auc_1"], 0.75)
        self.assertAlmostEqual(s["auc_x"], 1.0)

    def test_tuned_threshold_when_none(self):
        s = group_fairness_metrics_fn(self.y_true, self.y_prob, self.groups)
        self.assertAlmostEqual(s["threshold"], 0.7)

    def test_single_class_group_gives_nan_auc(self):
        groups = np.array(["a", "a", "b", "b", "b", "b", "b", "b"])
        with self.assertWarns(UserWarning):
            s = group_fairness_metrics_fn(
                self.y_true, self.y_prob, groups, metrics=["auc", "worst_auc"]
            )
        self.assertTrue(np.isnan(s["auc_a"]))
        self.assertAlmostEqual(s["worst_auc"], s["auc_b"])

    def test_single_class_group_warns_for_rates(self):
        groups = np.array(["a", "a", "b", "b", "b", "b", "b", "b"])
        with self.assertWarnsRegex(UserWarning, r"\['a'\]"):
            s = group_fairness_metrics_fn(
                self.y_true, self.y_prob, groups, metrics=["eqodd"], threshold=0.5
            )
        self.assertTrue(np.isnan(s["eqodd"]))

    def test_no_warning_without_class_dependent_metrics(self):
        groups = np.array(["a", "a", "b", "b", "b", "b", "b", "b"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s = group_fairness_metrics_fn(
                self.y_true, self.y_prob, groups, metrics=["acc"], threshold=0.5
            )
        self.assertIn("acc_a", s)

    def test_integer_groups(self):
        groups = np.array([0] * 4 + [1] * 4)
        s = group_fairness_metrics_fn(self.y_true, self.y_prob, groups, threshold=0.5)
        self.assertAlmostEqual(s["auc_0"], 0.75)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            group_fairness_metrics_fn(self.y_true, self.y_prob, self.groups[:-1])

    def test_bad_labels_or_probs_raise(self):
        with self.assertRaises(ValueError):
            group_fairness_metrics_fn(self.y_true * 2, self.y_prob, self.groups)
        for bad in (1.5, -0.1, np.nan):
            y_prob = self.y_prob.copy()
            y_prob[0] = bad
            with self.assertRaises(ValueError):
                group_fairness_metrics_fn(self.y_true, y_prob, self.groups)

    def test_float_labels_accepted(self):
        s = group_fairness_metrics_fn(
            self.y_true.astype(float), self.y_prob, self.groups, threshold=0.5
        )
        self.assertAlmostEqual(s["auc_M"], 0.75)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            self.score(metrics=["auroc"])

    def test_accepts_column_vector_probs(self):
        s = group_fairness_metrics_fn(
            self.y_true, self.y_prob[:, None], self.groups, threshold=0.5
        )
        self.assertAlmostEqual(s["auc"], 0.875)


class TestHelpers(unittest.TestCase):
    def test_best_f1_threshold(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.7, 0.9])
        self.assertAlmostEqual(best_f1_threshold(y_true, y_prob), 0.7)


if __name__ == "__main__":
    unittest.main()
