import unittest

import numpy as np

from pyhealth.metrics.multiclass import multiclass_metrics_fn


class TestMulticlassMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n_samples, n_classes = 200, 4
        self.y_true = rng.integers(0, n_classes, size=n_samples)
        logits = rng.normal(size=(n_samples, n_classes))
        logits[np.arange(n_samples), self.y_true] += 1.5
        self.y_prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    def test_default_metrics(self):
        result = multiclass_metrics_fn(self.y_true, self.y_prob)
        self.assertEqual(set(result), {"accuracy", "f1_macro", "f1_micro"})
        for value in result.values():
            self.assertTrue(0.0 <= value <= 1.0)

    def test_roc_auc_variants(self):
        metrics = [
            "roc_auc_macro_ovo",
            "roc_auc_macro_ovr",
            "roc_auc_weighted_ovo",
            "roc_auc_weighted_ovr",
        ]
        result = multiclass_metrics_fn(self.y_true, self.y_prob, metrics=metrics)
        self.assertEqual(set(result), set(metrics))
        for value in result.values():
            self.assertTrue(0.0 <= value <= 1.0)

    def test_balanced_accuracy_jaccard_cohen_kappa(self):
        metrics = [
            "balanced_accuracy",
            "f1_weighted",
            "jaccard_micro",
            "jaccard_macro",
            "jaccard_weighted",
            "cohen_kappa",
        ]
        result = multiclass_metrics_fn(self.y_true, self.y_prob, metrics=metrics)
        self.assertEqual(set(result), set(metrics))

    def test_calibration_metrics(self):
        metrics = ["brier_top1", "ECE", "ECE_adapt", "cwECEt", "cwECEt_adapt"]
        result = multiclass_metrics_fn(self.y_true, self.y_prob, metrics=metrics)
        self.assertEqual(set(result), set(metrics))

    def test_prediction_set_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_predset = np.array(
            [
                [1, 0, 0],  # correct, singleton
                [0, 1, 0],  # correct, singleton
                [1, 1, 0],  # incorrect, rejected (size 2)
                [1, 0, 0],  # correct, singleton
                [1, 1, 0],  # correct but rejected (size 2)
                [0, 0, 1],  # correct, singleton
            ]
        )
        metrics = [
            "rejection_rate",
            "set_size",
            "miscoverage_ps",
            "miscoverage_mean_ps",
            "miscoverage_overall_ps",
            "error_ps",
            "error_mean_ps",
            "error_overall_ps",
        ]
        result = multiclass_metrics_fn(
            y_true,
            np.zeros((6, 3)),  # unused by prediction-set metrics
            metrics=metrics,
            y_predset=y_predset,
        )
        self.assertEqual(set(result), set(metrics))

    def test_prediction_set_metrics_skipped_without_y_predset(self):
        result = multiclass_metrics_fn(
            self.y_true, self.y_prob, metrics=["rejection_rate"]
        )
        self.assertEqual(result, {})

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            multiclass_metrics_fn(
                self.y_true, self.y_prob, metrics=["not_a_real_metric"]
            )


if __name__ == "__main__":
    unittest.main()
