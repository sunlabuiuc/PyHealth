import unittest

import numpy as np

from pyhealth.metrics import multiclass_metrics_fn


class TestMulticlassMetrics(unittest.TestCase):
    """Test cases for multiclass classification metrics."""

    def setUp(self):
        """Set up synthetic multiclass classification data."""
        np.random.seed(42)
        self.y_true = np.array([0, 1, 2, 2, 0, 1])
        self.y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.4, 0.3, 0.3],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
        ])

    def test_default_metrics(self):
        """Test that default metrics are returned."""
        result = multiclass_metrics_fn(self.y_true, self.y_prob)
        self.assertIn("accuracy", result)
        self.assertIn("f1_macro", result)
        self.assertIn("f1_micro", result)
        self.assertEqual(len(result), 3)

    def test_accuracy(self):
        """Test accuracy metric with known values."""
        result = multiclass_metrics_fn(
            self.y_true, self.y_prob, metrics=["accuracy"],
        )
        self.assertIn("accuracy", result)
        self.assertIsInstance(result["accuracy"], float)
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_all_f1_variants(self):
        """Test all F1 score averaging methods."""
        metrics = ["f1_micro", "f1_macro", "f1_weighted"]
        result = multiclass_metrics_fn(
            self.y_true, self.y_prob, metrics=metrics,
        )
        for m in metrics:
            self.assertIn(m, result)
            self.assertGreaterEqual(result[m], 0.0)
            self.assertLessEqual(result[m], 1.0)

    def test_roc_auc_variants(self):
        """Test all ROC AUC averaging methods."""
        metrics = [
            "roc_auc_macro_ovo", "roc_auc_macro_ovr",
            "roc_auc_weighted_ovo", "roc_auc_weighted_ovr",
        ]
        result = multiclass_metrics_fn(
            self.y_true, self.y_prob, metrics=metrics,
        )
        for m in metrics:
            self.assertIn(m, result)
            self.assertGreaterEqual(result[m], 0.0)
            self.assertLessEqual(result[m], 1.0)

    def test_jaccard_variants(self):
        """Test all Jaccard score averaging methods."""
        metrics = ["jaccard_micro", "jaccard_macro", "jaccard_weighted"]
        result = multiclass_metrics_fn(
            self.y_true, self.y_prob, metrics=metrics,
        )
        for m in metrics:
            self.assertIn(m, result)
            self.assertGreaterEqual(result[m], 0.0)
            self.assertLessEqual(result[m], 1.0)

    def test_calibration_metrics(self):
        """Test calibration metrics (ECE, brier_top1)."""
        metrics = ["ECE", "ECE_adapt", "brier_top1"]
        result = multiclass_metrics_fn(
            self.y_true, self.y_prob, metrics=metrics,
        )
        for m in metrics:
            self.assertIn(m, result)
            self.assertIsInstance(result[m], float)
            self.assertGreaterEqual(result[m], 0.0)

    def test_classwise_ece(self):
        """Test classwise ECE metrics."""
        metrics = ["cwECEt", "cwECEt_adapt"]
        result = multiclass_metrics_fn(
            self.y_true, self.y_prob, metrics=metrics,
        )
        for m in metrics:
            self.assertIn(m, result)
            self.assertGreaterEqual(result[m], 0.0)

    def test_cohen_kappa(self):
        """Test Cohen's kappa score."""
        result = multiclass_metrics_fn(
            self.y_true, self.y_prob, metrics=["cohen_kappa"],
        )
        self.assertIn("cohen_kappa", result)
        self.assertGreaterEqual(result["cohen_kappa"], -1.0)
        self.assertLessEqual(result["cohen_kappa"], 1.0)

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 2])
        y_prob = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = multiclass_metrics_fn(
            y_true, y_prob, metrics=["accuracy"],
        )
        self.assertEqual(result["accuracy"], 1.0)

    def test_unknown_metric_raises(self):
        """Test that unknown metric name raises ValueError."""
        with self.assertRaises(ValueError):
            multiclass_metrics_fn(
                self.y_true, self.y_prob, metrics=["nonexistent"],
            )

    def test_hits_and_rank_metrics(self):
        """Test hits@n and mean_rank metrics."""
        result = multiclass_metrics_fn(
            self.y_true, self.y_prob,
            metrics=["hits@n", "mean_rank"],
        )
        self.assertIn("HITS@1", result)
        self.assertIn("HITS@5", result)
        self.assertIn("HITS@10", result)
        self.assertIn("mean_rank", result)
        self.assertIn("mean_reciprocal_rank", result)


if __name__ == "__main__":
    unittest.main()
