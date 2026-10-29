"""Deterministic hand-calculated anomaly detection metric fixtures; no data.

Contributors: Thi Van Trinh Tran and Zexi Li.
"""

import math
import unittest

import numpy as np

from pyhealth.metrics import (
    anomaly_detection_metrics,
    point_adjust,
    reconstruction_anomaly_scores,
)


class TestAnomalyDetectionMetrics(unittest.TestCase):
    def test_scores_average_squared_error_over_channels(self):
        pred = np.array([[1.0, 2.0], [3.0, 0.0]])
        scores = reconstruction_anomaly_scores(pred, np.zeros((2, 2)), False)
        self.assertEqual(scores.tolist(), [2.5, 4.5])

    def test_feature_normalization_divides_by_channel_mean_error(self):
        pred = np.array([[1.0, 10.0], [3.0, 30.0]])
        scores = reconstruction_anomaly_scores(pred, np.zeros((2, 2)), True)
        # errors [1, 9] and [100, 900] have means 5 and 500 -> both [0.2, 1.8]
        np.testing.assert_allclose(scores, [0.2, 1.8])

    def test_windows_are_concatenated_in_order(self):
        pred = np.arange(12, dtype=float).reshape(2, 3, 2)
        scores = reconstruction_anomaly_scores(pred, np.zeros_like(pred), False)
        flat = reconstruction_anomaly_scores(
            pred.reshape(6, 2), np.zeros((6, 2)), False
        )
        np.testing.assert_array_equal(scores, flat)

    def test_point_adjust_fills_detected_segments_only(self):
        adjusted = point_adjust([0, 1, 0, 0, 0, 1, 0, 1], [1, 1, 1, 0, 1, 1, 0, 0])
        self.assertEqual(adjusted.tolist(), [0, 1, 1, 0, 1, 1, 0, 1])
        # the original leaves step 0 alone when a segment starting there is
        # first hit later; a hit at step 0 fills the segment
        self.assertEqual(point_adjust([1, 0, 0], [1, 1, 0]).tolist(), [1, 1, 0])
        self.assertEqual(point_adjust([0, 0, 0], [1, 1, 0]).tolist(), [0, 0, 0])

    def test_quantile_threshold_and_point_adjusted_scores(self):
        scores = [0.1, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        labels = [0, 1, 1, 1, 0, 0, 0, 0]
        raw = anomaly_detection_metrics(scores, labels, 0.125, point_adjustment=False)
        self.assertAlmostEqual(raw["precision"], 1.0)
        self.assertAlmostEqual(raw["recall"], 1 / 3)
        adjusted = anomaly_detection_metrics(scores, labels, 0.125)
        self.assertAlmostEqual(adjusted["recall"], 1.0)
        self.assertAlmostEqual(adjusted["f1"], 1.0)
        self.assertAlmostEqual(adjusted["roc_auc"], 1.0)
        # the threshold-free AUROC ignores the threshold and the adjustment
        self.assertAlmostEqual(raw["score_roc_auc"], adjusted["score_roc_auc"])
        self.assertAlmostEqual(raw["score_roc_auc"], 7 / 15)

    def test_single_class_labels_give_nan_auroc(self):
        metrics = anomaly_detection_metrics([0.1, 0.5, 0.2], [0, 0, 0], 0.1)
        self.assertTrue(math.isnan(metrics["roc_auc"]))
        self.assertTrue(math.isnan(metrics["score_roc_auc"]))
        self.assertEqual(metrics["f1"], 0.0)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            anomaly_detection_metrics([0.1, 0.2], [0, 1], 1.5)
        with self.assertRaises(ValueError):
            anomaly_detection_metrics([0.1, 0.2], [0, 1, 0], 0.1)
        with self.assertRaises(ValueError):
            point_adjust([[0, 1]], [[0, 1]])
        with self.assertRaises(ValueError):
            reconstruction_anomaly_scores(np.zeros(3), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
