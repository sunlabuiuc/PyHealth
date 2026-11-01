"""Tests for the prediction interval metrics and their regression_metrics_fn hook."""

import unittest

import numpy as np

import pyhealth.metrics.prediction_interval as pint
from pyhealth.metrics import regression_metrics_fn


class TestPredictionIntervalHelpers(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([1.0, 2.0, 5.0, 0.0])
        self.y_lower = np.array([0.0, 2.0, 0.0, 1.0])
        self.y_upper = np.array([2.0, 3.0, 4.0, 3.0])

    def test_miscoverage_counts_bounds_as_covered(self):
        # Covered: 1 in [0, 2], 2 in [2, 3]. Missed: 5 > 4, 0 < 1.
        self.assertEqual(
            pint.miscoverage_pi(self.y_lower, self.y_upper, self.y_true), 0.5
        )

    def test_width(self):
        # Widths [2, 1, 4, 2].
        self.assertEqual(pint.width(self.y_lower, self.y_upper), 2.25)

    def test_infinite_width(self):
        self.assertEqual(pint.width(np.array([-np.inf]), np.array([1.0])), np.inf)

    def test_rolling_miscoverage(self):
        np.testing.assert_array_equal(
            pint.rolling_miscoverage_pi(
                self.y_lower, self.y_upper, self.y_true, window=2
            ),
            [0.0, 0.5, 1.0],
        )

    def test_column_inputs_are_flattened(self):
        self.assertEqual(
            pint.miscoverage_pi(
                self.y_lower[:, None], self.y_upper[:, None], self.y_true[:, None]
            ),
            0.5,
        )

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            pint.miscoverage_pi(self.y_lower[:3], self.y_upper, self.y_true)
        with self.assertRaises(ValueError):
            pint.width(self.y_lower[:1], self.y_upper)

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            pint.rolling_miscoverage_pi(
                self.y_lower, self.y_upper, self.y_true, window=5
            )


class TestRegressionMetricsFn(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([1.0, 2.0, 5.0, 0.0])
        self.y_pred = np.array([1.0, 2.5, 2.0, 2.0])
        self.y_lower = np.array([0.0, 2.0, 0.0, 1.0])
        self.y_upper = np.array([2.0, 3.0, 4.0, 3.0])

    def test_interval_metrics(self):
        out = regression_metrics_fn(
            self.y_true,
            self.y_pred,
            metrics=["mae", "miscoverage_pi", "interval_width"],
            y_lower=self.y_lower,
            y_upper=self.y_upper,
        )
        self.assertEqual(set(out), {"mae", "miscoverage_pi", "interval_width"})
        self.assertEqual(out["miscoverage_pi"], 0.5)
        self.assertEqual(out["interval_width"], 2.25)

    def test_rolling_metric_uses_window_from_name(self):
        out = regression_metrics_fn(
            self.y_true,
            self.y_pred,
            metrics=["rolling_miscoverage_pi@3"],
            y_lower=self.y_lower,
            y_upper=self.y_upper,
        )
        np.testing.assert_allclose(out["rolling_miscoverage_pi@3"], [1 / 3, 2 / 3])

    def test_interval_metrics_ignored_without_bounds(self):
        out = regression_metrics_fn(
            self.y_true,
            self.y_pred,
            metrics=["mae", "miscoverage_pi", "rolling_miscoverage_pi@2"],
            y_lower=self.y_lower,
        )
        self.assertEqual(set(out), {"mae"})

    def test_invalid_window_name_raises(self):
        for metric in ("rolling_miscoverage_pi@0", "rolling_miscoverage_pi@x"):
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                regression_metrics_fn(
                    self.y_true,
                    self.y_pred,
                    metrics=[metric],
                    y_lower=self.y_lower,
                    y_upper=self.y_upper,
                )


if __name__ == "__main__":
    unittest.main()
