"""Tests for pyhealth.metrics.regression.regression_metrics_fn.

Author: Syed Abdul Basit
Description: Unit tests for the regression metrics (kl_divergence, mse,
    mae), including the shape check that now runs before flattening.
    Uses small synthetic arrays only, so the tests run in milliseconds.
"""

import unittest

import numpy as np

from pyhealth.metrics.regression import regression_metrics_fn


class TestRegressionMetrics(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0.1, 0.2, 0.3, 0.4])
        self.x_rec = np.array([0.1, 0.25, 0.25, 0.4])

    def test_default_metrics(self):
        result = regression_metrics_fn(self.x, self.x_rec)
        self.assertEqual(set(result), {"kl_divergence", "mse", "mae"})

    def test_mse_and_mae_values(self):
        result = regression_metrics_fn(self.x, self.x_rec, metrics=["mse", "mae"])
        self.assertAlmostEqual(result["mse"], 0.00125)
        self.assertAlmostEqual(result["mae"], 0.025)

    def test_returns_python_floats(self):
        result = regression_metrics_fn(self.x, self.x_rec)
        for value in result.values():
            self.assertIs(type(value), float)

    def test_kl_divergence_matches_manual_value(self):
        p = self.x / self.x.sum()
        q = self.x_rec / self.x_rec.sum()
        expected = float(np.sum(q * np.log(q / p)))
        result = regression_metrics_fn(self.x, self.x_rec, metrics=["kl_divergence"])
        self.assertAlmostEqual(result["kl_divergence"], expected)

    def test_identical_inputs_give_zero(self):
        result = regression_metrics_fn(self.x, self.x.copy())
        self.assertAlmostEqual(result["kl_divergence"], 0.0)
        self.assertAlmostEqual(result["mse"], 0.0)
        self.assertAlmostEqual(result["mae"], 0.0)

    def test_inputs_are_not_mutated(self):
        x = np.array([0.0, 0.5, 0.5])
        x_rec = np.array([0.2, 0.0, 0.8])
        x_before, x_rec_before = x.copy(), x_rec.copy()
        regression_metrics_fn(x, x_rec)
        np.testing.assert_array_equal(x, x_before)
        np.testing.assert_array_equal(x_rec, x_rec_before)

    def test_multidimensional_inputs_with_same_shape(self):
        x = self.x.reshape(2, 2)
        x_rec = self.x_rec.reshape(2, 2)
        result = regression_metrics_fn(x, x_rec, metrics=["mse"])
        self.assertAlmostEqual(result["mse"], 0.00125)

    def test_different_shapes_with_same_size_raise(self):
        """(2, 3) and (3, 2) have the same size, so they used to pass the
        check silently because it ran after flattening."""
        x = np.arange(6, dtype=float).reshape(2, 3)
        x_rec = np.arange(6, dtype=float).reshape(3, 2)
        with self.assertRaises(ValueError):
            regression_metrics_fn(x, x_rec)

    def test_different_sizes_raise(self):
        with self.assertRaises(ValueError):
            regression_metrics_fn(np.zeros(3), np.zeros(4))

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            regression_metrics_fn(self.x, self.x_rec, metrics=["r2"])


if __name__ == "__main__":
    unittest.main()
