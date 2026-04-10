import unittest

import numpy as np

from pyhealth.metrics import regression_metrics_fn


class TestRegressionMetrics(unittest.TestCase):
    """Test cases for regression metrics."""

    def setUp(self):
        """Set up synthetic regression data."""
        np.random.seed(42)
        self.x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.x_rec = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

    def test_default_metrics(self):
        """Test that default metrics are returned."""
        result = regression_metrics_fn(self.x, self.x_rec)
        self.assertIn("kl_divergence", result)
        self.assertIn("mse", result)
        self.assertIn("mae", result)
        self.assertEqual(len(result), 3)

    def test_mse(self):
        """Test mean squared error with known values."""
        result = regression_metrics_fn(
            self.x, self.x_rec, metrics=["mse"],
        )
        self.assertIn("mse", result)
        self.assertIsInstance(result["mse"], float)
        self.assertGreaterEqual(result["mse"], 0.0)

    def test_mae(self):
        """Test mean absolute error with known values."""
        result = regression_metrics_fn(
            self.x, self.x_rec, metrics=["mae"],
        )
        self.assertIn("mae", result)
        self.assertIsInstance(result["mae"], float)
        self.assertGreaterEqual(result["mae"], 0.0)

    def test_kl_divergence(self):
        """Test KL divergence metric."""
        result = regression_metrics_fn(
            self.x, self.x_rec, metrics=["kl_divergence"],
        )
        self.assertIn("kl_divergence", result)
        self.assertIsInstance(result["kl_divergence"], float)

    def test_perfect_reconstruction(self):
        """Test that identical arrays yield zero MSE and MAE."""
        x = np.array([1.0, 2.0, 3.0])
        result = regression_metrics_fn(x, x, metrics=["mse", "mae"])
        self.assertAlmostEqual(result["mse"], 0.0)
        self.assertAlmostEqual(result["mae"], 0.0)

    def test_shape_mismatch_raises(self):
        """Test that mismatched shapes raise ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        x_rec = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            regression_metrics_fn(x, x_rec)

    def test_unknown_metric_raises(self):
        """Test that unknown metric name raises ValueError."""
        with self.assertRaises(ValueError):
            regression_metrics_fn(
                self.x, self.x_rec, metrics=["nonexistent"],
            )

    def test_2d_arrays(self):
        """Test that 2D arrays are handled via flattening."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        x_rec = np.array([[1.1, 2.1], [3.1, 4.1]])
        result = regression_metrics_fn(x, x_rec, metrics=["mse", "mae"])
        self.assertIn("mse", result)
        self.assertIn("mae", result)
        self.assertGreater(result["mse"], 0.0)

    def test_single_metric(self):
        """Test requesting a single metric."""
        result = regression_metrics_fn(
            self.x, self.x_rec, metrics=["mae"],
        )
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
