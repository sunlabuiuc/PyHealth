"""Regression tests for KL divergence on integer-valued arrays.

Author: Pema Sangpo Lama
Description: Exercise zero clipping using tiny synthetic count arrays.
"""

import math
import unittest

import numpy as np

from pyhealth.metrics import regression_metrics_fn, timeseries_regression_metrics_fn


class TestIntegerKLDivergence(unittest.TestCase):
    def setUp(self):
        # KL([1, eps] / (1 + eps) || [eps, 1] / (1 + eps)).
        eps = 1e-6
        self.expected = (1 - eps) / (1 + eps) * math.log(1 / eps)

    def test_zero_counts_are_clipped_in_either_input(self):
        for dtype in (np.int32, np.int64, np.uint64):
            for integer_target, integer_prediction in (
                (True, False),
                (False, True),
                (True, True),
            ):
                with self.subTest(
                    dtype=dtype, target=integer_target, prediction=integer_prediction
                ):
                    x = np.array([0, 1], dtype=dtype if integer_target else float)
                    x_rec = np.array(
                        [1, 0], dtype=dtype if integer_prediction else float
                    )
                    with np.errstate(divide="raise", invalid="raise"):
                        result = regression_metrics_fn(x, x_rec, ["kl_divergence"])
                    self.assertAlmostEqual(result["kl_divergence"], self.expected)

    def test_all_zero_integer_inputs_give_zero_divergence(self):
        for dtype in (np.int32, np.int64, np.uint64):
            with self.subTest(dtype=dtype):
                with np.errstate(divide="raise", invalid="raise"):
                    result = regression_metrics_fn(
                        np.zeros(2, dtype=dtype),
                        np.zeros(2, dtype=dtype),
                        ["kl_divergence"],
                    )
                self.assertEqual(result["kl_divergence"], 0.0)

    def test_floating_point_inputs_still_match_clipped_divergence(self):
        for dtype, places in ((np.float32, 5), (np.float64, 7)):
            with self.subTest(dtype=dtype):
                result = regression_metrics_fn(
                    np.array([0, 1], dtype=dtype),
                    np.array([1, 0], dtype=dtype),
                    ["kl_divergence"],
                )
                self.assertAlmostEqual(result["kl_divergence"], self.expected, places)

    def test_other_metrics_and_inputs_keep_original_values(self):
        x = np.array([0, 1], dtype=np.int64)
        x_rec = np.array([1, 0], dtype=np.int64)
        for metrics in (None, ["mse", "mae", "kl_divergence"]):
            with self.subTest(metrics=metrics):
                with np.errstate(divide="raise", invalid="raise"):
                    result = regression_metrics_fn(x, x_rec, metrics)
                self.assertAlmostEqual(result["kl_divergence"], self.expected)
                self.assertEqual(result["mse"], 1.0)
                self.assertEqual(result["mae"], 1.0)
                np.testing.assert_array_equal(x, [0, 1])
                np.testing.assert_array_equal(x_rec, [1, 0])
                self.assertEqual(x.dtype, np.dtype(np.int64))
                self.assertEqual(x_rec.dtype, np.dtype(np.int64))

    def test_timeseries_integer_counts_use_safe_clipping(self):
        with np.errstate(divide="raise", invalid="raise"):
            result = timeseries_regression_metrics_fn(
                [np.array([0]), np.array([1])],
                [np.array([1]), np.array([0])],
                ["kl_divergence"],
            )
        self.assertAlmostEqual(result["kl_divergence"], self.expected)


if __name__ == "__main__":
    unittest.main()
