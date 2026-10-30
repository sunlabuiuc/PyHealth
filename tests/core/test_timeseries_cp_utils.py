"""Tests for the shared time series conformal helpers.

The quantile tests use hand-computed values.
"""

import math
import unittest

import numpy as np

from pyhealth.calib.predictionset.timeseries import conformal_quantile


class TestConformalQuantile(unittest.TestCase):
    def test_rank_convention(self):
        # n = 9, alpha = 0.2: rank ceil(0.8 * 10) = 8, so the 8th smallest.
        self.assertEqual(conformal_quantile(np.arange(1.0, 10.0), 0.2), 8.0)

    def test_exact_rank_is_not_bumped_by_float_error(self):
        # (1 - 0.1) * 10 is 9.000000000000002 in floating point; rank is 9.
        self.assertEqual(conformal_quantile(np.arange(1.0, 10.0), 0.1), 9.0)

    def test_presorted_list_matches_unsorted(self):
        scores = [5.0, 1.0, 4.0, 2.0, 3.0]
        for alpha in (-0.5, 0.0, 0.3, 0.6, 1.0):
            with self.subTest(alpha=alpha):
                self.assertEqual(
                    conformal_quantile(sorted(scores), alpha, presorted=True),
                    conformal_quantile(np.array(scores), alpha),
                )

    def test_unordered_input(self):
        self.assertEqual(conformal_quantile(np.array([3.0, 1.0, 2.0]), 0.5), 2.0)

    def test_window_too_small_gives_inf(self):
        # n = 4, alpha = 0.1: rank ceil(0.9 * 5) = 5 > 4.
        self.assertEqual(conformal_quantile(np.ones(4), 0.1), math.inf)

    def test_empty_window_gives_inf(self):
        self.assertEqual(conformal_quantile(np.array([]), 0.1), math.inf)

    def test_alpha_at_or_below_zero_gives_inf(self):
        self.assertEqual(conformal_quantile(np.arange(100.0), 0.0), math.inf)
        self.assertEqual(conformal_quantile(np.arange(100.0), -0.3), math.inf)

    def test_alpha_at_or_above_one_gives_minus_inf(self):
        self.assertEqual(conformal_quantile(np.arange(100.0), 1.0), -math.inf)
        self.assertEqual(conformal_quantile(np.arange(100.0), 1.4), -math.inf)


if __name__ == "__main__":
    unittest.main()
