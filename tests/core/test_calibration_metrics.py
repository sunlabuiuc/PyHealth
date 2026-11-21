"""Tests for the CaliForest calibration metrics in pyhealth.metrics.calibration."""

import unittest

import numpy as np

from pyhealth.metrics.calibration import (
    hosmer_lemeshow,
    reliability_in_the_large,
    reliability_in_the_small,
    scaled_brier,
    spiegelhalter,
)

# 20 synthetic predictions and outcomes. With 20 samples and 10 groups, every
# risk group has exactly 2 samples, so grouping is unambiguous.
PROB = np.array(
    [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
     0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90, 0.95]
)  # fmt: skip
LABEL = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1])

# Expected values for PROB/LABEL computed with the CaliForest authors' own
# implementation (github.com/yubin-park/califorest,
# califorest/metrics/eval_metrics.py), which our functions reproduce.
REFERENCE = {
    "hosmer_lemeshow_p": 0.3719176914034547,
    "spiegelhalter_p": 0.9889818781066071,
    "reliability_in_the_small": 0.0639325,
    "reliability_in_the_large": 0.00330625,
    "scaled_brier": 0.35614,
}


class TestAgainstReference(unittest.TestCase):
    """Values match the paper authors' implementation."""

    def test_hosmer_lemeshow(self):
        stat, p = hosmer_lemeshow(PROB, LABEL)
        self.assertAlmostEqual(p, REFERENCE["hosmer_lemeshow_p"], places=10)
        self.assertGreater(stat, 0)

    def test_spiegelhalter(self):
        _, p = spiegelhalter(PROB, LABEL)
        self.assertAlmostEqual(p, REFERENCE["spiegelhalter_p"], places=10)

    def test_reliability_in_the_small(self):
        self.assertAlmostEqual(
            reliability_in_the_small(PROB, LABEL),
            REFERENCE["reliability_in_the_small"],
            places=10,
        )

    def test_reliability_in_the_large(self):
        self.assertAlmostEqual(
            reliability_in_the_large(PROB, LABEL),
            REFERENCE["reliability_in_the_large"],
            places=10,
        )

    def test_scaled_brier(self):
        self.assertAlmostEqual(
            scaled_brier(PROB, LABEL), REFERENCE["scaled_brier"], places=10
        )


class TestHandComputed(unittest.TestCase):
    """Small cases worked out by hand."""

    def test_hosmer_lemeshow_three_groups(self):
        # Groups (0.1, 0.2 | 0, 0), (0.3, 0.6 | 1, 0), (0.7, 0.9 | 1, 1):
        # 0.09 / 0.255 + 0.01 / 0.495 + 0.16 / 0.32 = 0.8731, df = 1.
        prob = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
        label = np.array([0, 0, 1, 0, 1, 1])
        stat, p = hosmer_lemeshow(prob, label, ngroups=3)
        self.assertAlmostEqual(stat, 0.09 / 0.255 + 0.01 / 0.495 + 0.5, places=10)
        self.assertAlmostEqual(p, 0.350086606, places=6)

    def test_spiegelhalter_two_samples(self):
        # num = (0 - 0.1)(0.8) + (1 - 0.9)(-0.8) = -0.16
        # den = sqrt(2 * 0.64 * 0.09)
        z, _ = spiegelhalter(np.array([0.1, 0.9]), np.array([0, 1]))
        self.assertAlmostEqual(z, -0.16 / np.sqrt(2 * 0.64 * 0.09), places=10)


class TestProperties(unittest.TestCase):
    """Values that follow directly from the definitions."""

    def test_perfect_predictions(self):
        label = np.array([0, 0, 1, 1, 0, 1])
        prob = label.astype(float)
        self.assertEqual(scaled_brier(prob, label), 1.0)
        self.assertEqual(reliability_in_the_small(prob, label, ngroups=3), 0.0)
        self.assertEqual(reliability_in_the_large(prob, label), 0.0)

    def test_predicting_the_event_rate(self):
        prob = np.full(len(LABEL), LABEL.mean())
        self.assertAlmostEqual(scaled_brier(prob, LABEL), 0.0, places=10)
        self.assertAlmostEqual(reliability_in_the_large(prob, LABEL), 0.0)

    def test_order_does_not_matter(self):
        order = np.random.default_rng(0).permutation(len(PROB))
        prob, label = PROB[order], LABEL[order]
        self.assertAlmostEqual(
            reliability_in_the_small(prob, label),
            reliability_in_the_small(PROB, LABEL),
        )
        self.assertAlmostEqual(
            hosmer_lemeshow(prob, label)[1], hosmer_lemeshow(PROB, LABEL)[1]
        )
        self.assertAlmostEqual(
            spiegelhalter(prob, label)[1], spiegelhalter(PROB, LABEL)[1]
        )

    def test_p_values_in_unit_interval(self):
        for prob in (PROB, PROB**2, np.sqrt(PROB)):
            for _, p in (hosmer_lemeshow(prob, LABEL), spiegelhalter(prob, LABEL)):
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_overconfident_model_is_flagged(self):
        # Cubing pushes predictions toward 0; the tests should reject it more
        # strongly than the original predictions.
        self.assertLess(
            hosmer_lemeshow(PROB**3, LABEL)[1], hosmer_lemeshow(PROB, LABEL)[1]
        )
        self.assertGreater(
            reliability_in_the_large(PROB**3, LABEL),
            reliability_in_the_large(PROB, LABEL),
        )

    def test_accepts_lists_and_column_vectors(self):
        self.assertAlmostEqual(
            reliability_in_the_large(list(PROB), list(LABEL)),
            reliability_in_the_large(PROB.reshape(-1, 1), LABEL.reshape(-1, 1)),
        )


class TestErrors(unittest.TestCase):
    """Invalid inputs raise ValueError."""

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            scaled_brier(PROB, LABEL[:-1])

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            reliability_in_the_large(np.array([]), np.array([]))

    def test_probability_out_of_range(self):
        with self.assertRaises(ValueError):
            reliability_in_the_large(np.array([1.2, 0.5]), np.array([1, 0]))

    def test_non_binary_label(self):
        with self.assertRaises(ValueError):
            reliability_in_the_large(np.array([0.2, 0.5]), np.array([2, 0]))

    def test_scaled_brier_single_class(self):
        with self.assertRaises(ValueError):
            scaled_brier(np.array([0.2, 0.5]), np.array([1, 1]))

    def test_too_few_groups(self):
        with self.assertRaises(ValueError):
            hosmer_lemeshow(PROB, LABEL, ngroups=2)
        with self.assertRaises(ValueError):
            reliability_in_the_small(PROB, LABEL, ngroups=len(PROB) + 1)

    def test_spiegelhalter_undefined(self):
        with self.assertRaises(ValueError):
            spiegelhalter(np.full(4, 0.5), np.array([0, 1, 0, 1]))


if __name__ == "__main__":
    unittest.main()
