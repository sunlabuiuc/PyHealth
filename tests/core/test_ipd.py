"""Tests for the IPD, DTW and source-weighting utilities."""

import unittest
from typing import ClassVar

import numpy as np

from pyhealth.models.ipd import (
    dtw_distance,
    learning_rate_scales,
    paired_ipd,
    source_weights,
)


class TestDTW(unittest.TestCase):
    """Tests for dtw_distance."""

    def test_identical_series_have_zero_distance(self):
        x = np.arange(12, dtype=float).reshape(6, 2)
        self.assertEqual(dtw_distance(x, x), 0.0)

    def test_symmetric(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
        self.assertAlmostEqual(dtw_distance(a, b), dtw_distance(b, a))

    def test_known_value(self):
        a = np.array([[0.0], [1.0], [2.0]])
        b = np.array([[0.0], [2.0]])
        # Best alignment: 0-0, 1-2, 2-2 -> 0 + 1 + 0.
        self.assertAlmostEqual(dtw_distance(a, b), 1.0)

    def test_band_never_lowers_the_distance(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=(10, 2)), rng.normal(size=(10, 2))
        self.assertGreaterEqual(dtw_distance(a, b, window=1), dtw_distance(a, b))

    def test_different_lengths(self):
        a, b = np.zeros((5, 2)), np.zeros((9, 2))
        self.assertEqual(dtw_distance(a, b, window=1), 0.0)


class TestPairedIPD(unittest.TestCase):
    """Tests for paired_ipd."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.target = rng.normal(size=(3, 12, 4))
        self.near = self.target + 0.01 * rng.normal(size=self.target.shape)
        self.far = rng.normal(size=self.target.shape)

    def test_zero_for_identical(self):
        self.assertEqual(paired_ipd(self.target, self.target, window=3), 0.0)

    def test_closer_source_has_smaller_ipd(self):
        near = paired_ipd(self.near, self.target, window=3)
        far = paired_ipd(self.far, self.target, window=3)
        self.assertLess(near, far)

    def test_pairs_must_match(self):
        with self.assertRaises(ValueError):
            paired_ipd(self.near[:2], self.target)
        with self.assertRaises(ValueError):
            paired_ipd(self.near[:0], self.target[:0])


class TestSourceWeights(unittest.TestCase):
    """Tests for source_weights and learning_rate_scales."""

    distances: ClassVar[dict[str, float]] = {
        "near": 1.0,
        "mid": 5.0,
        "far": 20.0,
    }
    def test_weights_sum_to_one_in_every_mode(self):
        for mode in ("ipd", "equal", "inverse"):
            total = sum(source_weights(self.distances, mode).values())
            self.assertAlmostEqual(total, 1.0)

    def test_ipd_prefers_close_sources(self):
        w = source_weights(self.distances, "ipd")
        self.assertGreater(w["near"], w["mid"])
        self.assertGreater(w["mid"], w["far"])

    def test_inverse_reverses_the_direction(self):
        w = source_weights(self.distances, "inverse")
        self.assertGreater(w["far"], w["mid"])
        self.assertGreater(w["mid"], w["near"])

    def test_equal_weights(self):
        w = source_weights(self.distances, "equal")
        self.assertAlmostEqual(w["near"], w["far"])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            source_weights(self.distances, "random")
        with self.assertRaises(ValueError):
            source_weights({}, "ipd")

    def test_learning_rate_scales_have_mean_one(self):
        scales = learning_rate_scales(source_weights(self.distances, "ipd"))
        self.assertAlmostEqual(sum(scales.values()) / len(scales), 1.0)
        equal = learning_rate_scales(source_weights(self.distances, "equal"))
        self.assertTrue(all(abs(v - 1.0) < 1e-9 for v in equal.values()))


if __name__ == "__main__":
    unittest.main()
