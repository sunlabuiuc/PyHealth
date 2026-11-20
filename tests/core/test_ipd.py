"""Tests for the IPD, per-channel DTW and decay-rate utilities."""

import unittest

import numpy as np

from pyhealth.models.ipd import (
    decay_rates,
    dtw_per_channel,
    paired_differences,
    paired_ipd,
    smooth_bootstrap_ipd,
)


class TestDTWPerChannel(unittest.TestCase):
    """Tests for dtw_per_channel."""

    def test_identical_series_have_zero_distance(self):
        x = np.arange(12, dtype=float).reshape(6, 2)
        self.assertEqual(dtw_per_channel(x, x).tolist(), [0.0, 0.0])

    def test_known_values_per_channel(self):
        a = np.array([[0.0, 10.0], [1.0, 10.0], [2.0, 10.0]])
        b = np.array([[0.0, 0.0], [2.0, 0.0]])
        # Channel 0 aligns 0-0, 1-2, 2-2 (cost 1); channel 1 costs 10 per step.
        np.testing.assert_allclose(dtw_per_channel(a, b), [1.0, 30.0])

    def test_channels_are_independent(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
        both = dtw_per_channel(a, b)
        for channel in range(3):
            alone = dtw_per_channel(a[:, [channel]], b[:, [channel]])
            self.assertAlmostEqual(both[channel], alone[0])

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=(7, 2)), rng.normal(size=(7, 2))
        np.testing.assert_allclose(dtw_per_channel(a, b), dtw_per_channel(b, a))

    def test_band_never_lowers_the_distance(self):
        rng = np.random.default_rng(2)
        a, b = rng.normal(size=(10, 2)), rng.normal(size=(10, 2))
        self.assertTrue(
            np.all(dtw_per_channel(a, b, window=1) >= dtw_per_channel(a, b))
        )

    def test_different_lengths(self):
        a, b = np.zeros((5, 2)), np.zeros((9, 2))
        self.assertEqual(dtw_per_channel(a, b, window=1).tolist(), [0.0, 0.0])


class TestPairedIPD(unittest.TestCase):
    """Tests for the paired differences and the smooth-bootstrap IPD."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.target = rng.normal(size=(6, 12, 4))
        self.near = self.target + 0.01 * rng.normal(size=self.target.shape)
        self.far = rng.normal(size=self.target.shape) + 3.0

    def test_differences_shape_and_pairing_check(self):
        d = paired_differences(self.near, self.target, window=3)
        self.assertEqual(d.shape, (6, 4))
        with self.assertRaises(ValueError):
            paired_differences(self.near[:3], self.target)

    def test_identical_domains_have_near_zero_ipd(self):
        self.assertAlmostEqual(paired_ipd(self.target, self.target, window=3), 0.0, 3)

    def test_closer_source_has_smaller_ipd(self):
        near = paired_ipd(self.near, self.target, window=3)
        far = paired_ipd(self.far, self.target, window=3)
        self.assertLess(near, far)

    def test_reproducible_for_a_seed(self):
        d = paired_differences(self.far, self.target, window=3)
        self.assertEqual(
            smooth_bootstrap_ipd(d, seed=4), smooth_bootstrap_ipd(d, seed=4)
        )

    def test_needs_two_pairs(self):
        with self.assertRaises(ValueError):
            smooth_bootstrap_ipd(np.ones((1, 3)))


class TestDecayRates(unittest.TestCase):
    """Tests for decay_rates."""

    def setUp(self):
        self.distances = {"near": 1.0, "mid": 5.0, "far": 20.0}

    def test_rates_sum_to_one_in_every_mode(self):
        for mode in ("ipd", "equal", "inverse"):
            total = sum(decay_rates(self.distances, mode).values())
            self.assertAlmostEqual(total, 1.0)

    def test_ipd_gives_closer_sources_smaller_alpha(self):
        a = decay_rates(self.distances, "ipd")
        self.assertLess(a["near"], a["mid"])
        self.assertLess(a["mid"], a["far"])
        # alpha is the share of the total distance, as in the paper.
        self.assertAlmostEqual(a["far"], 20.0 / 26.0)

    def test_inverse_reverses_the_direction(self):
        a = decay_rates(self.distances, "inverse")
        self.assertGreater(a["near"], a["mid"])
        self.assertGreater(a["mid"], a["far"])

    def test_equal(self):
        a = decay_rates(self.distances, "equal")
        self.assertAlmostEqual(a["near"], a["far"])

    def test_single_source_gets_all_the_weight(self):
        self.assertEqual(decay_rates({"only": 3.0}, "ipd"), {"only": 1.0})

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            decay_rates(self.distances, "random")
        with self.assertRaises(ValueError):
            decay_rates({}, "ipd")


if __name__ == "__main__":
    unittest.main()
