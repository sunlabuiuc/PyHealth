# Authors: Aritra Saha and Ugochukwu Njoku.
# Survival Mixture Density Networks, Han et al., MLHC 2022.
# https://proceedings.mlr.press/v182/han22a.html
"""Tests for pyhealth.metrics.survival_curve on hand-computable fixtures."""

import unittest

import numpy as np
from pyhealth.metrics import (
    concordance_td,
    integrated_binomial_loglik_td,
    integrated_brier_score_td,
    survival_curve_metrics_fn,
)
from pyhealth.metrics.survival_curve import (
    censoring_survival,
    horizon,
    integration_grid,
)


def _lognormal_curves(mu, grid, sigma=0.6):
    from math import erf, sqrt

    z = (np.log(np.asarray(grid))[None, :] - np.asarray(mu)[:, None]) / sigma
    return 0.5 * (1.0 - np.vectorize(erf)(z / sqrt(2.0)))


class TestCensoringSurvival(unittest.TestCase):
    def test_kaplan_meier_with_ties(self):
        # Censored at 2 (one of two tied times) and 5.
        times = [1.0, 2.0, 2.0, 3.0, 5.0]
        events = [1, 0, 1, 1, 0]
        g = censoring_survival(times, events, [1.0, 2.0, 3.0, 5.0, 9.0])
        np.testing.assert_allclose(g, [1.0, 0.75, 0.75, 0.0, 0.0])

    def test_linear_interpolation_between_knots(self):
        g = censoring_survival([1.0, 2.0, 3.0], [1, 0, 1], [0.5, 1.5, 2.5])
        np.testing.assert_allclose(g, [1.0, 0.75, 0.5])

    def test_horizon_and_grid(self):
        times, events = [1.0, 2.0, 3.0, 4.0], [1, 0, 1, 0]
        self.assertEqual(horizon(times, events, 1e-8), 3.0)
        self.assertEqual(horizon(times, events, 0.7), 1.0)
        grid = integration_grid(times, events, 1e-8, num_points=5)
        np.testing.assert_allclose(grid, [1.0, 1.5, 2.0, 2.5, 3.0])


class TestConcordanceTD(unittest.TestCase):
    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    events = np.array([1, 0, 1, 1, 0])

    def test_hand_computed_ipcw(self):
        # G = 1 at t=1 and 0.75 at t=3, 4, so those events weigh 16/9.
        # Pairs: i=1 vs {2,3,4,5}: 4/4 concordant; i=3 vs {4,5}: 1/2;
        # i=4 vs {5}: a tie, counted discordant. C = (4 + 16/9) / (4 + 48/9).
        # Complete valid curves, preserving the hand-computed pair rankings.
        surv = np.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1],
                [0.9, 0.9, 0.9, 0.9, 0.9],
                [0.9, 0.9, 0.5, 0.5, 0.5],
                [0.9, 0.9, 0.6, 0.3, 0.3],
                [0.9, 0.9, 0.4, 0.3, 0.3],
            ]
        )
        c = concordance_td(self.times, self.events, surv, self.times)
        self.assertAlmostEqual(c, 13 / 21)

    def test_perfect_reversed_and_constant(self):
        risk = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        perfect = np.exp(-np.outer(risk, self.times))
        reversed_ = np.exp(-np.outer(risk[::-1], self.times))
        constant = np.full((5, 5), 0.5)
        self.assertEqual(
            concordance_td(self.times, self.events, perfect, self.times), 1.0
        )
        self.assertEqual(
            concordance_td(self.times, self.events, reversed_, self.times), 0.0
        )
        self.assertEqual(
            concordance_td(self.times, self.events, constant, self.times), 0.0
        )

    def test_tied_times_count_censored_partner(self):
        # Patient 1 is censored at the same time as patient 0's event.
        times = np.array([2.0, 2.0, 3.0])
        events = np.array([1, 0, 1])
        surv = np.array([[0.3, 0.2], [0.6, 0.5], [0.9, 0.4]])
        self.assertEqual(concordance_td(times, events, surv, [2.0, 3.0]), 1.0)

    def test_threshold_truncates_late_events(self):
        # With threshold 0.8 only the event at t=1 (G=1) is used.
        surv = np.full((5, 5), 0.9)
        surv[0, :] = 0.1
        surv[2, :] = 0.95  # would be discordant if included
        c = concordance_td(self.times, self.events, surv, self.times, 0.8)
        self.assertEqual(c, 1.0)

    def test_requires_event_times_in_grid(self):
        with self.assertRaises(ValueError):
            concordance_td(self.times, self.events, np.ones((5, 2)), [1.5, 2.5])


class TestIntegratedScores(unittest.TestCase):
    def test_hand_computed_brier_with_censoring(self):
        times, events = [1.0, 2.0, 3.0], [1, 0, 1]
        grid = [1.5, 2.5]  # G = 0.75 and 0.5 there
        surv = [[0.2, 0.1], [0.7, 0.6], [0.9, 0.4]]
        expected = (0.04 + 0.09 / 0.75 + 0.01 / 0.75 + 0.01 + 0.0 + 0.36 / 0.5) / 6
        self.assertAlmostEqual(
            integrated_brier_score_td(times, events, surv, grid), expected
        )

    def test_uninformative_prediction_without_censoring(self):
        times = np.array([1.0, 2.0, 3.0, 4.0])
        events = np.ones(4)
        grid = np.linspace(1.0, 4.0, 7)
        half = np.full((4, 7), 0.5)
        self.assertAlmostEqual(
            integrated_brier_score_td(times, events, half, grid), 0.25
        )
        self.assertAlmostEqual(
            integrated_binomial_loglik_td(times, events, half, grid), np.log(0.5)
        )

    def test_perfect_prediction(self):
        times, events = np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1])
        grid = np.array([1.0, 2.0, 3.0])
        perfect = (times[:, None] > grid[None, :]).astype(float)
        self.assertEqual(integrated_brier_score_td(times, events, perfect, grid), 0.0)
        self.assertGreater(
            integrated_binomial_loglik_td(times, events, perfect, grid), -1e-8
        )

    def test_zero_censoring_survival_is_dropped(self):
        # The last patient is censored at the final time, so G = 0 there.
        times, events = [1.0, 2.0, 3.0], [1, 1, 0]
        grid = [1.0, 2.0, 3.0]
        perfect = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        score = integrated_brier_score_td(times, events, perfect, grid)
        self.assertTrue(np.isfinite(score))
        self.assertEqual(score, 0.0)


class TestMetricsFn(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 300
        self.mu = 2.0 + rng.standard_normal(n)
        t_event = np.exp(self.mu + 0.6 * rng.standard_normal(n))
        t_cens = rng.exponential(15.0, n)
        self.times = np.minimum(t_event, t_cens)
        self.events = (t_event <= t_cens).astype(float)

    def _predict(self, mu):
        return lambda grid: _lognormal_curves(mu, grid)

    def test_keys_and_consistency(self):
        scores = survival_curve_metrics_fn(
            self.times, self.events, self._predict(self.mu), num_points=200
        )
        expected = {"c_td", "ibs", "ibll"}
        for eps in ("1e-08", "0.2", "0.4"):
            expected |= {f"c_td@{eps}", f"ibs@{eps}", f"ibll@{eps}"}
        self.assertEqual(set(scores), expected)
        self.assertEqual(scores["ibs"], scores["ibs@1e-08"])
        grid = integration_grid(self.times, self.events, 0.2, 200)
        direct = integrated_brier_score_td(
            self.times, self.events, _lognormal_curves(self.mu, grid), grid
        )
        self.assertAlmostEqual(scores["ibs@0.2"], direct)

    def test_score_direction(self):
        rng = np.random.default_rng(1)
        good = survival_curve_metrics_fn(
            self.times, self.events, self._predict(self.mu), num_points=200
        )
        noisy_mu = self.mu + 1.5 * rng.standard_normal(self.mu.size)
        bad = survival_curve_metrics_fn(
            self.times, self.events, self._predict(noisy_mu), num_points=200
        )
        self.assertGreater(good["c_td"], bad["c_td"])
        self.assertLess(good["ibs"], bad["ibs"])
        self.assertGreater(good["ibll"], bad["ibll"])
        self.assertGreater(good["c_td"], 0.7)

    def test_rejects_invalid_labels(self):
        predict = self._predict(self.mu[:3])
        with self.assertRaises(ValueError):
            survival_curve_metrics_fn([1.0, 0.0, 2.0], [1, 0, 1], predict)
        with self.assertRaises(ValueError):
            survival_curve_metrics_fn([1.0, 2.0, 3.0], [1, 2, 1], predict)
        with self.assertRaises(ValueError):
            survival_curve_metrics_fn([1.0, 2.0, 3.0], [0, 0, 0], predict)


if __name__ == "__main__":
    unittest.main()
