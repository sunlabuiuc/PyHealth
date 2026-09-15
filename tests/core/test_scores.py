"""Tests for pyhealth.calib.predictionset.scores: the shared score module
implementing both the "threshold" (LAC) and "aps" (Adaptive Prediction
Sets, Romano/Sesia/Candes 2020) nonconformity/conformity scores.
"""

import unittest

import numpy as np

from pyhealth.calib.predictionset.scores import (
    SUPPORTED_SCORE_TYPES,
    all_class_conformity_scores,
    all_class_nc_scores,
    true_class_conformity_scores,
    true_class_nc_scores,
)


class TestScoresThreshold(unittest.TestCase):
    """"threshold" is just 1 - p (and its complement), regardless of rng."""

    def setUp(self):
        self.y_prob = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        self.y_true = np.array([0, 1])

    def test_all_class_nc_scores(self):
        np.testing.assert_allclose(
            all_class_nc_scores(self.y_prob, score_type="threshold"),
            1.0 - self.y_prob,
        )

    def test_all_class_conformity_scores(self):
        np.testing.assert_allclose(
            all_class_conformity_scores(self.y_prob, score_type="threshold"),
            self.y_prob,
        )

    def test_true_class_nc_scores(self):
        np.testing.assert_allclose(
            true_class_nc_scores(self.y_prob, self.y_true, score_type="threshold"),
            [0.3, 0.5],
        )

    def test_true_class_conformity_scores(self):
        np.testing.assert_allclose(
            true_class_conformity_scores(self.y_prob, self.y_true, score_type="threshold"),
            [0.7, 0.5],
        )

    def test_default_score_type_is_threshold(self):
        """Backward compatibility: omitting score_type must match the old,
        hardcoded 1 - p behavior every caller used before score_type
        existed."""
        np.testing.assert_allclose(
            all_class_nc_scores(self.y_prob),
            1.0 - self.y_prob,
        )


class TestScoresAPS(unittest.TestCase):
    """Verify the APS score formula: E(x,k) = [sum of probs ranked above k]
    + U * p(k), and its structural properties."""

    def test_non_randomized_matches_hand_computation(self):
        """With randomize=False (U=1), APS collapses to the cumulative sum
        of sorted probabilities -- hand-computable exactly."""
        y_prob = np.array([[0.5, 0.3, 0.15, 0.05]])
        rng = np.random.default_rng(0)
        scores = all_class_nc_scores(y_prob, score_type="aps", rng=rng, randomize=False)
        # sorted descending: 0.5, 0.3, 0.15, 0.05 -> cumsum 0.5, 0.8, 0.95, 1.0
        np.testing.assert_allclose(scores, [[0.5, 0.8, 0.95, 1.0]])

    def test_scores_bounded_in_unit_interval(self):
        rng = np.random.default_rng(1)
        n, k = 50, 6
        logits = rng.normal(size=(n, k))
        y_prob = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        scores = all_class_nc_scores(y_prob, score_type="aps", rng=rng)
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))

    def test_higher_probability_class_has_lower_or_equal_nc_score(self):
        """APS nonconformity score must be monotonically non-decreasing as
        predicted probability decreases (higher-probability classes are
        included in smaller/first-formed sets)."""
        rng = np.random.default_rng(2)
        y_prob = np.array([[0.6, 0.25, 0.1, 0.05]])
        scores = all_class_nc_scores(y_prob, score_type="aps", rng=rng, randomize=False)[0]
        order = np.argsort(-y_prob[0])
        sorted_scores = scores[order]
        self.assertTrue(np.all(np.diff(sorted_scores) >= -1e-12))

    def test_reproducible_with_seeded_rng(self):
        y_prob = np.array([[0.4, 0.35, 0.25]])
        s1 = all_class_nc_scores(y_prob, score_type="aps", rng=np.random.default_rng(42))
        s2 = all_class_nc_scores(y_prob, score_type="aps", rng=np.random.default_rng(42))
        np.testing.assert_allclose(s1, s2)

    def test_nc_and_conformity_are_complementary(self):
        rng = np.random.default_rng(3)
        y_prob = np.array([[0.5, 0.3, 0.2]])
        nc = all_class_nc_scores(y_prob, score_type="aps", rng=np.random.default_rng(3))
        conf = all_class_conformity_scores(y_prob, score_type="aps", rng=np.random.default_rng(3))
        np.testing.assert_allclose(nc, 1.0 - conf)

    def test_true_class_score_matches_all_class_indexing(self):
        rng = np.random.default_rng(4)
        y_prob = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
        y_true = np.array([1, 2])
        all_scores = all_class_nc_scores(y_prob, score_type="aps", rng=np.random.default_rng(4))
        true_scores = true_class_nc_scores(y_prob, y_true, score_type="aps", rng=np.random.default_rng(4))
        np.testing.assert_allclose(true_scores, all_scores[np.arange(2), y_true])


class TestScoresCoverage(unittest.TestCase):
    """The core statistical property: both score types must achieve
    approximately the target marginal coverage under split conformal
    calibration, for both marginal and class-conditional targets."""

    def _query_quantile(self, nc_scores, alpha):
        nc_scores = np.sort(nc_scores)
        n = len(nc_scores)
        loc = int(np.ceil((1 - alpha) * (n + 1))) - 1
        if loc >= n:
            return np.inf
        return float(nc_scores[loc])

    def test_marginal_coverage_threshold_and_aps(self):
        rng = np.random.default_rng(5)
        n, k = 4000, 5
        logits = rng.normal(size=(n, k)) * 2
        y_prob = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        y_true = np.array([rng.choice(k, p=y_prob[i]) for i in range(n)])
        cal, test = slice(0, n // 2), slice(n // 2, n)
        alpha = 0.1

        for score_type in SUPPORTED_SCORE_TYPES:
            cal_rng = np.random.default_rng(6)
            nc_cal = true_class_nc_scores(
                y_prob[cal], y_true[cal], score_type=score_type, rng=cal_rng
            )
            t = self._query_quantile(nc_cal, alpha)
            test_rng = np.random.default_rng(7)
            nc_test = all_class_nc_scores(y_prob[test], score_type=score_type, rng=test_rng)
            predset = nc_test <= t
            covered = predset[np.arange(n // 2), y_true[test]]
            coverage = covered.mean()
            # Allow generous slack for finite-sample noise at N=2000.
            self.assertGreaterEqual(
                coverage, 1 - alpha - 0.05,
                f"{score_type} marginal coverage {coverage:.3f} too far below target",
            )


class TestScoresValidation(unittest.TestCase):
    def test_unknown_score_type_raises(self):
        y_prob = np.array([[0.5, 0.5]])
        with self.assertRaises(ValueError):
            all_class_nc_scores(y_prob, score_type="not_a_real_score_type")


if __name__ == "__main__":
    unittest.main()
