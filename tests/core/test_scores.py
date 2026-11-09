"""Tests for pyhealth.calib.predictionset.scores: the shared score module
implementing the "threshold" (LAC), "aps" (Adaptive Prediction Sets,
Romano/Sesia/Candes 2020), and "raps" (Regularized Adaptive Prediction
Sets, Angelopoulos et al., ICLR 2021) nonconformity/conformity scores.
"""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from pyhealth.calib.predictionset import RC3P, BaseConformal
from pyhealth.calib.predictionset.scores import (
    SUPPORTED_SCORE_TYPES,
    all_class_conformity_scores,
    all_class_nc_scores,
    true_class_conformity_scores,
    true_class_nc_scores,
)
from pyhealth.datasets import get_dataloader
from tests.core.label_shift_test_utils import make_sample_dataset_and_model


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


class TestScoresRAPS(unittest.TestCase):
    """Frozen 2026-11-09, agreed with ChatGPT before implementation.

    RAPS (Angelopoulos et al., ICLR 2021): S_RAPS(x, y, u) = S_APS(x, y, u)
    + lambda * (rank(x, y) - k_reg)^+, with rank 1-indexed descending,
    u ~ Uniform(0,1) drawn once per row (u=1 when randomize=False),
    lambda >= 0, k_reg >= 0. Defaults: raps_lambda=0.1, raps_kreg=1.

    These tests must not be edited to make an implementation pass; a change
    needs a stated, valid reason.
    """

    def setUp(self):
        self.y_prob = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2]])
        self.y_true = np.array([0, 1])

    def test_raps_in_supported_types(self):
        self.assertIn("raps", SUPPORTED_SCORE_TYPES)

    def test_hand_computed_lambda_0_1_kreg_1(self):
        # randomize=False so u=1. APS [0.7, 0.9, 1.0] plus penalties
        # [0, 0.1, 0.2] for ranks 1, 2, 3 with kreg=1.
        scores = all_class_nc_scores(
            np.array([[0.7, 0.2, 0.1]]),
            score_type="raps",
            randomize=False,
            raps_lambda=0.1,
            raps_kreg=1,
        )
        np.testing.assert_allclose(scores, [[0.7, 1.0, 1.2]])

    def test_hand_computed_two_rows(self):
        # Row 0: APS [0.7, 0.9, 1.0], penalties [0, 0, 0.2] at kreg=2.
        # Row 1: class order by prob 1, 0, 2; APS [0.8, 0.5, 1.0],
        # penalties [0, 0, 0.2].
        scores = all_class_nc_scores(
            self.y_prob,
            score_type="raps",
            randomize=False,
            raps_lambda=0.2,
            raps_kreg=2,
        )
        np.testing.assert_allclose(
            scores, [[0.7, 0.9, 1.2], [0.8, 0.5, 1.2]]
        )

    def test_reduces_to_aps_when_lambda_zero(self):
        # lambda=0 makes RAPS exactly APS: same APS computation, zero
        # penalty. Independently seeded RNGs (same seed) and exact
        # equality, including a tied-probability row, verify the shared
        # ranking convention (one rank order, one uniform draw).
        y_prob = np.array([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2]])
        aps = all_class_nc_scores(
            y_prob, score_type="aps", rng=np.random.default_rng(3)
        )
        raps = all_class_nc_scores(
            y_prob,
            score_type="raps",
            rng=np.random.default_rng(3),
            raps_lambda=0,
            raps_kreg=1,
        )
        np.testing.assert_array_equal(raps, aps)

    def test_no_penalty_below_kreg(self):
        # With kreg=3 and K=3 no rank exceeds kreg, so RAPS equals APS.
        aps = all_class_nc_scores(
            self.y_prob, score_type="aps", randomize=False
        )
        raps = all_class_nc_scores(
            self.y_prob,
            score_type="raps",
            randomize=False,
            raps_lambda=0.1,
            raps_kreg=3,
        )
        np.testing.assert_array_equal(raps, aps)

    def test_penalty_monotone_in_rank(self):
        # The penalty is non-decreasing in rank; with lambda > 0 the total
        # score is strictly increasing in rank beyond kreg (distinct
        # probabilities, randomize=False so u=1).
        y_prob = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.15]])
        scores = all_class_nc_scores(
            y_prob,
            score_type="raps",
            randomize=False,
            raps_lambda=0.1,
            raps_kreg=1,
        )
        for i in range(2):
            sorted_scores = np.sort(scores[i])
            self.assertTrue(np.all(np.diff(sorted_scores) > 0))

    def test_randomized_bounds(self):
        # Each randomized score lies within [APS lower + penalty,
        # APS upper + penalty] for its rank (u in [0, 1]).
        y_prob = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
        lam, kreg = 0.1, 1
        rng = np.random.default_rng(0)
        scores = all_class_nc_scores(
            y_prob, score_type="raps", rng=rng,
            raps_lambda=lam, raps_kreg=kreg,
        )
        for i in range(2):
            order = np.argsort(-y_prob[i])
            sorted_probs = y_prob[i][order]
            cumsum_excl = np.cumsum(sorted_probs) - sorted_probs
            for r in range(3):
                penalty = lam * max(r + 1 - kreg, 0)
                lo = cumsum_excl[r] + penalty
                hi = cumsum_excl[r] + sorted_probs[r] + penalty
                self.assertGreaterEqual(scores[i, order[r]], lo)
                self.assertLessEqual(scores[i, order[r]], hi)

    def test_lambda_effect(self):
        # A larger lambda leaves ranks <= kreg unchanged and strictly
        # increases ranks > kreg.
        kwargs = {"score_type": "raps", "randomize": False, "raps_kreg": 1}
        s_small = all_class_nc_scores(
            np.array([[0.7, 0.2, 0.1]]), raps_lambda=0.1, **kwargs
        )
        s_large = all_class_nc_scores(
            np.array([[0.7, 0.2, 0.1]]), raps_lambda=0.5, **kwargs
        )
        self.assertAlmostEqual(s_small[0, 0], s_large[0, 0])
        self.assertLess(s_small[0, 1], s_large[0, 1])
        self.assertLess(s_small[0, 2], s_large[0, 2])

    def test_kreg_effect(self):
        # A larger kreg lowers or leaves scores unchanged (fewer ranks
        # penalized).
        kwargs = {"score_type": "raps", "randomize": False, "raps_lambda": 0.1}
        s_small_kreg = all_class_nc_scores(
            np.array([[0.7, 0.2, 0.1]]), raps_kreg=1, **kwargs
        )
        s_large_kreg = all_class_nc_scores(
            np.array([[0.7, 0.2, 0.1]]), raps_kreg=3, **kwargs
        )
        np.testing.assert_allclose(s_small_kreg, [[0.7, 1.0, 1.2]])
        np.testing.assert_allclose(s_large_kreg, [[0.7, 0.9, 1.0]])
        self.assertTrue(np.all(s_large_kreg <= s_small_kreg))

    def test_lambda_validation(self):
        y_prob = np.array([[0.7, 0.2, 0.1]])
        for bad in (-0.1, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                all_class_nc_scores(
                    y_prob, score_type="raps", raps_lambda=bad
                )
        for bad in ("0.1", True, False, None):
            with self.assertRaises(TypeError):
                all_class_nc_scores(
                    y_prob, score_type="raps", raps_lambda=bad
                )
        # Python int/float and NumPy scalars are accepted.
        for good in (1, 0.1, np.float64(0.2), np.int64(1)):
            scores = all_class_nc_scores(
                y_prob, score_type="raps", raps_lambda=good
            )
            self.assertEqual(scores.shape, (1, 3))

    def test_kreg_validation(self):
        y_prob = np.array([[0.7, 0.2, 0.1]])
        for bad in (True, False, "5", 1.5, 5.0, None):
            with self.assertRaises(TypeError):
                all_class_nc_scores(
                    y_prob, score_type="raps", raps_kreg=bad
                )
        for bad in (-1, -5):
            with self.assertRaises(ValueError):
                all_class_nc_scores(
                    y_prob, score_type="raps", raps_kreg=bad
                )
        # Python int and NumPy integer scalars are accepted.
        for good in (0, 1, 5, np.int64(2)):
            scores = all_class_nc_scores(
                y_prob, score_type="raps", raps_kreg=good
            )
            self.assertEqual(scores.shape, (1, 3))

    def test_true_class_nc_scores(self):
        scores = true_class_nc_scores(
            self.y_prob,
            np.array([2, 0]),
            score_type="raps",
            randomize=False,
            raps_lambda=0.1,
            raps_kreg=1,
        )
        np.testing.assert_allclose(scores, [1.2, 0.9])

    def test_conformity_complement(self):
        rng = np.random.default_rng(0)
        nc = all_class_nc_scores(
            self.y_prob, score_type="raps", rng=rng,
            raps_lambda=0.1, raps_kreg=1,
        )
        rng = np.random.default_rng(0)
        conformity = all_class_conformity_scores(
            self.y_prob, score_type="raps", rng=rng,
            raps_lambda=0.1, raps_kreg=1,
        )
        np.testing.assert_allclose(conformity, 1.0 - nc)

    def test_reproducible_with_seeded_rng(self):
        kwargs = {"score_type": "raps", "raps_lambda": 0.1, "raps_kreg": 1}
        s1 = all_class_nc_scores(
            self.y_prob, rng=np.random.default_rng(42), **kwargs
        )
        s2 = all_class_nc_scores(
            self.y_prob, rng=np.random.default_rng(42), **kwargs
        )
        np.testing.assert_array_equal(s1, s2)

    def test_aps_ignores_raps_params(self):
        # RAPS parameters must not change any other score type's behavior,
        # including when they are invalid (validation runs only for raps).
        # NOTE: "saps" is not on this branch's base (upstream master), so
        # it cannot be exercised here; the same by-construction guarantee
        # applies to it once both scores land.
        for score_type in ("aps", "threshold"):
            kwargs = {"score_type": score_type, "rng": np.random.default_rng(3)}
            plain = all_class_nc_scores(self.y_prob, **kwargs)
            kwargs = {
                "score_type": score_type,
                "rng": np.random.default_rng(3),
                "raps_lambda": 0.5,
                "raps_kreg": 2,
            }
            with_params = all_class_nc_scores(self.y_prob, **kwargs)
            np.testing.assert_array_equal(plain, with_params)
            invalid = {
                "score_type": score_type,
                "rng": np.random.default_rng(3),
                "raps_lambda": -1,
                "raps_kreg": -1,
            }
            all_class_nc_scores(self.y_prob, **invalid)  # must not raise

    def test_marginal_coverage_raps(self):
        rng = np.random.default_rng(5)
        n, k = 4000, 5
        logits = rng.normal(size=(n, k)) * 2
        y_prob = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        y_true = np.array([rng.choice(k, p=y_prob[i]) for i in range(n)])
        cal, test = slice(0, n // 2), slice(n // 2, n)
        alpha = 0.1
        cal_rng = np.random.default_rng(6)
        nc_cal = true_class_nc_scores(
            y_prob[cal], y_true[cal], score_type="raps", rng=cal_rng,
            raps_lambda=0.1, raps_kreg=2,
        )
        nc_cal = np.sort(nc_cal)
        loc = int(np.ceil((1 - alpha) * (len(nc_cal) + 1))) - 1
        t = float(nc_cal[loc])
        test_rng = np.random.default_rng(7)
        nc_test = all_class_nc_scores(
            y_prob[test], score_type="raps", rng=test_rng,
            raps_lambda=0.1, raps_kreg=2,
        )
        predset = nc_test <= t
        coverage = predset[np.arange(n // 2), y_true[test]].mean()
        self.assertGreaterEqual(coverage, 1 - alpha - 0.05)

    def test_positional_random_state_compat(self):
        # Regression test: raps params were added after the pre-existing
        # parameters, so positional constructor calls from before RAPS
        # keep their meaning. BaseConformal(model, 0.1, "aps", 42) must
        # still bind 42 to random_state, not to a RAPS parameter.
        _, model = make_sample_dataset_and_model()
        c = BaseConformal(model, 0.1, "aps", 42)
        self.assertEqual(c.raps_lambda, 0.1)
        self.assertEqual(c.raps_kreg, 1)
        expected = np.random.default_rng(42).uniform()
        self.assertAlmostEqual(c.rng.uniform(), expected)

    def test_baseconformal_forwards_raps_params(self):
        # The public predictor path must actually use the RAPS params: a
        # larger lambda raises penalized-rank calibration scores, so the
        # calibrated threshold is monotone in lambda. kreg=1 is pinned
        # explicitly: with kreg=5 and 3 classes every penalty would be
        # zero and the test would be vacuous. Calibration uses synthetic
        # probabilities through the real calibrate() (the 6-sample
        # fixture gives inf thresholds at alpha=0.1 for any lambda, which
        # cannot show the forwarding).
        rng = np.random.default_rng(11)
        n, k = 2000, 3
        logits = rng.normal(size=(n, k))
        y_prob = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        y_true = np.array([rng.choice(k, p=y_prob[i]) for i in range(n)])
        cal_ds = object()

        def fake_prepare(model, dataset, keys, **kwargs):
            return {"y_prob": y_prob, "y_true": y_true}

        _, model = make_sample_dataset_and_model()
        c_small = BaseConformal(
            model=model,
            alpha=0.1,
            score_type="raps",
            raps_lambda=0.1,
            raps_kreg=1,
            random_state=0,
        )
        c_large = BaseConformal(
            model=model,
            alpha=0.1,
            score_type="raps",
            raps_lambda=0.5,
            raps_kreg=1,
            random_state=0,
        )
        with patch(
            "pyhealth.calib.predictionset.base_conformal"
            ".prepare_numpy_dataset",
            side_effect=fake_prepare,
        ):
            c_small.calibrate(cal_dataset=cal_ds)
            c_large.calibrate(cal_dataset=cal_ds)
        t_small = np.asarray(c_small.t, dtype=float)
        t_large = np.asarray(c_large.t, dtype=float)
        self.assertTrue(np.all(t_large >= t_small))
        self.assertFalse(np.allclose(t_small, t_large))
        dataset, _ = make_sample_dataset_and_model()
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            out = c_large(**batch)
        self.assertEqual(out["y_predset"].dtype, torch.bool)
        self.assertEqual(out["y_predset"].shape, out["y_prob"].shape)

    def test_rc3p_forwards_raps_calibrate_and_inference(self):
        # Regression test: RC3P implements its own forward(), so the
        # configured RAPS params must reach both the calibration-time
        # true_class_nc_scores() call and the inference-time
        # all_class_nc_scores() call; otherwise the calibrated threshold
        # and the candidate scores would use different RAPS scores.
        import pyhealth.calib.predictionset.rc3p as rc3p_mod

        rng = np.random.default_rng(21)
        n, k = 600, 3
        logits = rng.normal(size=(n, k))
        y_prob = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        y_true = np.array([rng.choice(k, p=y_prob[i]) for i in range(n)])

        def fake_prepare(model, dataset, keys, **kwargs):
            return {"y_prob": y_prob, "y_true": y_true}

        _, model = make_sample_dataset_and_model()
        rc3p = RC3P(
            model=model,
            alpha=0.2,
            score_type="raps",
            random_state=0,
            raps_lambda=0.5,
            raps_kreg=2,
        )

        seen = {}
        real_true = rc3p_mod.true_class_nc_scores
        real_all = rc3p_mod.all_class_nc_scores

        def spy_true(*args, **kwargs):
            seen["calibrate"] = (
                kwargs.get("raps_lambda"),
                kwargs.get("raps_kreg"),
            )
            return real_true(*args, **kwargs)

        def spy_all(*args, **kwargs):
            seen["inference"] = (
                kwargs.get("raps_lambda"),
                kwargs.get("raps_kreg"),
            )
            return real_all(*args, **kwargs)

        class ProbStub(torch.nn.Module):
            def __init__(self, probs):
                super().__init__()
                self._probs = torch.as_tensor(probs, dtype=torch.float32)
                self.mode = "multiclass"
                self.device = torch.device("cpu")

            def forward(self, **kwargs):
                return {"y_prob": self._probs}

        with patch(
            "pyhealth.calib.predictionset.rc3p.prepare_numpy_dataset",
            side_effect=fake_prepare,
        ), patch.object(rc3p_mod, "true_class_nc_scores", spy_true), patch.object(
            rc3p_mod, "all_class_nc_scores", spy_all
        ):
            rc3p.calibrate(cal_dataset=object())
            rc3p.model = ProbStub(y_prob)
            out = rc3p.forward()
        self.assertEqual(seen["calibrate"], (0.5, 2))
        self.assertEqual(seen["inference"], (0.5, 2))
        self.assertEqual(out["y_predset"].dtype, torch.bool)
        self.assertEqual(out["y_predset"].shape, (n, k))


if __name__ == "__main__":
    unittest.main()
