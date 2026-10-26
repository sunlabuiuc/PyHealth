import unittest
import warnings

import numpy as np
import torch

from pyhealth.calib.predictionset import LabelShift, bbse, em
from pyhealth.calib.predictionset.base_conformal import _query_quantile
from pyhealth.calib.predictionset.label_shift.label_shift import (
    _compute_label_shift_thresholds,
)
from pyhealth.calib.predictionset.label_shift.mlls import _posteriors
from pyhealth.datasets import get_dataloader
from tests.core.label_shift_test_utils import make_sample_dataset_and_model


def _assert_valid_distribution(test_case, q, num_classes):
    """Every estimate is a 1-D distribution over the classes."""
    test_case.assertEqual(q.ndim, 1)
    test_case.assertEqual(len(q), num_classes)
    test_case.assertTrue(np.all(np.isfinite(q)))
    test_case.assertTrue(np.all(q >= 0))
    test_case.assertAlmostEqual(q.sum(), 1.0, places=8)


def _gaussian_pdf(x, mu):
    """Standard-normal-shaped density (the normalizing constant cancels
    in posterior ratios)."""
    return np.exp(-0.5 * (x - mu) ** 2)


def _binary_probas(rng, n, mix, source_mix=None, mu1=2.0):
    """Calibrated binary classifier probabilities. The feature x is
    Gaussian given the class, and the returned probabilities are the
    true posteriors P(y | x) under source_mix, the distribution the
    classifier was trained on. The classes overlap, so the classifier
    is genuinely imperfect."""
    if source_mix is None:
        source_mix = mix
    y = rng.choice(2, size=n, p=mix)
    x = np.where(y == 0, rng.normal(0.0, 1.0, n), rng.normal(mu1, 1.0, n))
    like0 = _gaussian_pdf(x, 0.0)
    like1 = _gaussian_pdf(x, mu1)
    post1 = (source_mix[1] * like1) / (
        source_mix[0] * like0 + source_mix[1] * like1
    )
    return np.column_stack([1.0 - post1, post1]), y


def _multiclass_probas(rng, n, mix, source_mix=None):
    """Calibrated 3-class probabilities: x is Gaussian given the class
    with means -2.5, 0, 2.5, and the returned probabilities are the true
    posteriors P(y | x) under source_mix."""
    if source_mix is None:
        source_mix = mix
    means = (-2.5, 0.0, 2.5)
    num_classes = len(mix)
    y = rng.choice(num_classes, size=n, p=mix)
    x = rng.normal(np.array(means)[y], 1.0)
    likes = np.column_stack([_gaussian_pdf(x, mu) for mu in means])
    weighted = likes * np.asarray(source_mix)
    return weighted / weighted.sum(axis=1, keepdims=True), y


class TestEM(unittest.TestCase):
    """Tests for the MLLS target label distribution estimator."""

    def test_one_hot_target_recovers_empirical_mix(self):
        """One-hot target rows make the E-step posteriors one-hot, so
        one EM iteration returns the exact empirical target mix."""
        y_true_source = np.array([0, 0, 1, 1])
        y_prob_target = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        )
        estimate = em(y_true_source, y_prob_target)
        np.testing.assert_allclose(estimate, [0.75, 0.25], atol=1e-8)
        _assert_valid_distribution(self, estimate, 2)

    def test_no_shift_recovery(self):
        """Target probabilities generated under the source mix give the
        source mix back, on a fixed-seed synthetic dataset."""
        rng = np.random.default_rng(11)
        source_mix = np.array([0.6, 0.4])
        _, cal_true = _binary_probas(rng, 4000, source_mix)
        tgt_proba, _ = _binary_probas(rng, 8000, source_mix)
        estimate = em(cal_true, tgt_proba)
        np.testing.assert_allclose(estimate, source_mix, atol=0.02)
        _assert_valid_distribution(self, estimate, 2)

    def test_binary_shift_recovery(self):
        """Source mix [0.9, 0.1] shifting to [0.7, 0.3] with an
        imperfect calibrated classifier: the estimate lands within 0.02
        of the true target mix."""
        rng = np.random.default_rng(23)
        source_mix = np.array([0.9, 0.1])
        target_mix = np.array([0.7, 0.3])
        _, cal_true = _binary_probas(rng, 6000, source_mix)
        tgt_proba, _ = _binary_probas(
            rng, 20000, target_mix, source_mix=source_mix
        )
        estimate = em(cal_true, tgt_proba)
        np.testing.assert_allclose(estimate, target_mix, atol=0.02)
        _assert_valid_distribution(self, estimate, 2)

    def test_multiclass_shift_recovery(self):
        """A 3-class identifiable label shift problem: the estimate
        lands within 0.02 of the true target mix."""
        rng = np.random.default_rng(45)
        source_mix = np.array([0.5, 0.3, 0.2])
        target_mix = np.array([0.2, 0.3, 0.5])
        _, cal_true = _multiclass_probas(rng, 9000, source_mix)
        tgt_proba, _ = _multiclass_probas(
            rng, 30000, target_mix, source_mix=source_mix
        )
        estimate = em(cal_true, tgt_proba)
        np.testing.assert_allclose(estimate, target_mix, atol=0.02)
        _assert_valid_distribution(self, estimate, 3)

    def test_em_and_bbse_independently_recover(self):
        """EM and BBSE each recover the same known target mix within
        0.02 on the same simulated dataset. No ordering between their
        errors is asserted: that comparison is setup-dependent."""
        rng = np.random.default_rng(23)
        source_mix = np.array([0.9, 0.1])
        target_mix = np.array([0.7, 0.3])
        cal_proba, cal_true = _binary_probas(rng, 6000, source_mix)
        tgt_proba, _ = _binary_probas(
            rng, 20000, target_mix, source_mix=source_mix
        )
        em_estimate = em(cal_true, tgt_proba)
        np.testing.assert_allclose(em_estimate, target_mix, atol=0.02)
        bbse_estimate = bbse(
            cal_true,
            np.argmax(cal_proba, axis=1),
            np.argmax(tgt_proba, axis=1),
        )
        np.testing.assert_allclose(bbse_estimate, target_mix, atol=0.02)

    def test_deterministic(self):
        """Two calls on identical inputs give identical outputs."""
        rng = np.random.default_rng(23)
        _, cal_true = _binary_probas(rng, 2000, [0.9, 0.1])
        tgt_proba, _ = _binary_probas(
            rng, 4000, [0.7, 0.3], source_mix=[0.9, 0.1]
        )
        first = em(cal_true, tgt_proba)
        second = em(cal_true, tgt_proba)
        np.testing.assert_array_equal(first, second)

    def test_converges_without_warning(self):
        """On the binary and multiclass recovery fixtures EM converges
        with default settings: no UserWarning fires, and the estimate is
        accurate."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rng = np.random.default_rng(23)
            _, cal_true = _binary_probas(rng, 6000, [0.9, 0.1])
            tgt_proba, _ = _binary_probas(
                rng, 20000, [0.7, 0.3], source_mix=[0.9, 0.1]
            )
            estimate = em(cal_true, tgt_proba)
            np.testing.assert_allclose(estimate, [0.7, 0.3], atol=0.02)
            rng = np.random.default_rng(45)
            _, cal_true = _multiclass_probas(
                rng, 9000, [0.5, 0.3, 0.2]
            )
            tgt_proba, _ = _multiclass_probas(
                rng, 30000, [0.2, 0.3, 0.5], source_mix=[0.5, 0.3, 0.2]
            )
            estimate = em(cal_true, tgt_proba)
            np.testing.assert_allclose(estimate, [0.2, 0.3, 0.5], atol=0.02)

    def test_max_iter_one_warns_and_returns_one_step(self):
        """With max_iter=1 on a case that needs more iterations, EM emits
        the non-convergence UserWarning and returns the valid one-step
        estimate: rows [[0.9, 0.1], [0.9, 0.1]] average to [0.9, 0.1]."""
        y_true_source = np.array([0, 1])
        y_prob_target = np.array([[0.9, 0.1], [0.9, 0.1]])
        with self.assertWarns(UserWarning):
            estimate = em(
                y_true_source, y_prob_target, max_iter=1, tol=1e-8
            )
        np.testing.assert_allclose(estimate, [0.9, 0.1], atol=1e-8)
        _assert_valid_distribution(self, estimate, 2)

    def test_invalid_target_inputs(self):
        """Invalid target probability data raises ValueError."""
        y_true_source = np.array([0, 0, 1, 1])
        cases = {
            "negative entry": np.array([[1.1, -0.1], [0.5, 0.5]]),
            "NaN entry": np.array([[np.nan, 1.0], [0.5, 0.5]]),
            "infinite entry": np.array([[np.inf, 0.0], [0.5, 0.5]]),
            "row sums below 1": np.array([[0.4, 0.4], [0.5, 0.5]]),
            "row sums above 1": np.array([[0.6, 0.6], [0.5, 0.5]]),
            "1-D array": np.array([0.5, 0.5]),
            "empty array": np.empty((0, 2)),
        }
        for name, y_prob_target in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                em(y_true_source, y_prob_target)

    def test_invalid_source_inputs(self):
        """Invalid source inputs and settings raise ValueError."""
        y_prob_target = np.array([[0.8, 0.2], [0.4, 0.6]])
        good_source = np.array([0, 0, 1, 1])
        empty_source = np.array([], dtype=int)
        cases = {
            "empty source labels": (empty_source, None, {}),
            "prior wrong length": (good_source, [0.5, 0.3, 0.2], {}),
            "prior negative": (good_source, [0.6, -0.1], {}),
            "prior nonfinite": (good_source, [0.6, np.nan], {}),
            "prior not summing to 1": (good_source, [0.6, 0.6], {}),
            "prior zero for observed class": (good_source, [1.0, 0.0], {}),
            "tol zero": (good_source, None, {"tol": 0.0}),
            "tol negative": (good_source, None, {"tol": -1e-8}),
            "tol infinite": (good_source, None, {"tol": np.inf}),
            "max_iter zero": (good_source, None, {"max_iter": 0}),
            "max_iter negative": (good_source, None, {"max_iter": -3}),
            "max_iter non-integer": (good_source, None, {"max_iter": 2.5}),
            "max_iter bool": (good_source, None, {"max_iter": True}),
        }
        for name, (source, prior, kwargs) in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                em(
                    source,
                    y_prob_target,
                    source_label_dist=prior,
                    **kwargs,
                )
        # A supplied prior that differs from the empirical source
        # proportions is accepted.
        estimate = em(
            good_source, y_prob_target, source_label_dist=[0.9, 0.1]
        )
        _assert_valid_distribution(self, estimate, 2)

    def test_supplied_prior_overrides_empirical(self):
        """A supplied prior is actually used, not just accepted.

        Source labels [0, 0, 1, 1] give empirical prior [0.5, 0.5].
        With target rows all [0.9, 0.1] and a supplied prior of
        [0.9, 0.1], the first E-step leaves the rows unchanged, so
        [0.9, 0.1] is an exact fixed point and EM returns it at once.
        With the empirical prior instead, the same rows push the
        estimate away from [0.9, 0.1] toward [1, 0], so the two
        estimates must differ clearly.
        """
        y_true_source = np.array([0, 0, 1, 1])
        y_prob_target = np.array([[0.9, 0.1], [0.9, 0.1]])
        with_prior = em(
            y_true_source,
            y_prob_target,
            source_label_dist=[0.9, 0.1],
        )
        np.testing.assert_allclose(with_prior, [0.9, 0.1], atol=1e-8)
        without_prior = em(y_true_source, y_prob_target)
        self.assertGreater(
            np.max(np.abs(with_prior - without_prior)), 0.05
        )

    def test_exact_fixed_point(self):
        """Source labels with empirical prior [0.6, 0.4] and target
        rows whose average is the prior itself: the first M-step leaves
        the prior unchanged, so convergence is immediate."""
        y_true_source = np.array([0, 0, 0, 1, 1])
        y_prob_target = np.array(
            [[0.8, 0.2], [0.4, 0.6], [0.6, 0.4], [0.6, 0.4]]
        )
        estimate = em(y_true_source, y_prob_target)
        np.testing.assert_allclose(estimate, [0.6, 0.4], atol=1e-8)

    def test_hand_computable_single_step(self):
        """Prior [0.5, 0.5] with rows (0.8, 0.2) and (0.4, 0.6): the
        first M-step averages the rows to [0.6, 0.4] exactly. One step
        does not meet tol, so the non-convergence warning fires."""
        y_true_source = np.array([0, 1])
        y_prob_target = np.array([[0.8, 0.2], [0.4, 0.6]])
        with self.assertWarns(UserWarning):
            estimate = em(
                y_true_source, y_prob_target, max_iter=1, tol=1e-8
            )
        np.testing.assert_allclose(estimate, [0.6, 0.4], atol=1e-8)

    def test_zero_prior_class_errors(self):
        """A class with zero source prior that receives positive
        predicted probability has no defined importance weight, so EM
        raises ValueError. If its target column is identically zero, the
        estimate is accepted and stays [1.0, 0.0]."""
        y_true_source = np.array([0, 0, 0, 0])
        with self.assertRaises(ValueError):
            em(
                y_true_source,
                np.array([[1.0, 0.0], [0.7, 0.3]]),
            )
        estimate = em(
            y_true_source,
            np.array([[1.0, 0.0], [1.0, 0.0]]),
        )
        np.testing.assert_allclose(estimate, [1.0, 0.0], atol=1e-12)

    def test_e_step_denominator_guard(self):
        """A constructed numerically degenerate E-step update raises
        ValueError, never a silent NaN or a division by zero."""
        with self.assertRaises(ValueError):
            _posteriors(
                np.array([[1.0, 0.0]]),
                np.array([0.0, 0.0]),
                np.array([0.5, 0.5]),
            )
        with self.assertRaises(ValueError):
            _posteriors(
                np.array([[1.0, 0.0]]),
                np.array([np.nan, 0.5]),
                np.array([0.5, 0.5]),
            )

    def test_api_integration_with_label_shift(self):
        """The em() estimate feeds LabelShift.calibrate as
        target_label_dist and produces correctly shaped prediction
        sets."""
        y_true_source = np.repeat([0, 1, 2], 4)
        y_prob_target = np.array(
            [[1.0, 0.0, 0.0]] * 6
            + [[0.0, 1.0, 0.0]] * 3
            + [[0.0, 0.0, 1.0]] * 3
        )
        estimate = em(y_true_source, y_prob_target)
        np.testing.assert_allclose(
            estimate, [0.5, 0.25, 0.25], atol=1e-8
        )

        dataset, model = make_sample_dataset_and_model()
        cal_model = LabelShift(model=model, alpha=0.3)
        cal_model.calibrate(
            cal_dataset=dataset, target_label_dist=estimate
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            output = cal_model(**batch)
        self.assertEqual(output["y_predset"].dtype, torch.bool)
        self.assertEqual(output["y_predset"].shape, output["y_prob"].shape)

    def test_coverage_recovery(self):
        """Thresholds weighted with the EM-estimated mix restore coverage
        under label shift with a genuinely imperfect classifier, on an
        independent target test set. The EM estimate lands within 0.02
        of the true mix per class, and weighted coverage is closer to
        the nominal 90% than unweighted coverage."""
        rng = np.random.default_rng(7)
        alpha = 0.1
        source_mix = np.array([0.9, 0.1])
        target_mix = np.array([0.7, 0.3])

        cal_proba, cal_true = _binary_probas(rng, 6000, source_mix)
        tgt_proba, _ = _binary_probas(
            rng, 20000, target_mix, source_mix=source_mix
        )
        estimate = em(cal_true, tgt_proba)
        np.testing.assert_allclose(estimate, target_mix, atol=0.02)

        nc_scores = 1.0 - cal_proba[np.arange(len(cal_true)), cal_true]
        thresholds, _ = _compute_label_shift_thresholds(
            nc_scores,
            cal_true,
            target_label_dist=estimate,
            source_label_dist=source_mix,
            alpha=alpha,
        )
        plain_threshold = _query_quantile(nc_scores, alpha)

        tst_proba, tst_true = _binary_probas(
            rng, 30000, target_mix, source_mix=source_mix
        )
        tst_nc = 1.0 - tst_proba[np.arange(len(tst_true)), tst_true]
        weighted_coverage = np.mean(tst_nc <= thresholds[tst_true])
        plain_coverage = np.mean(tst_nc <= plain_threshold)
        # Weighted coverage must be closer to the nominal 90% than
        # unweighted coverage, not merely higher: a weighted coverage of
        # 0.99 with unweighted 0.94 would pass a one-sided comparison
        # while being farther from nominal.
        self.assertLessEqual(abs(weighted_coverage - 0.9), 0.02)
        self.assertLess(
            abs(weighted_coverage - 0.9),
            abs(plain_coverage - 0.9) - 0.05,
        )

    def test_stopping_condition_is_strict(self):
        """The EM loop stops when the largest update is strictly below
        tol, not when it merely equals tol. With prior [0.5, 0.5] and
        rows [[0.75, 0.25], [0.75, 0.25]], the first update is exactly
        0.25, so tol=0.25 must not declare convergence: the loop runs a
        second iteration and settles at [0.9, 0.1]."""
        y_true_source = np.array([0, 1])
        y_prob_target = np.array([[0.75, 0.25], [0.75, 0.25]])
        estimate = em(y_true_source, y_prob_target, tol=0.25)
        np.testing.assert_allclose(estimate, [0.9, 0.1], atol=1e-12)

    def test_constant_target_rows_warn(self):
        """Constant target rows carry no prevalence information, so EM
        warns instead of silently returning the source prior. With
        source prior [0.75, 0.25] and every target row [0.75, 0.25],
        the source prior is an exact fixed point, exactly the case
        Logiquo flagged."""
        y_true_source = np.array([0, 0, 0, 1])
        y_prob_target = np.array(
            [[0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimate = em(y_true_source, y_prob_target)
        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, UserWarning))
        self.assertIn("constant", str(caught[0].message))
        np.testing.assert_allclose(estimate, [0.75, 0.25], atol=1e-12)

    def test_near_constant_target_rows_warn(self):
        """Rows that differ by only 1e-4 still warn: a spread that
        small carries almost no prevalence information, even though
        the rows are not exactly identical."""
        y_true_source = np.array([0, 0, 0, 1])
        y_prob_target = np.array(
            [[0.75, 0.25], [0.7501, 0.2499], [0.7499, 0.2501]]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimate = em(y_true_source, y_prob_target)
        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, UserWarning))
        self.assertIn("nearly constant", str(caught[0].message))
        _assert_valid_distribution(self, estimate, 2)

    def test_varying_target_rows_do_not_warn(self):
        """Informative target rows must not trigger the constant-rows
        warning."""
        y_true_source = np.array([0, 0, 1, 1])
        y_prob_target = np.array([[0.9, 0.1], [0.1, 0.9]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            estimate = em(y_true_source, y_prob_target)
        _assert_valid_distribution(self, estimate, 2)


if __name__ == "__main__":
    unittest.main()
