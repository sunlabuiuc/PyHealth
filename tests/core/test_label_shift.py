import copy
import unittest

import numpy as np
import torch

from pyhealth.calib.predictionset import LabelShift
from pyhealth.calib.predictionset.base_conformal import (
    _query_quantile,
    _query_weighted_quantile,
)
from pyhealth.calib.predictionset.label_shift.label_shift import (
    _compute_label_shift_thresholds,
    _label_shift_threshold,
    _validate_calibration_labels,
    _validate_label_distribution,
)
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP
from tests.core.label_shift_test_utils import (
    SAMPLES,
    make_sample_dataset_and_model,
)


class TestLabelShiftHelpers(unittest.TestCase):
    """Tests for the weighted threshold math, at the numpy level."""

    def test_hand_computed_thresholds(self):
        """Small example worked out by hand.

        Calibration scores [0.1, 0.3, 0.5, 0.7] with labels [0, 0, 1, 1], so
        the source mix is [0.5, 0.5]. With target mix [0.25, 0.75] the label
        ratios are [0.5, 1.5] and the calibration weights are
        [0.5, 0.5, 1.5, 1.5]. At alpha=0.25:

        - Candidate 0 reserves 0.5 of mass at +inf (total 4.5); the
          cumulative weights are [0.111, 0.222, 0.556, 0.889], so the first
          score reaching 0.75 is 0.7.
        - Candidate 1 reserves 1.5 of mass at +inf (total 5.5); the
          cumulative weights only reach 0.727, so the threshold is +inf.
        """
        scores = np.array([0.1, 0.3, 0.5, 0.7])
        y_true = np.array([0, 0, 1, 1])
        thresholds, ratios = _compute_label_shift_thresholds(
            scores,
            y_true,
            np.array([0.25, 0.75]),
            np.array([0.5, 0.5]),
            alpha=0.25,
        )
        np.testing.assert_allclose(ratios, [0.5, 1.5])
        self.assertEqual(thresholds[0], 0.7)
        self.assertEqual(thresholds[1], np.inf)

    def test_no_shift_recovers_base_conformal(self):
        """Unit weights and a unit candidate weight must reproduce the
        standard split conformal quantile exactly, for any calibration set
        size and alpha."""
        rng = np.random.default_rng(0)
        for n in (10, 57, 200):
            scores = rng.random(n)
            for alpha in (0.05, 0.1, 0.25):
                expected = _query_quantile(scores, alpha)
                got = _label_shift_threshold(scores, np.ones(n), alpha, 1.0)
                self.assertEqual(got, expected)

    def test_absent_class_threshold_differs_under_no_shift(self):
        """A class absent from calibration is the exception to the
        no-shift equivalence with BaseConformal.

        Scores [0.1, 0.3, 0.5, 0.7] with labels [0, 0, 1, 1] and
        source = target = [0.5, 0.5, 0.0] at alpha=0.1: classes 0 and
        1 reserve a unit candidate mass, so their cumulative mass only
        reaches 0.8 and their thresholds are +inf, matching
        BaseConformal. Class 2 has ratio 0 and reserves no candidate
        mass, so its cumulative mass reaches 1.0 at 0.7 and its
        threshold is 0.7, not +inf.
        """
        scores = np.array([0.1, 0.3, 0.5, 0.7])
        y_true = np.array([0, 0, 1, 1])
        source = np.array([0.5, 0.5, 0.0])
        thresholds, ratios = _compute_label_shift_thresholds(
            scores, y_true, source, source, alpha=0.1
        )
        np.testing.assert_allclose(ratios, [1.0, 1.0, 0.0])
        self.assertEqual(thresholds[0], np.inf)
        self.assertEqual(thresholds[1], np.inf)
        self.assertEqual(thresholds[2], 0.7)
        self.assertEqual(_query_quantile(scores, 0.1), np.inf)

    def test_weighted_quantile_candidate_weight_and_exhaustion(self):
        """The shared helper's candidate-mass and exhaustion branches.

        Scores [0.1, 0.3, 0.5] with unit weights and a candidate weight
        of 1 give total weight 4, so the cumulative weights only reach
        0.75. Querying 0.8 therefore exhausts the calibration mass: it
        returns +inf when return_inf_on_exhaustion is True, and falls
        back to the largest score when it is False.
        """
        scores = np.array([0.1, 0.3, 0.5])
        weights = np.ones(3)
        self.assertEqual(
            _query_weighted_quantile(
                scores, 0.8, weights, candidate_weight=1.0,
                return_inf_on_exhaustion=True,
            ),
            np.inf,
        )
        self.assertEqual(
            _query_weighted_quantile(
                scores, 0.8, weights, candidate_weight=1.0,
                return_inf_on_exhaustion=False,
            ),
            0.5,
        )

    def test_weighted_quantile_empty_scores(self):
        """Empty scores return +inf on exhaustion and -inf otherwise,
        whether or not a candidate mass is reserved."""
        empty = np.array([])
        empty_weights = np.array([])
        for candidate in (0.0, 2.0):
            self.assertEqual(
                _query_weighted_quantile(
                    empty, 0.9, empty_weights, candidate_weight=candidate,
                    return_inf_on_exhaustion=True,
                ),
                np.inf,
            )
            self.assertEqual(
                _query_weighted_quantile(
                    empty, 0.9, empty_weights, candidate_weight=candidate,
                    return_inf_on_exhaustion=False,
                ),
                -np.inf,
            )

    def test_target_equal_to_source_gives_single_threshold(self):
        """When the target distribution equals the empirical calibration
        distribution, every ratio is 1 and all per-class thresholds coincide
        with the BaseConformal threshold."""
        rng = np.random.default_rng(1)
        y_true = np.array([0, 0, 0, 1, 1, 2])
        scores = rng.random(len(y_true))
        source = np.bincount(y_true) / len(y_true)
        thresholds, ratios = _compute_label_shift_thresholds(
            scores, y_true, source, source, alpha=0.2
        )
        np.testing.assert_allclose(ratios, np.ones(3))
        expected = _query_quantile(scores, 0.2)
        np.testing.assert_allclose(thresholds, np.full(3, expected))

    def test_class_missing_from_calibration_is_always_included(self):
        """A class with target mass but no calibration examples gets an
        infinite ratio and an infinite threshold."""
        scores = np.array([0.2, 0.4, 0.6])
        y_true = np.array([0, 0, 1])
        source = np.array([2 / 3, 1 / 3, 0.0])
        thresholds, ratios = _compute_label_shift_thresholds(
            scores, y_true, np.array([0.3, 0.3, 0.4]), source, alpha=0.1
        )
        self.assertEqual(ratios[2], np.inf)
        self.assertEqual(thresholds[2], np.inf)

    def test_large_candidate_ratio_gives_inf_despite_positive_source(self):
        """A positive source mass does not by itself guarantee a finite
        threshold: with one calibration score of weight about 0.5 and a
        candidate ratio of 50, the calibration mass reaches only about
        0.01 of the total weight, well below 1 - alpha, so the threshold
        is +inf even though the class has positive source mass and the
        scores are computed over all calibration examples, not a
        per-class subset."""
        scores = np.array([0.5])
        y_true = np.array([0])
        thresholds, ratios = _compute_label_shift_thresholds(
            scores,
            y_true,
            np.array([0.5, 0.5]),
            np.array([0.99, 0.01]),
            alpha=0.1,
        )
        self.assertAlmostEqual(ratios[1], 50.0)
        self.assertEqual(thresholds[1], np.inf)

    def test_simulated_label_shift_coverage(self):
        """Under a known label shift, the oracle-weighted thresholds should
        hit the nominal coverage on target data while the unweighted split
        conformal threshold undercovers.

        Setup: binary problem, class 1 is harder (higher NC scores) and rare
        in the source mix (10%) but common in the target mix (30%). The
        class-conditional score distributions do not change (label shift).
        """
        rng = np.random.default_rng(42)
        alpha = 0.1
        source_mix = np.array([0.9, 0.1])
        target_mix = np.array([0.7, 0.3])

        def draw(n, mix):
            y = rng.choice(2, size=n, p=mix)
            # Probability assigned to the true class; class 1 is harder.
            p_true = np.where(
                y == 0, rng.uniform(0.70, 0.98, n), rng.uniform(0.50, 0.85, n)
            )
            y_prob = np.empty((n, 2))
            y_prob[np.arange(n), y] = p_true
            y_prob[np.arange(n), 1 - y] = 1 - p_true
            return y_prob, y

        cal_prob, cal_true = draw(1500, source_mix)
        cal_scores = 1 - cal_prob[np.arange(len(cal_true)), cal_true]

        test_prob, test_true = draw(30000, target_mix)
        test_scores = 1 - test_prob  # NC score of every class

        # Weighted (oracle source and target mixes): per-class thresholds.
        thresholds, _ = _compute_label_shift_thresholds(
            cal_scores, cal_true, target_mix, source_mix, alpha
        )
        weighted_sets = test_scores <= thresholds
        weighted_coverage = weighted_sets[
            np.arange(len(test_true)), test_true
        ].mean()

        # Unweighted: one global split conformal threshold.
        t_global = _query_quantile(cal_scores, alpha)
        plain_sets = test_scores <= t_global
        plain_coverage = plain_sets[np.arange(len(test_true)), test_true].mean()

        self.assertLess(plain_coverage, 0.85)
        self.assertGreater(weighted_coverage, 0.88)
        self.assertLess(weighted_coverage, 0.92)
        self.assertGreater(weighted_coverage, plain_coverage + 0.03)

    def test_tied_scores(self):
        """Tied scores: the threshold is the tied value when the cumulative
        mass reaches 1 - alpha inside the tie, and the next distinct score
        when it does not."""
        scores = np.array([0.4, 0.4, 0.4, 0.8])
        weights = np.ones(4)
        self.assertEqual(
            _label_shift_threshold(scores, weights, 0.5, 1.0), 0.4
        )
        self.assertEqual(
            _label_shift_threshold(scores, weights, 0.25, 1.0), 0.8
        )

    def test_exact_cumulative_boundary(self):
        """When the cumulative weight hits 1 - alpha exactly at a score,
        that score is the threshold (the construction uses >=), and a
        slightly smaller alpha moves past it."""
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        weights = np.ones(4)
        # Cumulative weights are [0.2, 0.4, 0.6, 0.8] out of a total of 5.
        self.assertEqual(
            _label_shift_threshold(scores, weights, 0.4, 1.0), 0.3
        )
        self.assertEqual(
            _label_shift_threshold(scores, weights, 0.39, 1.0), 0.4
        )

    def test_zero_weight_points_do_not_move_threshold(self):
        """A calibration point with weight 0 counts for nothing, whatever
        its score."""
        with_point = _label_shift_threshold(
            np.array([0.1, 0.2, 0.99]), np.array([1.0, 1.0, 0.0]), 0.5, 1.0
        )
        without_point = _label_shift_threshold(
            np.array([0.1, 0.2]), np.array([1.0, 1.0]), 0.5, 1.0
        )
        self.assertEqual(with_point, without_point)
        self.assertEqual(with_point, 0.2)

    def test_all_zero_calibration_weights(self):
        """With no calibration mass at all, the threshold is +inf whether
        the candidate weight is positive or also zero."""
        scores = np.array([0.3, 0.9])
        weights = np.zeros(2)
        self.assertEqual(
            _label_shift_threshold(scores, weights, 0.1, 2.0), np.inf
        )
        self.assertEqual(
            _label_shift_threshold(scores, weights, 0.1, 0.0), np.inf
        )

    def test_single_point_needing_infinite_threshold(self):
        """One calibration point rarely carries enough mass on its own."""
        self.assertEqual(
            _label_shift_threshold(
                np.array([0.42]), np.array([1.0]), 0.1, 1.0
            ),
            np.inf,
        )

    def test_no_shift_equivalence_at_rank_boundaries(self):
        """No-shift equivalence also holds when (n + 1)(1 - alpha) falls
        exactly on an integer rank."""
        rng = np.random.default_rng(3)
        cases = (
            (1, [0.5]),
            (3, [0.25, 0.5, 0.75]),
            (7, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]),
        )
        for n, alphas in cases:
            scores = rng.random(n)
            for alpha in alphas:
                expected = _query_quantile(scores, alpha)
                got = _label_shift_threshold(scores, np.ones(n), alpha, 1.0)
                self.assertEqual(got, expected)

    def test_zero_target_mass_class_points_are_ignored(self):
        """Changing the scores of a zero-target-mass class cannot change
        the other classes' thresholds, since its points have weight 0."""
        y_true = np.array([0, 0, 1, 1, 2])
        source = np.array([0.4, 0.4, 0.2])
        target = np.array([0.5, 0.5, 0.0])
        scores_a = np.array([0.1, 0.2, 0.3, 0.4, 0.05])
        scores_b = np.array([0.1, 0.2, 0.3, 0.4, 0.99])
        thresholds_a, ratios = _compute_label_shift_thresholds(
            scores_a, y_true, target, source, alpha=0.25
        )
        thresholds_b, _ = _compute_label_shift_thresholds(
            scores_b, y_true, target, source, alpha=0.25
        )
        self.assertEqual(ratios[2], 0.0)
        self.assertEqual(thresholds_a[0], thresholds_b[0])
        self.assertEqual(thresholds_a[1], thresholds_b[1])
        # The class itself still gets the weighted quantile of the
        # remaining mass with a candidate weight of 0.
        expected = _label_shift_threshold(
            scores_a, ratios[y_true], 0.25, 0.0
        )
        self.assertEqual(thresholds_a[2], expected)

    def test_validate_label_distribution(self):
        # Very large finite values must not overflow the normalization.
        dist = _validate_label_distribution(
            [1e308, 1e308], 2, "target_label_dist"
        )
        np.testing.assert_allclose(dist, [0.5, 0.5])
        self.assertTrue(np.all(np.isfinite(dist)))
        # Unnormalized input is fine.
        dist = _validate_label_distribution(
            [2.0, 1.0, 1.0], 3, "target_label_dist"
        )
        np.testing.assert_allclose(dist, [0.5, 0.25, 0.25])
        for bad in (
            [np.nan, 1.0],
            [np.inf, 1.0],
            [0.5, -0.1],
            [0.0, 0.0],
            [0.5],
            [[0.5, 0.5]],
        ):
            with self.assertRaises(ValueError):
                _validate_label_distribution(bad, 2, "target_label_dist")

    def test_validate_calibration_labels(self):
        with self.assertRaises(ValueError):
            _validate_calibration_labels(
                np.array([], dtype=np.int64), 0, 3
            )
        with self.assertRaises(ValueError):
            _validate_calibration_labels(np.array([[0, 1]]), 1, 3)
        with self.assertRaises(ValueError):
            _validate_calibration_labels(np.array([0, 1, 2]), 2, 3)
        with self.assertRaises(ValueError):
            _validate_calibration_labels(np.array([0.5, 1.0]), 2, 3)
        with self.assertRaises(ValueError):
            _validate_calibration_labels(np.array([0, 3]), 2, 3)
        with self.assertRaises(ValueError):
            _validate_calibration_labels(np.array([-1, 0]), 2, 3)
        # Integral float labels are accepted and cast.
        out = _validate_calibration_labels(np.array([0.0, 2.0]), 2, 3)
        self.assertEqual(out.dtype, np.int64)
        np.testing.assert_array_equal(out, [0, 2])
        # Ordinary integer labels pass through unchanged.
        out = _validate_calibration_labels(np.array([0, 1]), 2, 3)
        np.testing.assert_array_equal(out, [0, 1])

    def test_simulated_coverage_repeated_draws(self):
        """Coverage across repeated calibration draws, oracle vs plug-in.

        With known source and target mixes (oracle ratios) the weighted
        construction has a finite-sample guarantee, so average coverage
        should sit at or above the nominal level. With the source mix
        estimated from the calibration labels (plug-in ratios) there is no
        exact guarantee; coverage should still be close, and this test
        only requires it to stay near the oracle level, not equal to it.
        """
        alpha = 0.1
        source_mix = np.array([0.9, 0.1])
        target_mix = np.array([0.7, 0.3])

        def draw(rng, n, mix):
            y = rng.choice(2, size=n, p=mix)
            p_true = np.where(
                y == 0, rng.uniform(0.70, 0.98, n), rng.uniform(0.50, 0.85, n)
            )
            y_prob = np.empty((n, 2))
            y_prob[np.arange(n), y] = p_true
            y_prob[np.arange(n), 1 - y] = 1 - p_true
            return y_prob, y

        oracle_coverages = []
        plugin_coverages = []
        for rep in range(25):
            rng = np.random.default_rng(1000 + rep)
            cal_prob, cal_true = draw(rng, 400, source_mix)
            cal_scores = 1 - cal_prob[np.arange(len(cal_true)), cal_true]
            test_prob, test_true = draw(rng, 4000, target_mix)
            test_scores = 1 - test_prob

            thresholds, _ = _compute_label_shift_thresholds(
                cal_scores, cal_true, target_mix, source_mix, alpha
            )
            oracle_coverages.append(
                (test_scores <= thresholds)[
                    np.arange(len(test_true)), test_true
                ].mean()
            )

            plugin_source = np.bincount(cal_true, minlength=2) / len(cal_true)
            thresholds, _ = _compute_label_shift_thresholds(
                cal_scores, cal_true, target_mix, plugin_source, alpha
            )
            plugin_coverages.append(
                (test_scores <= thresholds)[
                    np.arange(len(test_true)), test_true
                ].mean()
            )

        self.assertGreaterEqual(np.mean(oracle_coverages), 0.885)
        self.assertGreaterEqual(min(oracle_coverages), 0.83)
        self.assertGreaterEqual(np.mean(plugin_coverages), 0.86)


class TestLabelShift(unittest.TestCase):
    """End-to-end tests with a small sample dataset and an MLP."""

    def setUp(self):
        # A copy, not an alias: mutating self.samples in a test must
        # never mutate the shared module-level fixture.
        self.samples = copy.deepcopy(SAMPLES)
        self.dataset, self.model = make_sample_dataset_and_model()

    def test_initialization(self):
        cal_model = LabelShift(model=self.model, alpha=0.1)
        self.assertIsInstance(cal_model, LabelShift)
        self.assertEqual(cal_model.mode, "multiclass")
        self.assertEqual(cal_model.alpha, 0.1)
        self.assertIsNone(cal_model.t)

    def test_initialization_rejects_bad_alpha(self):
        with self.assertRaises(ValueError):
            LabelShift(model=self.model, alpha=[0.1, 0.1, 0.1])
        with self.assertRaises(ValueError):
            LabelShift(model=self.model, alpha=1.5)
        with self.assertRaises(ValueError):
            LabelShift(model=self.model, alpha=0.0)

    def test_bad_alpha_does_not_freeze_model(self):
        """Alpha is validated before the parent init, so a rejected
        alpha must not freeze the base model's parameters."""
        with self.assertRaises(ValueError):
            LabelShift(model=self.model, alpha=1.5)
        self.assertTrue(
            all(p.requires_grad for p in self.model.parameters())
        )

    def test_initialization_unknown_score_type(self):
        with self.assertRaises(ValueError):
            LabelShift(model=self.model, alpha=0.1, score_type="nope")

    def test_initialization_non_multiclass_raises_error(self):
        binary_samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-1"],
                "procedures": [1.0],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-2"],
                "procedures": [2.0],
                "label": 1,
            },
        ]
        binary_dataset = create_sample_dataset(
            samples=binary_samples,
            input_schema={"conditions": "sequence", "procedures": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="test",
        )
        binary_model = MLP(
            dataset=binary_dataset,
            feature_keys=["conditions"],
            label_key="label",
            mode="binary",
        )
        with self.assertRaises(NotImplementedError):
            LabelShift(model=binary_model, alpha=0.1)

    def test_calibrate_sets_per_class_thresholds(self):
        cal_model = LabelShift(model=self.model, alpha=0.3)
        cal_model.calibrate(
            cal_dataset=self.dataset, target_label_dist=[0.5, 0.3, 0.2]
        )
        self.assertIsNotNone(cal_model.t)
        self.assertIsInstance(cal_model.t, torch.Tensor)
        self.assertEqual(cal_model.t.shape, (3,))
        # The 6 calibration samples are uniform over classes.
        np.testing.assert_allclose(cal_model.source_label_dist_, [1 / 3] * 3)
        np.testing.assert_allclose(cal_model.label_ratios_, [1.5, 0.9, 0.6])
        np.testing.assert_allclose(cal_model.target_label_dist_, [0.5, 0.3, 0.2])

    def test_calibrate_normalizes_target_dist(self):
        cal_model = LabelShift(model=self.model, alpha=0.3)
        cal_model.calibrate(
            cal_dataset=self.dataset, target_label_dist=[2.0, 1.0, 1.0]
        )
        np.testing.assert_allclose(
            cal_model.target_label_dist_, [0.5, 0.25, 0.25]
        )

    def test_calibrate_validates_target_dist(self):
        cal_model = LabelShift(model=self.model, alpha=0.3)
        with self.assertRaises(ValueError):
            cal_model.calibrate(
                cal_dataset=self.dataset, target_label_dist=[0.5, 0.5]
            )
        with self.assertRaises(ValueError):
            cal_model.calibrate(
                cal_dataset=self.dataset, target_label_dist=[0.5, -0.1, 0.6]
            )
        with self.assertRaises(ValueError):
            cal_model.calibrate(
                cal_dataset=self.dataset, target_label_dist=[0.0, 0.0, 0.0]
            )

    def test_calibrate_with_known_source_dist(self):
        # The calibration samples are uniform over classes, so passing the
        # known source distribution explicitly matches the plug-in default.
        default_model = LabelShift(model=self.model, alpha=0.3)
        default_model.calibrate(
            cal_dataset=self.dataset, target_label_dist=[0.5, 0.3, 0.2]
        )
        known_model = LabelShift(model=self.model, alpha=0.3)
        known_model.calibrate(
            cal_dataset=self.dataset,
            target_label_dist=[0.5, 0.3, 0.2],
            source_label_dist=[1 / 3, 1 / 3, 1 / 3],
        )
        np.testing.assert_allclose(
            known_model.t.numpy(), default_model.t.numpy()
        )
        np.testing.assert_allclose(known_model.source_label_dist_, [1 / 3] * 3)
        # A non-uniform known source gives the corresponding ratios.
        other_model = LabelShift(model=self.model, alpha=0.3)
        other_model.calibrate(
            cal_dataset=self.dataset,
            target_label_dist=[0.5, 0.3, 0.2],
            source_label_dist=[0.5, 0.25, 0.25],
        )
        np.testing.assert_allclose(other_model.label_ratios_, [1.0, 1.2, 0.8])

    def test_calibrate_validates_source_dist(self):
        cal_model = LabelShift(model=self.model, alpha=0.3)
        for bad in ([0.5, 0.5], [0.5, -0.1, 0.6], [np.nan, 0.5, 0.5],
                    [0.0, 0.0, 0.0]):
            with self.assertRaises(ValueError):
                cal_model.calibrate(
                    cal_dataset=self.dataset,
                    target_label_dist=[0.5, 0.3, 0.2],
                    source_label_dist=bad,
                )

    def test_calibrate_rejects_labels_outside_source_support(self):
        """A supplied source distribution that gives a class zero
        probability is inconsistent with calibration data containing that
        class, and is rejected instead of producing infinite calibration
        weights (and NaN thresholds)."""
        cal_model = LabelShift(model=self.model, alpha=0.3)
        with self.assertRaises(ValueError):
            cal_model.calibrate(
                cal_dataset=self.dataset,
                target_label_dist=[0.5, 0.3, 0.2],
                source_label_dist=[0.5, 0.5, 0.0],
            )

    def test_unobserved_target_class_gets_infinite_threshold(self):
        """A class with target mass but no calibration examples and zero
        supplied source probability keeps its infinite threshold."""
        subset_samples = [s for s in self.samples if s["label"] in (0, 1)]
        subset_dataset = create_sample_dataset(
            samples=subset_samples,
            input_schema={"conditions": "sequence", "procedures": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test",
        )
        cal_model = LabelShift(model=self.model, alpha=0.3)
        cal_model.calibrate(
            cal_dataset=subset_dataset,
            target_label_dist=[0.4, 0.3, 0.3],
            source_label_dist=[0.6, 0.4, 0.0],
        )
        self.assertEqual(cal_model.label_ratios_[2], np.inf)
        self.assertTrue(np.isinf(cal_model.t.numpy()[2]))

    def test_calibrate_extreme_target_dist_no_overflow(self):
        cal_model = LabelShift(model=self.model, alpha=0.3)
        cal_model.calibrate(
            cal_dataset=self.dataset,
            target_label_dist=[1e308, 1e308, 1e308],
        )
        np.testing.assert_allclose(cal_model.target_label_dist_, [1 / 3] * 3)
        self.assertTrue(np.all(np.isfinite(cal_model.target_label_dist_)))
        self.assertFalse(np.isnan(cal_model.t.numpy()).any())

    def test_calibrate_empty_dataset_raises(self):
        empty_dataset = create_sample_dataset(
            samples=[],
            input_schema={"conditions": "sequence", "procedures": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test",
        )
        cal_model = LabelShift(model=self.model, alpha=0.3)
        with self.assertRaises(ValueError):
            cal_model.calibrate(
                cal_dataset=empty_dataset, target_label_dist=[0.5, 0.3, 0.2]
            )

    def test_forward_candidatewise_with_known_thresholds(self):
        """With known thresholds, a class is in the set exactly when its
        NC score is at or below its own threshold; an infinite threshold
        includes the class for every input."""
        cal_model = LabelShift(model=self.model, alpha=0.3)
        cal_model.calibrate(
            cal_dataset=self.dataset, target_label_dist=[0.5, 0.3, 0.2]
        )
        thresholds = np.array([0.5, np.inf, 0.1], dtype=np.float32)
        cal_model.t = torch.tensor(thresholds)
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            output = cal_model(**batch)
        scores = 1 - output["y_prob"].numpy().astype(np.float32)
        expected = scores <= thresholds
        np.testing.assert_array_equal(
            output["y_predset"].numpy(), expected
        )
        # The infinite-threshold class is always included.
        self.assertTrue(output["y_predset"][:, 1].all())

    def test_aps_reproducible_with_fixed_seed(self):
        """Two instances with the same seed calibrate to the same
        thresholds and produce the same prediction sets."""

        def build_and_run():
            cal_model = LabelShift(
                model=self.model, alpha=0.3, score_type="aps",
                random_state=42,
            )
            cal_model.calibrate(
                cal_dataset=self.dataset, target_label_dist=[0.5, 0.3, 0.2]
            )
            loader = get_dataloader(
                self.dataset, batch_size=2, shuffle=False
            )
            batch = next(iter(loader))
            with torch.no_grad():
                output = cal_model(**batch)
            return cal_model.t.numpy(), output["y_predset"].numpy()

        t1, sets1 = build_and_run()
        t2, sets2 = build_and_run()
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(sets1, sets2)

    def test_forward_before_calibrate_raises(self):
        cal_model = LabelShift(model=self.model, alpha=0.3)
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        with self.assertRaises(RuntimeError), torch.no_grad():
            cal_model(**batch)

    def test_forward_returns_predset(self):
        cal_model = LabelShift(model=self.model, alpha=0.3)
        cal_model.calibrate(
            cal_dataset=self.dataset, target_label_dist=[0.5, 0.3, 0.2]
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            output = cal_model(**batch)
        self.assertIn("y_predset", output)
        self.assertEqual(output["y_predset"].dtype, torch.bool)
        self.assertEqual(output["y_predset"].shape, output["y_prob"].shape)

    def test_score_type_aps_runs_end_to_end(self):
        cal_model = LabelShift(
            model=self.model, alpha=0.3, score_type="aps", random_state=42
        )
        cal_model.calibrate(
            cal_dataset=self.dataset, target_label_dist=[0.5, 0.3, 0.2]
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            output = cal_model(**batch)
        self.assertEqual(output["y_predset"].dtype, torch.bool)
        self.assertEqual(output["y_predset"].shape, output["y_prob"].shape)


if __name__ == "__main__":
    unittest.main()
