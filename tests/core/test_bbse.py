import unittest

import numpy as np
import torch

from pyhealth.calib.predictionset import LabelShift, bbse
from pyhealth.calib.predictionset.base_conformal import _query_quantile
from pyhealth.calib.predictionset.label_shift.black_box_shift import _project_to_simplex
from pyhealth.calib.predictionset.label_shift.label_shift import (
    _compute_label_shift_thresholds,
)
from pyhealth.datasets import get_dataloader
from tests.core.label_shift_test_utils import make_sample_dataset_and_model


class TestBBSE(unittest.TestCase):
    """Tests for the BBSE target label distribution estimator."""

    def test_exact_recovery_from_counts(self):
        """With exact confusion and prediction counts, BBSE recovers the
        target mix used to generate the target predictions.

        The source confusion is P(pred | true) = [[0.9, 0.2], [0.1, 0.8]].
        A target mix of [0.7, 0.3] then predicts class 0 with probability
        0.9 * 0.7 + 0.2 * 0.3 = 0.69.
        """
        y_true_source = np.repeat([0, 1], 100)
        y_pred_source = np.array([0] * 90 + [1] * 10 + [0] * 20 + [1] * 80)
        y_pred_target = np.array([0] * 69 + [1] * 31)
        estimate = bbse(y_true_source, y_pred_source, y_pred_target)
        np.testing.assert_allclose(estimate, [0.7, 0.3], atol=1e-10)

    def test_no_shift_recovers_source_distribution(self):
        """Target predictions generated under the source mix give the
        source mix back. The source labels themselves follow the mix
        [0.75, 0.25], with the same class-conditional confusion as the
        exact recovery test."""
        y_true_source = np.array([0] * 300 + [1] * 100)
        y_pred_source = np.array([0] * 270 + [1] * 30 + [0] * 20 + [1] * 80)
        # P(pred = 0) under the mix is 0.9 * 0.75 + 0.2 * 0.25 = 0.725.
        y_pred_target = np.array([0] * 725 + [1] * 275)
        estimate = bbse(y_true_source, y_pred_source, y_pred_target)
        np.testing.assert_allclose(estimate, [0.75, 0.25], atol=1e-10)

    def test_simulated_estimation_accuracy(self):
        """On sampled data the estimate lands close to the true mix."""
        rng = np.random.default_rng(0)
        source_mix = np.array([0.9, 0.1])
        target_mix = np.array([0.7, 0.3])

        def draw(n, mix):
            y = rng.choice(2, size=n, p=mix)
            # Class-dependent accuracy: 0.85 for class 0, 0.70 for class 1.
            accuracy = np.where(y == 0, 0.85, 0.70)
            correct = rng.random(n) < accuracy
            y_pred = np.where(correct, y, 1 - y)
            return y, y_pred

        y_true_source, y_pred_source = draw(20000, source_mix)
        _, y_pred_target = draw(20000, target_mix)
        estimate = bbse(y_true_source, y_pred_source, y_pred_target)
        np.testing.assert_allclose(estimate, target_mix, atol=0.03)

    def test_multiclass_exact_recovery(self):
        """Exact recovery in a 3-class problem with a non-identity,
        full-rank confusion matrix.

        The columns of the confusion matrix (P(pred | true)) are
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], and [0.1, 0.2, 0.7]. A target
        mix of [0.2, 0.3, 0.5] then predicts with distribution
        [0.24, 0.36, 0.40].
        """
        y_true_source = np.repeat([0, 1, 2], 1000)
        y_pred_source = np.array(
            [0] * 800 + [1] * 100 + [2] * 100
            + [0] * 100 + [1] * 800 + [2] * 100
            + [0] * 100 + [1] * 200 + [2] * 700
        )
        y_pred_target = np.array([0] * 240 + [1] * 360 + [2] * 400)
        estimate = bbse(y_true_source, y_pred_source, y_pred_target)
        np.testing.assert_allclose(estimate, [0.2, 0.3, 0.5], atol=1e-10)

    def test_multiclass_estimate_is_a_distribution(self):
        rng = np.random.default_rng(1)
        y_true_source = rng.choice(3, size=5000, p=[0.5, 0.3, 0.2])
        y_pred_source = y_true_source.copy()
        flip = rng.random(5000) < 0.2
        y_pred_source[flip] = rng.choice(3, size=flip.sum())
        y_pred_target = rng.choice(3, size=5000, p=[0.2, 0.3, 0.5])
        estimate = bbse(y_true_source, y_pred_source, y_pred_target)
        self.assertEqual(estimate.shape, (3,))
        self.assertTrue(np.all(estimate >= 0))
        self.assertAlmostEqual(estimate.sum(), 1.0)

    def test_project_to_simplex(self):
        # The Euclidean projection of (0.8, 0.3, -0.1) is (0.75, 0.25, 0);
        # clipping and renormalizing would give a different point.
        projected = _project_to_simplex(np.array([0.8, 0.3, -0.1]))
        np.testing.assert_allclose(projected, [0.75, 0.25, 0.0])
        # A point already on the simplex is unchanged.
        projected = _project_to_simplex(np.array([0.2, 0.3, 0.5]))
        np.testing.assert_allclose(projected, [0.2, 0.3, 0.5])

    def test_projection_back_to_simplex(self):
        """A nearly singular confusion matrix can make the raw least
        squares solution leave the simplex; the returned estimate is its
        Euclidean projection."""
        y_true_source = np.repeat([0, 1], 100)
        # Confusion [[0.51, 0.49], [0.49, 0.51]]: nearly uninformative.
        y_pred_source = np.array([0] * 51 + [1] * 49 + [0] * 49 + [1] * 51)
        # The raw solution for this target is about [20.5, -19.5].
        y_pred_target = np.array([0] * 90 + [1] * 10)
        estimate = bbse(y_true_source, y_pred_source, y_pred_target)
        np.testing.assert_allclose(estimate, [1.0, 0.0])
        self.assertAlmostEqual(estimate.sum(), 1.0)

    def test_rank_deficient_confusion_raises(self):
        """If the classifier's predictions do not separate the classes,
        the target mix is not identifiable and BBSE refuses instead of
        returning a made-up estimate."""
        # The classifier always predicts class 0: identical columns.
        y_true_source = np.repeat([0, 1], 100)
        y_pred_source = np.zeros(200, dtype=int)
        y_pred_target = np.array([0] * 20 + [1] * 80)
        with self.assertRaises(ValueError):
            bbse(y_true_source, y_pred_source, y_pred_target)

    def test_rank_deficient_confusion_raises_multiclass(self):
        """Classes 1 and 2 are both always predicted as class 1, so their
        confusion columns coincide and the system is rank deficient."""
        y_true_source = np.repeat([0, 1, 2], 100)
        y_pred_source = np.array([0] * 90 + [1] * 210)
        y_pred_target = np.repeat([0, 1, 2], 10)
        with self.assertRaises(ValueError):
            bbse(y_true_source, y_pred_source, y_pred_target)

    def test_near_singular_confusion_warns_but_recovers(self):
        """A full-rank but ill-conditioned confusion matrix (condition
        number about 1000) still identifies the mix: with exact counts
        the estimate is accurate, and the instability warning fires.

        Confusion [[0.5005, 0.4995], [0.4995, 0.5005]]; a target mix of
        [0.6, 0.4] predicts class 0 with probability 0.5001.
        """
        y_true_source = np.repeat([0, 1], 2000)
        y_pred_source = np.array(
            [0] * 1001 + [1] * 999 + [0] * 999 + [1] * 1001
        )
        y_pred_target = np.array([0] * 5001 + [1] * 4999)
        with self.assertWarns(UserWarning):
            estimate = bbse(y_true_source, y_pred_source, y_pred_target)
        np.testing.assert_allclose(estimate, [0.6, 0.4], atol=1e-6)

    def test_missing_source_class_raises(self):
        y_true_source = np.zeros(200, dtype=int)
        y_pred_source = np.repeat([0, 1], 100)
        y_pred_target = np.repeat([0, 1], 50)
        with self.assertRaises(ValueError):
            bbse(y_true_source, y_pred_source, y_pred_target, num_classes=2)

    def test_validation(self):
        y_true_source = np.array([0, 1, 0, 1])
        y_pred_source = np.array([0, 1, 1, 1])
        y_pred_target = np.array([0, 1, 1])
        # Mismatched source lengths.
        with self.assertRaises(ValueError):
            bbse(y_true_source, y_pred_source[:3], y_pred_target)
        # Empty target predictions.
        with self.assertRaises(ValueError):
            bbse(y_true_source, y_pred_source, np.array([], dtype=int))
        # Negative label.
        with self.assertRaises(ValueError):
            bbse(y_true_source, y_pred_source, np.array([0, -1]))
        # Non-integer label.
        with self.assertRaises(ValueError):
            bbse(y_true_source, y_pred_source, np.array([0.5, 1.0]))
        # Label outside the declared number of classes.
        with self.assertRaises(ValueError):
            bbse(
                y_true_source,
                y_pred_source,
                np.array([0, 2]),
                num_classes=2,
            )
        # num_classes must be a positive integer.
        with self.assertRaises(ValueError):
            bbse(y_true_source, y_pred_source, y_pred_target, num_classes=2.5)
        with self.assertRaises(ValueError):
            bbse(y_true_source, y_pred_source, y_pred_target, num_classes=0)
        # String labels.
        with self.assertRaises(ValueError):
            bbse(y_true_source, y_pred_source, np.array(["a", "b"]))
        # Nonfinite labels.
        with self.assertRaises(ValueError):
            bbse(y_true_source, y_pred_source, np.array([0.0, np.nan]))
        with self.assertRaises(ValueError):
            bbse(y_true_source, y_pred_source, np.array([0.0, np.inf]))
        # Integers too large for int64 arrive as object arrays.
        with self.assertRaises(ValueError):
            bbse(
                y_true_source,
                y_pred_source,
                np.array([0, 2**70], dtype=object),
            )

    def test_bbse_mix_restores_coverage(self):
        """End to end: thresholds weighted with the BBSE-estimated mix
        cover at close to the nominal rate under label shift, well above
        the unweighted threshold.

        Same simulation setup as the LabelShift oracle test: class 1 is
        harder and grows from 10% of the source mix to 30% of the target
        mix. Unlike that test, the classifier here is genuinely imperfect:
        the probability on the true class ranges well below 0.5, so the
        argmax prediction is wrong about 25% of the time for class 0 and
        about 40% for class 1. The confusion matrix is far from the
        identity and BBSE has real estimation work to do. BBSE only sees
        predicted labels, never target labels.
        """
        rng = np.random.default_rng(7)
        alpha = 0.1
        source_mix = np.array([0.9, 0.1])
        target_mix = np.array([0.7, 0.3])

        def draw(n, mix):
            y = rng.choice(2, size=n, p=mix)
            p_true = np.where(
                y == 0, rng.uniform(0.35, 0.95, n), rng.uniform(0.25, 0.85, n)
            )
            y_prob = np.empty((n, 2))
            y_prob[np.arange(n), y] = p_true
            y_prob[np.arange(n), 1 - y] = 1 - p_true
            return y_prob, y

        cal_prob, cal_true = draw(6000, source_mix)
        cal_scores = 1 - cal_prob[np.arange(len(cal_true)), cal_true]
        test_prob, test_true = draw(30000, target_mix)
        test_scores = 1 - test_prob

        estimated_mix = bbse(
            cal_true,
            cal_prob.argmax(axis=1),
            test_prob.argmax(axis=1),
        )
        np.testing.assert_allclose(estimated_mix, target_mix, atol=0.03)

        thresholds, _ = _compute_label_shift_thresholds(
            cal_scores, cal_true, estimated_mix, source_mix, alpha
        )
        weighted_sets = test_scores <= thresholds
        weighted_coverage = weighted_sets[
            np.arange(len(test_true)), test_true
        ].mean()

        t_global = _query_quantile(cal_scores, alpha)
        plain_coverage = (test_scores <= t_global)[
            np.arange(len(test_true)), test_true
        ].mean()

        self.assertGreater(weighted_coverage, 0.87)
        self.assertGreater(weighted_coverage, plain_coverage + 0.03)


class TestBBSELabelShiftInterface(unittest.TestCase):
    """The bbse estimate plugs into the public LabelShift interface."""

    def setUp(self):
        self.dataset, self.model = make_sample_dataset_and_model()

    def test_estimate_feeds_label_shift_calibrate(self):
        # The exact multiclass estimate from TestBBSE ([0.2, 0.3, 0.5]).
        y_true_source = np.repeat([0, 1, 2], 1000)
        y_pred_source = np.array(
            [0] * 800 + [1] * 100 + [2] * 100
            + [0] * 100 + [1] * 800 + [2] * 100
            + [0] * 100 + [1] * 200 + [2] * 700
        )
        y_pred_target = np.array([0] * 240 + [1] * 360 + [2] * 400)
        estimate = bbse(y_true_source, y_pred_source, y_pred_target)

        cal_model = LabelShift(model=self.model, alpha=0.3)
        cal_model.calibrate(
            cal_dataset=self.dataset, target_label_dist=estimate
        )
        np.testing.assert_allclose(
            cal_model.target_label_dist_, [0.2, 0.3, 0.5]
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            output = cal_model(**batch)
        self.assertEqual(output["y_predset"].dtype, torch.bool)
        self.assertEqual(output["y_predset"].shape, output["y_prob"].shape)


if __name__ == "__main__":
    unittest.main()
