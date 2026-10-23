import unittest

import numpy as np
import torch

from pyhealth.calib.predictionset import LabelShift, bbse, em
from pyhealth.calib.predictionset.base_conformal import _query_quantile
from pyhealth.calib.predictionset.label_shift.label_shift import (
    _compute_label_shift_thresholds,
)
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP


class TestEM(unittest.TestCase):
    """Tests for the MLLS target label distribution estimator."""

    def test_exact_recovery_with_perfect_classifier(self):
        """A perfect classifier makes the E-step posteriors one-hot on
        the true class, so one EM iteration averages the target rows to
        the exact target mix [0.75, 0.25]."""
        y_true_source = np.array([0, 0, 1, 1])
        y_proba_target = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        )
        estimate = em(y_true_source, y_proba_target)
        np.testing.assert_allclose(estimate, [0.75, 0.25], atol=1e-10)

    def test_single_iteration_matches_column_means(self):
        """With max_iter=1 the EM estimate is exactly one M-step from the
        source prior: with uniform source prior the importance weights
        start at 1, so the estimate is the column means of the target
        probabilities. A single iteration cannot have converged, so the
        non-convergence warning fires."""
        y_true_source = np.array([0, 0, 1, 1])
        y_proba_target = np.array([[0.6, 0.4], [0.2, 0.8]])
        with self.assertWarns(UserWarning):
            estimate = em(
                y_true_source, y_proba_target, max_iter=1, tol=0.0
            )
        np.testing.assert_allclose(estimate, [0.4, 0.6], atol=1e-12)

    @staticmethod
    def _gaussian_sim(rng, source_mix):
        """Calibrated simulation for the label shift tests.

        Class-conditional features are N(0, 1) and N(1.5, 1); the
        predicted probabilities are the exact Bayes posteriors under the
        source mix, so the classifier estimates the source class
        probabilities P(y | x) as MLLS assumes. Returns a draw(n, mix)
        function producing (probabilities, labels) pairs.
        """

        def posterior(x):
            log_odds = (
                np.log(source_mix[1])
                - np.log(source_mix[0])
                + 1.5 * x
                - 1.125
            )
            p1 = 1.0 / (1.0 + np.exp(-log_odds))
            return np.stack([1 - p1, p1], axis=1)

        def draw(n, mix):
            y = rng.choice(2, size=n, p=mix)
            x = rng.normal(loc=np.where(y == 0, 0.0, 1.5), size=n)
            return posterior(x), y

        return draw

    def test_no_shift_recovers_source_distribution(self):
        """Target probabilities generated under the source mix give the
        source mix back."""
        rng = np.random.default_rng(3)
        source_mix = np.array([0.9, 0.1])
        draw = self._gaussian_sim(rng, source_mix)
        cal_proba, cal_true = draw(10000, source_mix)
        y_proba_target, _ = draw(10000, source_mix)
        estimate = em(cal_true, y_proba_target)
        np.testing.assert_allclose(estimate, source_mix, atol=0.05)

    def test_simulated_estimation_accuracy(self):
        """On sampled data the estimate lands close to the true mix."""
        rng = np.random.default_rng(4)
        source_mix = np.array([0.9, 0.1])
        target_mix = np.array([0.7, 0.3])
        draw = self._gaussian_sim(rng, source_mix)
        cal_proba, cal_true = draw(10000, source_mix)
        y_proba_target, _ = draw(20000, target_mix)
        estimate = em(cal_true, y_proba_target)
        np.testing.assert_allclose(estimate, target_mix, atol=0.05)

    def test_multiclass_exact_recovery(self):
        """Exact recovery in a 3-class problem with a perfect classifier
        and target counts [2, 3, 5]."""
        y_true_source = np.repeat([0, 1, 2], 100)
        y_proba_target = np.zeros((10, 3))
        y_proba_target[np.arange(10), [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]] = 1.0
        estimate = em(y_true_source, y_proba_target)
        np.testing.assert_allclose(estimate, [0.2, 0.3, 0.5], atol=1e-10)

    def test_multiclass_estimate_is_a_distribution(self):
        """Multiclass simulation with calibrated Bayes posteriors: the
        estimate is a distribution and lands close to the true mix.

        Class-conditional features are N(0, 1), N(1.5, 1), N(3.0, 1)
        with source mix [0.5, 0.3, 0.2] and target mix [0.2, 0.3, 0.5].
        """
        rng = np.random.default_rng(5)
        source_mix = np.array([0.5, 0.3, 0.2])
        target_mix = np.array([0.2, 0.3, 0.5])
        means = np.array([0.0, 1.5, 3.0])

        def draw(n, mix):
            y = rng.choice(3, size=n, p=mix)
            x = rng.normal(loc=means[y], size=n)
            log_numerators = (
                np.log(source_mix)[None, :] - 0.5 * (x[:, None] - means) ** 2
            )
            shifted = log_numerators - log_numerators.max(axis=1, keepdims=True)
            proba = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            return proba, y

        cal_proba, cal_true = draw(5000, source_mix)
        y_proba_target, _ = draw(5000, target_mix)
        estimate = em(cal_true, y_proba_target)
        self.assertEqual(estimate.shape, (3,))
        self.assertTrue(np.all(estimate >= 0))
        self.assertAlmostEqual(estimate.sum(), 1.0)
        np.testing.assert_allclose(estimate, target_mix, atol=0.05)

    def test_deterministic(self):
        """Repeated calls on the same inputs give the same estimate."""
        rng = np.random.default_rng(6)
        y_true_source = rng.choice(2, size=2000, p=[0.9, 0.1])
        y_proba_target = rng.dirichlet([2.0, 1.0], size=2000)
        first = em(y_true_source, y_proba_target)
        second = em(y_true_source, y_proba_target)
        np.testing.assert_array_equal(first, second)

    def test_agrees_with_bbse(self):
        """On the same simulated shift, the EM and BBSE estimates both
        land close to the true mix and close to each other. EM sees the
        full predicted probabilities; BBSE only the argmax labels."""
        rng = np.random.default_rng(8)
        source_mix = np.array([0.9, 0.1])
        target_mix = np.array([0.7, 0.3])
        draw = self._gaussian_sim(rng, source_mix)
        cal_proba, cal_true = draw(10000, source_mix)
        target_proba, _ = draw(20000, target_mix)

        em_estimate = em(cal_true, target_proba)
        bbse_estimate = bbse(
            cal_true,
            cal_proba.argmax(axis=1),
            target_proba.argmax(axis=1),
        )
        np.testing.assert_allclose(em_estimate, target_mix, atol=0.05)
        np.testing.assert_allclose(bbse_estimate, target_mix, atol=0.05)
        np.testing.assert_allclose(em_estimate, bbse_estimate, atol=0.05)

    def test_validation(self):
        y_true_source = np.array([0, 1, 0, 1])
        good_proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
        # One-dimensional probabilities.
        with self.assertRaises(ValueError):
            em(y_true_source, np.array([0.7, 0.3, 0.6]))
        # Empty target.
        with self.assertRaises(ValueError):
            em(y_true_source, np.empty((0, 2)))
        # Negative probability.
        with self.assertRaises(ValueError):
            em(y_true_source, np.array([[-0.1, 1.1], [0.2, 0.8]]))
        # Nonfinite probability.
        with self.assertRaises(ValueError):
            em(y_true_source, np.array([[np.nan, 1.0], [0.2, 0.8]]))
        with self.assertRaises(ValueError):
            em(y_true_source, np.array([[np.inf, 0.0], [0.2, 0.8]]))
        # Rows not summing to 1.
        with self.assertRaises(ValueError):
            em(y_true_source, np.array([[0.5, 0.3], [0.2, 0.8]]))
        # Probability width mismatching num_classes.
        with self.assertRaises(ValueError):
            em(y_true_source, good_proba, num_classes=3)
        # Source label outside the declared number of classes.
        with self.assertRaises(ValueError):
            em(
                np.array([0, 1, 2]),
                np.array([[0.7, 0.3], [0.2, 0.8]]),
                num_classes=2,
            )
        # A class missing from the source labels.
        with self.assertRaises(ValueError):
            em(
                np.zeros(4, dtype=int),
                np.array([[0.7, 0.3, 0.0], [0.2, 0.3, 0.5]]),
                num_classes=3,
            )
        # num_classes must be a positive integer.
        with self.assertRaises(ValueError):
            em(y_true_source, good_proba, num_classes=2.5)
        with self.assertRaises(ValueError):
            em(y_true_source, good_proba, num_classes=0)
        # String source labels.
        with self.assertRaises(ValueError):
            em(np.array(["a", "b"]), good_proba)
        # max_iter must be a positive integer.
        with self.assertRaises(ValueError):
            em(y_true_source, good_proba, max_iter=0)
        with self.assertRaises(ValueError):
            em(y_true_source, good_proba, max_iter=1.5)
        # tol must be non-negative.
        with self.assertRaises(ValueError):
            em(y_true_source, good_proba, tol=-1e-6)
        with self.assertRaises(ValueError):
            em(y_true_source, good_proba, tol=np.inf)

    def test_non_convergence_warns(self):
        """A single EM iteration cannot have converged, so the
        non-convergence warning fires. The partial estimate is still
        returned as a valid distribution."""
        rng = np.random.default_rng(10)
        source_mix = np.array([0.9, 0.1])
        draw = self._gaussian_sim(rng, source_mix)
        cal_proba, cal_true = draw(2000, source_mix)
        target_proba, _ = draw(2000, np.array([0.7, 0.3]))
        with self.assertWarns(UserWarning):
            estimate = em(cal_true, target_proba, max_iter=1)
        self.assertEqual(estimate.shape, (2,))
        self.assertTrue(np.all(estimate >= 0))
        self.assertAlmostEqual(estimate.sum(), 1.0)

    def test_em_mix_restores_coverage(self):
        """End to end: thresholds weighted with the EM-estimated mix
        cover at close to the nominal rate under label shift, well above
        the unweighted threshold.

        Same simulation setup as the BBSE coverage test: class 1 is
        harder and grows from 10% of the source mix to 30% of the target
        mix, with a genuinely imperfect classifier. EM only sees
        predicted probabilities, never target labels.
        """
        rng = np.random.default_rng(9)
        alpha = 0.1
        source_mix = np.array([0.9, 0.1])
        target_mix = np.array([0.7, 0.3])
        draw = self._gaussian_sim(rng, source_mix)

        cal_prob, cal_true = draw(6000, source_mix)
        cal_scores = 1 - cal_prob[np.arange(len(cal_true)), cal_true]
        test_prob, test_true = draw(30000, target_mix)
        test_scores = 1 - test_prob

        estimated_mix = em(cal_true, test_prob)
        np.testing.assert_allclose(estimated_mix, target_mix, atol=0.05)

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


class TestEMLabelShiftInterface(unittest.TestCase):
    """The em estimate plugs into the public LabelShift interface."""

    def setUp(self):
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": [1.0, 2.0, 3.5, 4.0],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": [5.0, 2.0, 3.5, 4.0],
                "label": 1,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "conditions": ["cond-10", "cond-20", "cond-30"],
                "procedures": [2.0, 3.0, 4.5, 5.0],
                "label": 2,
            },
            {
                "patient_id": "patient-3",
                "visit_id": "visit-3",
                "conditions": ["cond-40", "cond-50"],
                "procedures": [1.5, 2.5, 3.0, 4.5],
                "label": 0,
            },
            {
                "patient_id": "patient-4",
                "visit_id": "visit-4",
                "conditions": ["cond-60", "cond-70", "cond-80"],
                "procedures": [3.0, 4.0, 5.0, 6.0],
                "label": 1,
            },
            {
                "patient_id": "patient-5",
                "visit_id": "visit-5",
                "conditions": ["cond-90", "cond-100"],
                "procedures": [2.5, 3.5, 4.0, 5.5],
                "label": 2,
            },
        ]
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"conditions": "sequence", "procedures": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test",
        )
        self.model = MLP(
            dataset=self.dataset,
            feature_keys=["conditions", "procedures"],
            label_key="label",
            mode="multiclass",
        )
        self.model.eval()

    def test_estimate_feeds_label_shift_calibrate(self):
        # A perfect classifier on 10 target examples with class counts
        # [2, 3, 5] gives the exact mix [0.2, 0.3, 0.5].
        y_true_source = np.repeat([0, 1, 2], 100)
        y_proba_target = np.zeros((10, 3))
        y_proba_target[
            np.arange(10), [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
        ] = 1.0
        estimate = em(y_true_source, y_proba_target)
        np.testing.assert_allclose(estimate, [0.2, 0.3, 0.5], atol=1e-10)

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
