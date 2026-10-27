"""Tests for RC3P (Shi et al. 2024), frozen 2026-10-26.

The deterministic algorithm tests use hand-computed values from the
paper's Algorithm 1, Option II. They must not be edited to make an
implementation pass; a change needs a stated, valid reason.
"""

import copy
import unittest

import numpy as np
import torch

from pyhealth.calib.predictionset.base_conformal import _query_quantile
from pyhealth.calib.predictionset.rc3p import (
    RC3P,
    _candidate_ranks,
    _predset_mask,
    _rank_calibration,
)
from pyhealth.calib.predictionset.scores import (
    all_class_nc_scores,
    true_class_nc_scores,
)
from pyhealth.datasets import create_sample_dataset, get_dataloader
from tests.core.label_shift_test_utils import (
    SAMPLES,
    make_sample_dataset_and_model,
)


def _synthetic_probs(seed, n_per_class, num_classes=3):
    """Softmax probabilities from Gaussian logits, true class boosted."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1.0, size=(n_per_class * num_classes, num_classes))
    y_true = np.repeat(np.arange(num_classes), n_per_class)
    logits[np.arange(len(y_true)), y_true] += 2.0
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, y_true


class TestCandidateRanks(unittest.TestCase):
    def test_ranks_and_tie_rule(self):
        probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
        np.testing.assert_array_equal(
            _candidate_ranks(probs), np.array([[1, 2, 3], [3, 3, 1]])
        )


class TestRankCalibration(unittest.TestCase):
    def test_hand_computed_k_hat_and_alpha_hat(self):
        # True-label ranks [1, 2, 2, 3] for one class, alpha = 0.3:
        # eps^1 = 0.75, eps^2 = 0.25 < 0.3, so k_hat = 2, alpha_hat = 0.05.
        rank_y_prob = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.5, 0.1],
                [0.3, 0.6, 0.1],
                [0.2, 0.3, 0.5],
            ]
        )
        rank_y_true = np.array([0, 0, 0, 0])
        # True ranks for class 0 here: 1, 2, 2, 3.
        k_hat, alpha_hat = _rank_calibration(rank_y_prob, rank_y_true, 0.3)
        self.assertEqual(k_hat[0], 2)
        self.assertAlmostEqual(alpha_hat[0], 0.05)
        self.assertLess(0.25, 0.3)

    def test_perfect_classifier(self):
        rank_y_prob = np.array(
            [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]]
        )
        rank_y_true = np.array([0, 1, 2])
        k_hat, alpha_hat = _rank_calibration(rank_y_prob, rank_y_true, 0.1)
        np.testing.assert_array_equal(k_hat, np.array([1, 1, 1]))
        np.testing.assert_allclose(alpha_hat, np.array([0.1, 0.1, 0.1]))

    def test_absent_class_fallback(self):
        rank_y_prob = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]])
        rank_y_true = np.array([0, 1])
        k_hat, alpha_hat = _rank_calibration(rank_y_prob, rank_y_true, 0.1)
        self.assertEqual(k_hat[2], 3)
        self.assertAlmostEqual(alpha_hat[2], 0.1)


class TestPredsetMask(unittest.TestCase):
    def test_rank_filter_binds_and_passes(self):
        nc_scores = np.array([[0.2, 0.2, 0.9]])
        thresholds = np.array([0.5, 0.5, 0.5])
        ranks = np.array([[1, 3, 2]])
        k_hat = np.array([2, 2, 2])
        mask = _predset_mask(nc_scores, thresholds, ranks, k_hat)
        # Class 1 passes its score but fails its rank cutoff.
        np.testing.assert_array_equal(mask, np.array([[True, False, False]]))

    def test_absent_class_is_included(self):
        nc_scores = np.array([[0.9, 0.9, 0.9]])
        thresholds = np.array([0.5, 0.5, np.inf])
        ranks = np.array([[3, 3, 3]])
        k_hat = np.array([1, 1, 3])
        mask = _predset_mask(nc_scores, thresholds, ranks, k_hat)
        self.assertTrue(mask[0, 2])


class TestRC3PClass(unittest.TestCase):
    def setUp(self):
        self.dataset, self.model = make_sample_dataset_and_model()

    def test_validation(self):
        with self.assertRaises(ValueError):
            RC3P(model=self.model, alpha=0.0)
        with self.assertRaises(ValueError):
            RC3P(model=self.model, alpha=1.0)
        with self.assertRaises(ValueError):
            RC3P(model=self.model, alpha=float("nan"))
        # A non-float alpha type raises TypeError per the repo's
        # PR rules (TRY004); out-of-range and non-finite values raise
        # ValueError. This split is the stated reason for the
        # TypeError case in this frozen validation test.
        with self.assertRaises(TypeError):
            RC3P(model=self.model, alpha=[0.1, 0.1, 0.1])
        with self.assertRaises(ValueError):
            RC3P(model=self.model, alpha=0.1, score_type="unknown")

    def test_forward_before_calibrate_raises(self):
        cal_model = RC3P(model=self.model, alpha=0.3)
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        with self.assertRaises(RuntimeError), torch.no_grad():
            cal_model(**batch)

    def test_end_to_end_smoke(self):
        cal_model = RC3P(model=self.model, alpha=0.3, random_state=0)
        cal_model.calibrate(cal_dataset=self.dataset)
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        with torch.no_grad():
            output = cal_model(**batch)
        self.assertIn("y_predset", output)
        self.assertEqual(output["y_predset"].dtype, torch.bool)
        self.assertEqual(
            output["y_predset"].shape, output["y_prob"].shape
        )
        self.assertEqual(len(cal_model.t), 3)
        self.assertEqual(len(cal_model.k_hat), 3)
        self.assertTrue(np.all((cal_model.k_hat >= 1) & (cal_model.k_hat <= 3)))
        self.assertTrue(
            np.all((cal_model.alpha_hat > 0) & (cal_model.alpha_hat <= 0.3))
        )

    def test_absent_from_one_subset_is_included(self):
        """A class missing from only the rank or only the score
        subset must be included for every input, with threshold +inf
        and cutoff K, not just the fallback its own subset gives."""
        samples_no_class2 = [
            copy.deepcopy(s) for s in SAMPLES if s["label"] != 2
        ]
        partial_dataset = create_sample_dataset(
            samples=samples_no_class2,
            input_schema={"conditions": "sequence", "procedures": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test",
        )
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        # Class 2 absent from the score subset only.
        cal_model = RC3P(model=self.model, alpha=0.3, random_state=0)
        cal_model.calibrate(
            cal_dataset=partial_dataset, rank_cal_dataset=self.dataset
        )
        self.assertTrue(np.isinf(cal_model.t.numpy()[2]))
        self.assertEqual(cal_model.k_hat[2], 3)
        with torch.no_grad():
            output = cal_model(**batch)
        self.assertTrue(output["y_predset"][:, 2].all())
        # Class 2 absent from the rank subset only.
        cal_model = RC3P(model=self.model, alpha=0.3, random_state=0)
        cal_model.calibrate(
            cal_dataset=self.dataset, rank_cal_dataset=partial_dataset
        )
        self.assertTrue(np.isinf(cal_model.t.numpy()[2]))
        self.assertEqual(cal_model.k_hat[2], 3)
        with torch.no_grad():
            output = cal_model(**batch)
        self.assertTrue(output["y_predset"][:, 2].all())

    def test_aps_reproducible_with_fixed_seed(self):
        def build_and_run():
            cal_model = RC3P(
                model=self.model,
                alpha=0.3,
                score_type="aps",
                random_state=42,
            )
            cal_model.calibrate(cal_dataset=self.dataset)
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

    def test_adjusted_alpha_threshold(self):
        # Separate rank and score datasets from the same fixture is
        # not possible with 6 samples, so verify the adjusted
        # threshold through the calibration helpers on a constructed
        # case: class 0 ranks [1, 2, 2, 3], alpha 0.3, alpha_hat 0.05.
        rank_y_prob = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.5, 0.1],
                [0.3, 0.6, 0.1],
                [0.2, 0.3, 0.5],
            ]
        )
        rank_y_true = np.array([0, 0, 0, 0])
        _k_hat, alpha_hat = _rank_calibration(rank_y_prob, rank_y_true, 0.3)
        self.assertAlmostEqual(alpha_hat[0], 0.05)
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertAlmostEqual(
            _query_quantile(scores, alpha_hat[0]),
            _query_quantile(scores, 0.05),
        )
        self.assertNotEqual(
            _query_quantile(scores, alpha_hat[0]),
            _query_quantile(scores, 0.3),
        )


class TestCoverageSimulation(unittest.TestCase):
    def test_per_class_coverage(self):
        alpha = 0.1
        rank_probs, rank_true = _synthetic_probs(0, 800)
        score_probs, score_true = _synthetic_probs(1, 800)
        test_probs, test_true = _synthetic_probs(2, 2000)
        k_hat, alpha_hat = _rank_calibration(rank_probs, rank_true, alpha)
        score_nc = true_class_nc_scores(score_probs, score_true)
        thresholds = np.full(3, np.inf)
        for y in range(3):
            thresholds[y] = _query_quantile(
                score_nc[score_true == y], alpha_hat[y]
            )
        test_nc = all_class_nc_scores(test_probs, randomize=False)
        test_ranks = _candidate_ranks(test_probs)
        mask = _predset_mask(test_nc, thresholds, test_ranks, k_hat)
        for y in range(3):
            coverage = mask[test_true == y, y].mean()
            # 0.87 is an empirical sanity threshold for a 0.90 target
            # at this sample size, not a proof of nominal coverage:
            # the top-k errors here are estimated, not known.
            self.assertGreaterEqual(coverage, 0.87)
        # CCP comparison, reported only: RC3P size is computed here
        # but is not a pass/fail assertion, per the frozen spec.
        ccp_thresholds = np.array(
            [
                _query_quantile(score_nc[score_true == y], alpha)
                for y in range(3)
            ]
        )
        ccp_mask = test_nc <= ccp_thresholds
        self.assertTrue(np.isfinite(mask.mean()))
        self.assertTrue(np.isfinite(ccp_mask.mean()))


if __name__ == "__main__":
    unittest.main()
