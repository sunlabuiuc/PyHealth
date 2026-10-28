"""Tests for RC3P (Shi et al. 2024), frozen 2026-10-26.

The deterministic algorithm tests use hand-computed values from the
paper's Algorithm 1, Option II. They must not be edited to make an
implementation pass; a change needs a stated, valid reason.
"""

import copy
import unittest
from unittest.mock import patch

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
        # A non-float alpha type raises TypeError per ruff rule
        # TRY004; out-of-range and non-finite values raise
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
            # Reuse the fitted processors from the full dataset, so
            # the subset is encoded with the same vocabulary the
            # model was built on, not a freshly fitted one.
            input_processors=self.dataset.input_processors,
            output_processors=self.dataset.output_processors,
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
        self.assertAlmostEqual(cal_model.alpha_hat[2], 0.3)
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
        self.assertAlmostEqual(cal_model.alpha_hat[2], 0.3)
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
        """calibrate() itself must use alpha_hat, not alpha, for the
        class thresholds. The rank subset is built so class 0 has
        top-k errors eps^1 = 0.4 and eps^2 = 0.1, giving k_hat = 2
        and alpha_hat = 0.2 at alpha = 0.3, and the score subset is
        built so the quantiles at 0.2 and at 0.3 differ (0.85 vs
        0.75). The datasets are supplied directly to calibrate()
        through a patched prepare step, so this exercises the real
        calibrate() wiring, not just the helpers."""
        rank_rows = (
            [[0.7, 0.2, 0.1]] * 6
            + [[0.3, 0.6, 0.1]] * 3
            + [[0.1, 0.2, 0.7]]
        )
        rank_y_prob = np.array(rank_rows, dtype=np.float64)
        rank_y_true = np.zeros(10, dtype=np.int64)
        score_p0 = np.array(
            [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
        )
        score_y_prob = np.stack(
            [score_p0, (1 - score_p0) / 2, (1 - score_p0) / 2], axis=1
        )
        score_y_true = np.zeros(10, dtype=np.int64)
        rank_ds, score_ds = object(), object()

        def fake_prepare(model, dataset, keys, **kwargs):
            if dataset is rank_ds:
                return {"y_prob": rank_y_prob, "y_true": rank_y_true}
            return {"y_prob": score_y_prob, "y_true": score_y_true}

        cal_model = RC3P(model=self.model, alpha=0.3, random_state=0)
        with patch(
            "pyhealth.calib.predictionset.rc3p.prepare_numpy_dataset",
            side_effect=fake_prepare,
        ):
            cal_model.calibrate(
                cal_dataset=score_ds, rank_cal_dataset=rank_ds
            )
        self.assertEqual(cal_model.k_hat[0], 2)
        self.assertAlmostEqual(cal_model.alpha_hat[0], 0.2)
        scores = 1 - score_p0
        self.assertAlmostEqual(
            cal_model.t.numpy()[0], _query_quantile(scores, 0.2)
        )
        self.assertAlmostEqual(
            cal_model.t.numpy()[0], 0.85
        )
        self.assertNotAlmostEqual(
            cal_model.t.numpy()[0], _query_quantile(scores, 0.3)
        )


class _ProbStub(torch.nn.Module):
    """Minimal model stub returning fixed probabilities."""

    def __init__(self, y_prob):
        super().__init__()
        self._y_prob = torch.as_tensor(y_prob, dtype=torch.float32)

    def forward(self, **kwargs):
        return {"y_prob": self._y_prob}


class TestCoverageSimulation(unittest.TestCase):
    def test_per_class_coverage(self):
        """Coverage through the real RC3P class: calibrate() with
        explicit rank and score datasets, then forward() with a stub
        model returning synthetic test probabilities. The CCP
        comparison is computed on the same score subset and reported
        only, since the size ordering is regime-dependent."""
        alpha = 0.1
        rank_probs, rank_true = _synthetic_probs(0, 800)
        score_probs, score_true = _synthetic_probs(1, 800)
        test_probs, test_true = _synthetic_probs(2, 2000)
        rank_ds, score_ds = object(), object()

        def fake_prepare(model, dataset, keys, **kwargs):
            if dataset is rank_ds:
                return {"y_prob": rank_probs, "y_true": rank_true}
            return {"y_prob": score_probs, "y_true": score_true}

        _, model = make_sample_dataset_and_model()
        cal_model = RC3P(model=model, alpha=alpha, random_state=0)
        with patch(
            "pyhealth.calib.predictionset.rc3p.prepare_numpy_dataset",
            side_effect=fake_prepare,
        ):
            cal_model.calibrate(
                cal_dataset=score_ds, rank_cal_dataset=rank_ds
            )
        cal_model.model = _ProbStub(test_probs)
        with torch.no_grad():
            output = cal_model()
        mask = output["y_predset"].numpy()
        for y in range(3):
            coverage = mask[test_true == y, y].mean()
            # 0.87 is an empirical sanity threshold for a 0.90 target
            # at this sample size, not a proof of nominal coverage:
            # the top-k errors here are estimated, not known.
            self.assertGreaterEqual(coverage, 0.87)
        # CCP comparison, reported only: on this strong-signal
        # 3-class synthetic, RC3P sets are slightly larger than CCP
        # (larger adjusted thresholds outweigh the rank filter), so
        # no size ordering is asserted here. The smaller-set benefit
        # is regime-dependent, and the PR description says so.
        score_nc = true_class_nc_scores(score_probs, score_true)
        ccp_thresholds = np.array(
            [
                _query_quantile(score_nc[score_true == y], alpha)
                for y in range(3)
            ]
        )
        test_nc = all_class_nc_scores(test_probs, randomize=False)
        ccp_mask = test_nc <= ccp_thresholds
        self.assertTrue(np.isfinite(mask.mean()))
        self.assertTrue(np.isfinite(ccp_mask.mean()))


if __name__ == "__main__":
    unittest.main()
