import unittest

import numpy as np
import torch

from pyhealth.calib.predictionset import LabelShift
from pyhealth.calib.predictionset.base_conformal import _query_quantile
from pyhealth.calib.predictionset.label_shift.label_shift import (
    _compute_label_shift_thresholds,
    _label_shift_threshold,
)
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP


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
        thresholds, ratios, source = _compute_label_shift_thresholds(
            scores, y_true, np.array([0.25, 0.75]), alpha=0.25
        )
        np.testing.assert_allclose(source, [0.5, 0.5])
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

    def test_target_equal_to_source_gives_single_threshold(self):
        """When the target distribution equals the empirical calibration
        distribution, every ratio is 1 and all per-class thresholds coincide
        with the BaseConformal threshold."""
        rng = np.random.default_rng(1)
        y_true = np.array([0, 0, 0, 1, 1, 2])
        scores = rng.random(len(y_true))
        source = np.bincount(y_true) / len(y_true)
        thresholds, ratios, _ = _compute_label_shift_thresholds(
            scores, y_true, source, alpha=0.2
        )
        np.testing.assert_allclose(ratios, np.ones(3))
        expected = _query_quantile(scores, 0.2)
        np.testing.assert_allclose(thresholds, np.full(3, expected))

    def test_class_missing_from_calibration_is_always_included(self):
        """A class with target mass but no calibration examples gets an
        infinite ratio and an infinite threshold."""
        scores = np.array([0.2, 0.4, 0.6])
        y_true = np.array([0, 0, 1])
        thresholds, ratios, source = _compute_label_shift_thresholds(
            scores, y_true, np.array([0.3, 0.3, 0.4]), alpha=0.1
        )
        self.assertEqual(source[2], 0.0)
        self.assertEqual(ratios[2], np.inf)
        self.assertEqual(thresholds[2], np.inf)

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

        # Weighted (oracle target mix): per-class thresholds.
        thresholds, _, _ = _compute_label_shift_thresholds(
            cal_scores, cal_true, target_mix, alpha
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


class TestLabelShift(unittest.TestCase):
    """End-to-end tests with a small sample dataset and an MLP."""

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
