"""Tests for BaseConformal, standard split conformal prediction for
multiclass classification (Vovk, Gammerman, and Shafer 2005).

BaseConformal's default score_type ("threshold") is Sadinle, Lei, and
Wasserman's (2019) LAC score, matching every other score-then-quantile
class in this package (LABEL, ClusterLabel, CovariateLabel,
NeighborhoodLabel) for consistency. BaseConformal additionally supports
score_type="margin", the nonconformity measure Papadopoulos, Vovk, and
Gammerman defined for neural-network classifiers ("Conformal prediction
with neural networks," 19th IEEE ICTAI 2007, vol. 2, pp. 388-395) for
anyone who specifically wants that paper's own method. These tests verify
both score types work correctly, that coverage validity holds for both
(as it must for any nonconformity score), and that "margin" is a real,
distinct option rather than a no-op alias for "threshold".
"""

import unittest

import numpy as np
import torch

from pyhealth.calib.predictionset.base_conformal import BaseConformal
from pyhealth.calib.predictionset.label import LABEL
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP


class TestBaseConformal(unittest.TestCase):
    """Test cases for the BaseConformal prediction set constructor."""

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

        # 3-class multiclass task, split into a 6-sample train set (indices
        # 0-5) and a 12-sample calibration set (indices 6-17) so quantile
        # thresholds are well-defined even at small alpha.
        self.samples = [
            {
                "patient_id": f"patient-{i}",
                "visit_id": f"visit-{i}",
                "conditions": [f"cond-{i}", f"cond-{i+1}", f"cond-{i+2}"],
                "procedures": [float(i), float(i + 1), float(i + 2), float(i + 3)],
                "label": i % 3,
            }
            for i in range(18)
        ]

        self.input_schema = {"conditions": "sequence", "procedures": "tensor"}
        self.output_schema = {"label": "multiclass"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = MLP(
            dataset=self.dataset,
            feature_keys=["conditions", "procedures"],
            label_key="label",
            mode="multiclass",
        )
        self.model.eval()

        self.train_indices = list(range(6))
        self.cal_indices = list(range(6, 18))
        self.cal_dataset = self.dataset.subset(self.cal_indices)

    # -- initialization --------------------------------------------------

    def test_default_score_type_is_threshold(self):
        """BaseConformal's default must be "threshold" (Sadinle, Lei, and
        Wasserman 2019's LAC score), matching every other score-then-
        quantile class in this package (LABEL, ClusterLabel,
        CovariateLabel, NeighborhoodLabel) for consistency."""
        base_model = BaseConformal(model=self.model, alpha=0.1)
        self.assertEqual(base_model.score_type, "threshold")

    def test_initialization_with_array_alpha(self):
        alpha_per_class = [0.1, 0.15, 0.2]
        base_model = BaseConformal(model=self.model, alpha=alpha_per_class)
        self.assertIsInstance(base_model.alpha, np.ndarray)
        np.testing.assert_array_equal(base_model.alpha, alpha_per_class)

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
            BaseConformal(model=binary_model, alpha=0.1)

    def test_invalid_score_type_raises(self):
        with self.assertRaises(ValueError):
            BaseConformal(model=self.model, alpha=0.1, score_type="not_a_score")

    # -- calibration / forward, per score_type ----------------------------

    def test_calibrate_and_forward_marginal_default_score(self):
        base_model = BaseConformal(model=self.model, alpha=0.3)
        base_model.calibrate(cal_dataset=self.cal_dataset)
        self.assertIsNotNone(base_model.t)
        # Marginal coverage -> a single scalar threshold.
        self.assertEqual(base_model.t.numel(), 1)

        test_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(test_loader))
        with torch.no_grad():
            output = base_model(**data_batch)
        self.assertIn("y_predset", output)
        self.assertEqual(output["y_predset"].dtype, torch.bool)
        self.assertEqual(output["y_predset"].shape, output["y_prob"].shape)

    def test_calibrate_class_conditional(self):
        alpha_per_class = [0.3, 0.35, 0.3]
        base_model = BaseConformal(model=self.model, alpha=alpha_per_class)
        base_model.calibrate(cal_dataset=self.cal_dataset)
        self.assertEqual(base_model.t.numel(), 3)

    def test_score_type_threshold_runs_end_to_end(self):
        base_model = BaseConformal(model=self.model, alpha=0.3, score_type="threshold")
        base_model.calibrate(cal_dataset=self.cal_dataset)
        test_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        with torch.no_grad():
            for data_batch in test_loader:
                output = base_model(**data_batch)
                self.assertEqual(output["y_predset"].dtype, torch.bool)
                set_sizes = output["y_predset"].sum(dim=1)
                self.assertTrue(torch.all(set_sizes > 0))

    def test_score_type_aps_runs_end_to_end(self):
        base_model = BaseConformal(
            model=self.model, alpha=0.3, score_type="aps", random_state=42
        )
        base_model.calibrate(cal_dataset=self.cal_dataset)
        test_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        with torch.no_grad():
            for data_batch in test_loader:
                output = base_model(**data_batch)
                self.assertEqual(output["y_predset"].dtype, torch.bool)
                set_sizes = output["y_predset"].sum(dim=1)
                self.assertTrue(torch.all(set_sizes > 0))

    def test_forward_before_calibration_raises_error(self):
        base_model = BaseConformal(model=self.model, alpha=0.2)
        test_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(test_loader))
        with self.assertRaises(RuntimeError):
            with torch.no_grad():
                base_model(**data_batch)

    # -- the core claim: BaseConformal now genuinely differs from LABEL ---

    def test_default_matches_label_default(self):
        """BaseConformal's default (score_type="threshold") should exactly
        reproduce LABEL's thresholds and prediction sets on the same data,
        since both then use the identical Sadinle et al. 2019 score and
        calibration procedure -- this is intentional consistency across
        the package's CP classes, not a bug."""
        alpha = 0.3
        base_model = BaseConformal(model=self.model, alpha=alpha)
        base_model.calibrate(cal_dataset=self.cal_dataset)

        label_model = LABEL(model=self.model, alpha=alpha)
        label_model.calibrate(cal_dataset=self.cal_dataset)

        self.assertAlmostEqual(
            float(base_model.t.item()), float(label_model.t.item()), places=6
        )

        test_loader = get_dataloader(self.dataset, batch_size=18, shuffle=False)
        data_batch = next(iter(test_loader))
        with torch.no_grad():
            base_out = base_model(**data_batch)
            label_out = label_model(**data_batch)
        self.assertTrue(torch.equal(base_out["y_predset"], label_out["y_predset"]))

    def test_margin_score_type_diverges_from_default(self):
        """score_type="margin" (Papadopoulos, Vovk, and Gammerman 2007's
        own nonconformity measure) must produce a genuinely different
        threshold and prediction sets than the default "threshold" score
        on the same calibration data -- proving "margin" is a real,
        distinct, working option and not a no-op alias."""
        alpha = 0.3
        default_model = BaseConformal(model=self.model, alpha=alpha)
        default_model.calibrate(cal_dataset=self.cal_dataset)

        margin_model = BaseConformal(model=self.model, alpha=alpha, score_type="margin")
        margin_model.calibrate(cal_dataset=self.cal_dataset)

        self.assertNotAlmostEqual(
            float(default_model.t.item()),
            float(margin_model.t.item()),
            places=6,
            msg="score_type=\"margin\" produced the same threshold as the "
            "default \"threshold\" score -- expected a genuinely different "
            "nonconformity measure.",
        )

        test_loader = get_dataloader(self.dataset, batch_size=18, shuffle=False)
        data_batch = next(iter(test_loader))
        with torch.no_grad():
            default_out = default_model(**data_batch)
            margin_out = margin_model(**data_batch)

        self.assertFalse(
            torch.equal(default_out["y_predset"], margin_out["y_predset"]),
            "score_type=\"margin\" produced identical prediction sets to "
            "the default on every example.",
        )

    # -- coverage validity (holds regardless of score_type) ---------------

    def test_margin_score_achieves_approximate_marginal_coverage(self):
        """Monte Carlo check: with a larger synthetic exchangeable dataset,
        the "margin" score (available via score_type="margin") should
        achieve close to the target 1-alpha marginal coverage, per split
        conformal's validity guarantee (Vovk, Gammerman, and Shafer 2005),
        which holds for any nonconformity measure."""
        rng = np.random.default_rng(123)
        n, k = 3000, 4
        alpha = 0.1
        logits = rng.normal(size=(n, k)) * 2
        y_prob = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        y_true = np.array([rng.choice(k, p=y_prob[i]) for i in range(n)])

        from pyhealth.calib.predictionset.base_conformal import _query_quantile
        from pyhealth.calib.predictionset.scores import (
            all_class_nc_scores,
            true_class_nc_scores,
        )

        cal, test = slice(0, n // 2), slice(n // 2, n)
        nc_cal = true_class_nc_scores(y_prob[cal], y_true[cal], score_type="margin")
        t = _query_quantile(nc_cal, alpha)
        nc_test = all_class_nc_scores(y_prob[test], score_type="margin")
        predset = nc_test <= t
        coverage = predset[np.arange(n - n // 2), y_true[test]].mean()
        self.assertGreaterEqual(coverage, 1 - alpha - 0.03)


if __name__ == "__main__":
    unittest.main()
