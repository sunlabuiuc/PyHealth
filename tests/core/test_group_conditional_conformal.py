"""Tests for GroupConditionalConformal (Lu et al., AAAI 2022).

Contributor: Ibrahim Rabiu (GitHub: @alhibb)
"""

import unittest

import numpy as np
import torch

from pyhealth.calib.predictionset.group_conditional import GroupConditionalConformal


class DummyModel(torch.nn.Module):
    """Minimal dummy model returning fixed probabilities and logits."""

    def __init__(self, mode: str = "multiclass"):
        super().__init__()
        self.mode = mode

    def forward(self, **kwargs):
        if "y_prob" in kwargs:
            probs = kwargs["y_prob"]
        elif "x" in kwargs:
            probs = kwargs["x"]
        else:
            probs = torch.tensor([[0.7, 0.2, 0.1]])
        return {"logit": probs, "y_prob": probs}


class TestGroupConditionalConformal(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.rng = np.random.default_rng(42)

    def _generate_synthetic_data(self, n_samples=200, n_classes=3):
        # Generate synthetic probabilities with higher confidence for group 0 than group 1
        logits = self.rng.normal(0.0, 1.0, size=(n_samples, n_classes))
        y_true = self.rng.integers(0, n_classes, size=n_samples)
        groups = self.rng.choice(["male", "female"], size=n_samples)

        # Boost true class
        for i in range(n_samples):
            boost = 3.0 if groups[i] == "male" else 1.2
            logits[i, y_true[i]] += boost

        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs, y_true, groups

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            GroupConditionalConformal(self.model, alpha=1.5)
        with self.assertRaises(ValueError):
            GroupConditionalConformal(self.model, alpha=-0.1)
        with self.assertRaises(ValueError):
            GroupConditionalConformal(self.model, alpha={"male": 0.1, "female": 1.2})
        with self.assertRaises(ValueError):
            GroupConditionalConformal(self.model, alpha={})
        with self.assertRaises(TypeError):
            GroupConditionalConformal(self.model, alpha="invalid")

    def test_invalid_score_type(self):
        with self.assertRaises(ValueError):
            GroupConditionalConformal(self.model, score_type="unknown_score")

    def test_uncalibrated_forward_raises(self):
        conf = GroupConditionalConformal(self.model)
        with self.assertRaises(RuntimeError):
            conf(y_prob=torch.tensor([[0.7, 0.2, 0.1]]), subgroup=["male"])

    def test_calibrate_mismatched_lengths_raises(self):
        conf = GroupConditionalConformal(self.model)
        probs = np.zeros((10, 3))
        labels = np.zeros(10)
        groups = np.zeros(9)
        with self.assertRaises(ValueError):
            conf.calibrate(scores=probs, labels=labels, groups=groups)

    def test_calibration_and_group_thresholds(self):
        probs, y_true, groups = self._generate_synthetic_data(n_samples=300)
        conf = GroupConditionalConformal(
            self.model, alpha={"male": 0.1, "female": 0.2}, score_type="threshold"
        )
        conf.calibrate(scores=probs, labels=y_true, groups=groups)

        self.assertTrue(conf.calibrated)
        self.assertIn("male", conf.q_hat)
        self.assertIn("female", conf.q_hat)
        self.assertIsInstance(conf.q_hat["male"], float)
        self.assertIsInstance(conf.q_hat["female"], float)

        # Monotonicity on the same group with different alphas:
        # Lower alpha (higher coverage demand, e.g. 0.05) produces higher threshold than alpha 0.25
        conf_strict = GroupConditionalConformal(self.model, alpha=0.05)
        conf_loose = GroupConditionalConformal(self.model, alpha=0.25)
        conf_strict.calibrate(scores=probs, labels=y_true, groups=groups)
        conf_loose.calibrate(scores=probs, labels=y_true, groups=groups)
        for g in ["male", "female"]:
            self.assertGreaterEqual(conf_strict.q_hat[g], conf_loose.q_hat[g])

    def test_forward_output_shapes_and_types(self):
        probs, y_true, groups = self._generate_synthetic_data(n_samples=50)
        conf = GroupConditionalConformal(self.model, alpha=0.1)
        conf.calibrate(scores=probs, labels=y_true, groups=groups)

        test_probs = torch.tensor(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=torch.float32
        )
        test_groups = ["male", "female"]
        out = conf(y_prob=test_probs, subgroup=test_groups)

        self.assertIn("y_predset", out)
        self.assertIn("y_prob", out)
        self.assertEqual(out["y_predset"].shape, (2, 3))
        self.assertEqual(out["y_predset"].dtype, torch.bool)

    def test_group_conditional_coverage_guarantee(self):
        # Empirical test: on a large synthetic calibration set, coverage within each group >= 1 - alpha - eps
        cal_probs, cal_labels, cal_groups = self._generate_synthetic_data(n_samples=600)
        test_probs, test_labels, test_groups = self._generate_synthetic_data(
            n_samples=600
        )

        alpha = 0.1
        conf = GroupConditionalConformal(self.model, alpha=alpha, random_state=42)
        conf.calibrate(scores=cal_probs, labels=cal_labels, groups=cal_groups)

        out = conf(y_prob=torch.from_numpy(test_probs), subgroup=test_groups)
        predsets = out["y_predset"].numpy()

        for g in ["male", "female"]:
            idx = np.where(test_groups == g)[0]
            group_labels = test_labels[idx]
            group_predsets = predsets[idx]

            # Covered if true class is in prediction set
            covered = np.array(
                [group_predsets[i, group_labels[i]] for i in range(len(idx))]
            )
            empirical_coverage = np.mean(covered)
            # Coverage should approximate 1 - alpha = 0.90 within reasonable statistical sampling margin
            self.assertGreaterEqual(
                empirical_coverage,
                0.83,
                f"Group {g} under-covered: {empirical_coverage:.3f} < 0.83",
            )

    def test_aps_score_type(self):
        probs, y_true, groups = self._generate_synthetic_data(n_samples=100)
        conf = GroupConditionalConformal(
            self.model, alpha=0.15, score_type="aps", random_state=123
        )
        conf.calibrate(scores=probs, labels=y_true, groups=groups)

        out = conf(
            y_prob=torch.from_numpy(probs[:5]),
            subgroup=groups[:5],
        )
        self.assertEqual(out["y_predset"].shape, (5, 3))
        self.assertEqual(out["y_predset"].dtype, torch.bool)

    def test_fallback_when_group_key_absent_in_forward(self):
        probs, y_true, groups = self._generate_synthetic_data(n_samples=50)
        conf = GroupConditionalConformal(self.model, alpha=0.1)
        conf.calibrate(scores=probs, labels=y_true, groups=groups)

        # Call forward without providing subgroup in kwargs
        out = conf(y_prob=torch.from_numpy(probs[:4]))
        self.assertEqual(out["y_predset"].shape, (4, 3))

    def test_fallback_for_unseen_group(self):
        probs, y_true, groups = self._generate_synthetic_data(n_samples=50)
        conf = GroupConditionalConformal(self.model, alpha=0.1)
        conf.calibrate(scores=probs, labels=y_true, groups=groups)

        # Call forward with a new, unseen group
        out = conf(
            y_prob=torch.tensor([[0.5, 0.3, 0.2]]),
            subgroup=["unknown_group"],
        )
        self.assertEqual(out["y_predset"].shape, (1, 3))

    def test_set_size_monotonicity_with_alpha(self):
        probs, y_true, groups = self._generate_synthetic_data(n_samples=100)
        conf_strict = GroupConditionalConformal(self.model, alpha=0.05)
        conf_loose = GroupConditionalConformal(self.model, alpha=0.30)

        conf_strict.calibrate(scores=probs, labels=y_true, groups=groups)
        conf_loose.calibrate(scores=probs, labels=y_true, groups=groups)

        test_p = torch.from_numpy(probs[:20])
        test_g = groups[:20]

        sets_strict = conf_strict(y_prob=test_p, subgroup=test_g)["y_predset"]
        sets_loose = conf_loose(y_prob=test_p, subgroup=test_g)["y_predset"]

        # Strict alpha (higher confidence 95%) must produce sets >= loose alpha (70%)
        self.assertGreaterEqual(
            sets_strict.sum().item(),
            sets_loose.sum().item(),
        )


if __name__ == "__main__":
    unittest.main()

