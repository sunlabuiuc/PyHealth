"""Tests for GroupConditionalConformal (Lu et al., AAAI 2022).

Contributor: Ibrahim Rabiu (GitHub: @alhibb)
"""

import unittest
import warnings

import numpy as np
import torch

from pyhealth.calib.predictionset.group_conditional import GroupConditionalConformal
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP


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
        logits = self.rng.normal(0.0, 1.0, size=(n_samples, n_classes))
        y_true = self.rng.integers(0, n_classes, size=n_samples)
        groups = self.rng.choice(["male", "female"], size=n_samples)

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

    def test_eval_mode_enforced(self):
        train_model = DummyModel()
        train_model.train()
        self.assertTrue(train_model.training)
        conf = GroupConditionalConformal(train_model)
        self.assertFalse(conf.model.training)

    def test_non_multiclass_model_raises(self):
        binary_model = DummyModel(mode="binary")
        with self.assertRaises(NotImplementedError):
            GroupConditionalConformal(binary_model)

    def test_binary_scores_in_calibrate_raises(self):
        conf = GroupConditionalConformal(self.model)
        binary_probs = np.array([0.2, 0.8, 0.4])
        labels = np.array([0, 1, 0])
        groups = np.array(["A", "B", "A"])
        with self.assertRaises(ValueError):
            conf.calibrate(scores=binary_probs, labels=labels, groups=groups)

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
        out = conf(
            y_prob=test_probs,
            subgroup=test_groups,
            loss=torch.tensor(0.42),
            y_true=torch.tensor([0, 1]),
        )

        self.assertIn("y_predset", out)
        self.assertIn("y_prob", out)
        self.assertIn("loss", out)
        self.assertIn("y_true", out)
        self.assertEqual(out["y_predset"].shape, (2, 3))
        self.assertEqual(out["y_predset"].dtype, torch.bool)

    def test_group_conditional_coverage_guarantee(self):
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

            covered = np.array(
                [group_predsets[i, group_labels[i]] for i in range(len(idx))]
            )
            empirical_coverage = np.mean(covered)
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

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = conf(y_prob=torch.from_numpy(probs[:4]))
            self.assertEqual(out["y_predset"].shape, (4, 3))
            self.assertTrue(
                any("not found in forward kwargs" in str(w.message) for w in caught)
            )

    def test_fallback_for_unseen_group(self):
        probs, y_true, groups = self._generate_synthetic_data(n_samples=50)
        conf = GroupConditionalConformal(self.model, alpha=0.1)
        conf.calibrate(scores=probs, labels=y_true, groups=groups)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = conf(
                y_prob=torch.tensor([[0.5, 0.3, 0.2]]),
                subgroup=["unknown_group"],
            )
            self.assertEqual(out["y_predset"].shape, (1, 3))
            self.assertTrue(
                any("Unseen subgroup 'unknown_group'" in str(w.message) for w in caught)
            )

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

        self.assertGreaterEqual(
            sets_strict.sum().item(),
            sets_loose.sum().item(),
        )

    def test_calibrate_with_dataset_and_forward_loader(self):
        samples = [
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "conditions": [f"c{i % 3}"],
                "procedures": [float(i)],
                "label": i % 3,
                "subgroup": f"group_{i % 2}",
            }
            for i in range(12)
        ]
        input_schema = {"conditions": "sequence", "procedures": "tensor"}
        output_schema = {"label": "multiclass"}
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_gc",
        )
        mlp = MLP(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="label",
            mode="multiclass",
        )
        conf = GroupConditionalConformal(mlp, alpha=0.2, group_key="subgroup")
        cal_ds = dataset.subset(list(range(6, 12)))
        conf.calibrate(cal_dataset=cal_ds)

        self.assertTrue(conf.calibrated)
        self.assertIn("group_0", conf.q_hat)
        self.assertIn("group_1", conf.q_hat)

        test_loader = get_dataloader(
            dataset.subset([0, 1]), batch_size=2, shuffle=False
        )
        batch = next(iter(test_loader))
        with torch.no_grad():
            out = conf(**batch)

        self.assertEqual(out["y_predset"].shape, (2, 3))
        self.assertEqual(out["y_predset"].dtype, torch.bool)
        self.assertIn("y_prob", out)


if __name__ == "__main__":
    unittest.main()
