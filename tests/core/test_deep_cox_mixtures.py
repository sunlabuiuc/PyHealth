"""Unit tests for the DeepCoxMixtures survival model."""

import unittest
import warnings

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DeepCoxMixtures


def _make_synthetic_dataset(n: int = 24, feature_dim: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, feature_dim)).astype(np.float32)
    latent = rng.integers(0, 2, size=n)
    log_hr = np.where(latent == 0, features[:, 0], -features[:, 1])
    scale = np.where(latent == 0, 8.0, 4.0)
    true_time = -np.log(rng.uniform(0.05, 1.0, size=n)) * scale / np.exp(log_hr)
    censor = rng.exponential(scale=10.0, size=n)
    observed = np.minimum(true_time, censor)
    event = (true_time <= censor).astype(int)
    samples = [
        {
            "patient_id": f"p{i}",
            "features": features[i].tolist(),
            "time": float(observed[i]),
            "event": int(event[i]),
        }
        for i in range(n)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"time": "regression", "event": "binary"},
        dataset_name="test_dcm",
    )


class TestDeepCoxMixtures(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.dataset = _make_synthetic_dataset(n=24)
        self.loader = get_dataloader(self.dataset, batch_size=8, shuffle=False)

    def test_forward_returns_expected_keys_and_shapes(self):
        model = DeepCoxMixtures(dataset=self.dataset, k=3, hidden_dims=(8,))
        batch = next(iter(self.loader))
        out = model(**batch)
        for key in ("logit", "y_prob", "gate_probs", "loss", "y_true"):
            self.assertIn(key, out)
        self.assertEqual(out["logit"].shape, (8, 3))
        self.assertEqual(out["gate_probs"].shape, (8, 3))
        self.assertEqual(out["y_prob"].shape, (8, 1))
        self.assertEqual(out["y_true"].shape, (8, 2))
        self.assertEqual(out["loss"].dim(), 0)

    def test_loss_is_finite_and_gradients_flow(self):
        model = DeepCoxMixtures(dataset=self.dataset, k=2, hidden_dims=(8,))
        batch = next(iter(self.loader))
        out = model(**batch)
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        for name, param in model.named_parameters():
            if param.requires_grad and "_dummy_param" not in name:
                self.assertIsNotNone(param.grad, f"No grad on {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"NaN/Inf grad on {name}")

    def test_log_hr_is_clamped_within_gamma(self):
        model = DeepCoxMixtures(dataset=self.dataset, k=3, hidden_dims=(8,), gamma=5.0)
        with torch.no_grad():
            model.expert_head.weight.mul_(1000.0)
            model.expert_head.bias.fill_(100.0)
        batch = next(iter(self.loader))
        out = model(**batch)
        self.assertTrue((out["logit"].abs() <= 5.0 + 1e-5).all())

    def test_fit_runs_and_populates_breslow(self):
        model = DeepCoxMixtures(dataset=self.dataset, k=2, hidden_dims=(8,))
        history = model.fit(self.loader, epochs=2, learning_rate=1e-2)
        self.assertEqual(len(history["epoch_loss"]), 2)
        self.assertTrue(all(np.isfinite(x) for x in history["epoch_loss"]))
        self.assertTrue(any(s is not None for s in model._breslow_splines))

    def test_predict_survival_curve_monotone_non_increasing(self):
        model = DeepCoxMixtures(dataset=self.dataset, k=2, hidden_dims=(8,))
        model.fit(self.loader, epochs=2, learning_rate=1e-2)
        batch = next(iter(self.loader))
        times = [0.5, 1.5, 3.0, 6.0, 10.0]
        surv = model.predict_survival_curve(batch["features"], times)
        self.assertEqual(surv.shape, (batch["features"].shape[0], len(times)))
        self.assertTrue((surv >= 0.0).all() and (surv <= 1.0).all())
        diffs = np.diff(surv, axis=1)
        self.assertTrue((diffs <= 1e-6).all(), "Survival curves must be non-increasing")

    def test_predict_latent_z_and_risk_shapes(self):
        model = DeepCoxMixtures(dataset=self.dataset, k=3, hidden_dims=(8,))
        batch = next(iter(self.loader))
        z = model.predict_latent_z(batch["features"])
        self.assertEqual(z.shape, (batch["features"].shape[0], 3))
        self.assertTrue(torch.allclose(z.sum(dim=-1), torch.ones(z.shape[0]), atol=1e-4))
        risk = model.predict_risk(batch["features"])
        self.assertEqual(risk.shape, (batch["features"].shape[0],))
        self.assertTrue(np.isfinite(risk).all())

    def test_k_equals_one_reduces_to_single_cox(self):
        model = DeepCoxMixtures(dataset=self.dataset, k=1, hidden_dims=(8,))
        model.fit(self.loader, epochs=2, learning_rate=1e-2)
        batch = next(iter(self.loader))
        z = model.predict_latent_z(batch["features"])
        self.assertTrue(torch.allclose(z, torch.ones_like(z), atol=1e-5))
        self.assertIsNotNone(model._breslow_splines[0])
        surv = model.predict_survival_curve(batch["features"], [1.0, 5.0])
        self.assertTrue((surv[:, 0] >= surv[:, 1] - 1e-6).all())

    def test_init_rejects_missing_labels(self):
        bad = create_sample_dataset(
            samples=[
                {"patient_id": "p0", "features": [1.0, 2.0], "label": 0},
                {"patient_id": "p1", "features": [0.5, 1.5], "label": 1},
            ],
            input_schema={"features": "tensor"},
            output_schema={"label": "binary"},
        )
        with self.assertRaises(ValueError):
            DeepCoxMixtures(dataset=bad)

    def test_breslow_refit_fallback_preserves_previous(self):
        model = DeepCoxMixtures(dataset=self.dataset, k=2, hidden_dims=(8,))
        model.fit(self.loader, epochs=1, learning_rate=1e-2)
        prev = list(model._breslow_splines)
        # Force gate to put virtually all mass on component 0.
        with torch.no_grad():
            model.gate_head.weight.zero_()
            model.gate_head.bias[:] = torch.tensor([10.0, -10.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.refit_breslow(self.loader)
        self.assertIsNotNone(model._breslow_splines[0])
        self.assertTrue(
            model._breslow_splines[1] is not None or prev[1] is None
        )


if __name__ == "__main__":
    unittest.main()
