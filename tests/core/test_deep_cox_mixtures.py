"""Fast synthetic tests for DeepCoxMixtures; no real data or downloads.

Contributor: <AUTHOR NAME> (NetID: liu323)
Paper: Deep Cox Mixtures for Survival Regression (Nagpal et al., MLHC 2021),
https://proceedings.mlr.press/v149/nagpal21a.html
"""

import math
import os
import tempfile
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import BaseModel, DeepCoxMixtures
from pyhealth.models.deep_cox_mixtures import (
    breslow_cumulative_hazard,
    cox_partial_log_likelihood,
)


def synthetic(n=24, dim=3, seed=0):
    """Tiny right-censored data: times in (0.1, 5.1], ~70% events."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    t = torch.rand(n, generator=g) * 5 + 0.1
    e = (torch.rand(n, generator=g) < 0.7).float()
    e[0] = 1.0
    return x, t, e


def fitted_model(k=2, hidden=(4,), seed=0):
    torch.manual_seed(seed)
    model = DeepCoxMixtures(
        None,
        input_dim=3,
        num_components=k,
        hidden_dims=hidden,
        time_grid=[0.5, 1.0, 2.0],
    )
    x, t, e = synthetic()
    model.fit_baselines(x, t, e, generator=torch.Generator().manual_seed(seed))
    return model, (x, t, e)


class TestInstantiation(unittest.TestCase):
    def test_from_sample_dataset(self):
        samples = [
            {
                "patient_id": f"p{i}",
                "features": [float(i), 1.0, -1.0, 0.5],
                "duration": [1.0 + i],
                "event": [float(i % 2)],
            }
            for i in range(4)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"features": "tensor"},
            output_schema={"duration": "tensor", "event": "tensor"},
            dataset_name="dcm_test",
        )
        model = DeepCoxMixtures(dataset, num_components=2, hidden_dims=(4,))
        self.assertIsInstance(model, BaseModel)
        self.assertEqual(model.input_dim, 4)
        self.assertIsNone(model.mode)
        self.assertEqual(model.label_processors, {})
        self.assertEqual(model.feature_keys, ["features"])
        self.assertFalse(model.is_fitted)
        batch = next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))
        out = model(**batch)
        self.assertEqual(out["log_risk"].shape, (4, 2))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_configurations(self):
        linear = DeepCoxMixtures(None, input_dim=3, num_components=1, hidden_dims=())
        self.assertEqual(len(linear.encoder), 0)
        deep = DeepCoxMixtures(
            None,
            input_dim=3,
            num_components=4,
            hidden_dims=(5, 6),
            activation="relu",
            use_bias=True,
            log_risk_bound=None,
        )
        self.assertEqual(deep.risk.out_features, 4)
        self.assertIsNotNone(deep.encoder[0].bias)
        self.assertIsNone(deep.risk.bias)

    def test_invalid_configuration(self):
        bad = [
            {"input_dim": None},
            {"input_dim": 0},
            {"input_dim": 3, "num_components": 0},
            {"input_dim": 3, "hidden_dims": (0,)},
            {"input_dim": 3, "activation": "gelu"},
            {"input_dim": 3, "log_risk_bound": -1.0},
            {"input_dim": 3, "time_grid": [-1.0]},
            {"input_dim": True},
        ]
        for kwargs in bad:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                DeepCoxMixtures(None, **kwargs)

    def test_dataset_keys_are_checked(self):
        dataset = create_sample_dataset(
            samples=[
                {
                    "patient_id": "p",
                    "features": [1.0],
                    "duration": [1.0],
                    "event": [1.0],
                }
            ],
            input_schema={"features": "tensor"},
            output_schema={"duration": "tensor", "event": "tensor"},
            dataset_name="dcm_keys",
        )
        with self.assertRaises(ValueError):
            DeepCoxMixtures(dataset, feature_key="x")
        with self.assertRaises(ValueError):
            DeepCoxMixtures(dataset, event_key="status")


class TestForward(unittest.TestCase):
    def test_shapes_probabilities_and_gradients(self):
        model, (x, t, e) = fitted_model(k=3)
        out = model(features=x[:5], duration=t[:5], event=e[:5])
        self.assertEqual(out["log_risk"].shape, (5, 3))
        self.assertEqual(out["mixture_weights"].shape, (5, 3))
        self.assertEqual(out["posterior"].shape, (5, 3))
        self.assertEqual(out["y_prob"].shape, (5, 3))
        self.assertEqual(out["y_true"].shape, (5, 2))
        self.assertEqual(out["loss"].ndim, 0)
        for key in ("mixture_weights", "posterior"):
            self.assertTrue(torch.all(out[key] >= 0))
            torch.testing.assert_close(out[key].sum(-1), torch.ones(5))
        out["loss"].backward()
        for name, param in model.named_parameters():
            if name == "_dummy_param":
                continue
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertGreater(float(param.grad.abs().sum()), 0.0, name)

    def test_singleton_batch_and_unfitted_forward(self):
        model = DeepCoxMixtures(
            None, input_dim=3, num_components=2, hidden_dims=(4,), time_grid=[1.0]
        )
        out = model(features=torch.zeros(1, 3), duration=[2.0], event=[1.0])
        self.assertNotIn("y_prob", out)  # no baselines yet
        self.assertEqual(out["mixture_weights"].shape, (1, 2))
        # Before baselines exist the E-step returns the gate probabilities.
        torch.testing.assert_close(out["posterior"], out["mixture_weights"])

    def test_survival_range_and_monotonicity(self):
        model, (x, _, _) = fitted_model(k=3)
        times = torch.linspace(0.0, 10.0, 25)
        surv = model.predict_survival(x, times)
        self.assertEqual(surv.shape, (len(x), 25))
        self.assertTrue(torch.isfinite(surv).all())
        self.assertTrue(torch.all((surv >= 0) & (surv <= 1)))
        self.assertTrue(torch.all(surv[:, 1:] <= surv[:, :-1] + 1e-7))
        torch.testing.assert_close(surv[:, 0], torch.ones(len(x)))
        # unsorted request gives the same values in the requested order
        flipped = model.predict_survival(x, times.flip(0))
        torch.testing.assert_close(flipped, surv.flip(1))

    def test_mixture_formula(self):
        model, (x, _, _) = fitted_model(k=2)
        times = torch.tensor([0.7, 3.0])
        pi = model.predict_latent(x)
        f = model.predict_log_risk(x).double()
        cumulative, _ = model._baseline_at(times)  # (T, K)
        manual = (
            pi.double().unsqueeze(1)
            * torch.exp(-cumulative.unsqueeze(0) * f.exp().unsqueeze(1))
        ).sum(-1)
        torch.testing.assert_close(
            model.predict_survival(x, times).double(), manual, atol=1e-6, rtol=1e-5
        )


class TestCoxAndBreslow(unittest.TestCase):
    def setUp(self):
        # Ties at t = 2 (two events) and a censored tie at t = 3.
        self.f = torch.tensor([0.1, 0.2, -0.3, 0.4, 0.0], dtype=torch.float64)
        self.t = torch.tensor([1.0, 2.0, 2.0, 3.0, 3.0], dtype=torch.float64)
        self.e = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float64)

    def test_partial_likelihood_breslow_ties(self):
        r = self.f.exp()
        expected = (
            (0.1 - math.log(r.sum()))
            + (0.2 - math.log(r[1:].sum()))
            + (-0.3 - math.log(r[1:].sum()))
            + (0.4 - math.log(r[3:].sum()))
        )
        got = cox_partial_log_likelihood(self.f, self.t, self.e)
        self.assertAlmostEqual(float(got), expected, places=10)
        # order of the rows does not matter
        perm = torch.tensor([4, 2, 0, 3, 1])
        self.assertAlmostEqual(
            float(cox_partial_log_likelihood(self.f[perm], self.t[perm], self.e[perm])),
            expected,
            places=10,
        )
        self.assertEqual(
            float(cox_partial_log_likelihood(self.f, self.t, torch.zeros(5))), 0.0
        )
        self.assertEqual(
            float(
                cox_partial_log_likelihood(
                    torch.zeros(0), torch.zeros(0), torch.zeros(0)
                )
            ),
            0.0,
        )

    def test_breslow_hand_computed(self):
        r = self.f.exp()
        times, hazard = breslow_cumulative_hazard(self.f, self.t, self.e)
        h1 = 1 / r.sum()
        h2 = h1 + 2 / r[1:].sum()
        h3 = h2 + 1 / r[3:].sum()
        self.assertEqual(times.tolist(), [1.0, 2.0, 3.0])
        torch.testing.assert_close(hazard, torch.stack([h1, h2, h3]))
        empty_t, empty_h = breslow_cumulative_hazard(self.f, self.t, torch.zeros(5))
        self.assertEqual(len(empty_t), 0)
        self.assertEqual(len(empty_h), 0)

    def test_single_component_is_neural_cox(self):
        torch.manual_seed(1)
        model = DeepCoxMixtures(None, input_dim=3, num_components=1, hidden_dims=(4,))
        x, t, e = synthetic()
        model.fit_baselines(x, t, e)
        out = model(features=x, duration=t, event=e)
        torch.testing.assert_close(out["mixture_weights"], torch.ones(len(x), 1))
        expected = -cox_partial_log_likelihood(out["log_risk"][:, 0], t, e)
        torch.testing.assert_close(out["loss"].double(), expected.double() / len(x))
        # baseline is the Breslow estimate with the network's hazard ratios
        times, hazard = breslow_cumulative_hazard(out["log_risk"][:, 0], t, e)
        f = model.predict_log_risk(x)[:, 0].double()
        surv = model.predict_survival(x, times).double()
        torch.testing.assert_close(
            surv,
            torch.exp(-hazard.unsqueeze(0) * f.exp().unsqueeze(1)),
            atol=1e-6,
            rtol=1e-5,
        )


class TestEStep(unittest.TestCase):
    def test_posterior_matches_manual_and_uses_censoring(self):
        model, (x, t, e) = fitted_model(k=2)
        log_gate, log_risk = model._network(x)
        cumulative, rate = model._baseline_at(t)
        f = log_risk.double()
        log_s = -cumulative * f.exp()
        log_d = f + rate.log() + log_s
        ll = torch.where(e.bool().unsqueeze(1), log_d, log_s).clamp_min(math.log(1e-10))
        manual = torch.softmax(log_gate.double() + ll, dim=-1)
        got = model._log_posterior(log_gate, log_risk, t.double(), e.double()).exp()
        torch.testing.assert_close(got, manual)
        # The same row gets a different posterior as an event or censored.
        as_event = model._log_posterior(
            log_gate[:1], log_risk[:1], t[:1].double(), torch.ones(1)
        )
        as_censored = model._log_posterior(
            log_gate[:1], log_risk[:1], t[:1].double(), torch.zeros(1)
        )
        self.assertFalse(torch.allclose(as_event, as_censored))

    def test_interpolation_and_support(self):
        model = DeepCoxMixtures(None, input_dim=1, num_components=1, hidden_dims=())
        model._set_baselines(
            [
                (
                    torch.tensor([1.0, 3.0], dtype=torch.float64),
                    torch.tensor([0.2, 0.6], dtype=torch.float64),
                )
            ]
        )
        cumulative, rate = model._baseline_at(torch.tensor([0.0, 0.5, 2.0, 3.0, 9.0]))
        torch.testing.assert_close(
            cumulative[:, 0],
            torch.tensor([0.0, 0.1, 0.4, 0.6, 0.6], dtype=torch.float64),
        )
        torch.testing.assert_close(
            rate[:, 0], torch.tensor([0.2, 0.2, 0.2, 0.2, 0.0], dtype=torch.float64)
        )


class TestEdgeCases(unittest.TestCase):
    def test_zero_event_and_empty_components(self):
        torch.manual_seed(0)
        model = DeepCoxMixtures(None, input_dim=3, num_components=3, hidden_dims=())
        x, t, _ = synthetic()
        model.fit_baselines(x, t, torch.zeros(len(t)))  # nobody has an event
        surv = model.predict_survival(x, [1.0, 4.0])
        torch.testing.assert_close(surv, torch.ones_like(surv))
        out = model(features=x, duration=t, event=torch.ones(len(t)))
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(torch.isfinite(out["posterior"]).all())
        # Force every patient into group 0: groups 1 and 2 keep old baselines.
        before = model._cumulative_hazard[1:, 1:].clone()
        with torch.no_grad():
            model.gate.weight.zero_()
            model.risk.weight.zero_()
            model.gate.weight[0] = 50.0
        x_pos = x.abs() + 1.0  # positive inputs -> group 0 logit dominates
        model.fit_baselines(x_pos, t, torch.ones(len(t)))
        # groups 1 and 2 still have no event knot and a zero hazard
        self.assertTrue(torch.isinf(model._knots[1:, 1:]).all())
        self.assertEqual(float(before.abs().sum()), 0.0)
        self.assertEqual(float(model._cumulative_hazard[1:].abs().sum()), 0.0)
        self.assertGreater(int(torch.isfinite(model._knots[0]).sum()), 1)
        self.assertTrue(torch.isfinite(model.predict_survival(x, [2.0])).all())

    def test_first_fit_with_empty_component_uses_everyone(self):
        model = DeepCoxMixtures(None, input_dim=2, num_components=2, hidden_dims=())
        with torch.no_grad():
            model.gate.weight.copy_(torch.tensor([[50.0, 0.0], [0.0, 0.0]]))
        x = torch.ones(6, 2)
        model.fit_baselines(x, torch.arange(1.0, 7.0), torch.ones(6))
        self.assertEqual(int(torch.isfinite(model._knots[1]).sum()), 7)


class TestPersistenceAndValidation(unittest.TestCase):
    def test_state_dict_round_trip(self):
        model, (x, _, _) = fitted_model(k=2)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "dcm.pt")
            torch.save(model.state_dict(), path)
            restored = DeepCoxMixtures(
                None, input_dim=3, num_components=2, hidden_dims=(4,)
            )
            self.assertFalse(restored.is_fitted)
            restored.load_state_dict(torch.load(path))
        self.assertFalse(os.path.exists(root))
        torch.testing.assert_close(
            restored.predict_survival(x, [0.5, 2.0]),
            model.predict_survival(x, [0.5, 2.0]),
        )
        torch.testing.assert_close(restored.time_grid, model.time_grid)
        torch.testing.assert_close(
            restored(features=x)["y_prob"], model(features=x)["y_prob"]
        )
        with self.assertRaises(ValueError):
            DeepCoxMixtures(
                None, input_dim=3, num_components=3, hidden_dims=(4,)
            ).set_extra_state(model.get_extra_state())

    def test_invalid_inputs(self):
        model, (x, t, e) = fitted_model(k=2)
        with self.assertRaises(ValueError):
            model(features=x, duration=-t, event=e)
        with self.assertRaises(ValueError):
            model(features=x, duration=t, event=e + 1)
        with self.assertRaises(ValueError):
            model(features=x, duration=t)  # event missing
        with self.assertRaises(ValueError):
            model(features=x[:, :2])
        with self.assertRaises(ValueError):
            model(duration=t, event=e)
        nan_x = x.clone()
        nan_x[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model.predict_latent(nan_x)
        with self.assertRaises(ValueError):
            model.predict_survival(x, [-1.0])
        with self.assertRaises(ValueError):
            model(features=x, duration=t[:3], event=e[:3])
        unfitted = DeepCoxMixtures(None, input_dim=3, num_components=2)
        with self.assertRaises(RuntimeError):
            unfitted.predict_survival(x, [1.0])
        with self.assertRaises(RuntimeError):
            unfitted.negative_log_likelihood(x, t, e)
        with self.assertRaises(ValueError):
            unfitted.fit((x, t, e), epochs=0)
        with self.assertRaises(ValueError):
            unfitted.fit((x, t, e), patience=0)


class TestFit(unittest.TestCase):
    def test_fit_tuple_with_validation(self):
        torch.manual_seed(0)
        model = DeepCoxMixtures(
            None, input_dim=3, num_components=2, hidden_dims=(4,), time_grid=[1.0, 2.0]
        )
        x, t, e = synthetic(n=32)
        history = model.fit(
            (x, t, e), (x[:8], t[:8], e[:8]), epochs=3, batch_size=8, patience=1, seed=0
        )
        self.assertLessEqual(len(history["train_loss"]), 3)
        self.assertEqual(len(history["val_nll"]), len(history["train_loss"]))
        self.assertTrue(all(math.isfinite(v) for v in history["val_nll"]))
        self.assertTrue(model.is_fitted)
        self.assertFalse(model.training)
        self.assertEqual(model(features=x)["y_prob"].shape, (32, 2))
        self.assertGreater(model.negative_log_likelihood(x, t, e), 0)

    def test_fit_is_reproducible_with_seed(self):
        x, t, e = synthetic(n=20)
        preds = []
        for _ in range(2):
            torch.manual_seed(3)
            model = DeepCoxMixtures(
                None, input_dim=3, num_components=2, hidden_dims=(4,)
            )
            model.fit(
                {"features": x, "duration": t, "event": e},
                epochs=2,
                batch_size=10,
                seed=11,
            )
            preds.append(model.predict_survival(x, [1.0, 3.0]))
        torch.testing.assert_close(preds[0], preds[1])

    def test_trainer_inference_after_fit(self):
        from pyhealth.trainer import Trainer

        x, t, e = synthetic(n=12)
        samples = [
            {
                "patient_id": str(i),
                "features": x[i].tolist(),
                "duration": [float(t[i])],
                "event": [float(e[i])],
            }
            for i in range(12)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"features": "tensor"},
            output_schema={"duration": "tensor", "event": "tensor"},
            dataset_name="dcm_trainer",
        )
        model = DeepCoxMixtures(
            dataset, num_components=2, hidden_dims=(4,), time_grid=[1.0, 2.0]
        )
        model.fit(dataset, epochs=1, batch_size=6, seed=0)
        trainer = Trainer(model=model, enable_logging=False)
        loader = get_dataloader(dataset, batch_size=6, shuffle=False)
        y_true, y_prob, loss = trainer.inference(loader)
        self.assertEqual(y_true.shape, (12, 2))
        self.assertEqual(y_prob.shape, (12, 2))
        self.assertTrue(math.isfinite(loss))

    def test_fit_sample_dataset(self):
        x, t, e = synthetic(n=12)
        samples = [
            {
                "patient_id": str(i),
                "features": x[i].tolist(),
                "duration": [float(t[i])],
                "event": [float(e[i])],
            }
            for i in range(12)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"features": "tensor"},
            output_schema={"duration": "tensor", "event": "tensor"},
            dataset_name="dcm_fit",
        )
        model = DeepCoxMixtures(dataset, num_components=2, hidden_dims=(4,))
        history = model.fit(dataset, epochs=1, batch_size=6, seed=0)
        self.assertEqual(len(history["train_loss"]), 1)
        self.assertTrue(model.is_fitted)


if __name__ == "__main__":
    unittest.main()
