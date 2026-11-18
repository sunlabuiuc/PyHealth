"""Tests of DynamicSurvivalRNN on tiny synthetic data."""

import unittest
from datetime import datetime, timedelta

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN_V2, DynamicSurvivalRNN
from pyhealth.models.dynamic_survival import (
    hazard_prior,
    risk_from_hazards,
    survival_nll,
)
from pyhealth.models.dynamic_survival_rnn import hazards_to_risk, survival_loss
from pyhealth.trainer import Trainer

HORIZON = 4
T0 = datetime(2020, 1, 1)
HOURS = [T0 + timedelta(hours=h) for h in range(4)]


def _dataset(outputs=None):
    """Two stays with three labeled steps each (hours 1, 2, 3)."""
    steps = [1.0, 2.0, 3.0]
    samples = [
        {  # dies: the event falls in hour 3, 2 and 1 after the steps
            "patient_id": "p0",
            "x": (HOURS, [[1.0], [2.0], [3.0], [4.0]]),
            "decompensation": {"time": steps, "values": [[1.0], [1.0], [1.0]]},
            "time_to_event": {"time": steps, "values": [[3.0], [2.0], [1.0]]},
            "event": {"time": steps, "values": [[1.0], [1.0], [1.0]]},
        },
        {  # censored: 2, 1 and 0 whole hours observed
            "patient_id": "p1",
            "x": (HOURS, [[0.5], [0.4], [0.3], [0.2]]),
            "decompensation": {"time": steps, "values": [[0.0], [0.0], [0.0]]},
            "time_to_event": {"time": steps, "values": [[2.0], [1.0], [0.0]]},
            "event": {"time": steps, "values": [[0.0], [0.0], [0.0]]},
        },
    ]
    schema = outputs or {
        "decompensation": ("timeseries_binary", {"timeline": ["x"]}),
        "time_to_event": ("timeseries_regression", {"timeline": ["x"]}),
        "event": ("timeseries_binary", {"timeline": ["x"]}),
    }
    samples = [{k: v for k, v in s.items() if k in schema or k in {"patient_id", "x"}}
               for s in samples]
    return create_sample_dataset(samples, {"x": "timeseries"}, schema)


def _model(dataset, **kwargs):
    model = DynamicSurvivalRNN(
        dataset,
        horizon=HORIZON,
        embedding_dim={"x": [{"values": 3}]},
        hidden_dim=4,
        dropout=0.0,
        **kwargs,
    )
    model.eval()
    return model


def _batch(dataset):
    return next(iter(get_dataloader(dataset, batch_size=len(dataset))))


class TestFunctions(unittest.TestCase):
    def test_loss_matches_the_numpy_reference(self):
        torch.manual_seed(0)
        logits = torch.randn(5, 6)
        tte = torch.tensor([1.0, 3.0, 9.0, 2.0, 0.0])
        event = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
        got = survival_loss(logits, tte, event).item()
        want = survival_nll(logits.numpy(), tte.numpy(), event.numpy())
        self.assertAlmostEqual(got, want, places=5)

    def test_loss_without_counted_hours_is_zero_with_a_graph(self):
        logits = torch.zeros(2, 3, requires_grad=True)
        loss = survival_loss(logits, torch.zeros(2), torch.zeros(2))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()  # the graph is kept

    def test_risk_matches_the_numpy_reference(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 7) * 3
        got = hazards_to_risk(logits).numpy()
        want = risk_from_hazards(torch.sigmoid(logits).numpy())
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_risk_is_stable_for_extreme_logits(self):
        risk = hazards_to_risk(torch.tensor([[-80.0, -80.0], [80.0, 80.0]]))
        self.assertTrue(torch.isfinite(risk).all())
        self.assertAlmostEqual(risk[0].item(), 0.0, places=6)
        self.assertAlmostEqual(risk[1].item(), 1.0, places=6)


class TestModel(unittest.TestCase):
    def test_outputs_and_shapes(self):
        dataset = _dataset()
        out = _model(dataset)(**_batch(dataset))
        self.assertEqual(out["logit"].shape, (6, HORIZON))  # 3 steps x 2 stays
        self.assertEqual(out["y_prob"].shape, (6, 1))
        self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(len(out["y_true"]), 3)  # packed binary label

    def test_scored_as_the_binary_label(self):
        dataset = _dataset()
        model = _model(dataset)
        self.assertEqual(model.label_keys, ["decompensation"])
        self.assertEqual(list(model.label_processors), ["decompensation"])
        self.assertEqual(model.mode, "timeseries_binary")

    def test_needs_the_survival_outputs(self):
        dataset = _dataset(
            {"decompensation": ("timeseries_binary", {"timeline": ["x"]})}
        )
        with self.assertRaisesRegex(ValueError, "survival mode"):
            DynamicSurvivalRNN(
                dataset, horizon=HORIZON, embedding_dim={"x": [{"values": 3}]}
            )

    def test_rejects_bad_horizon_and_prior(self):
        dataset = _dataset()
        kwargs = {"embedding_dim": {"x": [{"values": 3}]}}
        with self.assertRaises(ValueError):
            DynamicSurvivalRNN(dataset, horizon=0, **kwargs)
        with self.assertRaises(ValueError):
            DynamicSurvivalRNN(dataset, horizon=HORIZON, hazard_prior=[0.1], **kwargs)
        with self.assertRaises(ValueError):
            DynamicSurvivalRNN(
                dataset, horizon=HORIZON, hazard_prior=[0.1, 0.1, 0.1, 1.0], **kwargs
            )

    def test_bias_starts_at_the_prior(self):
        prior = np.array([0.001, 0.002, 0.003, 0.004])
        model = _model(_dataset(), hazard_prior=prior)
        logit = np.log(prior / (1 - prior))
        np.testing.assert_allclose(
            model.fc.bias.detach().numpy(), logit, rtol=1e-5
        )

    def test_default_init_starts_near_half(self):
        torch.manual_seed(0)
        model = _model(_dataset())
        self.assertLess(model.fc.bias.abs().max().item(), 1.0)

    def test_prior_makes_the_initial_risk_the_base_rate(self):
        # zero the head's weights so only the bias acts
        prior = np.full(HORIZON, 0.01)
        model = _model(_dataset(), hazard_prior=prior)
        with torch.no_grad():
            model.fc.weight.zero_()
        dataset = _dataset()
        out = model(**_batch(dataset))
        np.testing.assert_allclose(
            out["y_prob"].detach().numpy(), 1 - 0.99**HORIZON, rtol=1e-4
        )

    def test_prior_from_samples(self):
        dataset = _dataset()
        got = DynamicSurvivalRNN.prior_from_samples(dataset, HORIZON)
        # steps: (3,1) (2,1) (1,1) (2,0) (1,0) (0,0)
        want = hazard_prior(
            [3, 2, 1, 2, 1, 0], [1, 1, 1, 0, 0, 0], HORIZON
        )
        np.testing.assert_allclose(got, want)

    def test_trains(self):
        dataset = _dataset()
        model = DynamicSurvivalRNN(
            dataset,
            horizon=HORIZON,
            hazard_prior=DynamicSurvivalRNN.prior_from_samples(dataset, HORIZON),
            embedding_dim={"x": [{"values": 3}]},
            hidden_dim=4,
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=True, seed=0)
        Trainer(model=model, enable_logging=False).train(
            train_dataloader=loader, epochs=2
        )
        out = model(**_batch(dataset))
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters())
        )

    def test_evaluates_with_the_trainer(self):
        dataset = _dataset()
        model = _model(dataset)
        scores = Trainer(
            model=model, metrics=["roc_auc"], enable_logging=False
        ).evaluate(get_dataloader(dataset, batch_size=2))
        self.assertIn("loss", scores)

    def test_is_an_rnn_v2(self):
        self.assertTrue(issubclass(DynamicSurvivalRNN, RNN_V2))


if __name__ == "__main__":
    unittest.main()
