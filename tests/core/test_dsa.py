import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DSA
from pyhealth.tasks import DecompensationMIMIC3


MAX_HORIZON = 25
PREDICTION_HORIZON = 24


def _record(steps, width, seed):
    rng = np.random.default_rng(seed)
    return {
        "time": [float(i + 1) for i in range(steps)],
        "values": rng.normal(size=(steps, width)).tolist(),
    }


def _dataset(horizon=MAX_HORIZON):
    task = DecompensationMIMIC3(horizon_hours=horizon)
    samples = []

    for i, steps in enumerate([5, 7, 6, 9]):
        hazard_label = [0.0] * horizon
        hazard_mask = [1.0] * horizon
        y_true = i % 2

        if y_true:
            event_index = 4 + i
            hazard_label[event_index] = 1.0
            for k in range(event_index + 1, horizon):
                hazard_mask[k] = 0.0

        samples.append(
            {
                "patient_id": f"p{i}",
                "dynamic_continuous": _record(
                    steps,
                    task.N_CONTINUOUS,
                    10 + i,
                ),
                "dynamic_categorical": _record(
                    steps,
                    task.N_CATEGORICAL,
                    20 + i,
                ),
                "dynamic_mask": {
                    "time": [
                        float(j + 1)
                        for j in range(steps)
                    ],
                    "values": np.ones(
                        (steps, task.N_MASKS)
                    ).tolist(),
                },
                "hazard_label": hazard_label,
                "hazard_mask": hazard_mask,
                "y_true": y_true,
            }
        )

    return create_sample_dataset(
        samples=samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name="dsa_test",
    )


class TestDSA(unittest.TestCase):
    def test_default_model_matches_default_task_horizon(self):
        dataset = _dataset(horizon=24)
        model = DSA(dataset, hidden_dim=8)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        output = model(**batch)
        self.assertEqual(model.max_horizon, 24)
        self.assertEqual(model.prediction_horizon, 23)
        self.assertEqual(output["hazard"].shape, (4, 24))
        torch.testing.assert_close(
            output["y_prob"], output["cumulative_risk"][:, 23:24]
        )

    def test_forward_backward_and_shapes(self):
        dataset = _dataset()
        model = DSA(
            dataset,
            hidden_dim=8,
            max_horizon=MAX_HORIZON,
            prediction_horizon=PREDICTION_HORIZON,
            dropout=0.0,
        )

        self.assertEqual(model.label_processors, {})
        self.assertEqual(model.input_dim, 76)
        self.assertEqual(model.max_horizon, 25)
        self.assertEqual(model.prediction_horizon, 24)

        batch = next(
            iter(get_dataloader(dataset, batch_size=4))
        )
        out = model(**batch)

        self.assertEqual(
            out["hazard"].shape,
            (4, MAX_HORIZON),
        )
        self.assertEqual(
            out["pmf"].shape,
            (4, MAX_HORIZON),
        )
        self.assertEqual(
            out["survival"].shape,
            (4, MAX_HORIZON),
        )
        self.assertEqual(
            out["cumulative_risk"].shape,
            (4, MAX_HORIZON),
        )
        self.assertEqual(out["y_prob"].shape, (4, 1))
        self.assertEqual(out["y_true"].shape, (4, 1))
        self.assertEqual(out["embed"].shape, (4, 8))

        torch.testing.assert_close(
            out["y_prob"],
            out["cumulative_risk"][
                :,
                PREDICTION_HORIZON:
                PREDICTION_HORIZON + 1,
            ],
        )

        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()

        grads = [
            p.grad
            for p in model.parameters()
            if p.requires_grad
        ]
        self.assertTrue(
            any(
                g is not None
                and torch.any(g != 0)
                for g in grads
            )
        )

    def test_survival_probability_identities(self):
        dataset = _dataset()
        model = DSA(
            dataset,
            hidden_dim=8,
            max_horizon=MAX_HORIZON,
            prediction_horizon=PREDICTION_HORIZON,
        )

        batch = next(
            iter(get_dataloader(dataset, batch_size=4))
        )

        model.eval()

        with torch.no_grad():
            out = model(**batch)

        hazard = out["hazard"]
        survival = out["survival"]
        pmf = out["pmf"]
        cumulative = out["cumulative_risk"]

        expected_survival = torch.cumprod(
            1.0 - hazard,
            dim=-1,
        )

        torch.testing.assert_close(
            survival,
            expected_survival,
        )
        torch.testing.assert_close(
            cumulative,
            1.0 - survival,
        )

        survival_before = torch.cat(
            [
                torch.ones(
                    hazard.shape[0],
                    1,
                    dtype=hazard.dtype,
                ),
                survival[:, :-1],
            ],
            dim=-1,
        )

        torch.testing.assert_close(
            pmf,
            survival_before * hazard,
        )

        self.assertTrue(
            torch.all(
                cumulative[:, 1:]
                >= cumulative[:, :-1] - 1e-7
            )
        )

        self.assertTrue(
            torch.all(
                survival[:, 1:]
                <= survival[:, :-1] + 1e-7
            )
        )

    def test_prevalence_bias_initialization(self):
        dataset = _dataset()

        prevalence = torch.linspace(
            0.001,
            0.025,
            MAX_HORIZON,
        )

        model = DSA(
            dataset,
            hidden_dim=8,
            max_horizon=MAX_HORIZON,
            prediction_horizon=PREDICTION_HORIZON,
            hazard_prevalence=prevalence,
        )

        expected = torch.logit(prevalence)

        torch.testing.assert_close(
            model.hazard_head.bias.detach(),
            expected,
        )

        torch.testing.assert_close(
            torch.sigmoid(
                model.hazard_head.bias.detach()
            ),
            prevalence,
        )

    def test_rejects_wrong_bias_length(self):
        dataset = _dataset()

        with self.assertRaises(ValueError):
            DSA(
                dataset,
                max_horizon=MAX_HORIZON,
                prediction_horizon=PREDICTION_HORIZON,
                hazard_prevalence=[0.01] * 24,
            )

    def test_ragged_sequences(self):
        dataset = _dataset()

        model = DSA(
            dataset,
            hidden_dim=8,
            max_horizon=MAX_HORIZON,
            prediction_horizon=PREDICTION_HORIZON,
        )

        batch = next(
            iter(get_dataloader(dataset, batch_size=4))
        )

        offsets = batch["dynamic_continuous"][2]
        lengths = offsets[1:] - offsets[:-1]

        self.assertEqual(
            lengths.tolist(),
            [5, 7, 6, 9],
        )

        out = model(**batch)

        self.assertEqual(
            out["embed"].shape,
            (4, 8),
        )

    def test_invalid_horizons(self):
        dataset = _dataset()

        with self.assertRaises(ValueError):
            DSA(
                dataset,
                max_horizon=24,
                prediction_horizon=24,
            )


if __name__ == "__main__":
    unittest.main()

def test_dsa_state_dict_roundtrip():
    dataset = _dataset()
    model = DSA(
        dataset=dataset,
        hidden_dim=8,
        max_horizon=24,
        prediction_horizon=23,
    )

    restored = DSA(
        dataset=dataset,
        hidden_dim=8,
        max_horizon=24,
        prediction_horizon=23,
    )
    restored.load_state_dict(model.state_dict())

    for original, loaded in zip(model.parameters(), restored.parameters()):
        assert torch.equal(original, loaded)
