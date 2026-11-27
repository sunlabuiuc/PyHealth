"""Synthetic correctness tests for SurvivalMoE.

Author: Shuxian Xu (NetID: sx30).
Paper: `Let the Experts Speak: Improving Survival Prediction & Calibration
via Mixture-of-Experts Heads <https://proceedings.mlr.press/v297/morrill26a.html>`_.
"""

import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import SurvivalMoE, SurvivalMoEModel
from pyhealth.trainer import Trainer
from support2_experiment import _bin_durations, _evaluation_times, prepare_split


def _tied_duration_data() -> pd.DataFrame:
    durations = np.tile(np.arange(1, 301), 2)
    return pd.DataFrame(
        {
            "sno": [f"patient-{index}" for index in range(len(durations))],
            "age": np.linspace(20, 90, len(durations)),
            "d.time": durations,
            "death": np.tile([0, 1], 300),
        }
    )


class SurvivalMoETest(unittest.TestCase):
    def test_time_grid_preserves_requested_bin_count_with_ties(self):
        split = prepare_split(_tied_duration_data(), seed=11, num_time_bins=100)

        self.assertEqual(len(split.time_grid), 100)
        self.assertTrue(np.all(np.diff(split.time_grid) > 0))

    def test_censored_time_on_grid_point_maps_to_next_bin(self):
        bins = _bin_durations(
            np.array([20.0, 20.0, 15.0, 35.0]),
            np.array([1, 0, 0, 0]),
            np.array([10.0, 20.0, 30.0]),
        )
        np.testing.assert_array_equal(bins, [1, 2, 1, 2])

    def test_ece_times_do_not_depend_on_time_grid(self):
        data = _tied_duration_data()
        coarse = _evaluation_times(prepare_split(data, seed=11, num_time_bins=25))
        fine = _evaluation_times(prepare_split(data, seed=11, num_time_bins=200))
        np.testing.assert_array_equal(coarse[1], fine[1])

    def test_pyhealth_base_model_forward_and_trainer(self):
        samples = [
            {
                "features": [float(index), 1.0, -1.0],
                "duration_bin": index % 4,
                "event": index % 2,
            }
            for index in range(4)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"features": "tensor"},
            output_schema={
                "duration_bin": ("tensor", {"dtype": torch.int64}),
                "event": ("tensor", {"dtype": torch.int64}),
            },
            in_memory=True,
        )
        model = SurvivalMoEModel(
            dataset=dataset,
            input_dim=3,
            num_time_bins=4,
            hidden_dim=8,
            num_experts=2,
            time_grid=[10.0, 20.0, 30.0, 40.0],
        )
        batch = next(iter(DataLoader(dataset, batch_size=4)))
        output = model(**batch)

        self.assertIsNone(model.mode)
        self.assertEqual(tuple(output["pmf"].shape), (4, 4))
        self.assertEqual(tuple(output["y_prob"].shape), (4, 4))
        self.assertEqual(tuple(output["y_true"].shape), (4, 2))
        self.assertTrue(torch.isfinite(output["loss"]))
        torch.testing.assert_close(
            model.time_grid,
            torch.tensor([10.0, 20.0, 30.0, 40.0]),
        )

        trainer = Trainer(model, device="cpu", enable_logging=False)
        trainer.train(DataLoader(dataset, batch_size=4), epochs=1)

    def test_pmf_and_survival_shapes_and_constraints(self):
        model = SurvivalMoE(
            input_dim=6,
            num_time_bins=12,
            hidden_dim=16,
            num_experts=4,
            expert_type="personalized",
        )
        output = model(torch.randn(5, 6))

        self.assertEqual(tuple(output["pmf"].shape), (5, 12))
        self.assertEqual(tuple(output["survival"].shape), (5, 12))
        self.assertEqual(tuple(output["routing"].shape), (5, 4))
        torch.testing.assert_close(
            output["pmf"].sum(dim=1),
            torch.ones(5),
        )
        self.assertTrue(torch.all(output["pmf"] > 0))
        self.assertTrue(torch.all(output["survival"][:, :-1] >= output["survival"][:, 1:]))

    def test_censored_and_uncensored_loss_backpropagates(self):
        for expert_type in ("personalized", "fixed", "mtlr"):
            model = SurvivalMoE(
                input_dim=4,
                num_time_bins=9,
                hidden_dim=12,
                num_experts=3,
                expert_type=expert_type,
            )
            output = model(
                torch.randn(6, 4),
                torch.tensor([0, 1, 3, 5, 6, 8]),
                torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.bool),
            )

            self.assertTrue(torch.isfinite(output["loss"]))
            self.assertTrue(torch.isfinite(output["nll"]))
            output["loss"].backward()
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                )
            )

    def test_top_k_router_uses_only_selected_experts(self):
        model = SurvivalMoE(
            input_dim=5,
            num_time_bins=10,
            hidden_dim=16,
            num_experts=4,
            expert_type="personalized",
            top_k=2,
        )
        routing = model(torch.randn(7, 5))["routing"]

        self.assertTrue(torch.all((routing > 0).sum(dim=1) <= 2))
        torch.testing.assert_close(routing.sum(dim=1), torch.ones(7))

    def test_top_one_router_keeps_a_training_gradient(self):
        model = SurvivalMoE(
            input_dim=5,
            num_time_bins=10,
            hidden_dim=16,
            num_experts=4,
            expert_type="personalized",
            top_k=1,
        )
        output = model(
            torch.randn(7, 5),
            torch.tensor([0, 1, 2, 3, 4, 5, 6]),
            torch.tensor([True, False, True, False, True, False, True]),
        )
        output["loss"].backward()

        self.assertIsNotNone(model.router.weight.grad)
        self.assertTrue(torch.any(model.router.weight.grad != 0))

    def test_mtlr_increments_are_cumulatively_decoded(self):
        model = SurvivalMoE(
            input_dim=1,
            num_time_bins=3,
            hidden_dim=1,
            num_layers=1,
            expert_type="mtlr",
        )
        with torch.no_grad():
            model.backbone[0].weight.fill_(1.0)
            model.backbone[0].bias.zero_()
            model.mtlr_head.weight.copy_(torch.tensor([[0.0], [1.0], [0.0]]))
            model.mtlr_head.bias.zero_()

        pmf = model(torch.ones(1, 1))["pmf"][0]
        expected = torch.softmax(torch.tensor([1.0, 1.0, 0.0]), dim=0)
        torch.testing.assert_close(pmf, expected)


if __name__ == "__main__":
    unittest.main()
