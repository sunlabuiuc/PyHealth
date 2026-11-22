"""Small CPU tests for TPC using made-up ICU stays.

Contributor: Audrey Labrie.
Paper: Rocheteau et al., Temporal Pointwise Convolutional Networks for
Length of Stay Prediction in the Intensive Care Unit.
Paper link: https://arxiv.org/abs/2007.09483
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from pyhealth.datasets import create_sample_dataset
from pyhealth.models.tpc import TPC, masked_log_loss


def synthetic_batch(
    static_dim: int = 2, diagnosis_dim: int = 3
) -> Dict[str, torch.Tensor]:
    """Make two stays, with one padding step in the second stay."""
    hours = torch.arange(1, 9, dtype=torch.float32)
    x = torch.zeros(2, 6, 8)
    for patient in range(2):
        x[patient, 0] = hours / 24
        x[patient, 1] = 0.1 * hours + 0.05 * patient
        x[patient, 2] = 0.2 + 0.02 * hours + 0.1 * patient
        x[patient, 3] = 1.0
        x[patient, 4] = 0.75
        x[patient, 5] = hours / 23
    x[1, :, -1] = 0.0
    valid_mask = (hours > 5).repeat(2, 1)
    valid_mask[1, -1] = False
    targets = (3.0 - hours / 24).repeat(2, 1)
    targets[1, -1] = 1.0
    return {
        "x": x,
        "static": torch.full((2, static_dim), 0.25),
        "diagnoses": torch.ones(2, diagnosis_dim),
        "valid_mask": valid_mask,
        "targets": targets,
    }


class TestTPC(unittest.TestCase):
    """Test the TPC model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Use one CPU thread and restore the old setting after the tests."""
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)

    def setUp(self) -> None:
        """Save the random state and make test data."""
        rng_scope = torch.random.fork_rng(devices=[])
        rng_scope.__enter__()
        self.addCleanup(rng_scope.__exit__, None, None, None)
        torch.random.default_generator.manual_seed(17)
        self.batch = synthetic_batch()
        self.dataset = self.make_dataset(self.batch)

    def make_dataset(self, batch: Dict[str, torch.Tensor]) -> Any:
        """Put the two test stays in a PyHealth dataset."""
        samples = []
        for index, length in enumerate((8, 7)):
            samples.append({
                "patient_id": f"synthetic-person-{index}",
                "record_id": f"synthetic-stay-{index}",
                "x": batch["x"][index, :, :length].clone(),
                "static": batch["static"][index].clone(),
                "diagnoses": batch["diagnoses"][index].clone(),
                "valid_mask": batch["valid_mask"][index, :length].clone(),
                "targets": batch["targets"][index, :length].clone(),
            })
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                key: "raw" for key in ("x", "static", "diagnoses", "valid_mask")
            },
            output_schema={"targets": "raw"},
            dataset_name="synthetic-tpc-test",
            task_name="hourly_remaining_icu_los",
            in_memory=True,
        )
        close = getattr(dataset, "close", None)
        if callable(close):
            self.addCleanup(close)
        return dataset

    def make_model(self, **overrides: Any) -> TPC:
        """Make a small model; dropout is off by default."""
        configuration = {
            "dataset": self.dataset,
            "features": 2,
            "static_dim": 2,
            "diagnosis_dim": 3,
            "kernel_size": 2,
            "dilation_rates": (1, 2),
            "temporal_channels": 2,
            "point_channels": 2,
            "diagnosis_hidden": 3,
            "final_hidden": 4,
            "main_dropout": 0.0,
            "temporal_dropout": 0.0,
        }
        configuration.update(overrides)
        return TPC(**configuration).cpu().eval()

    def make_active_model(self) -> TPC:
        """Set small positive weights for gradient and input-change tests."""
        model = self.make_model()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv1d)):
                    module.weight.fill_(0.03)
                    if module.bias is not None:
                        module.bias.fill_(0.05)
        return model

    def test_configurable_architecture(self) -> None:
        configurations = (
            {"kernel_size": 2, "dilation_rates": (1,)},
            {"kernel_size": 3, "dilation_rates": (1, 3)},
            {
                "kernel_size": 4,
                "dilation_rates": (1, 1),
                "temporal_channels": 3,
                "point_channels": 1,
                "main_dropout": 0.2,
                "temporal_dropout": 0.1,
            },
        )
        for configuration in configurations:
            with self.subTest(**configuration):
                model = self.make_model(**configuration)
                rates = configuration["dilation_rates"]
                kernel = configuration["kernel_size"]
                self.assertEqual(len(model.layers), len(rates))
                self.assertEqual(
                    model.receptive_field, 1 + (kernel - 1) * sum(rates)
                )
                self.assertEqual(model.mode, "regression")
                with torch.no_grad():
                    output = model(**self.batch)
                self.assertEqual(output["predictions"].shape, (2, 8))
                self.assertTrue(torch.isfinite(output["loss"]).item())

    def test_optional_static_and_diagnosis_inputs(self) -> None:
        for static_dim, diagnosis_dim in ((0, 3), (2, 0), (0, 0)):
            with self.subTest(static_dim=static_dim, diagnosis_dim=diagnosis_dim):
                batch = synthetic_batch(static_dim, diagnosis_dim)
                model = self.make_model(
                    dataset=self.make_dataset(batch),
                    static_dim=static_dim,
                    diagnosis_dim=diagnosis_dim,
                )
                if diagnosis_dim == 0:
                    self.assertIsNone(model.diagnosis_encoder)
                with torch.no_grad():
                    output = model(**batch)
                self.assertEqual(output["y_prob"].shape, (5, 1))
                self.assertTrue(torch.isfinite(output["loss"]).item())

    def test_forward_outputs_and_ordinary_log_loss(self) -> None:
        model = self.make_active_model()
        with torch.no_grad():
            output = model(**self.batch)
        predictions = output["predictions"]
        valid = self.batch["valid_mask"]
        targets = self.batch["targets"]
        self.assertEqual(predictions.shape, (2, 8))
        self.assertEqual(output["y_prob"].shape, (5, 1))
        self.assertEqual(output["y_true"].shape, (5, 1))
        self.assertEqual(output["loss"].ndim, 0)
        self.assertTrue(torch.isfinite(predictions).all().item())
        self.assertTrue((predictions > 0).all().item())
        self.assertTrue((predictions >= 1 / 48 - 1e-7).all().item())
        self.assertTrue((predictions <= 100 + 1e-4).all().item())
        torch.testing.assert_close(
            output["y_prob"], predictions[valid].reshape(-1, 1)
        )
        torch.testing.assert_close(output["logit"], output["y_prob"])
        torch.testing.assert_close(output["y_true"], targets[valid].reshape(-1, 1))
        expected_loss = (predictions[valid].log() - targets[valid].log()).square()
        torch.testing.assert_close(output["loss"], expected_loss.mean())
        inference_batch = {
            key: value for key, value in self.batch.items() if key != "targets"
        }
        with torch.no_grad():
            inference = model(**inference_batch)
        self.assertNotIn("loss", inference)
        self.assertNotIn("y_true", inference)
        torch.testing.assert_close(inference["predictions"], predictions)

    def test_invalid_targets_and_right_padding_are_ignored(self) -> None:
        model = self.make_active_model()
        changed = {key: value.clone() for key, value in self.batch.items()}
        changed["targets"][~changed["valid_mask"]] = float("nan")
        changed["x"][1, :, -1] = 1234.0
        with torch.no_grad():
            original_output = model(**self.batch)
            changed_output = model(**changed)
        for key in ("loss", "y_prob", "y_true"):
            torch.testing.assert_close(changed_output[key], original_output[key])

    def test_causal_prefix_and_nonconstant_response(self) -> None:
        model = self.make_active_model()
        changed = dict(self.batch)
        changed["x"] = self.batch["x"].clone()
        changed["x"][:, 1:3, 6:] += 2.0
        prefix = dict(self.batch)
        prefix["x"] = self.batch["x"][:, :, :6]
        prefix["valid_mask"] = self.batch["valid_mask"][:, :6]
        prefix["targets"] = self.batch["targets"][:, :6]
        with torch.no_grad():
            before = model(**self.batch)["predictions"]
            after = model(**changed)["predictions"]
            prefix_output = model(**prefix)["predictions"]
        torch.testing.assert_close(before[:, :6], after[:, :6], rtol=0, atol=1e-6)
        torch.testing.assert_close(before[:, :6], prefix_output, rtol=0, atol=1e-6)
        self.assertGreater((before[:, 6:] - after[:, 6:]).abs().max().item(), 1e-6)

    def test_gradients_reach_temporal_pointwise_and_head_weights(self) -> None:
        model = self.make_active_model().train()
        output = model(**self.batch)
        output["loss"].backward()
        parameters = dict(model.named_parameters())
        names = [
            f"layers.{layer}.{branch}.weight"
            for layer in range(2)
            for branch in ("temporal", "point")
        ] + ["diagnosis_encoder.0.weight", "head.0.weight", "head.3.weight"]
        for name in names:
            with self.subTest(parameter=name):
                gradient = parameters[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_state_dict_round_trip(self) -> None:
        model = self.make_active_model()
        with torch.no_grad():
            expected = model(**self.batch)
        with tempfile.TemporaryDirectory(prefix="pyhealth-tpc-test-") as directory:
            checkpoint = Path(directory) / "weights.pt"
            torch.save(model.state_dict(), checkpoint)
            restored = self.make_model()
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            restored.load_state_dict(state, strict=True)
            with torch.no_grad():
                actual = restored(**self.batch)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_invalid_configuration_is_rejected(self) -> None:
        for configuration in (
            {"features": 0},
            {"kernel_size": 0},
            {"static_dim": -1},
            {"diagnosis_dim": -1},
            {"dilation_rates": ()},
            {"dilation_rates": (1, 0)},
            {"main_dropout": 1.0},
            {"temporal_dropout": -0.1},
        ):
            with self.subTest(**configuration), self.assertRaises(ValueError):
                self.make_model(**configuration)

    def test_invalid_input_shapes_are_rejected(self) -> None:
        model = self.make_model()
        replacements = {
            "x": torch.zeros(2, 5, 8),
            "static": torch.zeros(2, 1),
            "diagnoses": torch.zeros(2, 2),
            "valid_mask": torch.ones(2, 7, dtype=torch.bool),
            "targets": torch.ones(2, 7),
        }
        for key, value in replacements.items():
            with self.subTest(field=key), self.assertRaises(ValueError):
                model(**{**self.batch, key: value})


class TestMaskedLogLoss(unittest.TestCase):
    """Test the loss with valid and ignored values."""

    def test_ordinary_log_and_ignored_invalid_entries(self) -> None:
        predictions = torch.tensor([2.0, 0.0, float("nan")], requires_grad=True)
        targets = torch.tensor([1.0, -5.0, float("nan")])
        valid = torch.tensor([True, False, False])
        loss = masked_log_loss(predictions, targets, valid)
        self.assertAlmostEqual(loss.item(), math.log(2.0) ** 2, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all().item())
        self.assertGreater(predictions.grad[0].item(), 0.0)
        torch.testing.assert_close(predictions.grad[1:], torch.zeros(2))

    def test_empty_mask_and_inconsistent_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            masked_log_loss(torch.ones(2), torch.ones(2), torch.zeros(2).bool())
        with self.assertRaises(ValueError):
            masked_log_loss(torch.ones(2), torch.ones(1), torch.ones(2).bool())

    def test_invalid_selected_values_are_rejected(self) -> None:
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            for field in ("predictions", "targets"):
                with self.subTest(field=field, value=invalid):
                    arguments = {
                        "predictions": torch.ones(1),
                        "targets": torch.ones(1),
                        "valid_mask": torch.ones(1, dtype=torch.bool),
                    }
                    arguments[field][0] = invalid
                    with self.assertRaises(ValueError):
                        masked_log_loss(**arguments)


if __name__ == "__main__":
    unittest.main()
