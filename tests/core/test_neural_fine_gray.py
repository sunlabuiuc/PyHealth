"""Tiny analytical model tests independent of optional reference checkouts.

Contributors: Amey Deotale (ameysd2), Anushka Shinde (ashinde4),
Abid Hossain (ahossa5).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://proceedings.mlr.press/v209/jeanselme23a.html
"""

import math
import tempfile
import unittest
from pathlib import Path

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import NeuralFineGray
from pyhealth.processors import BinaryLabelProcessor
from pyhealth.trainer import Trainer


class TestNeuralFineGray(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = NeuralFineGray(None, 3, hidden_dims=(4,), survival_dims=(4,))
        self.features = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.1]])
        self.times = torch.tensor([0.2, 0.7])

    def test_analytical_constant_hazards(self) -> None:
        model = NeuralFineGray(None, 1, hidden_dims=(), survival_dims=())
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            for head, hazard in zip(model.outcome, (2.0, 3.0)):
                head[0].bias.fill_(math.log(math.expm1(hazard)))
        features = [[0.0], [0.0], [0.0]]
        times = torch.tensor([0.0, 0.4, 0.7], dtype=torch.float64)
        output = model(features, times, [0, 1, 2])
        cumulative = times[:, None] * torch.tensor([2.0, 3.0], dtype=torch.float64)
        expected = 0.5 * (1 - torch.exp(-cumulative))
        torch.testing.assert_close(output["cif"], expected, atol=1e-12, rtol=1e-12)
        expected_loss = (
            -(math.log(0.5 * 2.0) - 0.4 * 2.0 + math.log(0.5 * 3.0) - 0.7 * 3.0) / 3
        )
        self.assertAlmostEqual(float(output["loss"].detach()), expected_loss, places=12)
        output["loss"].backward()
        self.assertTrue(torch.isfinite(model.outcome[0][0].bias.grad).all())

    def test_mixed_gradients_and_no_grad_validation(self) -> None:
        output = self.model(self.features, self.times, [0, 2])
        output["loss"].backward()
        for parameter in self.model.parameters():
            if parameter.numel():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.model.eval()
        with torch.no_grad():
            validation = self.model(self.features, self.times, [0, 2])
        torch.testing.assert_close(output["loss"].detach(), validation["loss"])
        self.assertFalse(validation["loss"].requires_grad)

    def test_density_finite_difference(self) -> None:
        self.model.eval()
        epsilon = 1e-5
        before = self.model(self.features, self.times.double() - epsilon)["cif"][:, 0]
        after = self.model(self.features, self.times.double() + epsilon)["cif"][:, 0]
        expected = -((after - before) / (2 * epsilon)).log().mean()
        actual = self.model(self.features, self.times, [1, 1])["loss"]
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)

    def test_eval_respects_feature_parameter_and_time_gradients(self) -> None:
        self.model.eval()
        for objective in ("loss", "cif", "grid"):
            self.model.zero_grad()
            features = self.features.double().requires_grad_()
            times = self.times.double().requires_grad_()
            if objective == "grid":
                value = self.model.predict_cif(features, times).sum()
            else:
                output = self.model(
                    features, times, [0, 2] if objective == "loss" else None
                )
                value = output["loss"] if objective == "loss" else output["cif"].sum()
            self.assertTrue(value.requires_grad, objective)
            value.backward()
            for gradient in (features.grad, times.grad):
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
            for parameter in self.model.parameters():
                if parameter.numel():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
        with torch.no_grad():
            self.assertFalse(self.model.predict_cif(self.features, [1.0]).requires_grad)
            self.assertFalse(
                self.model(self.features, self.times, [0, 2])["loss"].requires_grad
            )

    def test_curves_checkpoint_and_inference(self) -> None:
        self.model.eval()
        horizons = [0.0, 0.1, 1.0, 100.0]
        with torch.inference_mode():
            curves = self.model.predict_cif(self.features, horizons)
            self.assertEqual(tuple(curves.shape), (2, 4, 2))
            self.assertTrue((curves[:, 0] == 0).all())
            self.assertTrue((torch.diff(curves, dim=1) >= -1e-10).all())
            self.assertTrue((curves >= 0).all())
            self.assertTrue((curves.sum(2) <= 1 + 1e-10).all())
            with self.assertRaises(RuntimeError):
                self.model(self.features, self.times, [0, 1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(self.model.state_dict(), path)
            self.model.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(
                curves, self.model.predict_cif(self.features, horizons)
            )
        self.assertEqual(self.model.predict_cif(self.features, []).shape, (2, 0, 2))

    def test_single_risk_and_all_censored(self) -> None:
        model = NeuralFineGray(None, 3, num_risks=1, hidden_dims=(4,))
        output = model(self.features, self.times, [0, 0])
        self.assertEqual(output["cif"].shape, (2, 1))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()

    def test_masks_and_capacity(self) -> None:
        models = [
            NeuralFineGray(None, 3, hidden_dims=(4,), mask_mode=mode)
            for mode in ("constant", "observed", "augmented")
        ]
        self.assertEqual(
            len({sum(p.numel() for p in model.parameters()) for model in models}), 1
        )
        constant, observed, augmented = models
        constant.eval()
        torch.testing.assert_close(
            constant.predict_cif(self.features, [1.0], torch.zeros(2, 3)),
            constant.predict_cif(self.features, [1.0], torch.ones(2, 3)),
        )
        with self.assertRaises(ValueError):
            observed(self.features, self.times, [0, 1])
        augmented.eval()
        zeros = torch.zeros(2, 3)
        torch.testing.assert_close(
            augmented.predict_cif(self.features, [1.0], zeros),
            augmented.predict_cif(self.features, [1.0], zeros),
        )
        with self.assertRaises(ValueError):
            observed(self.features, self.times, [0, 1], missing_mask=[[2, 0, 0]] * 2)

    def test_invalid_inputs(self) -> None:
        for events in ([1, 3], [0, 0.5], [0, float("nan")], [1]):
            with self.assertRaises(ValueError):
                self.model(self.features, self.times, events)
        for times in ([-1.0, 2.0], [float("nan"), 2.0]):
            with self.assertRaises(ValueError):
                self.model(self.features, times, [0, 1])
        for changes in (
            {"input_dim": 0},
            {"num_risks": 0},
            {"hidden_dims": (0,)},
            {"dropout": 1.0},
            {"mask_indices": (0, 0)},
            {"mask_indices": (3,)},
        ):
            with self.assertRaises(ValueError):
                NeuralFineGray(None, **{"input_dim": 3, **changes})
        with self.assertRaises(ValueError):
            self.model.predict_cif([[float("nan")] * 3], [])

    def test_training_augmentation_changes_only_eligible_features(self) -> None:
        model = NeuralFineGray(
            None,
            3,
            hidden_dims=(4,),
            survival_dims=(4,),
            mask_mode="augmented",
            augmentation_rate=1.0,
            mask_indices=(1,),
        )
        seen: list[torch.Tensor] = []
        hook = model.embed[0].register_forward_pre_hook(
            lambda module, inputs: seen.append(inputs[0].detach().clone())
        )
        try:
            model(
                [[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]],
                [0.1, 0.2],
                [0, 1],
                missing_mask=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            )
        finally:
            hook.remove()
        expected = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 1.0, 1.0], [4.0, 0.0, 0.0, 0.0, 1.0, 1.0]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(seen[0], expected)

    def test_joint_loss_ignores_auxiliary_classification_labels(self) -> None:
        samples = [
            {
                "patient_id": f"fake-{i}",
                "features": [0.1, 0.2, 0.3],
                "time": 0.2 + i / 10,
                "event": event,
                "auxiliary_label": i % 2,
            }
            for i, event in enumerate((0, 1, 2, 0))
        ]
        dataset = create_sample_dataset(
            samples,
            {"features": "raw"},
            {"time": "raw", "event": "raw", "auxiliary_label": BinaryLabelProcessor},
        )
        self.assertIsInstance(
            dataset.output_processors["auxiliary_label"], BinaryLabelProcessor
        )
        model = NeuralFineGray(dataset, 3, hidden_dims=(4,), survival_dims=(4,))
        self.assertEqual(model.label_processors, {})
        self.assertIsNone(model.mode)
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        trainer = Trainer(model, device="cpu", enable_logging=False)
        scores = trainer.evaluate(loader)
        self.assertEqual(set(scores), {"loss"})
        self.assertTrue(math.isfinite(scores["loss"]))

    def test_native_trainer_batch_and_validation(self) -> None:
        samples = [
            {
                "patient_id": f"fake-{i}",
                "features": [0.1, 0.2, 0.3],
                "time": 0.2 + i / 10,
                "event": event,
            }
            for i, event in enumerate((0, 1, 2, 0))
        ]
        dataset = create_sample_dataset(
            samples,
            {"features": "raw"},
            {"time": "raw", "event": "raw"},
        )
        model = NeuralFineGray(dataset, 3, hidden_dims=(4,), survival_dims=(4,))
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        trainer = Trainer(model, device="cpu", enable_logging=False)
        scores = trainer.evaluate(loader)
        self.assertTrue(math.isfinite(scores["loss"]))
        batch = next(iter(loader))
        model.train()
        model(**batch)["loss"].backward()
