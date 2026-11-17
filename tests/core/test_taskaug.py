"""Unit tests for TaskAug: operations, policy, ResNet1D, model and training.

All data is synthetic: eight random 3-lead signals of 128 samples, and
backbones a few channels wide, so every test runs in milliseconds.

Authors:
    Bjorn Elvar (bthor5@illinois.edu), Rohan Vanjani
"""

import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import (
    ResNet1D,
    SparcNet,
    TaskAug,
    TaskAugPolicy,
    hyper_step,
    train_taskaug,
)
from pyhealth.models import taskaug as ops

N_LEADS, LENGTH = 3, 128


def _dataset(output_schema=None, n_samples=8):
    rng = np.random.RandomState(0)
    samples = [
        {
            "patient_id": f"patient-{i}",
            "signal": rng.randn(N_LEADS, LENGTH).astype(np.float32),
            "label": i % 2,
        }
        for i in range(n_samples)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema=output_schema or {"label": "binary"},
        dataset_name="taskaug-test",
    )


def _small_model(dataset, policy=None):
    backbone = ResNet1D(dataset, kernel_size=5, channels=(4, 8), blocks=(1, 1))
    return TaskAug(dataset, backbone=backbone, policy=policy)


class TestOperations(unittest.TestCase):
    """The seven augmentation operations."""

    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(4, N_LEADS, LENGTH)
        self.label = torch.tensor([0, 1, 1, 0])

    def test_learned_operations_keep_shape_and_pass_gradients(self):
        for name, operation in ops._LEARNED_OPERATIONS.items():
            with self.subTest(operation=name):
                magnitude = torch.full((4, 1, 1), 0.5, requires_grad=True)
                out = operation(self.x, magnitude)
                self.assertEqual(out.shape, self.x.shape)
                self.assertTrue(torch.isfinite(out).all())
                (grad,) = torch.autograd.grad(out.square().sum(), magnitude)
                self.assertTrue((grad != 0).all(), "a strength got no gradient")

    def test_resample_matches_grid_sample(self):
        # Offsets large enough to read outside the signal, which gives zeros.
        # Double precision: grid_sample's normalized grid rounds in float32.
        x = self.x.double()
        flow = 40 * torch.randn(4, 1, LENGTH, dtype=torch.float64)
        position = torch.arange(LENGTH) + flow
        grid = torch.stack(
            [2 * position / (LENGTH - 1) - 1, torch.zeros_like(position)], dim=-1
        )
        expected = F.grid_sample(
            x.unsqueeze(2), grid, mode="bilinear", align_corners=True
        ).squeeze(2)
        self.assertTrue(torch.allclose(ops._resample(x, flow), expected, atol=1e-10))

    def test_resample_shifts_and_zero_pads(self):
        out = ops._resample(self.x, torch.full((4, 1, 1), 5.0))
        self.assertTrue(torch.equal(out[..., :-5], self.x[..., 5:]))
        self.assertTrue((out[..., -5:] == 0).all())

    def test_smoothing_kernel_matches_scipy(self):
        from scipy.ndimage import gaussian_filter1d

        impulse = np.zeros(51)
        impulse[25] = 1.0
        expected = gaussian_filter1d(impulse, 25).astype(np.float32)
        kernel = ops._smoothing_kernel().numpy()
        np.testing.assert_allclose(kernel, expected, atol=1e-7)
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=5)

    def test_temporal_warp_strength(self):
        zero = ops._temporal_warp(self.x, torch.zeros(4, 1, 1))
        self.assertTrue(torch.allclose(zero, self.x, atol=1e-6))
        warped = ops._temporal_warp(self.x, torch.ones(4, 1, 1))
        self.assertFalse(torch.allclose(warped, self.x, atol=1e-2))

    def test_time_mask_zeroes_one_window(self):
        out = ops._time_mask(torch.ones(4, N_LEADS, LENGTH), 0.25)
        zeroed = (out == 0).all(dim=0).all(dim=0)
        self.assertEqual(int(zeroed.sum()), int(0.25 * LENGTH))
        start = int(zeroed.float().argmax())
        self.assertTrue(zeroed[start : start + int(0.25 * LENGTH)].all())
        self.assertTrue((out[..., ~zeroed] == 1).all())

    def test_displacement_is_bounded_by_strength(self):
        impulse = torch.zeros(4, 1, LENGTH)
        impulse[..., LENGTH // 2] = 1.0
        out = ops._displace(impulse, torch.full((4, 1, 1), 0.3))  # up to 9 samples
        moved = out.abs().sum(dim=1).nonzero()[:, 1]
        self.assertTrue(((moved - LENGTH // 2).abs() <= 9).all())

    def test_magnitude_scale_range(self):
        out = ops._magnitude_scale(torch.ones(64, 1, 4), torch.zeros(64, 1, 1))
        # sigmoid(0) * Uniform(0.75, 1.25)
        self.assertTrue(((out >= 0.375) & (out <= 0.625)).all())

    def test_noise_and_wander_vanish_at_low_strength(self):
        low = torch.full((4, 1, 1), -30.0)
        for operation in (ops._gaussian_noise, ops._baseline_wander):
            self.assertTrue(torch.allclose(operation(self.x, low), self.x))

    def test_wander_is_shared_across_leads(self):
        drift = ops._baseline_wander(self.x, torch.zeros(4, 1, 1)) - self.x
        shared = drift[:, :1].expand_as(drift)
        self.assertTrue(torch.allclose(shared, drift, atol=1e-6))
        self.assertGreater(float(drift.abs().max()), 0.01)


class TestTaskAugPolicy(unittest.TestCase):
    """The augmentation policy."""

    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(4, N_LEADS, LENGTH)
        self.label = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    def test_initialization(self):
        policy = TaskAugPolicy()
        self.assertEqual(policy.operations, ops.OPERATIONS)
        self.assertEqual(tuple(policy.logits.shape), (2, 7))
        self.assertTrue((policy.logits == 1).all())
        self.assertEqual(
            sorted(policy.magnitudes), ["displace", "noise", "scale", "wander", "warp"]
        )
        for name, magnitude in policy.magnitudes.items():
            self.assertEqual(tuple(magnitude.shape), (2, 2))
            self.assertTrue((magnitude == ops._INITIAL_MAGNITUDE[name]).all())

    def test_custom_configuration(self):
        policy = TaskAugPolicy(operations=("noise", "mask"), num_stages=3)
        self.assertEqual(tuple(policy.logits.shape), (3, 2))
        self.assertEqual(list(policy.magnitudes), ["noise"])
        self.assertEqual(tuple(policy.magnitudes["noise"].shape), (3, 2))

    def test_invalid_configuration(self):
        for kwargs in (
            {"operations": ("blur",)},
            {"operations": ()},
            {"num_stages": 0},
            {"mask_width": 1.0},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                TaskAugPolicy(**kwargs)

    def test_forward_shape(self):
        out = TaskAugPolicy()(self.x, self.label)
        self.assertEqual(out.shape, self.x.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_noop_policy_is_identity(self):
        out = TaskAugPolicy(operations=("noop",))(self.x, self.label)
        self.assertTrue(torch.equal(out, self.x))

    def test_every_stage_applies_an_operation(self):
        policy = TaskAugPolicy(operations=("mask",), num_stages=2, mask_width=0.25)
        out = policy(torch.ones(4, N_LEADS, LENGTH), self.label)
        zeroed = int((out == 0).all(dim=0).all(dim=0).sum())
        self.assertGreater(zeroed, 0.25 * LENGTH - 1)  # one window or two
        self.assertLessEqual(zeroed, 2 * int(0.25 * LENGTH))

    def test_gradients_reach_selection_and_strength(self):
        policy = TaskAugPolicy()
        policy(self.x, self.label).square().sum().backward()
        self.assertTrue((policy.logits.grad != 0).any())
        grads = [m.grad for m in policy.magnitudes.values() if m.grad is not None]
        self.assertTrue(grads, "no strength received a gradient")

    def test_strengths_are_class_conditioned(self):
        policy = TaskAugPolicy(operations=("scale",), num_stages=1)
        positives = torch.ones(4)
        policy(self.x, positives).sum().backward()
        grad = policy.magnitudes["scale"].grad
        self.assertEqual(float(grad[0, 0]), 0.0)  # negative-class strength
        self.assertNotEqual(float(grad[0, 1]), 0.0)  # positive-class strength

        with torch.no_grad():
            policy.magnitudes["scale"][0, 0] = -30.0  # negatives scaled to ~0
        out = policy(self.x, torch.tensor([0, 1, 1, 0]))
        self.assertTrue(torch.allclose(out[[0, 3]], torch.zeros(()), atol=1e-6))
        self.assertGreater(float(out[[1, 2]].detach().abs().max()), 0.1)


class TestResNet1D(unittest.TestCase):
    """The paper's backbone."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=4)))

    def setUp(self):
        torch.manual_seed(0)
        self.model = ResNet1D(
            self.dataset, kernel_size=5, channels=(4, 8), blocks=(1, 1)
        )

    def test_default_architecture(self):
        model = ResNet1D(self.dataset)
        self.assertEqual(model.kernel_size, 15)
        convs = [m for m in model.encoder.modules() if isinstance(m, torch.nn.Conv1d)]
        # 1 stem + 8 blocks x 2 + 3 shortcuts, as in a ResNet-18.
        self.assertEqual(len(convs), 20)
        self.assertEqual(model.fc.in_features, 256)
        self.assertEqual(model.fc.out_features, 1)

    def test_forward(self):
        with torch.no_grad():
            out = self.model(**self.batch)
        self.assertEqual(sorted(out), ["logit", "loss", "y_prob", "y_true"])
        self.assertEqual(tuple(out["logit"].shape), (4, 1))
        self.assertEqual(tuple(out["y_prob"].shape), (4, 1))
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())

    def test_forward_without_labels(self):
        with torch.no_grad():
            out = self.model(signal=self.batch["signal"])
        self.assertEqual(sorted(out), ["logit", "y_prob"])

    def test_backward(self):
        self.model(**self.batch)["loss"].backward()
        for name, param in self.model.named_parameters():
            if param.numel():
                self.assertIsNotNone(param.grad, f"{name} has no gradient")

    def test_any_signal_length(self):
        with torch.no_grad():
            out = self.model(signal=torch.randn(2, N_LEADS, 301))
        self.assertEqual(tuple(out["logit"].shape), (2, 1))

    def test_multiclass_output(self):
        dataset = _dataset(output_schema={"label": "multiclass"})
        model = ResNet1D(dataset, kernel_size=5, channels=(4,), blocks=(1,))
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        self.assertEqual(tuple(model(**batch)["y_prob"].shape), (4, 2))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ResNet1D(self.dataset, channels=(4, 8), blocks=(1,))


class TestTaskAug(unittest.TestCase):
    """The model wrapper."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=4)))

    def setUp(self):
        torch.manual_seed(0)
        self.model = _small_model(self.dataset)

    def test_default_initialization(self):
        model = TaskAug(self.dataset)
        self.assertIsInstance(model.backbone, ResNet1D)
        self.assertIsInstance(model.policy, TaskAugPolicy)
        self.assertEqual(model.feature_keys, ["signal"])
        self.assertEqual(model.label_keys, ["label"])
        self.assertEqual(model.mode, "binary")

    def test_rejects_non_binary_labels(self):
        with self.assertRaises(ValueError):
            TaskAug(_dataset(output_schema={"label": "multiclass"}))

    def test_forward(self):
        out = self.model(**self.batch)
        self.assertEqual(sorted(out), ["logit", "loss", "y_prob", "y_true"])
        self.assertEqual(tuple(out["y_prob"].shape), (4, 1))
        self.assertEqual(tuple(out["y_true"].shape), (4, 1))

    def test_backward_reaches_backbone_and_policy(self):
        self.model(**self.batch)["loss"].backward()
        for name, param in self.model.backbone.named_parameters():
            if param.numel():
                self.assertIsNotNone(param.grad, f"{name} has no gradient")
        self.assertIsNotNone(self.model.policy.logits.grad)

    def test_augments_only_when_training_with_labels(self):
        policy = self.model.policy
        with patch.object(policy, "forward", wraps=policy.forward) as augment:
            self.model.train()
            self.model(**self.batch)
            self.assertEqual(augment.call_count, 1)
            self.model(signal=self.batch["signal"])  # no labels
            self.assertEqual(augment.call_count, 1)
            self.model.eval()
            self.model(**self.batch)
            self.assertEqual(augment.call_count, 1)

    def test_eval_is_deterministic_and_matches_backbone(self):
        self.model.eval()
        with torch.no_grad():
            first = self.model(**self.batch)["logit"]
            second = self.model(**self.batch)["logit"]
            clean = self.model.backbone(**self.batch)["logit"]
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first, clean))

    def test_other_backbone(self):
        model = TaskAug(self.dataset, backbone=SparcNet(self.dataset))
        out = model(**self.batch)
        self.assertEqual(tuple(out["y_prob"].shape), (4, 1))
        out["loss"].backward()


class TestHyperStep(unittest.TestCase):
    """The implicit-function-theorem hypergradient."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _dataset()
        loader = get_dataloader(cls.dataset, batch_size=4)
        cls.train_batch, cls.val_batch = list(loader)

    def test_sets_policy_gradients_only(self):
        torch.manual_seed(0)
        model = _small_model(self.dataset)
        before = [p.clone() for p in model.parameters()]
        hyper_step(model, self.train_batch, self.val_batch, lr=1e-3)
        for param in model.policy.parameters():
            self.assertEqual(param.grad.shape, param.shape)
            self.assertTrue(torch.isfinite(param.grad).all())
        self.assertTrue((model.policy.logits.grad != 0).any())
        for param in model.backbone.parameters():
            self.assertIsNone(param.grad)
        for param, old in zip(model.parameters(), before):
            self.assertTrue(torch.equal(param, old), "hyper_step changed a weight")

    def test_skips_frozen_policy_parameters(self):
        torch.manual_seed(0)
        model = _small_model(self.dataset)
        model.policy.magnitudes.requires_grad_(False)
        hyper_step(model, self.train_batch, self.val_batch, lr=1e-3)
        self.assertIsNotNone(model.policy.logits.grad)
        for magnitude in model.policy.magnitudes.values():
            self.assertIsNone(magnitude.grad)

    def test_neumann_terms_change_the_gradient(self):
        grads = []
        for terms in (0, 1, 3):
            torch.manual_seed(0)
            model = _small_model(self.dataset)
            torch.manual_seed(1)
            hyper_step(model, self.train_batch, self.val_batch, 0.1, terms)
            grads.append(model.policy.logits.grad.clone())
        self.assertFalse(torch.allclose(grads[0], grads[1]))
        self.assertFalse(torch.allclose(grads[1], grads[2]))

    def test_matches_finite_differences(self):
        """Without Neumann terms the hypergradient is -lr * d<v, g(p)>/dp.

        ``g`` is the training gradient as a function of a policy strength
        ``p`` and ``v`` the validation gradient. The right-hand side is
        estimated by central differences under a fixed random seed.
        """
        lr, eps, seed = 0.1, 1e-5, 3
        torch.manual_seed(0)
        policy = TaskAugPolicy(operations=("scale",), num_stages=1)
        model = _small_model(self.dataset, policy).double()

        def to_double(batch):
            return {
                key: value.double() if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

        train_batch, val_batch = to_double(self.train_batch), to_double(self.val_batch)
        weights = [p for p in model.backbone.parameters() if p.numel()]
        model.train()
        val_grads = torch.autograd.grad(model.backbone(**val_batch)["loss"], weights)

        magnitude = policy.magnitudes["scale"]

        def inner_product(label, shift):
            with torch.no_grad():
                magnitude[0, label] += shift
            torch.manual_seed(seed)
            loss = model(**train_batch)["loss"]
            grads = torch.autograd.grad(loss, weights)
            with torch.no_grad():
                magnitude[0, label] -= shift
            return sum((g * v).sum() for g, v in zip(grads, val_grads))

        torch.manual_seed(seed)
        hyper_step(model, train_batch, val_batch, lr=lr, neumann_terms=0)
        for label in (0, 1):
            upper, lower = inner_product(label, eps), inner_product(label, -eps)
            expected = -lr * (upper - lower) / (2 * eps)
            self.assertNotEqual(float(expected), 0.0)
            self.assertAlmostEqual(
                float(magnitude.grad[0, label]) / float(expected), 1.0, places=5
            )


class TestTrainTaskAug(unittest.TestCase):
    """The bilevel training loop."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = _dataset()

    def _loaders(self):
        return (
            get_dataloader(self.dataset, batch_size=4, shuffle=True, seed=0),
            get_dataloader(self.dataset, batch_size=4, shuffle=True, seed=1),
        )

    def test_trains_classifier_and_policy(self):
        torch.manual_seed(0)
        model = _small_model(self.dataset)
        weight = model.backbone.fc.weight.clone()
        history = train_taskaug(
            model, *self._loaders(), epochs=3, warmup_epochs=1, policy_lr=0.1
        )
        self.assertEqual(len(history["train_loss"]), 3)
        self.assertEqual(len(history["val_loss"]), 3)
        self.assertFalse(torch.equal(model.backbone.fc.weight, weight))
        self.assertFalse(torch.equal(model.policy.logits, torch.ones(2, 7)))
        self.assertTrue(torch.isfinite(model.policy.logits).all())

    def test_warmup_leaves_policy_unchanged(self):
        torch.manual_seed(0)
        model = _small_model(self.dataset)
        train_taskaug(model, *self._loaders(), epochs=2, warmup_epochs=2)
        self.assertTrue(torch.equal(model.policy.logits, torch.ones(2, 7)))

    def test_frozen_policy_is_a_fixed_augmentation(self):
        torch.manual_seed(0)
        policy = TaskAugPolicy(operations=("mask",), num_stages=1)
        policy.requires_grad_(False)
        model = _small_model(self.dataset, policy)
        with patch("pyhealth.models.taskaug.hyper_step") as step:
            history = train_taskaug(model, *self._loaders(), epochs=2, warmup_epochs=0)
        step.assert_not_called()
        self.assertEqual(len(history["val_loss"]), 2)
        self.assertTrue(torch.equal(policy.logits, torch.ones(1, 1)))

    def test_policy_every(self):
        torch.manual_seed(0)
        model = _small_model(self.dataset)
        with patch("pyhealth.models.taskaug.hyper_step") as step:
            train_taskaug(
                model, *self._loaders(), epochs=2, warmup_epochs=0, policy_every=2
            )
        self.assertEqual(step.call_count, 2)  # 4 classifier steps

    def test_restores_best_validation_checkpoint(self):
        torch.manual_seed(0)
        model = _small_model(self.dataset)
        train_loader, val_loader = self._loaders()
        history = train_taskaug(
            model, train_loader, val_loader, epochs=4, warmup_epochs=1, lr=0.05
        )
        model.eval()
        total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                total += model(**batch)["loss"].item() * len(batch["label"])
        self.assertAlmostEqual(
            total / len(self.dataset), min(history["val_loss"]), places=5
        )

    def test_resumes_from_checkpoint(self):
        # Unshuffled loaders on a CPU: a resumed run must equal a whole one.
        def train(epochs, path=None):
            loader = get_dataloader(self.dataset, batch_size=4)
            return train_taskaug(
                model, loader, loader, epochs, warmup_epochs=1, checkpoint_path=path
            )

        torch.manual_seed(0)
        model = _small_model(self.dataset)
        start = copy.deepcopy(model.state_dict())
        whole = train(4)
        expected = copy.deepcopy(model.state_dict())

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "run.pt")
            torch.manual_seed(0)
            model = _small_model(self.dataset)
            model.load_state_dict(start)
            train(2, path)
            torch.manual_seed(123)  # a new process has another random state
            model = _small_model(self.dataset)
            resumed = train(4, path)
            self.assertEqual(os.listdir(folder), ["run.pt"])
            with patch("pyhealth.models.taskaug.hyper_step") as step:
                train(4, path)  # a finished run trains no further
            step.assert_not_called()
        self.assertEqual(resumed["val_loss"], whole["val_loss"])
        self.assertEqual(len(resumed["seconds"]), 4)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]), name)
        self.assertFalse(torch.equal(model.policy.logits, torch.ones(2, 7)))


if __name__ == "__main__":
    unittest.main()
