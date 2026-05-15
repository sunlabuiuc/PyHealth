"""Tests for ContrastiveEDAModel using synthetic data."""

import os
import tempfile
import unittest

import numpy as np
import torch

from pyhealth.models.contrastive_eda import (
    ContrastiveEDAModel,
    NCELoss,
    EDAEncoder,
    apply_augmentation_pair,
    AUGMENTATION_GROUPS,
    AUGMENTATION_REGISTRY,
)


class TestEDAEncoder(unittest.TestCase):

    def setUp(self):
        self.window_size = 60
        self.batch_size = 8
        self.encoder = EDAEncoder(window_size=self.window_size)
        self.x = torch.randn(self.batch_size, self.window_size)

    def test_encode_output_shape(self):
        h = self.encoder.encode(self.x)
        self.assertEqual(h.shape, (self.batch_size, self.encoder.embed_dim))

    def test_forward_output_shape(self):
        z = self.encoder(self.x)
        self.assertEqual(z.shape, (self.batch_size, self.encoder.proj_dim))

    def test_no_nan_in_output(self):
        z = self.encoder(self.x)
        self.assertFalse(torch.isnan(z).any())


class TestNCELoss(unittest.TestCase):

    def test_loss_is_scalar(self):
        loss_fn = NCELoss(temperature=0.1)
        z1 = torch.randn(8, 64)
        z2 = torch.randn(8, 64)
        loss = loss_fn(z1, z2)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_loss_is_positive(self):
        loss_fn = NCELoss(temperature=0.1)
        z1 = torch.randn(8, 64)
        z2 = torch.randn(8, 64)
        loss = loss_fn(z1, z2)
        self.assertGreater(loss.item(), 0)

    def test_identical_views_lower_loss(self):
        loss_fn = NCELoss(temperature=0.1)
        z = torch.randn(8, 64)
        loss_same = loss_fn(z, z)
        loss_diff = loss_fn(z, torch.randn(8, 64))
        self.assertLess(loss_same.item(), loss_diff.item())


class TestAugmentations(unittest.TestCase):

    def setUp(self):
        self.x = np.random.rand(60).astype(np.float32)

    def test_all_augmentations_preserve_shape(self):
        for name, cls in AUGMENTATION_REGISTRY.items():
            with self.subTest(augmentation=name):
                aug = cls()
                out = aug(self.x.copy())
                self.assertEqual(
                    len(out), len(self.x),
                    f"{name} changed output length"
                )

    def test_augmentation_pair_shapes(self):
        v1, v2 = apply_augmentation_pair(self.x, AUGMENTATION_GROUPS["full"])
        self.assertEqual(v1.shape, self.x.shape)
        self.assertEqual(v2.shape, self.x.shape)

    def test_augmentation_pair_differs(self):
        v1, v2 = apply_augmentation_pair(self.x, AUGMENTATION_GROUPS["full"])
        self.assertFalse(np.allclose(v1, v2))


class TestContrastiveEDAModel(unittest.TestCase):

    def setUp(self):
        self.window_size = 60
        self.batch_size = 8
        self.x = torch.randn(self.batch_size, self.window_size)
        self.y = torch.randint(0, 2, (self.batch_size,))

    def test_pretrain_step_returns_scalar(self):
        model = ContrastiveEDAModel(window_size=self.window_size)
        loss = model.pretrain_step(self.x)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_pretrain_step_loss_positive(self):
        model = ContrastiveEDAModel(window_size=self.window_size)
        loss = model.pretrain_step(self.x)
        self.assertGreater(loss.item(), 0)

    def test_finetune_step_output_shapes(self):
        model = ContrastiveEDAModel(window_size=self.window_size, num_classes=2)
        model.set_finetune_mode()
        loss, logits = model.finetune_step(self.x, self.y)
        self.assertEqual(logits.shape, (self.batch_size, 2))
        self.assertEqual(loss.shape, torch.Size([]))

    def test_forward_pretrain_mode(self):
        model = ContrastiveEDAModel(window_size=self.window_size)
        z = model(self.x)
        self.assertEqual(z.shape[0], self.batch_size)

    def test_forward_finetune_mode(self):
        model = ContrastiveEDAModel(window_size=self.window_size, num_classes=2)
        model.set_finetune_mode()
        result = model(self.x)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["logit"].shape, (self.batch_size, 2))
        self.assertEqual(result["y_prob"].shape, (self.batch_size, 2))

    def test_forward_finetune_without_set_raises(self):
        model = ContrastiveEDAModel(window_size=self.window_size)
        model._mode = "finetune"
        with self.assertRaises(RuntimeError):
            model(self.x)

    def test_freeze_encoder(self):
        model = ContrastiveEDAModel(
            window_size=self.window_size,
            freeze_encoder=True,
        )
        model.set_finetune_mode()
        for param in model.encoder.parameters():
            self.assertFalse(param.requires_grad)
        self.assertTrue(model.classifier.weight.requires_grad)

    def test_augmentation_groups(self):
        for group in ["full", "generic_only", "eda_specific_only"]:
            with self.subTest(group=group):
                model = ContrastiveEDAModel(
                    window_size=self.window_size,
                    augmentation_group=group,
                )
                loss = model.pretrain_step(self.x)
                self.assertGreater(loss.item(), 0)

    def test_invalid_augmentation_group_raises(self):
        with self.assertRaises(ValueError):
            ContrastiveEDAModel(augmentation_group="nonexistent")

    def test_save_load_encoder(self):
        model = ContrastiveEDAModel(window_size=self.window_size)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "encoder.pt")
            model.save_encoder(path)
            model2 = ContrastiveEDAModel(window_size=self.window_size)
            model2.load_encoder(path)
            z1 = model.encoder.encode(self.x)
            z2 = model2.encoder.encode(self.x)
            self.assertTrue(torch.allclose(z1, z2))

    def test_three_class_finetune(self):
        model = ContrastiveEDAModel(window_size=self.window_size, num_classes=3)
        model.set_finetune_mode()
        y = torch.randint(0, 3, (self.batch_size,))
        loss, logits = model.finetune_step(self.x, y)
        self.assertEqual(logits.shape, (self.batch_size, 3))

    def test_mode_switching(self):
        model = ContrastiveEDAModel(window_size=self.window_size, num_classes=2)
        self.assertEqual(model._mode, "pretrain")
        model.set_finetune_mode()
        self.assertEqual(model._mode, "finetune")
        model.set_pretrain_mode()
        self.assertEqual(model._mode, "pretrain")


if __name__ == "__main__":
    unittest.main()