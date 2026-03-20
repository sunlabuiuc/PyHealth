"""Tests for the cardiology multilabel example.

These tests use a tiny synthetic ECG dataset so they stay fast. They check that
the model can be created, run a forward pass, return tensors with the expected
shapes, produce embeddings, and backpropagate without errors.
"""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SparcNet


class TestCardiologyMultilabelSparcNet(unittest.TestCase):
    """Fast synthetic tests for cardiology-style multilabel classification."""

    def setUp(self):
        rng = np.random.RandomState(7)
        n_channels = 12
        length = 256

        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "signal": rng.randn(n_channels, length).astype(np.float32),
                "labels": ["164889003", "427172004"],
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "signal": rng.randn(n_channels, length).astype(np.float32),
                "labels": ["164889003"],
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-0",
                "signal": rng.randn(n_channels, length).astype(np.float32),
                "labels": ["426627000", "713427006"],
            },
            {
                "patient_id": "patient-3",
                "visit_id": "visit-0",
                "signal": rng.randn(n_channels, length).astype(np.float32),
                "labels": ["427172004"],
            },
            {
                "patient_id": "patient-4",
                "visit_id": "visit-0",
                "signal": rng.randn(n_channels, length).astype(np.float32),
                "labels": ["426627000"],
            },
        ]

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"signal": "tensor"},
            output_schema={"labels": "multilabel"},
            dataset_name="test_cardiology_multilabel",
        )

        self.model = SparcNet(
            dataset=self.dataset,
            block_layers=2,
            growth_rate=8,
            bn_size=4,
            drop_rate=0.1,
        )

    def _get_batch(self):
        loader = get_dataloader(self.dataset, batch_size=5, shuffle=False)
        return next(iter(loader))

    def test_model_initialization(self):
        """Model initializes with the expected cardiology multilabel setup."""
        self.assertIsInstance(self.model, SparcNet)
        self.assertEqual(self.model.mode, "multilabel")
        self.assertEqual(self.model.feature_keys, ["signal"])
        self.assertEqual(self.model.label_keys, ["labels"])
        self.assertEqual(self.model.get_output_size(), 4)

    def test_model_forward_shapes(self):
        """Forward pass returns multilabel-shaped logits, probabilities, and targets."""
        batch = self._get_batch()

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape, (5, 4))
        self.assertEqual(ret["y_true"].shape, (5, 4))
        self.assertEqual(ret["logit"].shape, (5, 4))
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertTrue(torch.all(ret["y_prob"] >= 0.0))
        self.assertTrue(torch.all(ret["y_prob"] <= 1.0))

    def test_model_backward(self):
        """Backward pass computes gradients for trainable parameters."""
        batch = self._get_batch()
        ret = self.model(**batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No gradients were produced during backward.")

    def test_model_with_embedding(self):
        """Embedding requests return a batch-aligned 2D representation."""
        batch = self._get_batch()
        batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 5)
        self.assertEqual(ret["embed"].dim(), 2)
        self.assertGreater(ret["embed"].shape[1], 0)


if __name__ == "__main__":
    unittest.main()
