import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SparcNet


class TestSparcNet(unittest.TestCase):
    """Test cases for the SparcNet model."""

    def setUp(self):
        """Set up test data and model."""
        rng = np.random.RandomState(0)
        n_channels = 2
        length = 256

        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "signal": rng.randn(n_channels, length).astype(np.float32),
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "signal": rng.randn(n_channels, length).astype(np.float32),
                "label": 1,
            },
        ]

        self.input_schema = {"signal": "tensor"}
        self.output_schema = {"label": "multiclass"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = SparcNet(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the SparcNet model initializes correctly."""
        self.assertIsInstance(self.model, SparcNet)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.hidden_dim, 128)
        self.assertEqual(len(self.model.feature_keys), 1)
        self.assertIn("signal", self.model.feature_keys)
        self.assertEqual(len(self.model.label_keys), 1)
        self.assertIn("label", self.model.label_keys)

    def test_model_forward(self):
        """Test that the SparcNet model forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_prob"].shape[1], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the SparcNet model backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_model_with_embedding(self):
        """Test that the SparcNet model returns embeddings when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)
        self.assertEqual(ret["embed"].dim(), 2)
        self.assertGreater(ret["embed"].shape[1], 0)

    def test_custom_hyperparameters(self):
        """Test SparcNet model with custom hyperparameters."""
        model = SparcNet(
            dataset=self.dataset,
            embedding_dim=64,
            hidden_dim=64,
            block_layers=3,
            growth_rate=8,
            bn_size=8,
            drop_rate=0.3,
            conv_bias=False,
            batch_norm=False,
        )

        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.hidden_dim, 64)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_multiclass_classification(self):
        """Test SparcNet with multiple classes (sleep staging scenario)."""
        rng = np.random.RandomState(42)
        n_channels = 2
        length = 256

        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "signal": rng.randn(n_channels, length).astype(np.float32),
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "signal": rng.randn(n_channels, length).astype(np.float32),
                "label": 1,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-0",
                "signal": rng.randn(n_channels, length).astype(np.float32),
                "label": 2,
            },
            {
                "patient_id": "patient-3",
                "visit_id": "visit-0",
                "signal": rng.randn(n_channels, length).astype(np.float32),
                "label": 3,
            },
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"signal": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_multiclass",
        )

        model = SparcNet(dataset=dataset)

        train_loader = get_dataloader(dataset, batch_size=4, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertEqual(ret["y_prob"].shape[0], 4)
        self.assertEqual(ret["y_prob"].shape[1], 4)
        self.assertEqual(ret["logit"].shape[1], 4)


if __name__ == "__main__":
    unittest.main()
