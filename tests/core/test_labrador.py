import unittest
from typing import Dict, Type, Union

import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import Labrador
from pyhealth.processors.base_processor import FeatureProcessor


class TestLabrador(unittest.TestCase):
    """Test cases for the Labrador model."""

    def setUp(self):
        """Set up test data and model."""
        # Two tiny patients with short lab trajectories.
        # categorical_input: integer lab codes (0 = pad)
        # continuous_input: float lab values
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "categorical_input": [1, 2, 3, 0],
                "continuous_input": [0.5, -0.3, 1.2, 0.0],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "categorical_input": [4, 5, 0, 0],
                "continuous_input": [1.0, -0.7, 0.0, 0.0],
                "label": 0,
            },
        ]

        # Use simple processors:
        # - "sequence" for categorical token sequences
        # - "tensor" for continuous numeric values
        self.input_schema: Dict[str, Union[str, Type[FeatureProcessor]]] = {
            "categorical_input": "sequence",
            "continuous_input": "tensor",
        }
        self.output_schema: Dict[str, Union[str, Type[FeatureProcessor]]] = {
            "label": "binary"
        }

        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_labrador",
        )

        # Small model for quick tests
        self.model = Labrador(
            dataset=self.dataset,
            feature_keys=["categorical_input", "continuous_input"],
            label_key="label",
            mode="binary",
            max_seq_len=4,
            vocab_size=10,
            embedding_dim=16,
            transformer_heads=2,
            transformer_blocks=1,
            transformer_ff_dim=32,
            dropout_rate=0.1,
            add_extra_dense_layer=False,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def test_model_initialization(self):
        """Test that the Labrador model initializes correctly."""
        self.assertIsInstance(self.model, Labrador)

        # Encoder configuration
        self.assertEqual(self.model.encoder.embedding_dim, 16)
        self.assertEqual(len(self.model.encoder.blocks), 1)

        # Feature / label keys
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("categorical_input", self.model.feature_keys)
        self.assertIn("continuous_input", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

        # Mode and output head
        self.assertEqual(self.model.mode, "binary")
        # For binary mode, output head should have size 1
        self.assertEqual(self.model.output_layer.out_features, 1)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def test_model_forward(self):
        """Test that the Labrador forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        # Returned keys
        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        # Batch dimension
        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)

        # For binary mode, we expect a single logit per sample
        self.assertEqual(ret["y_prob"].shape[1], 1)
        self.assertEqual(ret["y_true"].shape[1], 1)
        self.assertEqual(ret["logit"].shape[1], 1)

        # Loss should be a scalar
        self.assertEqual(ret["loss"].dim(), 0)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def test_model_backward(self):
        """Test that the Labrador backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(
            has_gradient,
            "No parameters have gradients after backward pass",
        )

    # ------------------------------------------------------------------
    # Custom hyperparameters
    # ------------------------------------------------------------------
    def test_custom_hyperparameters(self):
        """Test Labrador with custom hyperparameters."""
        model = Labrador(
            dataset=self.dataset,
            feature_keys=["categorical_input", "continuous_input"],
            label_key="label",
            mode="binary",
            max_seq_len=4,
            vocab_size=10,
            embedding_dim=32,
            transformer_heads=4,
            transformer_blocks=2,
            transformer_ff_dim=64,
            dropout_rate=0.2,
            add_extra_dense_layer=True,
        )

        # Check that custom hyperparameters are applied
        self.assertEqual(model.encoder.embedding_dim, 32)
        self.assertEqual(len(model.encoder.blocks), 2)
        self.assertTrue(model.add_extra_dense_layer)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_prob"].shape[1], 1)


if __name__ == "__main__":
    unittest.main()
