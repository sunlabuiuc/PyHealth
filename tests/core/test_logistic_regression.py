import unittest

import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import LogisticRegression


class TestLogisticRegression(unittest.TestCase):
    """Test cases for the LogisticRegression model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": [1.0, 2.0, 3.5, 4],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": [5.0, 2.0, 3.5, 4],
                "label": 1,
            },
        ]

        # Define input and output schemas
        self.input_schema = {
            "conditions": "sequence",  # sequence of condition codes
            "procedures": "tensor",  # tensor of procedure values
        }
        self.output_schema = {"label": "binary"}  # binary classification

        # Create dataset
        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        # Create model
        self.model = LogisticRegression(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the LogisticRegression model initializes correctly."""
        self.assertIsInstance(self.model, LogisticRegression)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the LogisticRegression model forward pass works correctly."""
        # Create data loader
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        # Forward pass
        with torch.no_grad():
            ret = self.model(**data_batch)

        # Check output structure
        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        # Check tensor shapes
        self.assertEqual(ret["y_prob"].shape[0], 2)  # batch size
        self.assertEqual(ret["y_true"].shape[0], 2)  # batch size
        self.assertEqual(ret["logit"].shape[0], 2)  # batch size

        # Check that loss is a scalar
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the LogisticRegression model backward pass works correctly."""
        # Create data loader
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        # Forward pass
        ret = self.model(**data_batch)

        # Backward pass
        ret["loss"].backward()

        # Check that at least one parameter has gradients (backward working)
        has_gradient = False
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradient = True
                break
        self.assertTrue(
            has_gradient, "No parameters have gradients after backward pass"
        )

    def test_model_with_embedding(self):
        """Test that the LogisticRegression model returns embeddings when requested."""
        # Create data loader
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True

        # Forward pass
        with torch.no_grad():
            ret = self.model(**data_batch)

        # Check that embeddings are returned
        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)  # batch size
        # Check embedding dimension (2 features * embedding_dim)
        expected_embed_dim = len(self.model.feature_keys) * self.model.embedding_dim
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)

    def test_custom_embedding_dim(self):
        """Test LogisticRegression model with custom embedding dimension."""
        model = LogisticRegression(
            dataset=self.dataset,
            embedding_dim=64,
        )

        self.assertEqual(model.embedding_dim, 64)

        # Test forward pass
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_regression_task(self):
        """Test LogisticRegression model with a regression task."""
        # Create regression samples
        regression_samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": [1.0, 2.0, 3.5, 4],
                "score": 0.5,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86"],
                "procedures": [5.0, 2.0, 3.5, 4],
                "score": 0.8,
            },
        ]

        input_schema = {
            "conditions": "sequence",
            "procedures": "tensor",
        }
        output_schema = {"score": "regression"}

        regression_dataset = SampleDataset(
            samples=regression_samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="test_regression",
        )

        # Create model
        regression_model = LogisticRegression(dataset=regression_dataset)

        # Test forward pass
        train_loader = get_dataloader(regression_dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = regression_model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)  # For regression, this is just the raw output
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)


if __name__ == "__main__":
    unittest.main()
