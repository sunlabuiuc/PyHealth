import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import LabradorModel


class TestLabradorModel(unittest.TestCase):
    """Minimal smoke test for LabradorModel."""

    def setUp(self):
        """Set up test data and model."""
        # Create minimal synthetic samples with aligned lab codes and values
        # Both lab_codes and lab_values represent the same 4 labs per sample
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "lab_codes": ["lab-1", "lab-2", "lab-3", "lab-4"],
                "lab_values": [1.0, 2.5, 3.0, 4.5],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "lab_codes": ["lab-1", "lab-2", "lab-3", "lab-4"],
                "lab_values": [2.1, 1.8, 2.9, 3.5],
                "label": 1,
            },
        ]

        # Define input and output schemas
        self.input_schema = {
            "lab_codes": "sequence",    # Categorical lab codes
            "lab_values": "tensor",     # Continuous lab values
        }
        self.output_schema = {"label": "binary"}  # Binary classification

        # Create dataset
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_labrador",
        )

        # Create model
        self.model = LabradorModel(
            dataset=self.dataset,
            code_feature_key="lab_codes",
            value_feature_key="lab_values",
            embed_dim=32,
            num_heads=2,
            num_layers=1,
        )

    def test_model_initialization(self):
        """Test that LabradorModel initializes correctly."""
        self.assertIsInstance(self.model, LabradorModel)
        self.assertEqual(self.model.embed_dim, 32)
        self.assertEqual(self.model.num_heads, 2)
        self.assertEqual(self.model.num_layers, 1)
        self.assertEqual(self.model.code_feature_key, "lab_codes")
        self.assertEqual(self.model.value_feature_key, "lab_values")
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that LabradorModel forward pass works correctly."""
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
        """Test that LabradorModel backward pass works correctly."""
        # Create data loader
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        # Forward pass
        ret = self.model(**data_batch)

        # Backward pass
        ret["loss"].backward()

        # Check that at least one parameter has gradients
        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_model_with_embedding(self):
        """Test that LabradorModel returns embeddings when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)  # batch size
        self.assertEqual(ret["embed"].shape[1], 32)  # embed_dim


if __name__ == "__main__":
    unittest.main()
