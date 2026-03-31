import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RETAIN


class TestRETAIN(unittest.TestCase):
    """Test cases for the RETAIN model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["A", "B"], ["C", "D", "E"]],
                "procedures": [["P1"], ["P2", "P3"]],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": [["F"], ["G", "H"]],
                "procedures": [["P4", "P5"], ["P6"]],
                "label": 0,
            },
        ]

        self.input_schema = {
            "conditions": "nested_sequence",
            "procedures": "nested_sequence",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = RETAIN(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the RETAIN model initializes correctly."""
        self.assertIsInstance(self.model, RETAIN)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_keys[0], "label")

    def test_forward_input_format(self):
        """Test that the dataloader provides tensor inputs."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        self.assertIsInstance(data_batch["conditions"], torch.Tensor)
        self.assertIsInstance(data_batch["procedures"], torch.Tensor)
        self.assertIsInstance(data_batch["label"], torch.Tensor)

    def test_model_forward(self):
        """Test that the RETAIN model forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the RETAIN model backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_loss_is_finite(self):
        """Test that the loss is finite."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertTrue(torch.isfinite(ret["loss"]).all())

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        batch_size = 2
        num_labels = 1  # binary classification

        self.assertEqual(ret["y_prob"].shape, (batch_size, num_labels))
        self.assertEqual(ret["y_true"].shape, (batch_size, num_labels))
        self.assertEqual(ret["logit"].shape, (batch_size, num_labels))

    def test_model_with_embedding(self):
        """Test that the RETAIN model returns embeddings when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)  # batch size
        expected_embed_dim = len(self.model.feature_keys) * self.model.embedding_dim
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)


if __name__ == "__main__":
    unittest.main()
