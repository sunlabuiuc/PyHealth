import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN_attention


class TestRNNAttention(unittest.TestCase):
    """Test cases for the RNNAttention model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": [["A01", "B02"], ["C03"]],
                "procedures": [["P01"], ["P02"]],
                "label": 1,
            },
            {
                "patient_id": "p2",
                "visit_id": "v2",
                "conditions": [["A01"], ["D04"]],
                "procedures": [["P03"], ["P01"]],
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
            dataset_name="test_ds",
        )

        self.model = RNN_attention(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the RNNAttention model initializes correctly."""
        self.assertIsInstance(self.model, RNN_attention)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_initialization_custom_embedding(self):
        """Test that the RNNAttention model initializes with custom embedding dim."""
        model_custom = RNN_attention(dataset=self.dataset, embedding_dim=64)
        self.assertEqual(model_custom.embedding_dim, 64)

    def test_model_forward(self):
        """Test that the RNNAttention model forward pass works correctly."""
        train_loader = get_dataloader(
            self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        # Batch size 2, Binary label
        self.assertEqual(ret["logit"].shape, (2, 1))
        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertFalse(torch.isnan(ret["loss"]))

    def test_model_backward(self):
        """Test that the RNNAttention model backward pass works correctly."""
        train_loader = get_dataloader(
            self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        loss = ret["loss"]
        loss.backward()

        # Check that gradients exist
        self.assertIsNotNone(self.model.fc.weight.grad)
        self.assertGreater(torch.sum(self.model.fc.weight.grad ** 2).item(), 0)

    def test_model_with_embedding(self):
        """Test that the RNNAttention model returns embeddings when requested."""
        train_loader = get_dataloader(
            self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            output = self.model(**data_batch, embed=True)

        self.assertIn("embed", output)
        # Expected dimension: num_heads (4) * embedding_dim (128)
        expected_dim = 4 * 128
        self.assertEqual(output["embed"].shape, (2, expected_dim))

    def test_invalid_feature_size_error(self):
        """Test that passing feature_size raises ValueError."""
        with self.assertRaises(ValueError) as context:
            RNN_attention(dataset=self.dataset, feature_size=128)

        self.assertIn("feature_size is determined by embedding_dim",
                      str(context.exception))

    def test_model_with_different_attention_heads(self):
        """Test that the model works with different numbers of attention heads."""
        model_2heads = RNN_attention(dataset=self.dataset, h=2)
        self.assertEqual(model_2heads.num_heads, 2)

        train_loader = get_dataloader(
            self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model_2heads(**data_batch)

        self.assertIn("loss", ret)
        self.assertFalse(torch.isnan(ret["loss"]))

    def test_model_output_shapes(self):
        """Test that all output tensors have correct shapes."""
        train_loader = get_dataloader(
            self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        batch_size = 2

        # Check loss is scalar
        self.assertEqual(ret["loss"].shape, torch.Size([]))

        # Check logits shape
        self.assertEqual(ret["logit"].shape, (batch_size, 1))

        # Check y_prob shape
        self.assertEqual(ret["y_prob"].shape[0], batch_size)

        # Check y_true shape
        self.assertEqual(ret["y_true"].shape[0], batch_size)


if __name__ == "__main__":
    unittest.main()
