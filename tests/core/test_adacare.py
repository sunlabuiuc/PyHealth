import unittest
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import AdaCare


class TestAdaCare(unittest.TestCase):
    """Test cases for the AdaCare model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "vector": [[0.1], [0.2], [0.3]],
                "list_codes": ["505800458", "50580045810", "50580045811"],
                "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],
                "list_vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "list_list_vectors": [
                    [[1.8, 2.25, 3.41], [4.50, 5.9, 6.0]],
                    [[7.7, 8.5, 9.4]],
                ],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "vector": [[0.7], [0.8], [0.9]],
                "list_codes": [
                    "55154191800",
                    "551541928",
                    "55154192800",
                    "705182798",
                    "70518279800",
                ],
                "list_list_codes": [["A04A", "B035", "C129"]],
                "list_vectors": [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
                "list_list_vectors": [
                    [[1.0, 2.8, 3.3], [4.9, 5.0, 6.6], [7.7, 8.4, 1.3], [7.7, 8.4, 1.3]],
                ],
                "label": 0,
            },
        ]

        self.input_schema = {
            "vector": "nested_sequence_floats",
            "list_codes": "sequence",
            "list_list_codes": "nested_sequence",
            "list_vectors": "nested_sequence_floats",
            "list_list_vectors": "deep_nested_sequence_floats",
        }
        self.output_schema = {"label": "binary"}
        
        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = AdaCare(dataset=self.dataset, hidden_dim=64)

    def test_model_initialization(self):
        """Test that the AdaCare model initializes correctly."""
        self.assertIsInstance(self.model, AdaCare)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.hidden_dim, 64)
        self.assertEqual(len(self.model.feature_keys), 5)
        self.assertIn("vector", self.model.feature_keys)
        self.assertIn("list_codes", self.model.feature_keys)
        self.assertIn("list_list_codes", self.model.feature_keys)
        self.assertIn("list_vectors", self.model.feature_keys)
        self.assertIn("list_list_vectors", self.model.feature_keys)
        self.assertEqual(self.model.label_keys[0], "label")

    def test_forward_input_format(self):
        """Test that the dataloader provides tensor inputs."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        self.assertIsInstance(data_batch["vector"], torch.Tensor)
        self.assertIsInstance(data_batch["list_codes"], torch.Tensor)
        self.assertIsInstance(data_batch["list_list_codes"], torch.Tensor)
        self.assertIsInstance(data_batch["list_vectors"], torch.Tensor)
        self.assertIsInstance(data_batch["list_list_vectors"], torch.Tensor)
        self.assertIsInstance(data_batch["label"], torch.Tensor)

    def test_model_forward(self):
        """Test that the AdaCare model forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        
        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)
        self.assertIn("feature_importance", ret)
        self.assertIn("conv_feature_importance", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the AdaCare model backward pass works correctly."""
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
        """Test that the AdaCare model returns embeddings when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        
        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)  # batch size
        expected_embed_dim = len(self.model.feature_keys) * self.model.hidden_dim
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)


if __name__ == "__main__":
    unittest.main()
