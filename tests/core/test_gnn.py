"""Unit tests for GNN models (GCN and GAT)."""

import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import GCN, GAT


class TestGCN(unittest.TestCase):
    """Test cases for the GCN model."""

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
                "visit_id": "visit-0",
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
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        # Create model
        self.model = GCN(dataset=self.dataset)

    def _get_num_visits(self, data_batch):
        for feature_key in self.model.feature_keys:
            value = data_batch[feature_key]
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                return value.shape[1]
        return 1

    def test_model_initialization(self):
        """Test that the GCN model initializes correctly."""
        self.assertIsInstance(self.model, GCN)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.nhid, 64)
        self.assertEqual(self.model.num_layers, 2)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the GCN model forward pass works correctly."""
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

    def test_model_forward_visit_graph(self):
        """Test that the GCN model works with a custom visit adjacency."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        batch_size = data_batch[self.model.label_key].shape[0]
        num_visits = self._get_num_visits(data_batch)
        visit_adj = torch.eye(num_visits).unsqueeze(0).repeat(batch_size, 1, 1)
        inputs = dict(data_batch)
        inputs["visit_adj"] = visit_adj

        with torch.no_grad():
            ret = self.model(**inputs)

        self.assertIn("loss", ret)
        self.assertEqual(ret["logit"].shape[0], batch_size)

    def test_model_forward_feature_graph(self):
        """Test that the GCN model works with a custom feature adjacency."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        num_features = len(self.model.feature_keys)
        feature_adj = torch.ones(num_features, num_features)
        inputs = dict(data_batch)
        inputs["feature_adj"] = feature_adj

        with torch.no_grad():
            ret = self.model(**inputs)

        self.assertIn("loss", ret)
        self.assertEqual(ret["logit"].shape[0], data_batch[self.model.label_key].shape[0])

    def test_model_forward_feature_graph_batch_specific(self):
        """GCN should accept batch-specific feature adjacency tensors."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        num_features = len(self.model.feature_keys)
        batch_size = data_batch[self.model.label_key].shape[0]
        per_batch_adj = torch.stack(
            [torch.eye(num_features) + i for i in range(batch_size)], dim=0
        )
        inputs = dict(data_batch)
        inputs["feature_adj"] = per_batch_adj

        with torch.no_grad():
            ret = self.model(**inputs)

        self.assertIn("logit", ret)
        self.assertEqual(ret["logit"].shape[0], batch_size)

    def test_feature_adj_invalid_shape_raises(self):
        """Invalid feature adjacency shapes should raise ValueError."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        num_features = len(self.model.feature_keys)
        feature_adj = torch.ones(num_features + 1, num_features)
        inputs = dict(data_batch)
        inputs["feature_adj"] = feature_adj

        with self.assertRaises(ValueError):
            self.model(**inputs)

    def test_visit_adj_invalid_shape_raises(self):
        """Invalid visit adjacency shapes should raise ValueError."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        batch_size = data_batch[self.model.label_key].shape[0]
        num_visits = self._get_num_visits(data_batch)
        visit_adj = torch.ones(batch_size, num_visits + 1, num_visits)
        inputs = dict(data_batch)
        inputs["visit_adj"] = visit_adj

        with self.assertRaises(ValueError):
            self.model(**inputs)

    def test_model_backward(self):
        """Test that the GCN model backward pass works correctly."""
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
        """Test that the GCN model returns embeddings when requested."""
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
        # Check embedding dimension (output of last GCN layer)
        self.assertEqual(ret["embed"].shape[1], self.model.get_output_size())

    def test_custom_hyperparameters(self):
        """Test GCN model with custom hyperparameters."""
        model = GCN(
            dataset=self.dataset,
            embedding_dim=64,
            nhid=32,
            num_layers=3,
            dropout=0.3,
        )

        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.nhid, 32)
        self.assertEqual(model.num_layers, 3)
        self.assertEqual(model.dropout, 0.3)

        # Test forward pass
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)


class TestGAT(unittest.TestCase):
    """Test cases for the GAT model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": [1.0, 2.0, 3.5, 4],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": [5.0, 2.0, 3.5, 4],
                "label": 0,
            },
        ]

        # Define input and output schemas
        self.input_schema = {
            "conditions": "sequence",  # sequence of condition codes
            "procedures": "tensor",  # tensor of procedure values
        }
        self.output_schema = {"label": "binary"}  # binary classification

        # Create dataset
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        # Create model
        self.model = GAT(dataset=self.dataset)

    def _get_num_visits(self, data_batch):
        for feature_key in self.model.feature_keys:
            value = data_batch[feature_key]
            if isinstance(value, torch.Tensor) and value.dim() >= 2:
                return value.shape[1]
        return 1

    def test_model_initialization(self):
        """Test that the GAT model initializes correctly."""
        self.assertIsInstance(self.model, GAT)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.nhid, 64)
        self.assertEqual(self.model.num_layers, 2)
        self.assertEqual(self.model.nheads, 1)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the GAT model forward pass works correctly."""
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

    def test_model_forward_visit_graph(self):
        """Test that the GAT model works with a custom visit adjacency."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        batch_size = data_batch[self.model.label_key].shape[0]
        num_visits = self._get_num_visits(data_batch)
        visit_adj = torch.eye(num_visits).unsqueeze(0).repeat(batch_size, 1, 1)
        inputs = dict(data_batch)
        inputs["visit_adj"] = visit_adj

        with torch.no_grad():
            ret = self.model(**inputs)

        self.assertIn("loss", ret)
        self.assertEqual(ret["logit"].shape[0], batch_size)

    def test_model_forward_feature_graph(self):
        """Test that the GAT model works with a custom feature adjacency."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        num_features = len(self.model.feature_keys)
        batch_size = data_batch[self.model.label_key].shape[0]
        feature_adj = torch.ones(num_features, num_features)
        inputs = dict(data_batch)
        inputs["feature_adj"] = feature_adj

        with torch.no_grad():
            ret = self.model(**inputs)

        self.assertIn("loss", ret)
        self.assertEqual(ret["logit"].shape[0], batch_size)

    def test_model_forward_feature_graph_batch_specific(self):
        """GAT should accept batch-specific feature adjacency tensors."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        num_features = len(self.model.feature_keys)
        batch_size = data_batch[self.model.label_key].shape[0]
        per_batch_adj = torch.stack(
            [torch.eye(num_features) * (i + 1) for i in range(batch_size)], dim=0
        )
        inputs = dict(data_batch)
        inputs["feature_adj"] = per_batch_adj

        with torch.no_grad():
            ret = self.model(**inputs)

        self.assertIn("logit", ret)
        self.assertEqual(ret["logit"].shape[0], batch_size)

    def test_feature_adj_invalid_shape_raises(self):
        """Invalid feature adjacency shapes should raise ValueError in GAT."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        num_features = len(self.model.feature_keys)
        feature_adj = torch.ones(num_features, num_features + 1)
        inputs = dict(data_batch)
        inputs["feature_adj"] = feature_adj

        with self.assertRaises(ValueError):
            self.model(**inputs)

    def test_visit_adj_invalid_shape_raises(self):
        """Invalid visit adjacency shapes should raise ValueError in GAT."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        batch_size = data_batch[self.model.label_key].shape[0]
        num_visits = self._get_num_visits(data_batch)
        visit_adj = torch.ones(batch_size, num_visits, num_visits + 1)
        inputs = dict(data_batch)
        inputs["visit_adj"] = visit_adj

        with self.assertRaises(ValueError):
            self.model(**inputs)

    def test_model_backward(self):
        """Test that the GAT model backward pass works correctly."""
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
        """Test that the GAT model returns embeddings when requested."""
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
        # Check embedding dimension (output of last GAT layer)
        self.assertEqual(ret["embed"].shape[1], self.model.get_output_size())

    def test_custom_hyperparameters(self):
        """Test GAT model with custom hyperparameters."""
        model = GAT(
            dataset=self.dataset,
            embedding_dim=64,
            nhid=32,
            num_layers=3,
            nheads=2,
            dropout=0.3,
        )

        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.nhid, 32)
        self.assertEqual(model.num_layers, 3)
        self.assertEqual(model.nheads, 2)
        self.assertEqual(model.dropout, 0.3)

        # Test forward pass
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)


if __name__ == "__main__":
    unittest.main()
