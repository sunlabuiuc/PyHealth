import unittest
from typing import Dict, Type, Union

import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.models import Transformer
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.interpret.methods import CheferRelevance


class TestTransformer(unittest.TestCase):
    """Test cases for the Transformer model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "diagnoses": ["A", "B", "C"],
                "procedures": ["X", "Y"],
                "labs": [1.0, 2.0, 3.0],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "diagnoses": ["D", "E"],
                "procedures": ["Y"],
                "labs": [4.0, 5.0, 6.0],
                "label": 0,
            },
        ]

        self.input_schema: Dict[str, Union[str, Type[FeatureProcessor]]] = {
            "diagnoses": "sequence",
            "procedures": "sequence",
            "labs": "tensor",
        }
        self.output_schema: Dict[str, Union[str, Type[FeatureProcessor]]] = {
            "label": "binary"
        }

        self.dataset = SampleDataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = Transformer(dataset=self.dataset)

    def test_model_initialization(self):
        """Test that the Transformer model initializes correctly."""
        self.assertIsInstance(self.model, Transformer)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.heads, 1)
        self.assertEqual(self.model.num_layers, 1)
        self.assertEqual(len(self.model.feature_keys), 3)
        self.assertIn("diagnoses", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertIn("labs", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the Transformer forward pass works correctly."""
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
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["y_prob"].shape[1], 1)
        self.assertEqual(ret["y_true"].shape[1], 1)
        self.assertEqual(ret["logit"].shape[1], 1)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the Transformer backward pass works correctly."""
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
        """Test that the Transformer returns embeddings when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)
        expected_embed_dim = len(self.model.feature_keys) * self.model.embedding_dim
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)

    def test_custom_hyperparameters(self):
        """Test Transformer with custom hyperparameters."""
        model = Transformer(
            dataset=self.dataset,
            embedding_dim=64,
            heads=2,
            dropout=0.1,
            num_layers=2,
        )

        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.heads, 2)
        self.assertEqual(model.num_layers, 2)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_chefer_relevance(self):
        """Test CheferRelevance interpretability method with Transformer."""
        # Use a model with multiple heads for better attention testing
        model = Transformer(
            dataset=self.dataset,
            embedding_dim=64,
            heads=2,
            dropout=0.1,
            num_layers=2,
        )
        
        # Create a dataloader with batch size 1 for interpretability
        train_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(train_loader))
        
        # Initialize CheferRelevance
        relevance = CheferRelevance(model)
        
        # Test with explicitly specified class index
        data_batch["class_index"] = 0
        scores = relevance.get_relevance_matrix(**data_batch)
        
        # Verify that scores are returned for all feature keys
        self.assertIsInstance(scores, dict)
        for feature_key in model.feature_keys:
            self.assertIn(feature_key, scores)
            self.assertIsInstance(scores[feature_key], torch.Tensor)
            # Verify batch dimension matches
            self.assertEqual(scores[feature_key].shape[0], 1)
            # Verify scores are non-negative (due to clamping in relevance computation)
            self.assertTrue(torch.all(scores[feature_key] >= 0))
        
        # Test without specifying class_index (should use predicted class)
        data_batch_no_idx = {k: v for k, v in data_batch.items() if k != "class_index"}
        scores_auto = relevance.get_relevance_matrix(**data_batch_no_idx)
        
        # Verify that scores are returned
        self.assertIsInstance(scores_auto, dict)
        for feature_key in model.feature_keys:
            self.assertIn(feature_key, scores_auto)
            self.assertIsInstance(scores_auto[feature_key], torch.Tensor)
            self.assertEqual(scores_auto[feature_key].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
