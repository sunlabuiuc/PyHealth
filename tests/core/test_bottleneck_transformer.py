import unittest
from typing import Dict, Type, Union

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import BottleneckTransformer
from pyhealth.processors.base_processor import FeatureProcessor


class TestBottleneckTransformer(unittest.TestCase):
    """Test cases for the Bottleneck Transformer model."""

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

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = BottleneckTransformer(
            dataset=self.dataset,
            embedding_dim=128,
            bottlenecks_n=2,
            fusion_startidx=1,
            num_layers=2,
            heads=2
        )

    def test_model_initialization(self):
        """Test that the BottleneckTransformer model initializes correctly."""
        self.assertIsInstance(self.model, BottleneckTransformer)
        self.assertEqual(self.model.embedding_dim, 128)
        self.assertEqual(self.model.bottlenecks_n, 2)
        self.assertEqual(self.model.num_layers, 2)
        self.assertEqual(len(self.model.feature_keys), 3)
        self.assertIn("diagnoses", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertIn("labs", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the BottleneckTransformer forward pass works correctly."""
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
        """Test that the BottleneckTransformer backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

if __name__ == "__main__":
    unittest.main()
