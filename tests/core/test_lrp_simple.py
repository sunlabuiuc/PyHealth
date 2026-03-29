"""
Basic tests for Layer-wise Relevance Propagation (LRP).

This test suite covers core LRP functionality with PyHealth models.
"""

import unittest
import torch
import numpy as np
import tempfile
import shutil
import pickle
import litdata

from pyhealth.datasets import SampleDataset
from pyhealth.datasets.sample_dataset import SampleBuilder
from pyhealth.interpret.methods import LayerwiseRelevancePropagation
from pyhealth.models import MLP


class TestLRPBasic(unittest.TestCase):
    """Basic tests for LRP functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic dataset
        samples = [
            {
                "patient_id": f"patient-{i}",
                "visit_id": f"visit-0",
                "conditions": [f"cond-{j}" for j in range(3)],
                "label": i % 2,
            }
            for i in range(20)
        ]

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Build dataset using SampleBuilder
        builder = SampleBuilder(
            input_schema={"conditions": "sequence"},
            output_schema={"label": "binary"},
        )
        builder.fit(samples)
        builder.save(f"{self.temp_dir}/schema.pkl")
        
        # Optimize samples into dataset format
        def sample_generator():
            for sample in samples:
                yield {"sample": pickle.dumps(sample)}
        
        litdata.optimize(
            fn=builder.transform,
            inputs=list(sample_generator()),
            output_dir=self.temp_dir,
            num_workers=1,
            chunk_bytes="64MB",
        )
        
        # Create dataset
        self.dataset = SampleDataset(
            path=self.temp_dir,
            dataset_name="test_dataset",
        )
        
        # Create model
        self.model = MLP(
            dataset=self.dataset,
            feature_keys=["conditions"],
            embedding_dim=32,
            hidden_dim=32,
            dropout=0.0,
        )
        self.model.eval()

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'temp_dir') and self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lrp_initialization_epsilon(self):
        """Test LRP initialization with epsilon rule."""
        lrp = LayerwiseRelevancePropagation(
            self.model, rule="epsilon", epsilon=0.01
        )
        self.assertEqual(lrp.rule, "epsilon")
        self.assertEqual(lrp.epsilon, 0.01)
        self.assertIsNotNone(lrp.model)

    def test_lrp_initialization_alphabeta(self):
        """Test LRP initialization with alphabeta rule."""
        lrp = LayerwiseRelevancePropagation(
            self.model, rule="alphabeta", alpha=1.0, beta=0.0
        )
        self.assertEqual(lrp.rule, "alphabeta")
        self.assertEqual(lrp.alpha, 1.0)
        self.assertEqual(lrp.beta, 0.0)

    def test_lrp_attribution_computation(self):
        """Test that LRP can compute attributions."""
        lrp = LayerwiseRelevancePropagation(self.model, rule="epsilon")
        
        # Get a sample
        sample = self.dataset[0]
        batch = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0)
            elif key not in ["patient_id", "visit_id"]:
                if isinstance(value, (int, float)):
                    batch[key] = torch.tensor([value])
        
        # Compute attributions
        attributions = lrp.attribute(**batch)
        
        # Check that we have attributions
        self.assertIn("conditions", attributions)
        self.assertIsInstance(attributions["conditions"], torch.Tensor)
        self.assertEqual(attributions["conditions"].shape[0], 1)

    def test_lrp_no_nan_or_inf(self):
        """Test that attributions don't contain NaN or Inf values."""
        lrp = LayerwiseRelevancePropagation(self.model, rule="epsilon", epsilon=0.01)
        
        # Get a sample
        sample = self.dataset[0]
        batch = {}
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0)
            elif key not in ["patient_id", "visit_id"]:
                if isinstance(value, (int, float)):
                    batch[key] = torch.tensor([value])
        
        # Compute attributions
        attributions = lrp.attribute(**batch, target_class_idx=0)
        
        # Check for NaN or Inf
        for key, attr in attributions.items():
            self.assertFalse(torch.isnan(attr).any())
            self.assertFalse(torch.isinf(attr).any())


if __name__ == "__main__":
    unittest.main()
