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

import random

from pyhealth.datasets import SampleDataset, create_sample_dataset
from pyhealth.datasets.sample_dataset import SampleBuilder
from pyhealth.interpret.methods import LayerwiseRelevancePropagation
from pyhealth.models import MLP, StageNet


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


class TestLRPStageNet(unittest.TestCase):
    """Tests for LRP with StageNet (temporal EHR model)."""

    @classmethod
    def _make_dataset(cls):
        """Build a tiny in-memory dataset with stagenet schema."""
        random.seed(0)
        samples = []
        for i in range(30):
            num_visits = random.randint(3, 6)
            codes = [[f"D{j}" for j in random.choices(range(20), k=random.randint(2, 4))]
                     for _ in range(num_visits)]
            times = [v * 24.0 for v in range(num_visits)]
            labs = [[float(f + v * 0.1) for f in range(4)] for v in range(num_visits)]
            samples.append({
                "patient_id": f"P{i}",
                "visit_id": f"V{i}",
                "diagnoses": (times, codes),
                "labs": (times, labs),
                "label": i % 2,
            })
        return create_sample_dataset(
            samples=samples,
            input_schema={"diagnoses": "stagenet", "labs": "stagenet_tensor"},
            output_schema={"label": "binary"},
            in_memory=True,
        )

    def setUp(self):
        self.dataset = self._make_dataset()
        self.model = StageNet(
            dataset=self.dataset,
            embedding_dim=32,
            chunk_size=32,
            levels=2,
        )
        self.model.eval()

        # Build a single-sample batch
        sample = self.dataset[0]
        self.batch = {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor)
            else tuple(t.unsqueeze(0) if isinstance(t, torch.Tensor) else t for t in v)
            if isinstance(v, tuple) else v
            for k, v in sample.items()
        }

    def test_stagenet_lrp_epsilon(self):
        """LRP epsilon rule runs on StageNet without errors."""
        lrp = LayerwiseRelevancePropagation(
            self.model, rule="epsilon", epsilon=0.01, use_embeddings=True
        )
        attrs = lrp.attribute(**self.batch)
        self.assertIn("diagnoses", attrs)
        self.assertIn("labs", attrs)
        for attr in attrs.values():
            self.assertIsInstance(attr, torch.Tensor)
            self.assertFalse(torch.isnan(attr).any())
            self.assertFalse(torch.isinf(attr).any())

    def test_stagenet_lrp_alphabeta(self):
        """LRP alphabeta rule runs on StageNet without errors."""
        lrp = LayerwiseRelevancePropagation(
            self.model, rule="alphabeta", alpha=1.0, beta=0.0, use_embeddings=True
        )
        attrs = lrp.attribute(**self.batch)
        self.assertIn("diagnoses", attrs)
        for attr in attrs.values():
            self.assertFalse(torch.isnan(attr).any())
            self.assertFalse(torch.isinf(attr).any())

    def test_stagenet_logit_conservation(self):
        """Sum of LRP attributions should approximate the model logit."""
        lrp = LayerwiseRelevancePropagation(
            self.model, rule="epsilon", epsilon=0.01, use_embeddings=True
        )
        with torch.no_grad():
            out = self.model(**self.batch)
        logit = out["logit"][0, 0].item()

        attrs = lrp.attribute(**self.batch, target_class_idx=0)
        total = sum(a.sum().item() for a in attrs.values())

        rel = abs(total - logit) / max(abs(logit), 1e-6)
        self.assertLess(rel, 3.0,
            f"Conservation violated: logit={logit:.4f}, sum={total:.4f}, rel={rel:.2%}")

    def test_stagenet_target_class_differs(self):
        """Attributions for class 0 and class 1 should differ."""
        lrp = LayerwiseRelevancePropagation(
            self.model, rule="epsilon", epsilon=0.01, use_embeddings=True
        )
        attrs0 = lrp.attribute(**self.batch, target_class_idx=0)
        attrs1 = lrp.attribute(**self.batch, target_class_idx=1)
        for key in attrs0:
            diff = (attrs0[key] - attrs1[key]).abs().mean().item()
            if diff > 1e-8:
                return  # Found a difference — test passes
        self.fail("Attributions for class 0 and class 1 are identical")


if __name__ == "__main__":
    unittest.main()
