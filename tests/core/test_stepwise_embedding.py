"""Tests for the StepwiseEmbedding model.

Uses synthetic data only. All tests should complete in milliseconds.
"""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.datasets.mimic3_tls import MIMIC3TLSDataset
from pyhealth.models.stepwise_embedding import (
    StepwiseEmbedding,
    StepwiseEmbeddingLayer,
    LinearEmbedding,
    MLPEmbedding,
    FTTransformerEmbedding,
    TransformerBackbone,
)


class TestStepwiseEmbedding(unittest.TestCase):
    """Test cases for the StepwiseEmbedding model."""

    def setUp(self):
        """Set up synthetic dataset and data loader."""
        np.random.seed(42)
        self.samples = [
            {
                "patient_id": f"p-{i}",
                "time_series": np.random.randn(48, 42).tolist(),
                "ihm": i % 2,
            }
            for i in range(4)
        ]
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"time_series": "tensor"},
            output_schema={"ihm": "binary"},
            dataset_name="test",
        )

    def _get_batch(self, batch_size: int = 2):
        loader = get_dataloader(self.dataset, batch_size=batch_size)
        return next(iter(loader))

    def _check_output(self, ret, batch_size: int = 2):
        """Verify the model output dictionary structure."""
        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertEqual(ret["y_prob"].shape[0], batch_size)
        self.assertEqual(ret["y_true"].shape[0], batch_size)
        self.assertEqual(ret["logit"].shape[0], batch_size)

    def test_backbone_only(self):
        """Test model with no step-wise embedding (backbone only)."""
        model = StepwiseEmbedding(
            dataset=self.dataset,
            embedding_type=None,
            input_dim=42,
            hidden_dim=32,
            backbone_depth=1,
            backbone_heads=1,
        )
        batch = self._get_batch()
        with torch.no_grad():
            ret = model(**batch)
        self._check_output(ret)

    def test_ftt_no_grouping(self):
        """Test FTT embedding without feature grouping."""
        model = StepwiseEmbedding(
            dataset=self.dataset,
            embedding_type="ftt",
            input_dim=42,
            hidden_dim=32,
            backbone_depth=1,
            backbone_heads=1,
        )
        batch = self._get_batch()
        with torch.no_grad():
            ret = model(**batch)
        self._check_output(ret)

    def test_ftt_organ_groups(self):
        """Test FTT embedding with organ-based grouping."""
        model = StepwiseEmbedding(
            dataset=self.dataset,
            embedding_type="ftt",
            group_indices=MIMIC3TLSDataset.ORGAN_GROUPS_INDICES,
            aggregation="mean",
            input_dim=42,
            hidden_dim=32,
            backbone_depth=1,
            backbone_heads=1,
        )
        batch = self._get_batch()
        with torch.no_grad():
            ret = model(**batch)
        self._check_output(ret)

    def test_ftt_type_groups(self):
        """Test FTT embedding with type-based grouping."""
        model = StepwiseEmbedding(
            dataset=self.dataset,
            embedding_type="ftt",
            group_indices=MIMIC3TLSDataset.TYPE_GROUPS_INDICES,
            aggregation="mean",
            input_dim=42,
            hidden_dim=32,
            backbone_depth=1,
            backbone_heads=1,
        )
        batch = self._get_batch()
        with torch.no_grad():
            ret = model(**batch)
        self._check_output(ret)

    def test_linear_embedding(self):
        """Test linear embedding without grouping."""
        model = StepwiseEmbedding(
            dataset=self.dataset,
            embedding_type="linear",
            input_dim=42,
            hidden_dim=32,
            backbone_depth=1,
            backbone_heads=1,
        )
        batch = self._get_batch()
        with torch.no_grad():
            ret = model(**batch)
        self._check_output(ret)

    def test_mlp_embedding(self):
        """Test MLP embedding without grouping."""
        model = StepwiseEmbedding(
            dataset=self.dataset,
            embedding_type="mlp",
            input_dim=42,
            hidden_dim=32,
            backbone_depth=1,
            backbone_heads=1,
        )
        batch = self._get_batch()
        with torch.no_grad():
            ret = model(**batch)
        self._check_output(ret)

    def test_concat_aggregation(self):
        """Test concat aggregation with grouped features."""
        model = StepwiseEmbedding(
            dataset=self.dataset,
            embedding_type="linear",
            group_indices=MIMIC3TLSDataset.TYPE_GROUPS_INDICES,
            aggregation="concat",
            input_dim=42,
            hidden_dim=32,
            backbone_depth=1,
            backbone_heads=1,
        )
        batch = self._get_batch()
        with torch.no_grad():
            ret = model(**batch)
        self._check_output(ret)

    def test_attention_cls_aggregation(self):
        """Test attention_cls aggregation with organ groups."""
        model = StepwiseEmbedding(
            dataset=self.dataset,
            embedding_type="linear",
            group_indices=MIMIC3TLSDataset.ORGAN_GROUPS_INDICES,
            aggregation="attention_cls",
            input_dim=42,
            hidden_dim=32,
            backbone_depth=1,
            backbone_heads=1,
        )
        batch = self._get_batch()
        with torch.no_grad():
            ret = model(**batch)
        self._check_output(ret)

    def test_backward_pass(self):
        """Test that gradients flow through the model."""
        model = StepwiseEmbedding(
            dataset=self.dataset,
            embedding_type="ftt",
            input_dim=42,
            hidden_dim=32,
            backbone_depth=1,
            backbone_heads=1,
        )
        batch = self._get_batch()
        ret = model(**batch)
        ret["loss"].backward()

        has_gradient = any(
            p.grad is not None
            for p in model.parameters()
            if p.requires_grad
        )
        self.assertTrue(
            has_gradient,
            "No parameters have gradients after backward pass",
        )

    def test_custom_hyperparameters(self):
        """Test model with small custom hyperparameters."""
        model = StepwiseEmbedding(
            dataset=self.dataset,
            embedding_type="ftt",
            input_dim=42,
            hidden_dim=16,
            backbone_depth=2,
            backbone_heads=2,
            dropout=0.2,
            ff_hidden_mult=4,
        )
        self.assertEqual(model.hidden_dim, 16)
        batch = self._get_batch()
        with torch.no_grad():
            ret = model(**batch)
        self._check_output(ret)

    def test_model_initialization_attributes(self):
        """Test that model attributes are set correctly."""
        model = StepwiseEmbedding(
            dataset=self.dataset,
            embedding_type="ftt",
            input_dim=42,
            hidden_dim=64,
        )
        self.assertEqual(model.hidden_dim, 64)
        self.assertEqual(model.input_dim, 42)
        self.assertEqual(model.feature_key, "time_series")
        self.assertEqual(model.label_key, "ihm")
        self.assertIsNotNone(model.stepwise)
        self.assertIsNotNone(model.backbone)
        self.assertIsNotNone(model.fc)

    def test_backbone_only_no_stepwise(self):
        """When embedding_type is None, stepwise should be None."""
        model = StepwiseEmbedding(
            dataset=self.dataset,
            embedding_type=None,
            input_dim=42,
            hidden_dim=32,
        )
        self.assertIsNone(model.stepwise)


class TestHelperLayers(unittest.TestCase):
    """Test individual helper layer components."""

    def test_linear_embedding_shape(self):
        """LinearEmbedding should produce correct output shape."""
        layer = LinearEmbedding(input_dim=42, hidden_dim=32)
        x = torch.randn(2, 48, 42)
        out = layer(x)
        self.assertEqual(out.shape, (2, 48, 32))

    def test_mlp_embedding_shape(self):
        """MLPEmbedding should produce correct output shape."""
        layer = MLPEmbedding(
            input_dim=42, hidden_dim=32, latent_dim=16, depth=2,
        )
        x = torch.randn(2, 48, 42)
        out = layer(x)
        self.assertEqual(out.shape, (2, 48, 32))

    def test_ftt_embedding_shape(self):
        """FTTransformerEmbedding should produce correct output shape."""
        layer = FTTransformerEmbedding(
            input_dim=10, hidden_dim=32, token_dim=8, n_heads=2,
        )
        x = torch.randn(2, 48, 10)
        out = layer(x)
        self.assertEqual(out.shape, (2, 48, 32))

    def test_transformer_backbone_shape(self):
        """TransformerBackbone should produce correct output shape."""
        backbone = TransformerBackbone(
            input_dim=32, hidden_dim=64, heads=2, depth=2,
        )
        x = torch.randn(2, 48, 32)
        out = backbone(x)
        self.assertEqual(out.shape, (2, 48, 64))

    def test_stepwise_layer_no_grouping(self):
        """StepwiseEmbeddingLayer without grouping."""
        layer = StepwiseEmbeddingLayer(
            input_dim=42, hidden_dim=32, embedding_type="linear",
        )
        x = torch.randn(2, 48, 42)
        out = layer(x)
        self.assertEqual(out.shape, (2, 48, 32))

    def test_stepwise_layer_with_grouping_mean(self):
        """StepwiseEmbeddingLayer with grouping and mean aggregation."""
        groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
        layer = StepwiseEmbeddingLayer(
            input_dim=10,
            hidden_dim=16,
            group_indices=groups,
            embedding_type="linear",
            aggregation="mean",
        )
        x = torch.randn(2, 48, 10)
        out = layer(x)
        self.assertEqual(out.shape, (2, 48, 16))

    def test_stepwise_layer_with_grouping_concat(self):
        """StepwiseEmbeddingLayer with concat aggregation."""
        groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        layer = StepwiseEmbeddingLayer(
            input_dim=10,
            hidden_dim=16,
            group_indices=groups,
            embedding_type="linear",
            aggregation="concat",
        )
        x = torch.randn(2, 48, 10)
        out = layer(x)
        self.assertEqual(out.shape, (2, 48, 16))


if __name__ == "__main__":
    unittest.main()
