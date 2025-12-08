"""Test cases for BulkRNABert models.

This module provides unit tests for the BulkRNABert cancer prognosis models,
including classification and survival prediction variants.
"""
import unittest

import torch

from pyhealth.models import BulkRNABert, BulkRNABertLayer, BulkRNABertForSurvival
from pyhealth.models.bulkrnabert import compute_c_index, TCGA_CANCER_TYPES


class TestBulkRNABertLayer(unittest.TestCase):
    """Test cases for the BulkRNABertLayer encoder."""

    def setUp(self):
        """Set up test data."""
        self.batch_size = 4
        self.input_dim = 1000  # Reduced for testing
        self.hidden_dim = 128
        self.layer = BulkRNABertLayer(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        )

    def test_layer_initialization(self):
        """Test that the layer initializes correctly."""
        self.assertIsInstance(self.layer, BulkRNABertLayer)
        self.assertEqual(self.layer.input_dim, self.input_dim)
        self.assertEqual(self.layer.hidden_dim, self.hidden_dim)

    def test_layer_forward(self):
        """Test layer forward pass."""
        x = torch.randn(self.batch_size, self.input_dim)
        with torch.no_grad():
            output = self.layer(x)

        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))

    def test_layer_forward_with_mask(self):
        """Test layer forward pass with mask."""
        x = torch.randn(self.batch_size, self.input_dim)
        # Note: mask is not typically used for single-gene input
        with torch.no_grad():
            output = self.layer(x, mask=None)

        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))


class TestBulkRNABert(unittest.TestCase):
    """Test cases for the BulkRNABert classification model."""

    def setUp(self):
        """Set up test data and model."""
        self.batch_size = 4
        self.input_dim = 1000
        self.num_classes = 10
        self.model = BulkRNABert(
            dataset=None,
            feature_keys=["gene_expression"],
            label_key="cancer_type",
            mode="multiclass",
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        )
        # Manually set output size for testing without dataset
        self.model.output_layer[-1] = torch.nn.Linear(64, self.num_classes)

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, BulkRNABert)
        self.assertEqual(self.model.mode, "multiclass")
        self.assertEqual(self.model.feature_keys, ["gene_expression"])
        self.assertEqual(self.model.label_key, "cancer_type")

    def test_model_forward_without_labels(self):
        """Test forward pass without labels."""
        gene_expression = torch.randn(self.batch_size, self.input_dim)

        with torch.no_grad():
            outputs = self.model(gene_expression=gene_expression)

        self.assertIn("logits", outputs)
        self.assertIn("y_prob", outputs)
        self.assertIn("embeddings", outputs)
        self.assertNotIn("loss", outputs)

        self.assertEqual(outputs["logits"].shape, (self.batch_size, self.num_classes))
        self.assertEqual(outputs["y_prob"].shape, (self.batch_size, self.num_classes))
        self.assertEqual(outputs["embeddings"].shape, (self.batch_size, 128))

    def test_model_forward_with_labels(self):
        """Test forward pass with labels."""
        gene_expression = torch.randn(self.batch_size, self.input_dim)
        labels = torch.randint(0, self.num_classes, (self.batch_size,))

        with torch.no_grad():
            outputs = self.model(gene_expression=gene_expression, labels=labels)

        self.assertIn("loss", outputs)
        self.assertEqual(outputs["loss"].dim(), 0)  # Scalar loss

    def test_model_backward(self):
        """Test backward pass."""
        gene_expression = torch.randn(self.batch_size, self.input_dim)
        labels = torch.randint(0, self.num_classes, (self.batch_size,))

        outputs = self.model(gene_expression=gene_expression, labels=labels)
        outputs["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_model_binary_mode(self):
        """Test model in binary classification mode."""
        model = BulkRNABert(
            dataset=None,
            feature_keys=["gene_expression"],
            label_key="is_cancer",
            mode="binary",
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
        )

        gene_expression = torch.randn(self.batch_size, self.input_dim)
        labels = torch.randint(0, 2, (self.batch_size, 1)).float()

        with torch.no_grad():
            outputs = model(gene_expression=gene_expression, labels=labels)

        self.assertIn("loss", outputs)
        # Binary mode uses sigmoid
        self.assertTrue(torch.all(outputs["y_prob"] >= 0))
        self.assertTrue(torch.all(outputs["y_prob"] <= 1))

    def test_model_kwargs_input(self):
        """Test model with inputs via kwargs."""
        gene_expression = torch.randn(self.batch_size, self.input_dim)
        labels = torch.randint(0, self.num_classes, (self.batch_size,))

        with torch.no_grad():
            outputs = self.model(
                gene_expression=gene_expression,
                cancer_type=labels,
            )

        self.assertIn("logits", outputs)


class TestBulkRNABertForSurvival(unittest.TestCase):
    """Test cases for the BulkRNABertForSurvival model."""

    def setUp(self):
        """Set up test data and model."""
        self.batch_size = 8
        self.input_dim = 1000
        self.model = BulkRNABertForSurvival(
            dataset=None,
            feature_keys=["gene_expression"],
            time_key="survival_time",
            event_key="event",
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
        )

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, BulkRNABertForSurvival)
        self.assertEqual(self.model.mode, "regression")
        self.assertEqual(self.model.time_key, "survival_time")
        self.assertEqual(self.model.event_key, "event")

    def test_model_forward_without_survival_data(self):
        """Test forward pass without survival data."""
        gene_expression = torch.randn(self.batch_size, self.input_dim)

        with torch.no_grad():
            outputs = self.model(gene_expression=gene_expression)

        self.assertIn("risk_scores", outputs)
        self.assertIn("y_prob", outputs)
        self.assertIn("embeddings", outputs)
        self.assertNotIn("loss", outputs)

        self.assertEqual(outputs["risk_scores"].shape, (self.batch_size, 1))

    def test_model_forward_with_survival_data(self):
        """Test forward pass with survival data."""
        gene_expression = torch.randn(self.batch_size, self.input_dim)
        survival_time = torch.rand(self.batch_size) * 100 + 1  # Random times > 0
        event = torch.randint(0, 2, (self.batch_size,)).float()

        with torch.no_grad():
            outputs = self.model(
                gene_expression=gene_expression,
                survival_time=survival_time,
                event=event,
            )

        self.assertIn("loss", outputs)
        self.assertEqual(outputs["loss"].dim(), 0)

    def test_model_backward(self):
        """Test backward pass."""
        gene_expression = torch.randn(self.batch_size, self.input_dim)
        survival_time = torch.rand(self.batch_size) * 100 + 1
        event = torch.randint(0, 2, (self.batch_size,)).float()

        outputs = self.model(
            gene_expression=gene_expression,
            survival_time=survival_time,
            event=event,
        )
        outputs["loss"].backward()

        has_gradient = any(
            param.requires_grad and param.grad is not None
            for param in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_cox_loss_no_events(self):
        """Test Cox loss when no events occur (all censored)."""
        gene_expression = torch.randn(self.batch_size, self.input_dim)
        survival_time = torch.rand(self.batch_size) * 100 + 1
        event = torch.zeros(self.batch_size)  # All censored

        with torch.no_grad():
            outputs = self.model(
                gene_expression=gene_expression,
                survival_time=survival_time,
                event=event,
            )

        # Loss should be 0 when no events
        self.assertEqual(outputs["loss"].item(), 0.0)


class TestComputeCIndex(unittest.TestCase):
    """Test cases for the C-index computation."""

    def test_perfect_ranking(self):
        """Test C-index with perfect ranking."""
        # Higher risk should have shorter survival
        risk_scores = torch.tensor([3.0, 2.0, 1.0])
        survival_times = torch.tensor([1.0, 2.0, 3.0])
        events = torch.tensor([1.0, 1.0, 1.0])

        c_index = compute_c_index(risk_scores, survival_times, events)
        self.assertEqual(c_index, 1.0)

    def test_inverse_ranking(self):
        """Test C-index with inverse ranking."""
        # Lower risk has shorter survival (worst case)
        risk_scores = torch.tensor([1.0, 2.0, 3.0])
        survival_times = torch.tensor([1.0, 2.0, 3.0])
        events = torch.tensor([1.0, 1.0, 1.0])

        c_index = compute_c_index(risk_scores, survival_times, events)
        self.assertEqual(c_index, 0.0)

    def test_random_ranking(self):
        """Test C-index with tied risk scores."""
        risk_scores = torch.tensor([1.0, 1.0, 1.0])
        survival_times = torch.tensor([1.0, 2.0, 3.0])
        events = torch.tensor([1.0, 1.0, 1.0])

        c_index = compute_c_index(risk_scores, survival_times, events)
        # With all tied scores, C-index should be 0.5
        self.assertEqual(c_index, 0.5)

    def test_with_censoring(self):
        """Test C-index with censored observations."""
        risk_scores = torch.tensor([3.0, 2.0, 1.0, 0.0])
        survival_times = torch.tensor([1.0, 2.0, 3.0, 4.0])
        events = torch.tensor([1.0, 0.0, 1.0, 0.0])  # 2 censored

        c_index = compute_c_index(risk_scores, survival_times, events)
        # Should still be able to compute with partial data
        self.assertGreaterEqual(c_index, 0.0)
        self.assertLessEqual(c_index, 1.0)

    def test_all_censored(self):
        """Test C-index when all observations are censored."""
        risk_scores = torch.tensor([3.0, 2.0, 1.0])
        survival_times = torch.tensor([1.0, 2.0, 3.0])
        events = torch.tensor([0.0, 0.0, 0.0])

        c_index = compute_c_index(risk_scores, survival_times, events)
        # With no events, return 0.5 (no information)
        self.assertEqual(c_index, 0.5)


class TestTCGACancerTypes(unittest.TestCase):
    """Test cases for TCGA cancer type constants."""

    def test_cancer_types_count(self):
        """Test that we have 33 TCGA cancer types."""
        self.assertEqual(len(TCGA_CANCER_TYPES), 33)

    def test_common_cancer_types_present(self):
        """Test that common cancer types are present."""
        common_types = ["BRCA", "LUAD", "KIRC", "GBM", "PRAD"]
        for cancer_type in common_types:
            self.assertIn(cancer_type, TCGA_CANCER_TYPES)


if __name__ == "__main__":
    unittest.main()
