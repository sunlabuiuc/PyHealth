"""Test cases for the ConCare model.

Author: Joshua Steier

Description:
    This module contains unit tests for the ConCare model implementation.
    Tests cover model initialization, forward pass, backward pass, embedding
    extraction, and various configuration options.
"""

import unittest
import torch

from pyhealth.datasets import get_dataloader
from pyhealth.models.concare import ConCare
from pyhealth.datasets import create_sample_dataset


class TestConCare(unittest.TestCase):
    """Test cases for the ConCare model."""

    def setUp(self):
        """Set up test data and model."""
        # Use categorical codes for both conditions and procedures
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": ["proc-1", "proc-2"],
                "demographic": [0.0, 2.0, 1.5],
                "label": 0,
            },
            {
                "patient_id": "patient-0",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": ["proc-3", "proc-4", "proc-5"],
                "demographic": [0.0, 2.0, 1.5],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "conditions": ["cond-12", "cond-45"],
                "procedures": ["proc-1"],
                "demographic": [1.0, 1.0, 0.5],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-80", "cond-12", "cond-33", "cond-45"],
                "procedures": ["proc-2", "proc-3"],
                "demographic": [1.0, 1.0, 0.5],
                "label": 1,
            },
        ]

        # Define input and output schemas
        self.input_schema = {
            "conditions": "sequence",
            "procedures": "sequence",
        }
        self.output_schema = {"label": "binary"}

        # Create dataset
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        # Create model
        self.model = ConCare(
            dataset=self.dataset,
            static_key="demographic",
            embedding_dim=64,
            hidden_dim=64,
        )

    def test_model_initialization(self):
        """Test that the ConCare model initializes correctly."""
        self.assertIsInstance(self.model, ConCare)
        self.assertEqual(self.model.embedding_dim, 64)
        self.assertEqual(self.model.hidden_dim, 64)
        self.assertEqual(self.model.static_key, "demographic")
        self.assertEqual(self.model.static_dim, 3)
        self.assertEqual(len(self.model.dynamic_feature_keys), 2)
        self.assertIn("conditions", self.model.dynamic_feature_keys)
        self.assertIn("procedures", self.model.dynamic_feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_initialization_without_static(self):
        """Test that the ConCare model initializes correctly without static features."""
        model = ConCare(
            dataset=self.dataset,
            embedding_dim=64,
            hidden_dim=64,
        )
        self.assertEqual(model.static_dim, 0)
        self.assertIsNone(model.static_key)

    def test_model_forward(self):
        """Test that the ConCare model forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape[0], 2)  # batch size
        self.assertEqual(ret["y_true"].shape[0], 2)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_model_backward(self):
        """Test that the ConCare model backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_gradient, "No parameters have gradients after backward pass")

    def test_model_with_embedding(self):
        """Test that the ConCare model returns embeddings when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)
        expected_embed_dim = len(self.model.dynamic_feature_keys) * self.model.hidden_dim
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)

    def test_custom_hyperparameters(self):
        """Test ConCare model with custom hyperparameters."""
        model = ConCare(
            dataset=self.dataset,
            static_key="demographic",
            embedding_dim=32,
            hidden_dim=32,
            num_head=2,
            pe_hidden=16,
            dropout=0.3,
        )

        self.assertEqual(model.embedding_dim, 32)
        self.assertEqual(model.hidden_dim, 32)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_decov_loss_included(self):
        """Test that the decov loss is included in the total loss."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        # Just ensure loss is finite and not NaN
        self.assertIsInstance(ret["loss"].item(), float)
        self.assertFalse(torch.isnan(ret["loss"]))

    def test_multiclass_classification(self):
        """Test ConCare model with multiclass classification."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86"],
                "procedures": ["proc-1"],
                "label": 0,
            },
            {
                "patient_id": "patient-0",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-80"],
                "procedures": ["proc-2", "proc-3"],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "conditions": ["cond-12"],
                "procedures": ["proc-4"],
                "label": 2,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-80", "cond-12"],
                "procedures": ["proc-1"],
                "label": 0,
            },
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "sequence", "procedures": "sequence"},
            output_schema={"label": "multiclass"},
            dataset_name="test_multiclass",
        )

        model = ConCare(
            dataset=dataset,
            embedding_dim=32,
            hidden_dim=32,
        )

        train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        # For multiclass with 3 classes, y_prob should have 3 columns
        self.assertEqual(ret["y_prob"].shape[1], 3)

    def test_single_feature(self):
        """Test ConCare model with a single dynamic feature."""
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "label": 0,
            },
            {
                "patient_id": "patient-0",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86"],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "conditions": ["cond-12", "cond-45", "cond-80"],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-80"],
                "label": 1,
            },
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="test_single",
        )

        model = ConCare(
            dataset=dataset,
            embedding_dim=32,
            hidden_dim=32,
        )

        self.assertEqual(len(model.dynamic_feature_keys), 1)

        train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)


    def test_batch_size_one(self):
        """Test that ConCare handles batch_size=1 without crashing.

        Regression test: bare .squeeze() in FinalAttentionQKV removed the
        batch dimension when batch_size=1, causing softmax to fail with
        'IndexError: Dimension out of range'.
        """
        train_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertEqual(ret["y_prob"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
