"""Test cases for the GRASP model.

Description:
    Unit tests for the GRASP model implementation covering initialization,
    forward pass, backward pass, embedding extraction, and custom configs.
"""

import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import GRASP


class TestGRASP(unittest.TestCase):
    """Test cases for the GRASP model."""

    def setUp(self):
        """Set up test data and model."""
        torch.manual_seed(42)
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
                "procedures": ["proc-1", "proc-2"],
                "label": 0,
            },
            {
                "patient_id": "patient-0",
                "visit_id": "visit-1",
                "conditions": ["cond-33", "cond-86", "cond-80"],
                "procedures": ["proc-3", "proc-4", "proc-5"],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-0",
                "conditions": ["cond-12", "cond-45"],
                "procedures": ["proc-1"],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": ["cond-80", "cond-12", "cond-33", "cond-45"],
                "procedures": ["proc-2", "proc-3"],
                "label": 1,
            },
        ]

        self.input_schema = {
            "conditions": "sequence",
            "procedures": "sequence",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = GRASP(
            dataset=self.dataset,
            embedding_dim=16,
            hidden_dim=16,
            cluster_num=2,
        )

    def test_model_initialization(self):
        """Test that the GRASP model initializes correctly."""
        self.assertIsInstance(self.model, GRASP)
        self.assertEqual(self.model.embedding_dim, 16)
        self.assertEqual(self.model.hidden_dim, 16)
        self.assertEqual(len(self.model.feature_keys), 2)
        self.assertIn("conditions", self.model.feature_keys)
        self.assertIn("procedures", self.model.feature_keys)
        self.assertEqual(self.model.label_key, "label")

    def test_model_forward(self):
        """Test that the GRASP model forward pass works correctly."""
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
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertFalse(torch.isnan(ret["loss"]))

    def test_model_backward(self):
        """Test that the GRASP model backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(
            has_gradient, "No parameters have gradients after backward pass"
        )

    def test_model_with_embedding(self):
        """Test that the GRASP model returns embeddings when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)
        expected_embed_dim = (
            len(self.model.feature_keys) * self.model.hidden_dim
        )
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)

    def test_custom_hyperparameters(self):
        """Test GRASP model with custom hyperparameters."""
        torch.manual_seed(42)
        model = GRASP(
            dataset=self.dataset,
            embedding_dim=4,
            hidden_dim=4,
            cluster_num=2,
            block="GRU",
            dropout=0.3,
        )

        self.assertEqual(model.embedding_dim, 4)
        self.assertEqual(model.hidden_dim, 4)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_lstm_backbone(self):
        """Test GRASP model with LSTM backbone."""
        model = GRASP(
            dataset=self.dataset,
            embedding_dim=16,
            hidden_dim=16,
            cluster_num=2,
            block="LSTM",
        )

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertFalse(torch.isnan(ret["loss"]))


    def test_batch_smaller_than_cluster_num(self):
        """Test GRASP handles batch_size < cluster_num without crashing.

        Regression test: random_init called random.sample(range(num_points),
        num_centers) which raises ValueError when num_centers > num_points.
        Fixed by clamping: num_centers = min(num_centers, num_points).
        """
        model = GRASP(
            dataset=self.dataset,
            embedding_dim=16,
            hidden_dim=16,
            cluster_num=4,  # more clusters than batch_size=1
            block="GRU",
        )

        train_loader = get_dataloader(self.dataset, batch_size=1, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertEqual(ret["y_prob"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
