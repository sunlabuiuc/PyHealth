"""Test cases for the GRASP model."""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.grasp import GRASP


class TestGRASP(unittest.TestCase):
    """Test cases for the GRASP model."""

    def setUp(self):
        """Set up test data and model."""
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

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={
                "conditions": "sequence",
                "procedures": "sequence",
            },
            output_schema={"label": "binary"},
            dataset_name="test",
        )

        self.model = GRASP(
            dataset=self.dataset,
            static_key="demographic",
            embedding_dim=32,
            hidden_dim=32,
            cluster_num=2,
        )

    def test_model_initialization(self):
        """Test that the GRASP model initializes correctly."""
        self.assertIsInstance(self.model, GRASP)
        self.assertEqual(self.model.embedding_dim, 32)
        self.assertEqual(self.model.hidden_dim, 32)
        self.assertEqual(self.model.static_key, "demographic")
        self.assertEqual(self.model.static_dim, 3)
        self.assertEqual(len(self.model.dynamic_feature_keys), 2)
        self.assertEqual(self.model.label_key, "label")

    def test_model_without_static(self):
        """Test GRASP initializes correctly without static features."""
        model = GRASP(
            dataset=self.dataset,
            embedding_dim=32,
            hidden_dim=32,
            cluster_num=2,
        )
        self.assertEqual(model.static_dim, 0)
        self.assertIsNone(model.static_key)

    def test_forward(self):
        """Test that forward pass produces correct output keys and shapes."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["y_true"].shape[0], 2)

    def test_backward(self):
        """Test that backward pass computes gradients."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_gradient)

    def test_embed_extraction(self):
        """Test that embeddings are returned when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        expected_dim = len(self.model.dynamic_feature_keys) * self.model.hidden_dim
        self.assertEqual(ret["embed"].shape[1], expected_dim)

    def test_gru_backbone(self):
        """Test GRASP with GRU backbone."""
        model = GRASP(
            dataset=self.dataset,
            embedding_dim=32,
            hidden_dim=32,
            cluster_num=2,
            block="GRU",
        )
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)
        self.assertIn("loss", ret)

    def test_lstm_backbone(self):
        """Test GRASP with LSTM backbone."""
        model = GRASP(
            dataset=self.dataset,
            embedding_dim=32,
            hidden_dim=32,
            cluster_num=2,
            block="LSTM",
        )
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=True)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)
        self.assertIn("loss", ret)


if __name__ == "__main__":
    unittest.main()
