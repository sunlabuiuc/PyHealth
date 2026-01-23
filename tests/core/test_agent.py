"""Test cases for the Dr. Agent model.

Author: REDACTED_AUTHOR
"""

import unittest

import torch

from pyhealth.datasets import get_dataloader, create_sample_dataset
from pyhealth.models.agent import Agent



class TestAgent(unittest.TestCase):
    """Test cases for the Agent model."""

    def setUp(self):
        """Set up test data and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "conditions": [["A05B", "A05C"], ["A06A"], ["A11D"]],  # 3 visits
                "procedures": [["P1", "P2"], ["P3"], []],
                "demographic": [65.0, 1.0, 25.5],
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "conditions": [["B01", "B02"], ["B03"]],  # 2 visits
                "procedures": [["P4"], ["P5"]],
                "demographic": [45.0, 0.0, 22.1],
                "label": 0,
            },
        ]

        self.input_schema = {
            "conditions": "nested_sequence",
            "procedures": "nested_sequence",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = Agent(
            dataset=self.dataset,
            static_key="demographic",
            embedding_dim=64,
            hidden_dim=64,
        )

    def test_model_initialization(self):
        """Test that the Agent model initializes correctly."""
        self.assertIsInstance(self.model, Agent)
        self.assertEqual(self.model.embedding_dim, 64)
        self.assertEqual(self.model.hidden_dim, 64)
        self.assertEqual(self.model.static_key, "demographic")
        self.assertEqual(self.model.static_dim, 3)
        self.assertEqual(len(self.model.seq_feature_keys), 2)
        self.assertIn("conditions", self.model.seq_feature_keys)
        self.assertIn("procedures", self.model.seq_feature_keys)
        self.assertEqual(self.model.label_key, "label")
        self.assertTrue(self.model.use_baseline)

    def test_model_forward(self):
        """Test that the Agent model forward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
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

    def test_model_backward(self):
        """Test that the Agent model backward pass works correctly."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        ret = self.model(**data_batch)
        ret["loss"].backward()

        has_gradient = False
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradient = True
                break
        self.assertTrue(
            has_gradient, "No parameters have gradients after backward pass"
        )

    def test_model_with_embedding(self):
        """Test that the Agent model returns embeddings when requested."""
        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))
        data_batch["embed"] = True

        with torch.no_grad():
            ret = self.model(**data_batch)

        self.assertIn("embed", ret)
        self.assertEqual(ret["embed"].shape[0], 2)
        expected_embed_dim = len(self.model.seq_feature_keys) * self.model.hidden_dim
        self.assertEqual(ret["embed"].shape[1], expected_embed_dim)

    def test_model_without_static(self):
        """Test Agent model without static features."""
        model = Agent(
            dataset=self.dataset,
            static_key=None,
            embedding_dim=64,
            hidden_dim=64,
        )

        self.assertIsNone(model.static_key)
        self.assertEqual(model.static_dim, 0)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_model_without_baseline(self):
        """Test Agent model without baseline for RL."""
        model = Agent(
            dataset=self.dataset,
            static_key="demographic",
            use_baseline=False,
            embedding_dim=64,
            hidden_dim=64,
        )

        self.assertFalse(model.use_baseline)

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        ret = model(**data_batch)
        ret["loss"].backward()

        self.assertIn("loss", ret)

    def test_custom_hyperparameters(self):
        """Test Agent model with custom hyperparameters."""
        model = Agent(
            dataset=self.dataset,
            embedding_dim=32,
            hidden_dim=48,
            static_key="demographic",
            n_actions=5,
            n_units=32,
            dropout=0.3,
            lamda=0.7,
            cell="lstm",
        )

        self.assertEqual(model.embedding_dim, 32)
        self.assertEqual(model.hidden_dim, 48)

        # Check agent layer parameters
        agent_layer = model.agent["conditions"]
        self.assertEqual(agent_layer.n_actions, 5)
        self.assertEqual(agent_layer.n_units, 32)
        self.assertEqual(agent_layer.dropout, 0.3)
        self.assertEqual(agent_layer.lamda, 0.7)
        self.assertEqual(agent_layer.cell, "lstm")

        train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)

    def test_multiclass_mode(self):
        """Test Agent model with multiclass classification."""
        samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": [["A01", "A02"], ["A03"]],
                "label": 0,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "conditions": [["B01"]],
                "label": 1,
            },
            {
                "patient_id": "p2",
                "visit_id": "v2",
                "conditions": [["C01", "C02"], ["C03"]],
                "label": 2,
            },
        ]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "nested_sequence"},
            output_schema={"label": "multiclass"},
            dataset_name="test_multiclass",
        )
        model = Agent(dataset=dataset, embedding_dim=32, hidden_dim=32)

        train_loader = get_dataloader(dataset, batch_size=3, shuffle=False)
        data_batch = next(iter(train_loader))

        with torch.no_grad():
            ret = model(**data_batch)

        self.assertEqual(ret["y_prob"].shape, (3, 3))
        self.assertEqual(ret["logit"].shape, (3, 3))
        self.assertIn("loss", ret)
        
        # Verify y_prob sums to 1 along class dimension (softmax property)
        self.assertTrue(
            torch.allclose(ret["y_prob"].sum(dim=1), torch.ones(3), atol=1e-5)
        )


class TestAgentLayer(unittest.TestCase):
    """Test cases for the AgentLayer module."""

    def setUp(self):
        """Set up test data."""
        from pyhealth.models.agent import AgentLayer

        self.layer = AgentLayer(
            input_dim=64,
            static_dim=12,
            n_hidden=128,
            n_actions=10,
        )

    def test_layer_initialization(self):
        """Test AgentLayer initialization."""
        self.assertEqual(self.layer.input_dim, 64)
        self.assertEqual(self.layer.static_dim, 12)
        self.assertEqual(self.layer.n_hidden, 128)
        self.assertEqual(self.layer.n_actions, 10)
        self.assertEqual(self.layer.cell, "gru")

    def test_layer_forward(self):
        """Test AgentLayer forward pass."""
        batch_size, seq_len = 4, 20
        x = torch.randn(batch_size, seq_len, 64)
        static = torch.randn(batch_size, 12)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        last_out, all_out = self.layer(x, static=static, mask=mask)

        self.assertEqual(last_out.shape, (batch_size, 128))
        self.assertEqual(all_out.shape, (batch_size, seq_len, 128))

    def test_layer_without_static(self):
        """Test AgentLayer without static features."""
        from pyhealth.models.agent import AgentLayer

        layer = AgentLayer(input_dim=64, static_dim=0, n_hidden=128)

        x = torch.randn(4, 20, 64)
        last_out, all_out = layer(x)

        self.assertEqual(last_out.shape, (4, 128))
        self.assertEqual(all_out.shape, (4, 20, 128))

    def test_layer_lstm_cell(self):
        """Test AgentLayer with LSTM cell."""
        from pyhealth.models.agent import AgentLayer

        layer = AgentLayer(input_dim=64, static_dim=0, n_hidden=128, cell="lstm")

        x = torch.randn(4, 20, 64)
        last_out, all_out = layer(x)

        self.assertEqual(last_out.shape, (4, 128))
        self.assertEqual(all_out.shape, (4, 20, 128))

    def test_invalid_cell_type(self):
        """Test that invalid cell type raises error."""
        from pyhealth.models.agent import AgentLayer

        with self.assertRaises(ValueError):
            AgentLayer(input_dim=64, cell="invalid")


if __name__ == "__main__":
    unittest.main()