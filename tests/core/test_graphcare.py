# Author: Josh Steier
# Description: Tests for GraphCare model

"""Test cases for GraphCare and BiAttentionGNNConv.

Run with: python -m pytest test_graphcare.py -v

Note: Requires torch-geometric to be installed.
"""

import random
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models.graphcare import (
    GraphCare,
    BiAttentionGNNConv,
    _check_torch_geometric,
)

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import global_mean_pool

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


def _make_fake_batch(
    batch_size=4,
    num_nodes=200,
    num_rels=30,
    max_visit=5,
    nodes_per_graph_range=(10, 30),
):
    """Create a fake PyG batch for testing.

    Returns:
        Tuple of (batched_data, node_ids, rel_ids, edge_index, batch_vec,
        visit_node, ehr_nodes, labels).
    """
    graphs = []
    for _ in range(batch_size):
        n = random.randint(*nodes_per_graph_range)
        e = random.randint(n, n * 3)
        src = torch.randint(0, n, (e,))
        dst = torch.randint(0, n, (e,))
        y = torch.randint(0, num_nodes, (n,))
        relation = torch.randint(0, num_rels, (e,))

        vpn = torch.zeros(max_visit, num_nodes)
        for v in range(max_visit):
            active = torch.randint(0, num_nodes, (random.randint(1, 10),))
            vpn[v, active] = 1.0

        ehr = torch.zeros(num_nodes)
        ehr[torch.randint(0, num_nodes, (5,))] = 1.0

        data = Data(
            edge_index=torch.stack([src, dst]),
            y=y,
            relation=relation,
            visit_padded_node=vpn,
            ehr_nodes=ehr,
            label=torch.tensor([1.0]),
        )
        data.num_nodes = n
        graphs.append(data)

    batched = Batch.from_data_list(graphs)
    node_ids = batched.y
    rel_ids = batched.relation
    edge_index = batched.edge_index
    batch_vec = batched.batch
    visit_node = batched.visit_padded_node.reshape(batch_size, max_visit, num_nodes)
    ehr_nodes = batched.ehr_nodes.reshape(batch_size, num_nodes)
    labels = batched.label.reshape(batch_size, -1)

    return node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, labels


@unittest.skipUnless(HAS_TORCH_GEOMETRIC, "torch-geometric not installed")
class TestBiAttentionGNNConv(unittest.TestCase):
    """Test cases for the BiAttentionGNNConv layer."""

    def test_forward_with_edge_attn(self):
        """Test BAT conv with edge attention enabled."""
        hidden_dim = 32
        conv = BiAttentionGNNConv(
            nn.Linear(hidden_dim, hidden_dim),
            edge_dim=hidden_dim,
            edge_attn=True,
        )

        num_nodes, num_edges = 20, 40
        x = torch.randn(num_nodes, hidden_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, hidden_dim)
        attn = torch.randn(num_edges, 1)

        out, w_rel = conv(x, edge_index, edge_attr, attn=attn)

        self.assertEqual(out.shape, (num_nodes, hidden_dim))
        self.assertIsNotNone(w_rel)
        self.assertEqual(w_rel.shape, (num_edges, 1))

    def test_forward_without_edge_attn(self):
        """Test BAT conv with edge attention disabled."""
        hidden_dim = 32
        conv = BiAttentionGNNConv(
            nn.Linear(hidden_dim, hidden_dim),
            edge_dim=hidden_dim,
            edge_attn=False,
        )

        num_nodes, num_edges = 20, 40
        x = torch.randn(num_nodes, hidden_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, hidden_dim)
        attn = torch.randn(num_edges, 1)

        out, w_rel = conv(x, edge_index, edge_attr, attn=attn)

        self.assertEqual(out.shape, (num_nodes, hidden_dim))
        self.assertIsNone(w_rel)

    def test_gradient_flow(self):
        """Test gradients flow through BAT conv."""
        hidden_dim = 16
        conv = BiAttentionGNNConv(
            nn.Linear(hidden_dim, hidden_dim),
            edge_dim=hidden_dim,
            edge_attn=True,
        )

        x = torch.randn(10, hidden_dim, requires_grad=True)
        edge_index = torch.randint(0, 10, (2, 20))
        edge_attr = torch.randn(20, hidden_dim)
        attn = torch.randn(20, 1)

        out, _ = conv(x, edge_index, edge_attr, attn=attn)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))

    def test_reset_parameters(self):
        """Test reset_parameters doesn't crash."""
        conv = BiAttentionGNNConv(
            nn.Linear(32, 32), edge_dim=32, edge_attn=True
        )
        conv.reset_parameters()  # Should not raise


@unittest.skipUnless(HAS_TORCH_GEOMETRIC, "torch-geometric not installed")
class TestGraphCare(unittest.TestCase):
    """Test cases for the GraphCare model."""

    NUM_NODES = 200
    NUM_RELS = 30
    MAX_VISIT = 5
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 64
    OUT_CHANNELS = 1
    BATCH_SIZE = 4

    def _make_model(self, gnn="BAT", patient_mode="joint", **kwargs):
        """Helper to create a GraphCare model with default test params."""
        defaults = dict(
            num_nodes=self.NUM_NODES,
            num_rels=self.NUM_RELS,
            max_visit=self.MAX_VISIT,
            embedding_dim=self.EMBEDDING_DIM,
            hidden_dim=self.HIDDEN_DIM,
            out_channels=self.OUT_CHANNELS,
            layers=2,
            dropout=0.5,
            decay_rate=0.01,
            node_emb=torch.randn(self.NUM_NODES, self.EMBEDDING_DIM),
            rel_emb=torch.randn(self.NUM_RELS, self.EMBEDDING_DIM),
            gnn=gnn,
            patient_mode=patient_mode,
        )
        defaults.update(kwargs)
        return GraphCare(**defaults)

    def _make_batch(self):
        """Helper to create fake batch data."""
        return _make_fake_batch(
            batch_size=self.BATCH_SIZE,
            num_nodes=self.NUM_NODES,
            num_rels=self.NUM_RELS,
            max_visit=self.MAX_VISIT,
        )

    # --- Output shape tests for all GNN × patient_mode combos ---

    def test_bat_joint_output_shape(self):
        """Test BAT/joint produces correct output shape."""
        model = self._make_model(gnn="BAT", patient_mode="joint")
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)

        self.assertEqual(logits.shape, (self.BATCH_SIZE, self.OUT_CHANNELS))

    def test_bat_graph_output_shape(self):
        """Test BAT/graph produces correct output shape."""
        model = self._make_model(gnn="BAT", patient_mode="graph")
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes=None)

        self.assertEqual(logits.shape, (self.BATCH_SIZE, self.OUT_CHANNELS))

    def test_bat_node_output_shape(self):
        """Test BAT/node produces correct output shape."""
        model = self._make_model(gnn="BAT", patient_mode="node")
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)

        self.assertEqual(logits.shape, (self.BATCH_SIZE, self.OUT_CHANNELS))

    def test_gat_joint_output_shape(self):
        """Test GAT/joint produces correct output shape."""
        model = self._make_model(gnn="GAT", patient_mode="joint")
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)

        self.assertEqual(logits.shape, (self.BATCH_SIZE, self.OUT_CHANNELS))

    def test_gin_joint_output_shape(self):
        """Test GIN/joint produces correct output shape."""
        model = self._make_model(gnn="GIN", patient_mode="joint")
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)

        self.assertEqual(logits.shape, (self.BATCH_SIZE, self.OUT_CHANNELS))

    # --- Multi-label / multi-class output ---

    def test_multilabel_output(self):
        """Test model works with multi-label output."""
        model = self._make_model(out_channels=50)
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)

        self.assertEqual(logits.shape, (self.BATCH_SIZE, 50))

    def test_multiclass_output(self):
        """Test model works with multi-class (10-way) output."""
        model = self._make_model(out_channels=10)
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)

        self.assertEqual(logits.shape, (self.BATCH_SIZE, 10))

    # --- Backward pass ---

    def test_backward_pass_bat(self):
        """Test gradients flow through full BAT model."""
        model = self._make_model(gnn="BAT", patient_mode="joint")
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.train()
        logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.ones(self.BATCH_SIZE, self.OUT_CHANNELS)
        )
        loss.backward()

        # Check at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        self.assertTrue(has_grad)

    def test_backward_pass_gat(self):
        """Test gradients flow through full GAT model."""
        model = self._make_model(gnn="GAT", patient_mode="joint")
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.train()
        logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.ones(self.BATCH_SIZE, self.OUT_CHANNELS)
        )
        loss.backward()

    def test_backward_pass_gin(self):
        """Test gradients flow through full GIN model."""
        model = self._make_model(gnn="GIN", patient_mode="joint")
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.train()
        logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.ones(self.BATCH_SIZE, self.OUT_CHANNELS)
        )
        loss.backward()

    # --- Edge dropout ---

    def test_edge_dropout(self):
        """Test edge dropout doesn't crash and produces valid output."""
        model = self._make_model(drop_rate=0.3)
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.train()
        logits = model(
            node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes,
            in_drop=True,
        )

        self.assertEqual(logits.shape, (self.BATCH_SIZE, self.OUT_CHANNELS))
        self.assertFalse(torch.isnan(logits).any())

    def test_no_edge_dropout_at_eval(self):
        """Test edge dropout is not applied during eval."""
        model = self._make_model(drop_rate=0.5)
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            logits1 = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)
            logits2 = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)

        # Without dropout, same input → same output
        self.assertTrue(torch.allclose(logits1, logits2))

    # --- store_attn ---

    def test_store_attn(self):
        """Test store_attn returns attention weights."""
        num_layers = 2
        model = self._make_model(gnn="BAT", patient_mode="joint", layers=num_layers)
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            result = model(
                node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes,
                store_attn=True,
            )

        logits, alphas, betas, attns, edge_ws = result

        self.assertEqual(logits.shape, (self.BATCH_SIZE, self.OUT_CHANNELS))
        self.assertEqual(len(alphas), num_layers)
        self.assertEqual(len(betas), num_layers)
        self.assertEqual(len(attns), num_layers)
        self.assertEqual(len(edge_ws), num_layers)

        # Alpha shape: (batch, max_visit, num_nodes)
        self.assertEqual(alphas[0].shape, (self.BATCH_SIZE, self.MAX_VISIT, self.NUM_NODES))
        # Beta shape: (batch, max_visit, 1)
        self.assertEqual(betas[0].shape, (self.BATCH_SIZE, self.MAX_VISIT, 1))

    def test_store_attn_disabled(self):
        """Test store_attn=False returns just logits."""
        model = self._make_model(gnn="BAT")
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            result = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)

        self.assertIsInstance(result, torch.Tensor)

    # --- Attention flags ---

    def test_no_alpha(self):
        """Test model works with alpha attention disabled."""
        model = self._make_model(use_alpha=False, use_beta=True)
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)

        self.assertEqual(logits.shape, (self.BATCH_SIZE, self.OUT_CHANNELS))

    def test_no_beta(self):
        """Test model works with beta attention disabled."""
        model = self._make_model(use_alpha=True, use_beta=False)
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)

        self.assertEqual(logits.shape, (self.BATCH_SIZE, self.OUT_CHANNELS))

    def test_no_alpha_no_beta(self):
        """Test model works with both attentions disabled."""
        model = self._make_model(use_alpha=False, use_beta=False)
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)

        self.assertEqual(logits.shape, (self.BATCH_SIZE, self.OUT_CHANNELS))

    # --- Embedding options ---

    def test_learned_embeddings(self):
        """Test model works with learned (not pre-trained) embeddings."""
        model = self._make_model(node_emb=None, rel_emb=None)
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)

        self.assertEqual(logits.shape, (self.BATCH_SIZE, self.OUT_CHANNELS))

    def test_frozen_embeddings(self):
        """Test frozen pre-trained embeddings don't get gradients."""
        node_emb = torch.randn(self.NUM_NODES, self.EMBEDDING_DIM)
        model = self._make_model(node_emb=node_emb, freeze=True)
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.train()
        logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)
        loss = logits.sum()
        loss.backward()

        self.assertFalse(model.node_emb.weight.requires_grad)

    def test_attn_init(self):
        """Test attention initialization with pre-computed weights."""
        attn_init = torch.randn(self.NUM_NODES)
        model = self._make_model(attn_init=attn_init)
        node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

        model.eval()
        with torch.no_grad():
            logits = model(node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes)

        self.assertEqual(logits.shape, (self.BATCH_SIZE, self.OUT_CHANNELS))

    # --- Device movement ---

    def test_to_device(self):
        """Test lambda_j is a registered buffer and moves with the model."""
        model = self._make_model()
        model = model.to("cpu")

        # lambda_j should be a registered buffer, visible in state_dict
        self.assertIn("lambda_j", model.state_dict())
        self.assertEqual(model.lambda_j.device, torch.device("cpu"))

    # --- Invalid inputs ---

    def test_invalid_gnn(self):
        """Test invalid GNN type raises error."""
        with self.assertRaises(AssertionError):
            self._make_model(gnn="INVALID")

    def test_invalid_patient_mode(self):
        """Test invalid patient_mode raises error."""
        with self.assertRaises(AssertionError):
            self._make_model(patient_mode="INVALID")

    # --- Parameter count ---

    def test_joint_has_more_params_than_graph(self):
        """Joint mode should have more params due to wider MLP."""
        model_joint = self._make_model(patient_mode="joint")
        model_graph = self._make_model(patient_mode="graph")

        params_joint = sum(p.numel() for p in model_joint.parameters())
        params_graph = sum(p.numel() for p in model_graph.parameters())

        self.assertGreater(params_joint, params_graph)

    # --- Numerical sanity ---

    def test_no_nan_in_output(self):
        """Test model output contains no NaN values."""
        for gnn in ["BAT", "GAT", "GIN"]:
            for mode in ["joint", "graph", "node"]:
                model = self._make_model(gnn=gnn, patient_mode=mode)
                node_ids, rel_ids, edge_index, batch_vec, visit_node, ehr_nodes, _ = self._make_batch()

                model.eval()
                with torch.no_grad():
                    logits = model(
                        node_ids, rel_ids, edge_index, batch_vec,
                        visit_node,
                        ehr_nodes if mode != "graph" else None,
                    )

                self.assertFalse(
                    torch.isnan(logits).any(),
                    f"NaN in output for gnn={gnn}, mode={mode}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)