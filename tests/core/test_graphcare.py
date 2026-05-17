"""Unit tests for GraphCare model.

Author: Joshua Steier

Tests cover:
    - BiAttentionGNNConv layer forward/backward
    - GraphCare model with BAT, GAT, GIN backbones
    - Patient representation modes: joint, graph, node
    - Forward/backward pass shapes
    - Pre-computed node features
    - Edge cases: single feature key, embed flag
    - Compatibility with PyHealth pipeline
"""

import unittest

import torch
import torch.nn as nn

# Optional dependency check
try:
    from torch_geometric.data import Data, Batch

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def _make_kg():
    """Create a small KnowledgeGraph for testing."""
    from pyhealth.graph import KnowledgeGraph

    triples = [
        ("aspirin", "treats", "headache"),
        ("headache", "symptom_of", "migraine"),
        ("ibuprofen", "treats", "headache"),
        ("migraine", "is_a", "neuro"),
        ("aspirin", "is_a", "nsaid"),
        ("ibuprofen", "is_a", "nsaid"),
        ("X", "used_for", "headache"),
        ("Y", "used_for", "migraine"),
    ]
    return KnowledgeGraph(triples=triples)


def _make_dataset(kg):
    """Create a minimal SampleDataset with graph features."""
    from pyhealth.datasets import create_sample_dataset

    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "conditions": ["aspirin", "headache"],
            "procedures": ["X"],
            "label": 1,
        },
        {
            "patient_id": "p1",
            "visit_id": "v0",
            "conditions": ["ibuprofen"],
            "procedures": ["Y", "X"],
            "label": 0,
        },
    ]
    input_schema = {
        "conditions": ("graph", {
            "knowledge_graph": kg,
            "num_hops": 2,
        }),
        "procedures": ("graph", {
            "knowledge_graph": kg,
            "num_hops": 2,
        }),
    }
    output_schema = {"label": "binary"}
    return create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="test_graphcare",
    )


def _make_batch(dataset):
    """Get a single batch from the dataset."""
    from pyhealth.datasets import get_dataloader

    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    return next(iter(loader))


# ------------------------------------------------------------------ #
# BiAttentionGNNConv tests
# ------------------------------------------------------------------ #


@unittest.skipUnless(HAS_PYG, "torch-geometric required")
class TestBiAttentionGNNConv(unittest.TestCase):
    """Tests for the BAT GNN layer."""

    def test_forward_shape(self):
        """Output has correct shape."""
        from pyhealth.models._graphcare.bat_gnn import BiAttentionGNNConv

        layer = BiAttentionGNNConv(hidden_dim=64, edge_attn=True)
        x = torch.randn(10, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 4]])
        edge_attr = torch.randn(4, 64)
        attn = torch.ones(10, 1)
        out, w_rel = layer(x, edge_index, edge_attr, attn)
        self.assertEqual(out.shape, (10, 64))
        self.assertIsNotNone(w_rel)
        self.assertEqual(w_rel.shape, (4, 1))

    def test_no_edge_attn(self):
        """Works without edge attention."""
        from pyhealth.models._graphcare.bat_gnn import BiAttentionGNNConv

        layer = BiAttentionGNNConv(hidden_dim=32, edge_attn=False)
        x = torch.randn(5, 32)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        edge_attr = torch.randn(2, 32)
        attn = torch.ones(5, 1)
        out, w_rel = layer(x, edge_index, edge_attr, attn)
        self.assertEqual(out.shape, (5, 32))
        self.assertIsNone(w_rel)

    def test_gradient_flow(self):
        """Gradients flow through the layer."""
        from pyhealth.models._graphcare.bat_gnn import BiAttentionGNNConv

        torch.manual_seed(42)
        layer = BiAttentionGNNConv(hidden_dim=32)
        x = torch.randn(5, 32, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_attr = torch.randn(3, 32)
        attn = torch.ones(5, 1)
        out, _ = layer(x, edge_index, edge_attr, attn)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_trainable_eps(self):
        """Epsilon is trainable when requested."""
        from pyhealth.models._graphcare.bat_gnn import BiAttentionGNNConv

        layer = BiAttentionGNNConv(hidden_dim=32, train_eps=True)
        self.assertIsInstance(layer.eps, nn.Parameter)

    def test_fixed_eps(self):
        """Epsilon is a buffer when not trainable."""
        from pyhealth.models._graphcare.bat_gnn import BiAttentionGNNConv

        layer = BiAttentionGNNConv(hidden_dim=32, train_eps=False)
        self.assertNotIsInstance(layer.eps, nn.Parameter)


# ------------------------------------------------------------------ #
# GraphCare model tests
# ------------------------------------------------------------------ #


@unittest.skipUnless(HAS_PYG, "torch-geometric required")
class TestGraphCareBAT(unittest.TestCase):
    """Tests for GraphCare with BAT backbone."""

    @classmethod
    def setUpClass(cls):
        cls.kg = _make_kg()
        cls.dataset = _make_dataset(cls.kg)
        cls.batch = _make_batch(cls.dataset)

    def _make_model(self, **kwargs):
        from pyhealth.models.graphcare import GraphCare

        defaults = dict(
            dataset=self.dataset,
            knowledge_graph=self.kg,
            hidden_dim=64,
            num_layers=2,
            gnn_type="bat",
        )
        defaults.update(kwargs)
        return GraphCare(**defaults)

    def test_forward_keys(self):
        """Output dict has expected keys."""
        model = self._make_model()
        out = model(**self.batch)
        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertIn("logit", out)

    def test_backward(self):
        """Loss backward pass succeeds."""
        model = self._make_model()
        out = model(**self.batch)
        out["loss"].backward()

    def test_output_shapes(self):
        """Logit and probability shapes match batch."""
        model = self._make_model()
        out = model(**self.batch)
        batch_size = self.batch["label"].shape[0]
        self.assertEqual(out["logit"].shape[0], batch_size)
        self.assertEqual(out["y_prob"].shape[0], batch_size)

    def test_joint_mode(self):
        """Joint patient mode concatenates graph and node reps."""
        model = self._make_model(patient_mode="joint")
        out = model(**self.batch)
        self.assertTrue(out["loss"].isfinite())
        out["loss"].backward()

    def test_graph_mode(self):
        """Graph-only patient mode works."""
        model = self._make_model(patient_mode="graph")
        out = model(**self.batch)
        self.assertTrue(out["loss"].isfinite())
        out["loss"].backward()

    def test_node_mode(self):
        """Node-only patient mode works."""
        model = self._make_model(patient_mode="node")
        out = model(**self.batch)
        self.assertTrue(out["loss"].isfinite())
        out["loss"].backward()

    def test_embed_flag(self):
        """embed=True returns patient embedding."""
        model = self._make_model()
        batch = dict(self.batch)
        batch["embed"] = True
        out = model(**batch)
        self.assertIn("embed", out)
        batch_size = self.batch["label"].shape[0]
        self.assertEqual(out["embed"].shape[0], batch_size)

    def test_loss_is_finite(self):
        """Loss is not NaN or Inf."""
        model = self._make_model()
        out = model(**self.batch)
        self.assertTrue(out["loss"].isfinite())


@unittest.skipUnless(HAS_PYG, "torch-geometric required")
class TestGraphCareGAT(unittest.TestCase):
    """Tests for GraphCare with GAT backbone."""

    @classmethod
    def setUpClass(cls):
        cls.kg = _make_kg()
        cls.dataset = _make_dataset(cls.kg)
        cls.batch = _make_batch(cls.dataset)

    def test_forward_backward(self):
        """GAT backbone forward + backward works."""
        from pyhealth.models.graphcare import GraphCare

        model = GraphCare(
            dataset=self.dataset,
            knowledge_graph=self.kg,
            hidden_dim=64,
            num_layers=2,
            gnn_type="gat",
            heads=4,
        )
        out = model(**self.batch)
        self.assertIn("loss", out)
        self.assertTrue(out["loss"].isfinite())
        out["loss"].backward()

    def test_all_patient_modes(self):
        """GAT works with all patient modes."""
        from pyhealth.models.graphcare import GraphCare

        for mode in ("joint", "graph", "node"):
            model = GraphCare(
                dataset=self.dataset,
                knowledge_graph=self.kg,
                hidden_dim=64,
                num_layers=2,
                gnn_type="gat",
                heads=4,
                patient_mode=mode,
            )
            out = model(**self.batch)
            self.assertTrue(out["loss"].isfinite(), msg=f"Failed for {mode}")
            out["loss"].backward()


@unittest.skipUnless(HAS_PYG, "torch-geometric required")
class TestGraphCareGIN(unittest.TestCase):
    """Tests for GraphCare with GIN backbone."""

    @classmethod
    def setUpClass(cls):
        cls.kg = _make_kg()
        cls.dataset = _make_dataset(cls.kg)
        cls.batch = _make_batch(cls.dataset)

    def test_forward_backward(self):
        """GIN backbone forward + backward works."""
        from pyhealth.models.graphcare import GraphCare

        model = GraphCare(
            dataset=self.dataset,
            knowledge_graph=self.kg,
            hidden_dim=64,
            num_layers=2,
            gnn_type="gin",
        )
        out = model(**self.batch)
        self.assertIn("loss", out)
        self.assertTrue(out["loss"].isfinite())
        out["loss"].backward()

    def test_all_patient_modes(self):
        """GIN works with all patient modes."""
        from pyhealth.models.graphcare import GraphCare

        for mode in ("joint", "graph", "node"):
            model = GraphCare(
                dataset=self.dataset,
                knowledge_graph=self.kg,
                hidden_dim=64,
                num_layers=2,
                gnn_type="gin",
                patient_mode=mode,
            )
            out = model(**self.batch)
            self.assertTrue(out["loss"].isfinite(), msg=f"Failed for {mode}")
            out["loss"].backward()


@unittest.skipUnless(HAS_PYG, "torch-geometric required")
class TestGraphCareEdgeCases(unittest.TestCase):
    """Edge case tests for GraphCare."""

    @classmethod
    def setUpClass(cls):
        cls.kg = _make_kg()

    def test_single_feature_key(self):
        """Model works with a single feature key."""
        from pyhealth.datasets import create_sample_dataset, get_dataloader
        from pyhealth.models.graphcare import GraphCare

        samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": ["aspirin", "headache"],
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v0",
                "conditions": ["ibuprofen"],
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "conditions": ("graph", {
                    "knowledge_graph": self.kg,
                    "num_hops": 2,
                }),
            },
            output_schema={"label": "binary"},
            dataset_name="test_single",
        )
        model = GraphCare(
            dataset=dataset,
            knowledge_graph=self.kg,
            hidden_dim=32,
            num_layers=1,
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertTrue(out["loss"].isfinite())
        out["loss"].backward()

    def test_pretrained_node_features(self):
        """Model works with pre-computed node features."""
        from pyhealth.graph import KnowledgeGraph
        from pyhealth.datasets import create_sample_dataset, get_dataloader
        from pyhealth.models.graphcare import GraphCare

        triples = [
            ("A", "r1", "B"),
            ("B", "r2", "C"),
            ("A", "r1", "C"),
        ]
        node_features = torch.randn(3, 16)
        kg = KnowledgeGraph(triples=triples, node_features=node_features)

        samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "codes": ["A", "B"],
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v0",
                "codes": ["C"],
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={
                "codes": ("graph", {
                    "knowledge_graph": kg,
                    "num_hops": 1,
                }),
            },
            output_schema={"label": "binary"},
            dataset_name="test_pretrained",
        )
        model = GraphCare(
            dataset=dataset,
            knowledge_graph=kg,
            hidden_dim=32,
            num_layers=1,
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        out = model(**batch)
        self.assertTrue(out["loss"].isfinite())
        out["loss"].backward()

    def test_invalid_gnn_type(self):
        """Invalid gnn_type raises AssertionError."""
        from pyhealth.models.graphcare import GraphCare

        dataset = _make_dataset(self.kg)
        with self.assertRaises(AssertionError):
            GraphCare(
                dataset=dataset,
                knowledge_graph=self.kg,
                gnn_type="invalid",
            )

    def test_invalid_patient_mode(self):
        """Invalid patient_mode raises AssertionError."""
        from pyhealth.models.graphcare import GraphCare

        dataset = _make_dataset(self.kg)
        with self.assertRaises(AssertionError):
            GraphCare(
                dataset=dataset,
                knowledge_graph=self.kg,
                patient_mode="invalid",
            )


if __name__ == "__main__":
    unittest.main()