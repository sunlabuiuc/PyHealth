"""Unit tests for pyhealth.processors.GraphProcessor.

Tests cover: construction, process method, multi-visit codes, unknown codes,
pruning, schema methods, and edge cases.

Run: python -m unittest tests/core/test_graph_processor.py -v
"""

import unittest

import torch


def _has_pyg():
    try:
        import torch_geometric
        return True
    except ImportError:
        return False


def _make_kg():
    """Helper: build a small KnowledgeGraph for testing."""
    from pyhealth.graph import KnowledgeGraph

    triples = [
        ("aspirin", "treats", "headache"),
        ("headache", "symptom_of", "migraine"),
        ("ibuprofen", "treats", "headache"),
        ("migraine", "is_a", "neurological_disorder"),
        ("aspirin", "is_a", "nsaid"),
        ("ibuprofen", "is_a", "nsaid"),
        ("nsaid", "treats", "inflammation"),
        ("inflammation", "symptom_of", "arthritis"),
    ]
    return KnowledgeGraph(triples=triples)


@unittest.skipUnless(_has_pyg(), "torch-geometric not installed")
class TestGraphProcessorConstruction(unittest.TestCase):
    """Tests for GraphProcessor initialization."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.processors.graph_processor import GraphProcessor

        cls.GraphProcessor = GraphProcessor
        cls.kg = _make_kg()

    def test_basic_construction(self):
        """GraphProcessor initializes with a KG."""
        processor = self.GraphProcessor(knowledge_graph=self.kg)
        self.assertIsNotNone(processor)

    def test_custom_params(self):
        """Custom num_hops and max_nodes are stored."""
        processor = self.GraphProcessor(
            knowledge_graph=self.kg, num_hops=3, max_nodes=10
        )
        self.assertEqual(processor.num_hops, 3)
        self.assertEqual(processor.max_nodes, 10)

    def test_repr(self):
        """__repr__ returns a readable string."""
        processor = self.GraphProcessor(knowledge_graph=self.kg)
        r = repr(processor)
        self.assertIn("GraphProcessor", r)
        self.assertIn("num_hops", r)


@unittest.skipUnless(_has_pyg(), "torch-geometric not installed")
class TestGraphProcessorProcess(unittest.TestCase):
    """Tests for the process method."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.processors.graph_processor import GraphProcessor
        from torch_geometric.data import Data

        cls.GraphProcessor = GraphProcessor
        cls.Data = Data
        cls.kg = _make_kg()
        cls.processor = GraphProcessor(knowledge_graph=cls.kg, num_hops=2)

    def test_basic_process(self):
        """process returns a PyG Data object."""
        graph = self.processor.process(["aspirin", "headache"])
        self.assertIsInstance(graph, self.Data)

    def test_has_required_attrs(self):
        """Output Data has x, edge_index, edge_type, node_ids, seed_mask."""
        graph = self.processor.process(["aspirin"])
        self.assertIsNotNone(graph.x)
        self.assertIsNotNone(graph.edge_index)
        self.assertIsNotNone(graph.edge_type)
        self.assertIsNotNone(graph.node_ids)
        self.assertIsNotNone(graph.seed_mask)

    def test_seed_mask_count(self):
        """Seed mask has correct number of True values."""
        graph = self.processor.process(["aspirin", "headache"])
        self.assertEqual(graph.seed_mask.sum().item(), 2)

    def test_single_code(self):
        """Single code produces a valid graph."""
        graph = self.processor.process(["aspirin"])
        self.assertGreater(graph.num_nodes, 0)

    def test_multi_visit_codes(self):
        """List of lists is flattened properly."""
        multi_visit = [["aspirin"], ["headache", "migraine"]]
        graph = self.processor.process(multi_visit)
        # Should have 3 seed nodes
        self.assertEqual(graph.seed_mask.sum().item(), 3)

    def test_unknown_codes_skipped(self):
        """Unknown codes are silently skipped."""
        graph = self.processor.process(["unknown_drug", "aspirin"])
        # Only aspirin is a seed
        self.assertEqual(graph.seed_mask.sum().item(), 1)

    def test_all_unknown_codes(self):
        """All unknown codes produce empty graph."""
        graph = self.processor.process(["unknown1", "unknown2"])
        self.assertEqual(graph.num_nodes, 0)
        self.assertEqual(graph.num_edges, 0)

    def test_edge_index_valid(self):
        """edge_index values are within [0, num_nodes)."""
        graph = self.processor.process(["aspirin", "headache"])
        if graph.num_edges > 0:
            self.assertLess(
                graph.edge_index.max().item(), graph.num_nodes
            )
            self.assertGreaterEqual(graph.edge_index.min().item(), 0)

    def test_edge_type_matches_edges(self):
        """edge_type length matches number of edges."""
        graph = self.processor.process(["aspirin"])
        self.assertEqual(
            graph.edge_type.shape[0], graph.edge_index.shape[1]
        )

    def test_more_hops_more_nodes(self):
        """Increasing num_hops includes more nodes."""
        proc1 = self.GraphProcessor(knowledge_graph=self.kg, num_hops=1)
        proc2 = self.GraphProcessor(knowledge_graph=self.kg, num_hops=3)
        g1 = proc1.process(["aspirin"])
        g2 = proc2.process(["aspirin"])
        self.assertGreaterEqual(g2.num_nodes, g1.num_nodes)


@unittest.skipUnless(_has_pyg(), "torch-geometric not installed")
class TestGraphProcessorPruning(unittest.TestCase):
    """Tests for max_nodes pruning."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.processors.graph_processor import GraphProcessor

        cls.GraphProcessor = GraphProcessor
        cls.kg = _make_kg()

    def test_pruning_respects_max(self):
        """Pruned graph has at most max_nodes nodes."""
        processor = self.GraphProcessor(
            knowledge_graph=self.kg, num_hops=3, max_nodes=3
        )
        graph = processor.process(["aspirin", "headache"])
        self.assertLessEqual(graph.num_nodes, 3)

    def test_pruning_keeps_seeds(self):
        """Seeds are always kept during pruning."""
        processor = self.GraphProcessor(
            knowledge_graph=self.kg, num_hops=3, max_nodes=3
        )
        graph = processor.process(["aspirin", "headache"])
        # Both seeds should still be present
        self.assertGreaterEqual(graph.seed_mask.sum().item(), 2)

    def test_pruning_edges_valid(self):
        """Pruned graph has valid edge_index."""
        processor = self.GraphProcessor(
            knowledge_graph=self.kg, num_hops=3, max_nodes=4
        )
        graph = processor.process(["aspirin", "headache"])
        if graph.num_edges > 0:
            self.assertLess(
                graph.edge_index.max().item(), graph.num_nodes
            )

    def test_no_pruning_when_under_limit(self):
        """No pruning when graph is smaller than max_nodes."""
        proc_small = self.GraphProcessor(
            knowledge_graph=self.kg, num_hops=1, max_nodes=100
        )
        proc_none = self.GraphProcessor(
            knowledge_graph=self.kg, num_hops=1
        )
        g1 = proc_small.process(["aspirin"])
        g2 = proc_none.process(["aspirin"])
        self.assertEqual(g1.num_nodes, g2.num_nodes)


@unittest.skipUnless(_has_pyg(), "torch-geometric not installed")
class TestGraphProcessorSchema(unittest.TestCase):
    """Tests for schema/metadata methods."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.processors.graph_processor import GraphProcessor

        cls.processor = GraphProcessor(knowledge_graph=_make_kg())

    def test_is_token_false(self):
        """is_token returns False."""
        self.assertFalse(self.processor.is_token())

    def test_schema(self):
        """schema returns ('graph',)."""
        self.assertEqual(self.processor.schema(), ("graph",))

    def test_dim(self):
        """dim returns (0,)."""
        self.assertEqual(self.processor.dim(), (0,))

    def test_spatial(self):
        """spatial returns (False,)."""
        self.assertEqual(self.processor.spatial(), (False,))

    def test_fit_is_noop(self):
        """fit does nothing and doesn't error."""
        self.processor.fit([], "field")  # should not raise


if __name__ == "__main__":
    unittest.main()