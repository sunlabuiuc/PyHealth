"""Tests for KEEP intrinsic evaluation (Resnik + co-occurrence correlation).

Uses small synthetic SNOMED hierarchy + embeddings to verify the
evaluation metrics behave correctly. Paper target numbers (Table 2,
0.68 Resnik / 0.62 co-occ) are for real UK Biobank data and can't be
reproduced on tiny synthetic fixtures — these tests verify correctness
of the computation, not absolute values.
"""

import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np


def _make_snomed_graph():
    """Small hierarchy for semantic similarity testing."""
    G = nx.DiGraph()
    G.add_edges_from([
        (200, 100), (500, 100),
        (300, 200), (400, 200),
        (600, 500), (700, 500),
    ])
    return G


NODE_IDS = [100, 200, 300, 400, 500, 600, 700]


def _similar_siblings_embeddings():
    """Embeddings where siblings are close (good — matches ontology)."""
    np.random.seed(0)
    return np.array([
        [0.0, 0.0, 1.0, 0.0],  # 100 (root)
        [1.0, 0.0, 0.0, 0.1],  # 200 (cardiovascular)
        [1.0, 0.1, 0.0, 0.0],  # 300 (heart failure)
        [1.0, 0.0, 0.1, 0.0],  # 400 (hypertension)
        [0.0, 1.0, 0.0, 0.1],  # 500 (metabolic)
        [0.0, 1.0, 0.1, 0.0],  # 600 (diabetes)
        [0.0, 1.0, 0.0, 0.1],  # 700 (obesity)
    ], dtype=np.float32)


def _random_embeddings():
    """Random embeddings — should have weak correlation with ontology."""
    np.random.seed(42)
    return np.random.randn(7, 4).astype(np.float32)


class _IntrinsicEvalBase(unittest.TestCase):
    """Shared setUp for intrinsic evaluation tests."""

    def setUp(self):
        self.snomed_graph = _make_snomed_graph()
        self.node_ids = list(NODE_IDS)
        self.similar_embeddings = _similar_siblings_embeddings()
        self.random_embeddings = _random_embeddings()


class TestComputeInformationContent(_IntrinsicEvalBase):
    """Tests for compute_information_content()."""

    def test_returns_dict_for_all_nodes(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            compute_information_content,
        )
        ic = compute_information_content(self.snomed_graph)
        self.assertEqual(set(ic.keys()), set(self.snomed_graph.nodes()))

    def test_root_has_lowest_ic(self):
        """Root has the most descendants, so lowest IC (least specific)."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            compute_information_content,
        )
        ic = compute_information_content(self.snomed_graph)
        self.assertLess(ic[100], ic[300])
        self.assertLess(ic[100], ic[600])

    def test_leaves_have_highest_ic(self):
        """Leaf nodes have no descendants, so highest IC (most specific)."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            compute_information_content,
        )
        ic = compute_information_content(self.snomed_graph)
        self.assertGreater(ic[300], ic[200])
        self.assertGreater(ic[400], ic[200])

    def test_corpus_based_ic(self):
        """Corpus-based IC: more frequent = lower IC."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            compute_information_content,
        )
        patient_codes = [
            {300, 200, 100},
            {300, 200, 100},
            {300, 200, 100},
            {700, 500, 100},
        ]
        ic = compute_information_content(self.snomed_graph, patient_codes)
        # 300 appears 3 times, 700 only 1 → 700 should have higher IC
        self.assertGreater(ic[700], ic[300])


class TestResnikSimilarity(_IntrinsicEvalBase):
    """Tests for resnik_similarity() pairwise function."""

    def test_siblings_higher_than_cousins(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            resnik_similarity, compute_information_content,
        )
        ic = compute_information_content(self.snomed_graph)
        siblings = resnik_similarity(self.snomed_graph, 300, 400, ic)
        cousins = resnik_similarity(self.snomed_graph, 300, 600, ic)
        self.assertGreater(siblings, cousins)

    def test_same_concept_is_max(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            resnik_similarity, compute_information_content,
        )
        ic = compute_information_content(self.snomed_graph)
        self_sim = resnik_similarity(self.snomed_graph, 300, 300, ic)
        sibling_sim = resnik_similarity(self.snomed_graph, 300, 400, ic)
        self.assertGreater(self_sim, sibling_sim)


class TestResnikCorrelation(_IntrinsicEvalBase):
    """Tests for resnik_correlation() — the paper's primary metric."""

    def test_similar_siblings_get_positive_correlation(self):
        """Embeddings that match ontology should give positive correlation."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            resnik_correlation,
        )
        result = resnik_correlation(
            self.similar_embeddings, self.node_ids, self.snomed_graph,
            k1=2, k2=4, num_runs=10, seed=42,
        )
        self.assertGreater(result["median"], 0)

    def test_random_embeddings_low_correlation(self):
        """Random embeddings should give near-zero correlation."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            resnik_correlation,
        )
        result = resnik_correlation(
            self.random_embeddings, self.node_ids, self.snomed_graph,
            k1=2, k2=4, num_runs=20, seed=42,
        )
        self.assertLess(abs(result["median"]), 0.6)

    def test_aligned_beats_random(self):
        """Aligned embeddings should have higher correlation than random."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            resnik_correlation,
        )
        aligned = resnik_correlation(
            self.similar_embeddings, self.node_ids, self.snomed_graph,
            k1=2, k2=4, num_runs=10, seed=42,
        )
        random_result = resnik_correlation(
            self.random_embeddings, self.node_ids, self.snomed_graph,
            k1=2, k2=4, num_runs=10, seed=42,
        )
        self.assertGreater(aligned["median"], random_result["median"])

    def test_returns_all_statistics(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            resnik_correlation,
        )
        result = resnik_correlation(
            self.similar_embeddings, self.node_ids, self.snomed_graph,
            k1=2, k2=4, num_runs=5, seed=42,
        )
        self.assertIn("mean", result)
        self.assertIn("median", result)
        self.assertIn("std", result)
        self.assertIn("min", result)
        self.assertIn("max", result)


class TestCooccurrenceCorrelation(_IntrinsicEvalBase):
    """Tests for cooccurrence_correlation()."""

    def test_matches_cooc_pattern(self):
        """Embeddings close to codes with high co-occurrence → positive corr."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            cooccurrence_correlation,
        )
        n = len(self.node_ids)
        cooc = np.zeros((n, n), dtype=np.float32)
        code_to_idx = {c: i for i, c in enumerate(self.node_ids)}
        cooc[code_to_idx[300], code_to_idx[400]] = 100
        cooc[code_to_idx[400], code_to_idx[300]] = 100
        cooc[code_to_idx[600], code_to_idx[700]] = 80
        cooc[code_to_idx[700], code_to_idx[600]] = 80

        result = cooccurrence_correlation(
            self.similar_embeddings, self.node_ids, cooc, code_to_idx,
            k1=2, k2=4, num_runs=5, seed=42,
        )
        self.assertGreaterEqual(result["median"], -0.3)


class TestLoadKeepEmbeddings(unittest.TestCase):
    """Tests for load_keep_embeddings() text file reader."""

    def test_round_trip(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            load_keep_embeddings,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "keep_snomed.txt"
            path.write_text(
                "84114007 0.1 0.2 0.3 0.4\n"
                "38341003 0.5 0.6 0.7 0.8\n"
            )
            emb, tokens = load_keep_embeddings(path, embedding_dim=4)
            self.assertEqual(emb.shape, (2, 4))
            self.assertEqual(tokens, ["84114007", "38341003"])
            np.testing.assert_allclose(emb[0], [0.1, 0.2, 0.3, 0.4])


class TestEvaluateEmbeddings(_IntrinsicEvalBase):
    """Tests for the evaluate_embeddings() convenience function."""

    def test_runs_resnik_only_when_no_cooc(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            evaluate_embeddings,
        )
        results = evaluate_embeddings(
            self.similar_embeddings, self.node_ids, self.snomed_graph,
            k1=2, k2=4, num_runs=3, seed=42,
        )
        self.assertIn("resnik", results)
        self.assertNotIn("cooccurrence", results)

    def test_runs_both_when_cooc_provided(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            evaluate_embeddings,
        )
        n = len(self.node_ids)
        cooc = np.ones((n, n), dtype=np.float32)
        code_to_idx = {c: i for i, c in enumerate(self.node_ids)}
        results = evaluate_embeddings(
            self.similar_embeddings, self.node_ids, self.snomed_graph,
            cooc_matrix=cooc, code_to_idx=code_to_idx,
            k1=2, k2=4, num_runs=3, seed=42,
        )
        self.assertIn("resnik", results)
        self.assertIn("cooccurrence", results)


if __name__ == "__main__":
    unittest.main()
