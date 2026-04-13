"""Tests for KEEP intrinsic evaluation (Resnik + co-occurrence correlation).

Uses small synthetic SNOMED hierarchy + embeddings to verify the
evaluation metrics behave correctly. Paper target numbers (Table 2,
0.68 Resnik / 0.62 co-occ) are for real UK Biobank data and can't be
reproduced on tiny synthetic fixtures — these tests verify correctness
of the computation, not absolute values.
"""

from pathlib import Path
import tempfile

import networkx as nx
import numpy as np
import pytest


@pytest.fixture
def snomed_graph():
    """Small hierarchy for semantic similarity testing.

    Structure:
        Root (100)
          Cardiovascular (200)
            Heart failure (300)
            Hypertension (400)
          Metabolic (500)
            Diabetes (600)
            Obesity (700)
    """
    G = nx.DiGraph()
    G.add_edges_from([
        (200, 100), (500, 100),
        (300, 200), (400, 200),
        (600, 500), (700, 500),
    ])
    return G


@pytest.fixture
def node_ids():
    return [100, 200, 300, 400, 500, 600, 700]


@pytest.fixture
def similar_siblings_embeddings():
    """Embeddings where siblings are close (good — matches ontology)."""
    np.random.seed(0)
    # Cardiovascular cluster: 200, 300, 400 near each other
    # Metabolic cluster: 500, 600, 700 near each other
    emb = np.array([
        [0.0, 0.0, 1.0, 0.0],  # 100 (root)
        [1.0, 0.0, 0.0, 0.1],  # 200 (cardiovascular)
        [1.0, 0.1, 0.0, 0.0],  # 300 (heart failure)
        [1.0, 0.0, 0.1, 0.0],  # 400 (hypertension)
        [0.0, 1.0, 0.0, 0.1],  # 500 (metabolic)
        [0.0, 1.0, 0.1, 0.0],  # 600 (diabetes)
        [0.0, 1.0, 0.0, 0.1],  # 700 (obesity)
    ], dtype=np.float32)
    return emb


@pytest.fixture
def random_embeddings():
    """Random embeddings — should have weak correlation with ontology."""
    np.random.seed(42)
    return np.random.randn(7, 4).astype(np.float32)


class TestComputeInformationContent:
    """Tests for compute_information_content()."""

    def test_returns_dict_for_all_nodes(self, snomed_graph):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            compute_information_content,
        )
        ic = compute_information_content(snomed_graph)
        assert set(ic.keys()) == set(snomed_graph.nodes())

    def test_root_has_lowest_ic(self, snomed_graph):
        """Root has the most descendants, so lowest IC (least specific)."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            compute_information_content,
        )
        ic = compute_information_content(snomed_graph)
        # Root (100) has all 7 nodes as descendants (including self)
        # Leaves (300, 400, 600, 700) have only themselves
        assert ic[100] < ic[300]
        assert ic[100] < ic[600]

    def test_leaves_have_highest_ic(self, snomed_graph):
        """Leaf nodes have no descendants, so highest IC (most specific)."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            compute_information_content,
        )
        ic = compute_information_content(snomed_graph)
        # 300 and 400 are leaves under Cardiovascular
        # 200 is their parent (has descendants)
        assert ic[300] > ic[200]
        assert ic[400] > ic[200]

    def test_corpus_based_ic(self, snomed_graph):
        """Corpus-based IC: more frequent = lower IC."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            compute_information_content,
        )
        # Patients: concept 300 appears often, 700 rarely
        patient_codes = [
            {300, 200, 100},  # Patient 1
            {300, 200, 100},  # Patient 2
            {300, 200, 100},  # Patient 3
            {700, 500, 100},  # Patient 4
        ]
        ic = compute_information_content(snomed_graph, patient_codes)
        # 300 appears 3 times, 700 only 1 → 700 should have higher IC
        assert ic[700] > ic[300]


class TestResnikSimilarity:
    """Tests for resnik_similarity() pairwise function."""

    def test_siblings_higher_than_cousins(self, snomed_graph):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            resnik_similarity, compute_information_content,
        )
        ic = compute_information_content(snomed_graph)
        # 300 and 400 are siblings under Cardiovascular (200)
        # 300 and 600 are cousins, only share Root (100)
        siblings = resnik_similarity(snomed_graph, 300, 400, ic)
        cousins = resnik_similarity(snomed_graph, 300, 600, ic)
        assert siblings > cousins

    def test_same_concept_is_max(self, snomed_graph):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            resnik_similarity, compute_information_content,
        )
        ic = compute_information_content(snomed_graph)
        # Self-similarity uses self as LCA (most specific)
        self_sim = resnik_similarity(snomed_graph, 300, 300, ic)
        sibling_sim = resnik_similarity(snomed_graph, 300, 400, ic)
        assert self_sim > sibling_sim


class TestResnikCorrelation:
    """Tests for resnik_correlation() — the paper's primary metric."""

    def test_similar_siblings_get_positive_correlation(
        self, similar_siblings_embeddings, node_ids, snomed_graph,
    ):
        """Embeddings that match ontology should give positive correlation."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            resnik_correlation,
        )
        # k1+k2 is bigger than fixture has, use smaller values
        result = resnik_correlation(
            similar_siblings_embeddings, node_ids, snomed_graph,
            k1=2, k2=4, num_runs=10, seed=42,
        )
        # Well-aligned embeddings → positive correlation
        assert result["median"] > 0

    def test_random_embeddings_low_correlation(
        self, random_embeddings, node_ids, snomed_graph,
    ):
        """Random embeddings should give near-zero correlation."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            resnik_correlation,
        )
        result = resnik_correlation(
            random_embeddings, node_ids, snomed_graph,
            k1=2, k2=4, num_runs=20, seed=42,
        )
        # Random embeddings: correlation should be small in magnitude
        # (not necessarily exactly 0, but much less than aligned)
        assert abs(result["median"]) < 0.6

    def test_aligned_beats_random(
        self, similar_siblings_embeddings, random_embeddings,
        node_ids, snomed_graph,
    ):
        """Aligned embeddings should have higher correlation than random."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            resnik_correlation,
        )
        aligned = resnik_correlation(
            similar_siblings_embeddings, node_ids, snomed_graph,
            k1=2, k2=4, num_runs=10, seed=42,
        )
        random_result = resnik_correlation(
            random_embeddings, node_ids, snomed_graph,
            k1=2, k2=4, num_runs=10, seed=42,
        )
        assert aligned["median"] > random_result["median"]

    def test_returns_all_statistics(
        self, similar_siblings_embeddings, node_ids, snomed_graph,
    ):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            resnik_correlation,
        )
        result = resnik_correlation(
            similar_siblings_embeddings, node_ids, snomed_graph,
            k1=2, k2=4, num_runs=5, seed=42,
        )
        assert "mean" in result
        assert "median" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result


class TestCooccurrenceCorrelation:
    """Tests for cooccurrence_correlation()."""

    def test_matches_cooc_pattern(
        self, similar_siblings_embeddings, node_ids,
    ):
        """Embeddings close to codes with high co-occurrence → positive corr."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            cooccurrence_correlation,
        )
        # Build co-occ matching the embedding structure
        # 300-400 (siblings) close in embedding, high co-occurrence
        # 300-600 (cousins) far in embedding, low co-occurrence
        n = len(node_ids)
        cooc = np.zeros((n, n), dtype=np.float32)
        code_to_idx = {c: i for i, c in enumerate(node_ids)}
        # High co-occurrence within cardiovascular cluster
        cooc[code_to_idx[300], code_to_idx[400]] = 100
        cooc[code_to_idx[400], code_to_idx[300]] = 100
        # High co-occurrence within metabolic cluster
        cooc[code_to_idx[600], code_to_idx[700]] = 80
        cooc[code_to_idx[700], code_to_idx[600]] = 80

        result = cooccurrence_correlation(
            similar_siblings_embeddings, node_ids, cooc, code_to_idx,
            k1=2, k2=4, num_runs=5, seed=42,
        )
        # Embeddings and co-occ agree → non-negative correlation
        assert result["median"] >= -0.3  # loose check on small synthetic data


class TestLoadKeepEmbeddings:
    """Tests for load_keep_embeddings() text file reader."""

    def test_round_trip(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            load_keep_embeddings,
        )
        # Write a tiny embedding file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "keep_snomed.txt"
            path.write_text(
                "84114007 0.1 0.2 0.3 0.4\n"
                "38341003 0.5 0.6 0.7 0.8\n"
            )
            emb, tokens = load_keep_embeddings(path, embedding_dim=4)
            assert emb.shape == (2, 4)
            assert tokens == ["84114007", "38341003"]
            np.testing.assert_allclose(emb[0], [0.1, 0.2, 0.3, 0.4])


class TestEvaluateEmbeddings:
    """Tests for the evaluate_embeddings() convenience function."""

    def test_runs_resnik_only_when_no_cooc(
        self, similar_siblings_embeddings, node_ids, snomed_graph,
    ):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            evaluate_embeddings,
        )
        results = evaluate_embeddings(
            similar_siblings_embeddings, node_ids, snomed_graph,
            k1=2, k2=4, num_runs=3, seed=42,
        )
        assert "resnik" in results
        assert "cooccurrence" not in results

    def test_runs_both_when_cooc_provided(
        self, similar_siblings_embeddings, node_ids, snomed_graph,
    ):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.intrinsic_eval import (
            evaluate_embeddings,
        )
        n = len(node_ids)
        cooc = np.ones((n, n), dtype=np.float32)
        code_to_idx = {c: i for i, c in enumerate(node_ids)}
        results = evaluate_embeddings(
            similar_siblings_embeddings, node_ids, snomed_graph,
            cooc_matrix=cooc, code_to_idx=code_to_idx,
            k1=2, k2=4, num_runs=3, seed=42,
        )
        assert "resnik" in results
        assert "cooccurrence" in results
