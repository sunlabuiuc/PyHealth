"""Tests for KEEP co-occurrence matrix construction.

Uses small synthetic patient data (3-5 patients, ~10 diagnosis rows)
to verify extraction, roll-up, and co-occurrence logic without
requiring real MIMIC data or network access.
"""

import networkx as nx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def icd9_to_snomed():
    """Minimal ICD-9 to SNOMED mapping (multi-target format)."""
    return {
        "428.0": [300],    # Heart failure
        "401.9": [400],    # Hypertension
        "250.00": [600],   # Diabetes
    }


@pytest.fixture
def icd10_to_snomed():
    """Minimal ICD-10 to SNOMED mapping (multi-target format)."""
    return {
        "I50.9": [300],    # Heart failure
        "E11": [600],      # Diabetes
    }


@pytest.fixture
def snomed_graph():
    """Small SNOMED hierarchy for roll-up testing.

    Structure:
        Root(100) -> Cardiovascular(200) -> Heart Failure(300)
                                         -> Hypertension(400)
                  -> Metabolic(500) -> Diabetes(600)
    Edges are child -> parent.
    """
    G = nx.DiGraph()
    G.add_edges_from([
        (200, 100),  # Cardiovascular -> Root
        (300, 200),  # Heart Failure -> Cardiovascular
        (400, 200),  # Hypertension -> Cardiovascular
        (500, 100),  # Metabolic -> Root
        (600, 500),  # Diabetes -> Metabolic
    ])
    return G


# ---------------------------------------------------------------------------
# Test extract_patient_codes_from_df
# ---------------------------------------------------------------------------

class TestExtractPatientCodes:
    """Tests for extract_patient_codes_from_df()."""

    def test_basic_extraction(self, icd9_to_snomed):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )

        # Patient P1 has 428.0 twice and 250.00 twice -> both pass min_occurrences=2
        patient_ids = ["P1", "P1", "P1", "P1"]
        codes = ["428.0", "428.0", "250.00", "250.00"]

        result = extract_patient_codes_from_df(
            patient_ids, codes, icd9_to_snomed, min_occurrences=2,
        )
        assert result["P1"] == {300, 600}

    def test_min_occurrences_filter(self, icd9_to_snomed):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )

        # P1: 428.0 appears 2x (passes), 250.00 appears 1x (filtered out)
        patient_ids = ["P1", "P1", "P1"]
        codes = ["428.0", "428.0", "250.00"]

        result = extract_patient_codes_from_df(
            patient_ids, codes, icd9_to_snomed, min_occurrences=2,
        )
        assert result["P1"] == {300}  # Only heart failure passes
        assert 600 not in result["P1"]

    def test_patient_with_no_qualifying_codes_excluded(self, icd9_to_snomed):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )

        # P1 has each code only once -> nothing passes min_occurrences=2
        patient_ids = ["P1", "P1"]
        codes = ["428.0", "250.00"]

        result = extract_patient_codes_from_df(
            patient_ids, codes, icd9_to_snomed, min_occurrences=2,
        )
        assert "P1" not in result

    def test_unmapped_codes_skipped(self, icd9_to_snomed):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )

        # "999.99" is not in the mapping
        patient_ids = ["P1", "P1", "P1", "P1"]
        codes = ["428.0", "428.0", "999.99", "999.99"]

        result = extract_patient_codes_from_df(
            patient_ids, codes, icd9_to_snomed, min_occurrences=2,
        )
        assert result["P1"] == {300}  # Only mapped code

    def test_mimic4_dual_icd(self, icd9_to_snomed, icd10_to_snomed):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )

        # P1 has ICD-9 "428.0" (2x) and ICD-10 "E11" (2x)
        patient_ids = ["P1", "P1", "P1", "P1"]
        codes = ["428.0", "428.0", "E11", "E11"]
        versions = ["9", "9", "10", "10"]

        result = extract_patient_codes_from_df(
            patient_ids,
            codes,
            icd9_to_snomed,
            version_col=versions,
            icd10_to_snomed=icd10_to_snomed,
            min_occurrences=2,
        )
        assert result["P1"] == {300, 600}  # Heart failure + Diabetes

    def test_multi_target_icd_dense_expansion(self):
        """Multi-target ICD codes count as multiple SNOMED occurrences.

        Patient has ICD "250.01" which maps to [300, 600]. Each occurrence
        of "250.01" increments BOTH SNOMED 300 and 600 (dense expansion).
        """
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )

        icd_map = {
            "250.01": [300, 600],  # multi-target combination code
            "428.0": [300],         # single-target
        }

        # Patient has 250.01 twice — that's 2x for SNOMED 300 AND 2x for 600
        # (dense expansion: each ICD occurrence counts for all targets)
        patient_ids = ["P1", "P1"]
        codes = ["250.01", "250.01"]

        result = extract_patient_codes_from_df(
            patient_ids, codes, icd_map, min_occurrences=2,
        )
        assert result["P1"] == {300, 600}  # both targets pass filter

    def test_standardize_functions(self, icd9_to_snomed):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )

        # Simulate MIMIC raw codes (no dots) with standardization
        def fake_standardize(code):
            if len(code) > 3 and "." not in code:
                return code[:3] + "." + code[3:]
            return code

        patient_ids = ["P1", "P1"]
        codes = ["4280", "4280"]  # No dots, like raw MIMIC

        result = extract_patient_codes_from_df(
            patient_ids,
            codes,
            icd9_to_snomed,
            standardize_icd9=fake_standardize,
            min_occurrences=2,
        )
        assert result["P1"] == {300}


# ---------------------------------------------------------------------------
# Test rollup_codes
# ---------------------------------------------------------------------------

class TestRollupCodes:
    """Tests for rollup_codes()."""

    def test_expands_to_ancestors(self, snomed_graph):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            rollup_codes,
        )

        # Patient has Diabetes (600) -> should expand to include
        # Metabolic (500) and Root (100)
        patient_codes = {"P1": {600}}
        result = rollup_codes(patient_codes, snomed_graph)
        assert result["P1"] == {600, 500, 100}

    def test_multiple_codes_expand_independently(self, snomed_graph):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            rollup_codes,
        )

        # Patient has Heart Failure (300) and Diabetes (600)
        # HF expands: 300 -> 200 -> 100
        # DM expands: 600 -> 500 -> 100
        patient_codes = {"P1": {300, 600}}
        result = rollup_codes(patient_codes, snomed_graph)
        assert result["P1"] == {300, 200, 100, 600, 500}

    def test_codes_not_in_graph_kept_but_not_expanded(self, snomed_graph):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            rollup_codes,
        )

        # Code 999 is not in the graph -> kept as-is, no ancestors added
        patient_codes = {"P1": {999, 600}}
        result = rollup_codes(patient_codes, snomed_graph)
        assert 999 in result["P1"]
        assert 600 in result["P1"]
        assert 500 in result["P1"]  # Ancestor of 600


# ---------------------------------------------------------------------------
# Test build_cooccurrence_matrix
# ---------------------------------------------------------------------------

class TestBuildCooccurrenceMatrix:
    """Tests for build_cooccurrence_matrix()."""

    def test_basic_cooccurrence(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            build_cooccurrence_matrix,
        )

        # P1: {100, 200}, P2: {100, 200}, P3: {200, 300}
        patient_codes = {
            "P1": {100, 200},
            "P2": {100, 200},
            "P3": {200, 300},
        }
        matrix, code_to_idx, idx_to_code = build_cooccurrence_matrix(
            patient_codes
        )
        # 100 and 200 co-occur in 2 patients (P1, P2)
        assert matrix[code_to_idx[100], code_to_idx[200]] == 2.0
        # 200 and 300 co-occur in 1 patient (P3)
        assert matrix[code_to_idx[200], code_to_idx[300]] == 1.0
        # 100 and 300 never co-occur
        assert matrix[code_to_idx[100], code_to_idx[300]] == 0.0

    def test_matrix_is_symmetric(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            build_cooccurrence_matrix,
        )

        patient_codes = {
            "P1": {100, 200, 300},
            "P2": {100, 300},
        }
        matrix, _, _ = build_cooccurrence_matrix(patient_codes)
        np.testing.assert_array_equal(matrix, matrix.T)

    def test_valid_codes_filter(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            build_cooccurrence_matrix,
        )

        patient_codes = {
            "P1": {100, 200, 999},
        }
        # Only include 100 and 200, not 999
        matrix, code_to_idx, idx_to_code = build_cooccurrence_matrix(
            patient_codes, valid_codes={100, 200}
        )
        assert 999 not in code_to_idx
        assert len(idx_to_code) == 2

    def test_diagonal_counts_patients(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            build_cooccurrence_matrix,
        )

        # Code 100 appears in 2 patients
        patient_codes = {
            "P1": {100, 200},
            "P2": {100},
        }
        matrix, code_to_idx, _ = build_cooccurrence_matrix(patient_codes)
        assert matrix[code_to_idx[100], code_to_idx[100]] == 2.0
        assert matrix[code_to_idx[200], code_to_idx[200]] == 1.0

    def test_empty_input(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            build_cooccurrence_matrix,
        )

        matrix, code_to_idx, idx_to_code = build_cooccurrence_matrix({})
        assert matrix.shape == (0, 0)
        assert len(code_to_idx) == 0


# ---------------------------------------------------------------------------
# Test apply_count_filter
# ---------------------------------------------------------------------------

class TestApplyCountFilter:
    """Tests for apply_count_filter().

    Drops SNOMED concepts with zero observations in the co-occurrence
    matrix diagonal. This ensures Stage 1 (Node2Vec) and Stage 2 (GloVe)
    operate on the same set of concepts, matching the paper's implicit
    count filter (visible in G2Lab's "_ct_filter" file suffix).
    """

    def test_drops_unobserved_concepts(self, snomed_graph):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            build_cooccurrence_matrix, apply_count_filter,
        )

        # P1 has 100, 200, 300. Concepts 400, 500, 600 never appear.
        patient_codes = {"P1": {100, 200, 300}}
        matrix, code_to_idx, idx_to_code = build_cooccurrence_matrix(
            patient_codes, valid_codes=set(snomed_graph.nodes()),
        )
        # Matrix includes only observed codes (100, 200, 300)
        # because build_cooccurrence_matrix filters to all_codes
        assert set(code_to_idx.keys()) == {100, 200, 300}

        # Apply filter — should keep all since all have non-zero diagonals
        (filtered_graph, filtered_matrix,
         filtered_code_to_idx, filtered_idx_to_code) = apply_count_filter(
            snomed_graph, matrix, code_to_idx, idx_to_code,
        )
        assert set(filtered_graph.nodes()) == {100, 200, 300}

    def test_filter_with_zero_diagonals(self, snomed_graph):
        """Codes in matrix but with zero diagonal should be dropped."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            apply_count_filter,
        )

        # Build a matrix where 500 has zero diagonal
        code_to_idx = {100: 0, 200: 1, 300: 2, 500: 3}
        idx_to_code = [100, 200, 300, 500]
        matrix = np.array([
            [3.0, 2.0, 1.0, 0.0],
            [2.0, 2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],  # 500 never observed
        ], dtype=np.float32)

        (filtered_graph, filtered_matrix,
         filtered_code_to_idx, filtered_idx_to_code) = apply_count_filter(
            snomed_graph, matrix, code_to_idx, idx_to_code,
        )
        # 500 should be dropped (zero diagonal)
        assert 500 not in filtered_graph.nodes()
        assert 500 not in filtered_code_to_idx
        # 100, 200, 300 should remain
        assert {100, 200, 300} == set(filtered_code_to_idx.keys())

    def test_matrix_reindexed_after_filter(self, snomed_graph):
        """Filtered matrix should have correct dimensions and values."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            apply_count_filter,
        )

        code_to_idx = {100: 0, 200: 1, 500: 2}
        idx_to_code = [100, 200, 500]
        matrix = np.array([
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # 500 unobserved
        ], dtype=np.float32)

        _, filtered_matrix, filtered_code_to_idx, filtered_idx_to_code = (
            apply_count_filter(
                snomed_graph, matrix, code_to_idx, idx_to_code,
            )
        )

        # New matrix is 2x2 (dropped 500)
        assert filtered_matrix.shape == (2, 2)
        # Values preserved
        assert filtered_matrix[filtered_code_to_idx[100],
                               filtered_code_to_idx[200]] == 1.0
        # Indices are 0-based and contiguous
        assert sorted(filtered_code_to_idx.values()) == [0, 1]

    def test_graph_subgraph_is_copy(self, snomed_graph):
        """Filtered graph should be a copy, not a view — safe to modify."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            apply_count_filter,
        )

        code_to_idx = {100: 0, 200: 1}
        idx_to_code = [100, 200]
        matrix = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)

        filtered_graph, _, _, _ = apply_count_filter(
            snomed_graph, matrix, code_to_idx, idx_to_code,
        )
        # Modifying filtered_graph should not affect original
        filtered_graph.add_node(999)
        assert 999 not in snomed_graph.nodes()
