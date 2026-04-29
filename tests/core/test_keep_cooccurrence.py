"""Tests for KEEP co-occurrence matrix construction.

Uses small synthetic patient data (3-5 patients, ~10 diagnosis rows)
to verify extraction, roll-up, and co-occurrence logic without
requiring real MIMIC data or network access.
"""

import unittest

import networkx as nx
import numpy as np


def _make_icd9_mapping():
    """Minimal ICD-9 to SNOMED mapping (multi-target format)."""
    return {
        "428.0": [300],    # Heart failure
        "401.9": [400],    # Hypertension
        "250.00": [600],   # Diabetes
    }


def _make_icd10_mapping():
    """Minimal ICD-10 to SNOMED mapping (multi-target format)."""
    return {
        "I50.9": [300],    # Heart failure
        "E11": [600],      # Diabetes
    }


def _make_snomed_graph():
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


class TestExtractPatientCodes(unittest.TestCase):
    """Tests for extract_patient_codes_from_df()."""

    def setUp(self):
        self.icd9_to_snomed = _make_icd9_mapping()
        self.icd10_to_snomed = _make_icd10_mapping()

    def test_basic_extraction(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )
        patient_ids = ["P1", "P1", "P1", "P1"]
        codes = ["428.0", "428.0", "250.00", "250.00"]
        result = extract_patient_codes_from_df(
            patient_ids, codes, self.icd9_to_snomed, min_occurrences=2,
        )
        self.assertEqual(result["P1"], {300, 600})

    def test_min_occurrences_filter(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )
        patient_ids = ["P1", "P1", "P1"]
        codes = ["428.0", "428.0", "250.00"]
        result = extract_patient_codes_from_df(
            patient_ids, codes, self.icd9_to_snomed, min_occurrences=2,
        )
        self.assertEqual(result["P1"], {300})
        self.assertNotIn(600, result["P1"])

    def test_patient_with_no_qualifying_codes_excluded(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )
        patient_ids = ["P1", "P1"]
        codes = ["428.0", "250.00"]
        result = extract_patient_codes_from_df(
            patient_ids, codes, self.icd9_to_snomed, min_occurrences=2,
        )
        self.assertNotIn("P1", result)

    def test_unmapped_codes_skipped(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )
        patient_ids = ["P1", "P1", "P1", "P1"]
        codes = ["428.0", "428.0", "999.99", "999.99"]
        result = extract_patient_codes_from_df(
            patient_ids, codes, self.icd9_to_snomed, min_occurrences=2,
        )
        self.assertEqual(result["P1"], {300})

    def test_mimic4_dual_icd(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )
        patient_ids = ["P1", "P1", "P1", "P1"]
        codes = ["428.0", "428.0", "E11", "E11"]
        versions = ["9", "9", "10", "10"]
        result = extract_patient_codes_from_df(
            patient_ids,
            codes,
            self.icd9_to_snomed,
            version_col=versions,
            icd10_to_snomed=self.icd10_to_snomed,
            min_occurrences=2,
        )
        self.assertEqual(result["P1"], {300, 600})

    def test_multi_target_icd_dense_expansion(self):
        """Multi-target ICD codes count as multiple SNOMED occurrences."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )
        icd_map = {
            "250.01": [300, 600],
            "428.0": [300],
        }
        patient_ids = ["P1", "P1"]
        codes = ["250.01", "250.01"]
        result = extract_patient_codes_from_df(
            patient_ids, codes, icd_map, min_occurrences=2,
        )
        self.assertEqual(result["P1"], {300, 600})

    def test_standardize_functions(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            extract_patient_codes_from_df,
        )

        def fake_standardize(code):
            if len(code) > 3 and "." not in code:
                return code[:3] + "." + code[3:]
            return code

        patient_ids = ["P1", "P1"]
        codes = ["4280", "4280"]
        result = extract_patient_codes_from_df(
            patient_ids,
            codes,
            self.icd9_to_snomed,
            standardize_icd9=fake_standardize,
            min_occurrences=2,
        )
        self.assertEqual(result["P1"], {300})


class TestRollupCodes(unittest.TestCase):
    """Tests for rollup_codes()."""

    def setUp(self):
        self.snomed_graph = _make_snomed_graph()

    def test_expands_to_ancestors(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            rollup_codes,
        )
        patient_codes = {"P1": {600}}
        result = rollup_codes(patient_codes, self.snomed_graph)
        self.assertEqual(result["P1"], {600, 500, 100})

    def test_multiple_codes_expand_independently(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            rollup_codes,
        )
        patient_codes = {"P1": {300, 600}}
        result = rollup_codes(patient_codes, self.snomed_graph)
        self.assertEqual(result["P1"], {300, 200, 100, 600, 500})

    def test_codes_not_in_graph_kept_but_not_expanded(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            rollup_codes,
        )
        patient_codes = {"P1": {999, 600}}
        result = rollup_codes(patient_codes, self.snomed_graph)
        self.assertIn(999, result["P1"])
        self.assertIn(600, result["P1"])
        self.assertIn(500, result["P1"])


class TestBuildCooccurrenceMatrix(unittest.TestCase):
    """Tests for build_cooccurrence_matrix()."""

    def test_basic_cooccurrence(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            build_cooccurrence_matrix,
        )
        patient_codes = {
            "P1": {100, 200},
            "P2": {100, 200},
            "P3": {200, 300},
        }
        matrix, code_to_idx, _idx_to_code = build_cooccurrence_matrix(
            patient_codes
        )
        self.assertEqual(matrix[code_to_idx[100], code_to_idx[200]], 2.0)
        self.assertEqual(matrix[code_to_idx[200], code_to_idx[300]], 1.0)
        self.assertEqual(matrix[code_to_idx[100], code_to_idx[300]], 0.0)

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
        _matrix, code_to_idx, idx_to_code = build_cooccurrence_matrix(
            patient_codes, valid_codes={100, 200}
        )
        self.assertNotIn(999, code_to_idx)
        self.assertEqual(len(idx_to_code), 2)

    def test_diagonal_counts_patients(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            build_cooccurrence_matrix,
        )
        patient_codes = {
            "P1": {100, 200},
            "P2": {100},
        }
        matrix, code_to_idx, _ = build_cooccurrence_matrix(patient_codes)
        self.assertEqual(matrix[code_to_idx[100], code_to_idx[100]], 2.0)
        self.assertEqual(matrix[code_to_idx[200], code_to_idx[200]], 1.0)

    def test_empty_input(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            build_cooccurrence_matrix,
        )
        matrix, code_to_idx, _idx_to_code = build_cooccurrence_matrix({})
        self.assertEqual(matrix.shape, (0, 0))
        self.assertEqual(len(code_to_idx), 0)


class TestApplyCountFilter(unittest.TestCase):
    """Tests for apply_count_filter()."""

    def setUp(self):
        self.snomed_graph = _make_snomed_graph()

    def test_drops_unobserved_concepts(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            build_cooccurrence_matrix, apply_count_filter,
        )
        patient_codes = {"P1": {100, 200, 300}}
        matrix, code_to_idx, idx_to_code = build_cooccurrence_matrix(
            patient_codes, valid_codes=set(self.snomed_graph.nodes()),
        )
        self.assertEqual(set(code_to_idx.keys()), {100, 200, 300})

        (filtered_graph, _filtered_matrix,
         _filtered_code_to_idx, _filtered_idx_to_code) = apply_count_filter(
            self.snomed_graph, matrix, code_to_idx, idx_to_code,
        )
        self.assertEqual(set(filtered_graph.nodes()), {100, 200, 300})

    def test_filter_with_zero_diagonals(self):
        """Codes in matrix but with zero diagonal should be dropped."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            apply_count_filter,
        )
        code_to_idx = {100: 0, 200: 1, 300: 2, 500: 3}
        idx_to_code = [100, 200, 300, 500]
        matrix = np.array([
            [3.0, 2.0, 1.0, 0.0],
            [2.0, 2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)

        (filtered_graph, _filtered_matrix,
         filtered_code_to_idx, _filtered_idx_to_code) = apply_count_filter(
            self.snomed_graph, matrix, code_to_idx, idx_to_code,
        )
        self.assertNotIn(500, filtered_graph.nodes())
        self.assertNotIn(500, filtered_code_to_idx)
        self.assertEqual({100, 200, 300}, set(filtered_code_to_idx.keys()))

    def test_matrix_reindexed_after_filter(self):
        """Filtered matrix should have correct dimensions and values."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            apply_count_filter,
        )
        code_to_idx = {100: 0, 200: 1, 500: 2}
        idx_to_code = [100, 200, 500]
        matrix = np.array([
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)

        _, filtered_matrix, filtered_code_to_idx, _filtered_idx_to_code = (
            apply_count_filter(
                self.snomed_graph, matrix, code_to_idx, idx_to_code,
            )
        )
        self.assertEqual(filtered_matrix.shape, (2, 2))
        self.assertEqual(
            filtered_matrix[filtered_code_to_idx[100],
                            filtered_code_to_idx[200]],
            1.0,
        )
        self.assertEqual(sorted(filtered_code_to_idx.values()), [0, 1])

    def test_graph_subgraph_is_copy(self):
        """Filtered graph should be a copy, not a view — safe to modify."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_cooccurrence import (
            apply_count_filter,
        )
        code_to_idx = {100: 0, 200: 1}
        idx_to_code = [100, 200]
        matrix = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)

        filtered_graph, _, _, _ = apply_count_filter(
            self.snomed_graph, matrix, code_to_idx, idx_to_code,
        )
        filtered_graph.add_node(999)
        self.assertNotIn(999, self.snomed_graph.nodes())


if __name__ == "__main__":
    unittest.main()
