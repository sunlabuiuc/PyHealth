"""Build patient co-occurrence matrix from MIMIC diagnosis codes.

Implements the co-occurrence matrix construction described in KEEP
Appendix A.4: "We construct the co-occurrence matrix using the same
codes from our previous graph analysis [...] Co-occurrence is determined
based on the patient's complete medical history, rather than being
restricted to individual visits."

For each patient, we collect all unique SNOMED concepts across their
entire history, then count every pairwise co-occurrence. The result is
a symmetric matrix X where X[i,j] = number of patients who have both
code i and code j.

Roll-up procedure (KEEP Appendix A.4):
    The paper maps each code to ALL of its parent nodes in the SNOMED
    hierarchy: "we implement a roll-up procedure that maps each code to
    its parent codes present in the graph." This densifies the matrix and
    ensures ancestor concepts accumulate co-occurrence signal even if they
    never appear directly in patient records. This is controlled by the
    ``rollup`` parameter in ``extract_patient_codes``.

This module handles both MIMIC-III (ICD-9 only) and MIMIC-IV (mixed
ICD-9 + ICD-10) by routing codes through the appropriate ICD-to-SNOMED
mapping based on the ``icd_version`` column.

Authors: Colton Loew, Desmond Fung, Lookman Olowo, Christiana Beard
Paper: Elhussein et al., "KEEP: Integrating Medical Ontologies with Clinical
       Data for Robust Code Embeddings", CHIL 2025.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def extract_patient_codes_from_df(
    patient_id_col: list,
    code_col: list,
    icd_to_snomed: Dict[str, List[int]],
    version_col: Optional[list] = None,
    icd10_to_snomed: Optional[Dict[str, List[int]]] = None,
    standardize_icd9: Optional[callable] = None,
    standardize_icd10: Optional[callable] = None,
    min_occurrences: int = 2,
) -> Dict[str, Set[int]]:
    """Extract per-patient SNOMED code sets from raw diagnosis data.

    Processes raw ICD diagnosis codes (as parallel lists/columns) and
    maps them to SNOMED concept IDs. Groups by patient, applies
    a minimum occurrence filter per patient, and returns unique
    SNOMED codes per patient.

    Handles both MIMIC-III (ICD-9 only, no version column) and
    MIMIC-IV (mixed ICD-9 + ICD-10, version column present).

    Multi-target ICD handling:
        When an ICD code maps to multiple SNOMED concepts (combination
        codes like "A01.04 Typhoid arthritis" mapping to BOTH typhoid
        fever AND inflammatory arthritis), each occurrence of the ICD
        code counts as an occurrence of ALL its SNOMED targets. This
        "dense" expansion matches the paper's atomic-concept assumption.

    The minimum occurrence filter follows the KEEP paper (Appendix A.4):
    "To establish the presence of a disease in a patient's history,
    we require at least two occurrences."

    No temporal censoring:
        This function uses the patient's COMPLETE medical history with
        no date cutoff. KEEP embedding training intentionally uses all
        history to capture population-level co-occurrence patterns
        (paper Appendix A.4: "Co-occurrence is determined based on the
        patient's complete medical history, rather than being restricted
        to individual visits").

        The reference code's ``create_cohort_sentence.py`` applies a
        censoring rule (2nd-occurrence date must precede the outcome
        date) for extrinsic evaluation tasks where patients have a
        specific index date. We do NOT apply that rule here because:

        1. KEEP embeddings are population-level, trained once and reused.
           They don't know about any specific prediction task.
        2. For downstream prediction, PyHealth's task processors
           (e.g., ``MortalityPredictionMIMIC3``) apply the temporal
           cutoff automatically when generating samples.
        3. Using all history gives richer co-occurrence signal for
           embedding training; censoring would discard real clinical
           patterns for no benefit in this phase.

        If you later need to reproduce the paper's exact UK Biobank
        tasks (which require per-patient index dates), implement the
        censoring rule in your task extractor, not in KEEP training.

    Args:
        patient_id_col: List of patient ID strings, one per diagnosis row.
        code_col: List of ICD code strings, one per diagnosis row.
        icd_to_snomed: Mapping from ICD-9 code strings (dotted format,
            e.g., "428.0") to lists of SNOMED concept IDs.
        version_col: List of ICD version strings ("9" or "10"), one per
            diagnosis row. If None, assumes all codes are ICD-9
            (MIMIC-III behavior).
        icd10_to_snomed: Mapping from ICD-10 code strings to lists of
            SNOMED concept IDs. Required if ``version_col`` is provided.
        standardize_icd9: Function to standardize ICD-9 codes (e.g.,
            "4280" -> "428.0"). If None, codes are used as-is.
        standardize_icd10: Function to standardize ICD-10 codes (e.g.,
            "I509" -> "I50.9"). If None, codes are used as-is.
        min_occurrences: Minimum times a SNOMED code must appear in a
            patient's history to be included. Default: 2 (KEEP paper
            Appendix A.4).

    Returns:
        Dict mapping patient_id (str) to a set of SNOMED concept IDs
        that passed the minimum occurrence filter. Patients with zero
        qualifying codes are excluded.

    Example:
        >>> patient_ids = ["P1", "P1", "P1", "P2", "P2"]
        >>> codes = ["428.0", "428.0", "250.00", "401.9", "401.9"]
        >>> icd9_map = {"428.0": [319835], "250.00": [201826],
        ...             "401.9": [320128]}
        >>> result = extract_patient_codes_from_df(
        ...     patient_ids, codes, icd9_map, min_occurrences=2,
        ... )
        >>> result["P1"]  # 428.0 appears 2x, 250.00 only 1x
        {319835}
    """
    # Count (patient_id, snomed_id) occurrences.
    # Multi-target ICD: each occurrence of the ICD increments ALL its
    # SNOMED targets (dense expansion, matches paper's atomic-concept model).
    patient_code_counts: Dict[str, Dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    unmapped = 0

    for i in range(len(patient_id_col)):
        pid = str(patient_id_col[i])
        raw_code = str(code_col[i])

        # Determine which mapping to use
        if version_col is not None and str(version_col[i]) == "10":
            if standardize_icd10 is not None:
                raw_code = standardize_icd10(raw_code)
            mapping = icd10_to_snomed or {}
        else:
            if standardize_icd9 is not None:
                raw_code = standardize_icd9(raw_code)
            mapping = icd_to_snomed

        snomed_ids = mapping.get(raw_code)
        if not snomed_ids:
            unmapped += 1
            continue

        # Increment count for each SNOMED target (dense expansion)
        for snomed_id in snomed_ids:
            patient_code_counts[pid][snomed_id] += 1

    # Apply min_occurrences filter
    patient_codes: Dict[str, Set[int]] = {}
    for pid, code_counts in patient_code_counts.items():
        codes = {
            sid for sid, count in code_counts.items()
            if count >= min_occurrences
        }
        if codes:
            patient_codes[pid] = codes

    logger.info(
        "Extracted codes for %d patients (%d diagnosis rows unmapped)",
        len(patient_codes),
        unmapped,
    )
    return patient_codes


def rollup_codes(
    patient_codes: Dict[str, Set[int]],
    graph: nx.DiGraph,
) -> Dict[str, Set[int]]:
    """Expand each patient's codes to include all ancestor codes in the graph.

    Implements the roll-up procedure from KEEP Appendix A.4:
    "we implement a roll-up procedure that maps each code to its parent
    codes present in the graph. Adopting a dense roll-up approach, we map
    every code to all of its parents, creating multiple entries when a
    code has multiple parent nodes."

    This densifies the co-occurrence matrix: if a patient has "Type 2
    Diabetes" (a leaf), they also get "Diabetes mellitus" and "Metabolic
    disease" (ancestors). Parent concepts accumulate co-occurrence signal
    even if no patient was ever diagnosed with the parent code directly.

    Args:
        patient_codes: Dict mapping patient_id to set of SNOMED
            concept IDs (from ``extract_patient_codes_from_df``).
        graph: The SNOMED hierarchy DiGraph (edges: child -> parent)
            from ``build_hierarchy_graph``.

    Returns:
        Dict mapping patient_id to expanded set of SNOMED concept IDs
        (original codes + all reachable ancestors in the graph).

    Example:
        >>> # Graph: Diabetes(600) -> Metabolic(500) -> Root(100)
        >>> patient_codes = {"P1": {600}}
        >>> expanded = rollup_codes(patient_codes, G)
        >>> expanded["P1"]
        {600, 500, 100}
    """
    graph_nodes = set(graph.nodes())

    # Pre-compute ancestors for each node (cache for efficiency)
    ancestor_cache: Dict[int, Set[int]] = {}

    def get_ancestors(node: int) -> Set[int]:
        if node in ancestor_cache:
            return ancestor_cache[node]
        ancestors = set()
        # Follow child -> parent edges (successors in our DiGraph)
        queue = [node]
        visited = set()
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            for parent in graph.successors(current):
                if parent in graph_nodes:
                    ancestors.add(parent)
                    queue.append(parent)
        ancestor_cache[node] = ancestors
        return ancestors

    expanded: Dict[str, Set[int]] = {}
    for pid, codes in patient_codes.items():
        all_codes = set(codes)
        for code in codes:
            if code in graph_nodes:
                all_codes.update(get_ancestors(code))
        expanded[pid] = all_codes

    total_original = sum(len(c) for c in patient_codes.values())
    total_expanded = sum(len(c) for c in expanded.values())
    logger.info(
        "Roll-up: %d total codes -> %d (%.1fx expansion)",
        total_original,
        total_expanded,
        total_expanded / max(total_original, 1),
    )
    return expanded


def build_cooccurrence_matrix(
    patient_codes: Dict[str, Set[int]],
    valid_codes: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, Dict[int, int], List[int]]:
    """Build a symmetric co-occurrence matrix from per-patient code sets.

    For each patient, counts every pairwise co-occurrence of SNOMED
    codes across their complete medical history. The result is a
    symmetric matrix where entry (i,j) = number of patients who have
    both code i and code j.

    This is the input to KEEP Stage 2 (regularized GloVe). GloVe
    learns: dot(emb_i, emb_j) ~= log(cooccurrence[i,j]).

    Args:
        patient_codes: Dict mapping patient_id to set of SNOMED
            concept IDs.
        valid_codes: If provided, only include these SNOMED IDs in the
            matrix. Codes not in this set are dropped. Use this to
            restrict to codes that appear in the SNOMED graph.

    Returns:
        Tuple of:
            - matrix: np.ndarray of shape ``(N, N)`` where N is the
              number of unique codes. Symmetric, dtype float32.
            - code_to_idx: Dict mapping SNOMED concept_id to matrix
              row/column index.
            - idx_to_code: List where ``idx_to_code[i]`` is the SNOMED
              concept_id for row/column i.

    Example:
        >>> patient_codes = {
        ...     "P1": {100, 200, 300},
        ...     "P2": {100, 200},
        ...     "P3": {200, 300},
        ... }
        >>> matrix, code_to_idx, idx_to_code = build_cooccurrence_matrix(
        ...     patient_codes
        ... )
        >>> matrix[code_to_idx[100], code_to_idx[200]]
        2.0
    """
    # Collect all unique codes
    all_codes: Set[int] = set()
    for codes in patient_codes.values():
        all_codes.update(codes)

    if valid_codes is not None:
        all_codes = all_codes & valid_codes

    idx_to_code = sorted(all_codes)
    code_to_idx = {code: i for i, code in enumerate(idx_to_code)}
    n = len(idx_to_code)

    logger.info(
        "Building co-occurrence matrix: %d unique codes from %d patients",
        n,
        len(patient_codes),
    )

    # Count pairwise co-occurrences
    matrix = np.zeros((n, n), dtype=np.float32)

    for pid, codes in patient_codes.items():
        # Filter to valid codes and convert to indices
        valid = [code_to_idx[c] for c in codes if c in code_to_idx]
        # Count all pairs
        for i, j in combinations(valid, 2):
            matrix[i, j] += 1.0
            matrix[j, i] += 1.0

    # Diagonal = number of patients who have each code
    for pid, codes in patient_codes.items():
        for c in codes:
            if c in code_to_idx:
                matrix[code_to_idx[c], code_to_idx[c]] += 1.0

    nonzero = np.count_nonzero(matrix)
    density = nonzero / max(n * n, 1)
    logger.info(
        "Co-occurrence matrix: %d x %d, %d non-zero entries (%.1f%% dense)",
        n,
        n,
        nonzero,
        density * 100,
    )
    return matrix, code_to_idx, idx_to_code


def apply_count_filter(
    graph: nx.DiGraph,
    cooc_matrix: np.ndarray,
    code_to_idx: Dict[int, int],
    idx_to_code: List[int],
) -> Tuple[nx.DiGraph, np.ndarray, Dict[int, int], List[int]]:
    """Drop SNOMED concepts with zero patient observations.

    Implements the count filter implicit in the KEEP paper (visible in
    G2Lab's ``_ct_filter`` file suffix). After dense rollup, the
    co-occurrence matrix diagonal contains the patient count per concept.
    Concepts with zero count never appeared in any patient's confirmed
    disease set — they're ontologically valid but clinically unobserved
    in the training data.

    Why this matters for KEEP:
        Without this filter, Stage 1 (Node2Vec) trains on ~68K concepts
        but Stage 2 (GloVe) only refines the ~5K observed ones. The
        unobserved concepts end up with Stage 1 embeddings only — mixed
        quality in the exported file. The filter ensures every exported
        embedding has undergone both stages.

    Why this matches the paper:
        The paper reports ~5,686 concepts at depth 5. Our raw graph has
        ~68K. The gap is this filter. G2Lab's pipeline applies it
        upstream (not in the public repo); we replicate it here.

    Args:
        graph: SNOMED hierarchy DiGraph (from ``build_hierarchy_graph``).
        cooc_matrix: Co-occurrence matrix from ``build_cooccurrence_matrix``.
            Diagonal contains per-concept patient counts.
        code_to_idx: Dict mapping concept_id to current matrix row/column.
        idx_to_code: List mapping matrix index to concept_id.

    Returns:
        Tuple of:
            - filtered_graph: Subgraph containing only observed concepts.
            - filtered_matrix: Reindexed co-occurrence matrix, shape
              ``(N_obs, N_obs)`` where N_obs is the observed concept count.
            - filtered_code_to_idx: New code_to_idx for filtered matrix.
            - filtered_idx_to_code: New idx_to_code for filtered matrix.

    Example:
        >>> patient_codes = {"P1": {100, 200}}  # only 100, 200 observed
        >>> matrix, c2i, i2c = build_cooccurrence_matrix(
        ...     patient_codes, valid_codes={100, 200, 300, 400}
        ... )
        >>> # matrix includes 300, 400 but their diagonals are zero
        >>> fg, fm, fc2i, fi2c = apply_count_filter(graph, matrix, c2i, i2c)
        >>> sorted(fc2i.keys())  # only observed concepts remain
        [100, 200]
    """
    # Identify observed concepts (non-zero on diagonal)
    diagonal = np.diagonal(cooc_matrix)
    observed_indices = [
        i for i, count in enumerate(diagonal) if count > 0
    ]
    observed_codes = {idx_to_code[i] for i in observed_indices}

    logger.info(
        "Count filter: %d / %d concepts have >0 patient observations",
        len(observed_codes),
        len(idx_to_code),
    )

    # Rebuild graph as subgraph, copy to be safe for downstream modifications
    filtered_graph = graph.subgraph(observed_codes).copy()

    # Rebuild matrix with only observed indices
    if observed_indices:
        filtered_matrix = cooc_matrix[np.ix_(observed_indices, observed_indices)]
    else:
        filtered_matrix = np.zeros((0, 0), dtype=cooc_matrix.dtype)

    # Rebuild code_to_idx / idx_to_code for the filtered matrix
    filtered_idx_to_code = [idx_to_code[i] for i in observed_indices]
    filtered_code_to_idx = {
        code: i for i, code in enumerate(filtered_idx_to_code)
    }

    logger.info(
        "After count filter: graph %d nodes, matrix %s",
        filtered_graph.number_of_nodes(),
        filtered_matrix.shape,
    )
    return (
        filtered_graph,
        filtered_matrix,
        filtered_code_to_idx,
        filtered_idx_to_code,
    )
