"""Intrinsic evaluation for KEEP embeddings — reproducible vs paper Table 2.

Implements the intrinsic evaluation methodology from the KEEP paper
(Appendix B.1): for each code, compute cosine similarity between its
embedding and all other embeddings, then correlate those similarities
with (1) Resnik semantic similarity on the SNOMED graph and
(2) co-occurrence counts.

Paper Table 2 targets (UK Biobank):
    Resnik correlation:      0.68
    Co-occurrence correlation: 0.62

These are the numbers we should reproduce if our embeddings are
paper-faithful. Use this script to validate ``keep_snomed.txt`` output.

Methodology from paper Appendix B.1:
    "For each code, we identified the ten most similar concepts based
    on cosine similarity, along with 150 randomly sampled concepts.
    For each selected concept, we computed its Resnik similarity and
    its co-occurrence frequency with the original code. We measured
    the correlation between cosine similarity and both Resnik similarity
    and co-occurrence values. To ensure statistical robustness, we
    repeated the experiment 250 times and report the median correlation
    values across all runs."

Authors: Colton Loew, Desmond Fung, Lookman Olowo, Christiana Beard
Paper: Elhussein et al., "KEEP: Integrating Medical Ontologies with Clinical
       Data for Robust Code Embeddings", CHIL 2025.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def compute_information_content(
    graph: nx.DiGraph,
    patient_code_sets: Optional[List[set]] = None,
) -> Dict[int, float]:
    """Compute information content (IC) for each concept in the graph.

    Two modes:
    1. Graph-based IC (default when ``patient_code_sets`` is None):
       IC(c) = -log(|descendants(c)| / |total nodes|)
       Concepts with many descendants = less specific = lower IC.

    2. Corpus-based IC (when ``patient_code_sets`` is provided):
       IC(c) = -log(count(c) / total_observations)
       where count(c) includes descendants (dense rollup).
       More faithful to the paper's methodology (uses patient data).

    Args:
        graph: SNOMED hierarchy DiGraph (edges: child -> parent).
        patient_code_sets: Optional list of per-patient SNOMED code sets
            for corpus-based IC. If None, uses graph topology.

    Returns:
        Dict mapping concept_id to IC value (higher = more specific).
    """
    ic: Dict[int, float] = {}

    if patient_code_sets is not None:
        # Corpus-based IC: count how often each concept appears across patients
        counts: Dict[int, int] = defaultdict(int)
        total = 0
        for codes in patient_code_sets:
            for code in codes:
                counts[code] += 1
                total += 1
        if total == 0:
            return {n: 0.0 for n in graph.nodes()}
        for node in graph.nodes():
            freq = counts.get(node, 0) / total
            ic[node] = -np.log(freq) if freq > 0 else 0.0
    else:
        # Graph-based IC: smaller subtree = more specific = higher IC
        total = graph.number_of_nodes()
        if total == 0:
            return ic
        # Count descendants per node (nodes that transitively have us as ancestor)
        reverse = graph.reverse()
        for node in graph.nodes():
            # +1 includes the node itself
            num_desc = len(nx.descendants(reverse, node)) + 1
            freq = num_desc / total
            ic[node] = -np.log(freq) if freq > 0 else 0.0

    return ic


def resnik_similarity(
    graph: nx.DiGraph,
    code_a: int,
    code_b: int,
    ic: Dict[int, float],
) -> float:
    """Resnik semantic similarity: IC of lowest common ancestor.

    Resnik(A, B) = IC(LCA(A, B))

    Two concepts with a specific (high-IC) common ancestor are more
    similar than two concepts whose only common ancestor is the root
    (low-IC = generic).

    Args:
        graph: SNOMED DiGraph (edges: child -> parent).
        code_a: First concept_id.
        code_b: Second concept_id.
        ic: Precomputed information content dict.

    Returns:
        IC of the lowest common ancestor. Returns 0.0 if no common
        ancestor exists (disconnected concepts).
    """
    # Ancestors in our DiGraph = nodes reachable via outgoing edges
    # (since edges go child -> parent). Include self.
    anc_a = {code_a} | nx.descendants(graph, code_a)
    anc_b = {code_b} | nx.descendants(graph, code_b)
    common = anc_a & anc_b
    if not common:
        return 0.0
    # LCA = common ancestor with highest IC (most specific)
    return max(ic.get(c, 0.0) for c in common)


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute N×N cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    normalized = embeddings / safe_norms
    return normalized @ normalized.T


def resnik_correlation(
    embeddings: np.ndarray,
    node_ids: List[int],
    graph: nx.DiGraph,
    k1: int = 10,
    k2: int = 150,
    num_runs: int = 250,
    seed: int = 42,
    ic: Optional[Dict[int, float]] = None,
) -> Dict[str, float]:
    """Paper-faithful Resnik correlation (paper Appendix B.1).

    For each concept, takes the top-K1 most similar concepts by cosine
    similarity + K2 random concepts, then correlates cosine similarity
    with Resnik similarity across that sample. Repeats num_runs times
    (randomness from the K2 sample) and reports median.

    Paper Table 2 target: 0.68 on UK Biobank KEEP embeddings.

    Args:
        embeddings: KEEP embedding matrix, shape ``(N, dim)``.
        node_ids: List of SNOMED concept_ids, same order as embedding rows.
        graph: SNOMED hierarchy DiGraph.
        k1: Number of top-similar concepts per source concept. Default: 10.
        k2: Number of random concepts per source. Default: 150.
        num_runs: Number of bootstrap iterations. Default: 250.
        seed: Random seed. Default: 42.
        ic: Precomputed information content. Computed from graph if None.

    Returns:
        Dict with keys:
            "mean": Mean Pearson correlation across runs
            "median": Median correlation across runs (paper's reported metric)
            "std": Std of correlations
            "min": Min correlation
            "max": Max correlation
    """
    if ic is None:
        ic = compute_information_content(graph)

    if len(node_ids) < k1 + k2 + 1:
        raise ValueError(
            f"Need at least {k1 + k2 + 1} concepts, got {len(node_ids)}"
        )

    # Precompute cosine similarity matrix
    cos_sim = _cosine_similarity_matrix(embeddings)

    # Precompute ancestor sets for faster LCA lookup
    # For each concept: ancestors = itself + all nodes reachable via outgoing edges
    ancestors: Dict[int, set] = {
        cid: {cid} | nx.descendants(graph, cid) for cid in node_ids
    }

    rng = np.random.default_rng(seed)
    n = len(node_ids)
    correlations: List[float] = []

    for run in range(num_runs):
        run_correlations: List[float] = []
        for i in range(n):
            source_id = node_ids[i]
            if source_id not in ancestors or not ancestors[source_id]:
                continue

            # Top-K1 most similar (excluding self)
            sim_row = cos_sim[i].copy()
            sim_row[i] = -np.inf  # exclude self
            top_k1_indices = np.argpartition(sim_row, -k1)[-k1:]

            # K2 random indices (excluding self and top-K1)
            exclude = set(top_k1_indices.tolist()) | {i}
            candidates = [j for j in range(n) if j not in exclude]
            if len(candidates) < k2:
                continue
            random_k2_indices = rng.choice(
                candidates, size=k2, replace=False,
            )

            sample_indices = np.concatenate([top_k1_indices, random_k2_indices])

            # Compute cosine and Resnik for each pair
            cos_values = cos_sim[i, sample_indices]
            resnik_values = []
            for j in sample_indices:
                target_id = node_ids[int(j)]
                # Resnik = max IC over common ancestors
                common = ancestors[source_id] & ancestors.get(target_id, set())
                if common:
                    resnik_values.append(max(ic.get(c, 0.0) for c in common))
                else:
                    resnik_values.append(0.0)
            resnik_values = np.array(resnik_values)

            # Pearson correlation (skip if degenerate)
            if np.std(cos_values) > 1e-10 and np.std(resnik_values) > 1e-10:
                corr = np.corrcoef(cos_values, resnik_values)[0, 1]
                if not np.isnan(corr):
                    run_correlations.append(corr)

        if run_correlations:
            correlations.append(float(np.mean(run_correlations)))

    if not correlations:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": float(np.mean(correlations)),
        "median": float(np.median(correlations)),
        "std": float(np.std(correlations)),
        "min": float(np.min(correlations)),
        "max": float(np.max(correlations)),
    }


def cooccurrence_correlation(
    embeddings: np.ndarray,
    node_ids: List[int],
    cooc_matrix: np.ndarray,
    code_to_idx: Dict[int, int],
    k1: int = 10,
    k2: int = 150,
    num_runs: int = 250,
    seed: int = 42,
) -> Dict[str, float]:
    """Correlation between cosine similarity and co-occurrence counts.

    Paper Table 2 target: 0.62 on UK Biobank KEEP embeddings.

    Same K1+K2 sampling methodology as resnik_correlation, but uses
    co-occurrence counts as the comparison signal instead of Resnik.

    Args:
        embeddings: KEEP embedding matrix.
        node_ids: List of SNOMED concept_ids.
        cooc_matrix: Co-occurrence matrix (N×N).
        code_to_idx: Mapping from concept_id to matrix index.
        k1, k2, num_runs, seed: Same as resnik_correlation.

    Returns:
        Dict with keys mean/median/std/min/max correlations.
    """
    if len(node_ids) < k1 + k2 + 1:
        raise ValueError(
            f"Need at least {k1 + k2 + 1} concepts, got {len(node_ids)}"
        )

    cos_sim = _cosine_similarity_matrix(embeddings)
    rng = np.random.default_rng(seed)
    n = len(node_ids)
    correlations: List[float] = []

    for run in range(num_runs):
        run_correlations: List[float] = []
        for i in range(n):
            source_id = node_ids[i]
            if source_id not in code_to_idx:
                continue

            sim_row = cos_sim[i].copy()
            sim_row[i] = -np.inf
            top_k1_indices = np.argpartition(sim_row, -k1)[-k1:]

            exclude = set(top_k1_indices.tolist()) | {i}
            candidates = [j for j in range(n) if j not in exclude]
            if len(candidates) < k2:
                continue
            random_k2_indices = rng.choice(
                candidates, size=k2, replace=False,
            )

            sample_indices = np.concatenate([top_k1_indices, random_k2_indices])

            cos_values = cos_sim[i, sample_indices]
            cooc_values = []
            src_idx = code_to_idx[source_id]
            for j in sample_indices:
                target_id = node_ids[int(j)]
                if target_id in code_to_idx:
                    cooc_values.append(cooc_matrix[src_idx, code_to_idx[target_id]])
                else:
                    cooc_values.append(0.0)
            cooc_values = np.array(cooc_values)

            # Log-transform co-occurrences (huge dynamic range)
            log_cooc = np.log(cooc_values + 1)

            if np.std(cos_values) > 1e-10 and np.std(log_cooc) > 1e-10:
                corr = np.corrcoef(cos_values, log_cooc)[0, 1]
                if not np.isnan(corr):
                    run_correlations.append(corr)

        if run_correlations:
            correlations.append(float(np.mean(run_correlations)))

    if not correlations:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return {
        "mean": float(np.mean(correlations)),
        "median": float(np.median(correlations)),
        "std": float(np.std(correlations)),
        "min": float(np.min(correlations)),
        "max": float(np.max(correlations)),
    }


def load_keep_embeddings(
    text_path: str | Path,
    embedding_dim: int = 100,
) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings from keep_snomed.txt format.

    Args:
        text_path: Path to GloVe-format text file.
        embedding_dim: Expected embedding dimension.

    Returns:
        Tuple of:
            embeddings: np.ndarray of shape (N, embedding_dim).
            tokens: List of N token strings (SNOMED concept codes).
    """
    tokens: List[str] = []
    vectors: List[List[float]] = []
    with open(text_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < embedding_dim + 1:
                continue
            tokens.append(parts[0])
            vectors.append([float(x) for x in parts[1:embedding_dim + 1]])
    return np.array(vectors, dtype=np.float32), tokens


def evaluate_embeddings(
    embeddings: np.ndarray,
    node_ids: List[int],
    graph: nx.DiGraph,
    cooc_matrix: Optional[np.ndarray] = None,
    code_to_idx: Optional[Dict[int, int]] = None,
    k1: int = 10,
    k2: int = 150,
    num_runs: int = 250,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Run both Resnik and co-occurrence correlations.

    Paper Table 2 targets (UK Biobank KEEP):
        Resnik correlation:      median ~0.68
        Co-occurrence correlation: median ~0.62

    Args:
        embeddings: KEEP embedding matrix.
        node_ids: List of concept_ids in same order as embedding rows.
        graph: SNOMED hierarchy DiGraph.
        cooc_matrix: Optional co-occurrence matrix for co-occ correlation.
        code_to_idx: Optional index mapping for co-occ matrix.
        num_runs: Bootstrap iterations. Default: 250.
        seed: Random seed. Default: 42.

    Returns:
        Dict with "resnik" (always) and "cooccurrence" (if cooc_matrix
        provided), each a dict of mean/median/std/min/max.
    """
    results: Dict[str, Dict[str, float]] = {}

    logger.info("Running Resnik correlation (%d runs)...", num_runs)
    results["resnik"] = resnik_correlation(
        embeddings, node_ids, graph,
        k1=k1, k2=k2, num_runs=num_runs, seed=seed,
    )

    if cooc_matrix is not None and code_to_idx is not None:
        logger.info("Running co-occurrence correlation (%d runs)...", num_runs)
        results["cooccurrence"] = cooccurrence_correlation(
            embeddings, node_ids, cooc_matrix, code_to_idx,
            k1=k1, k2=k2, num_runs=num_runs, seed=seed,
        )

    return results
