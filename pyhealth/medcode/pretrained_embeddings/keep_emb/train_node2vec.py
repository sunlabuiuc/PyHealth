"""Train Node2Vec embeddings on the SNOMED knowledge graph (KEEP Stage 1).

Implements KEEP Stage 1 as described in Section 4.2 of the paper:
"We first generate embeddings from knowledge graphs" using Node2Vec
(Grover & Leskovec, 2016) on the OMOP "is-a" hierarchy.

Node2Vec performs random walks on the graph and feeds the resulting
"sentences" of node IDs into Word2Vec (Skip-gram). Nodes that appear
in similar walk contexts receive similar embedding vectors.

With p=1, q=1 (KEEP's configuration), this is equivalent to DeepWalk
(Perozzi et al., 2014) -- completely unbiased random walks.

The output embeddings capture **ontological structure**: sibling
concepts (e.g., Type 1 DM and Type 2 DM) get similar vectors because
random walks through either pass through the same parent nodes.

These structural embeddings are used as:
  1. Initialization for KEEP Stage 2 (regularized GloVe).
  2. Regularization target -- Stage 2 penalizes drift from these vectors,
     preserving ontological structure while learning clinical co-occurrence.

Hyperparameters:
    Default values are sourced from the KEEP paper Appendix A.2, Table 5.
    The paper does not report sensitivity analysis on Node2Vec parameters.
    This is an opportunity for ablation -- particularly p/q values (BFS vs
    DFS bias on the SNOMED hierarchy) and walk_length (interaction with
    graph depth). See the ``train_node2vec`` docstring for details.

Evaluating Stage 1 ablations:
    Two options, cheap and expensive (see KEEP paper Section 5.2, Table 2):

    Option 1 -- Resnik similarity correlation (cheap, fast, no patient data):
        For every pair of SNOMED concepts in the graph, compute their Resnik
        similarity (information content of the lowest common ancestor). Then
        compute cosine similarity for every pair in the trained embeddings.
        Correlate the two. Higher correlation means the embeddings better
        preserve the ontological structure. This is pure matrix computation
        on the graph -- no patients, no downstream model, runs in minutes.
        Good enough to score p/q and walk_length configs cheaply.

    Option 2 -- End-to-end downstream task performance (expensive, definitive):
        Run the full pipeline (Stage 1 -> Stage 2 -> train prediction model)
        for each config and compare AUPRC on a held-out clinical task.
        This is the gold standard but costs hours per config.

Authors: Desmond Fung, Colton Loew, Lookman Olowo, Christiana Beard
Paper: Elhussein et al., "KEEP: Integrating Medical Ontologies with Clinical
       Data for Robust Code Embeddings", CHIL 2025.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def train_node2vec(
    graph: nx.DiGraph,
    embedding_dim: int = 100,
    walk_length: int = 30,
    num_walks: int = 750,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 10,
    min_count: int = 1,
    workers: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, List[int]]:
    """Train Node2Vec embeddings on a knowledge graph.

    Performs random walks on the graph, then trains Word2Vec (Skip-gram)
    on the resulting walk "sentences" to produce dense embeddings for
    each node.

    The graph is converted to undirected for walking, since the "Is a"
    hierarchy should be traversable in both directions (parent-to-child
    and child-to-parent) during random walks.

    Default hyperparameters are from the KEEP paper (Appendix A.2,
    Table 5). The paper reports no sensitivity analysis on these values.

    Ablation opportunities:
        - **p and q**: p=1, q=1 is DeepWalk (unbiased). Medical
          ontologies might benefit from BFS-biased walks (p=1, q=2)
          that stay local in a subtree, producing tighter sibling
          clusters. Or DFS-biased (p=1, q=0.5) to explore deeper
          hierarchical chains. No one has tested this on SNOMED.
        - **walk_length**: 30 is reasonable for depth-5 graphs, but
          larger graphs may benefit from shorter walks to avoid noise
          from distant subtrees.
        - **num_walks**: 750 walks/node on ~5K nodes = 3.75M walks.
          Diminishing returns likely set in before 750; lower values
          would speed up training significantly.
        - **embedding_dim**: 100 was chosen by the paper after a
          preliminary sweep showing reconstruction loss plateaus at 100.

    Args:
        graph: NetworkX DiGraph (e.g., SNOMED "Is a" hierarchy from
            ``build_hierarchy_graph``). Nodes should be integer
            concept IDs.
        embedding_dim: Dimensionality of output embeddings.
            Default: 100 (KEEP paper Table 5).
        walk_length: Length of each random walk.
            Default: 30 (KEEP paper Table 5).
        num_walks: Number of random walks per node.
            Default: 750 (KEEP paper Table 5).
        p: Node2Vec return parameter. Controls likelihood of
            immediately revisiting a node. p=1 means no preference.
            Default: 1.0 (KEEP paper Table 5).
        q: Node2Vec in-out parameter. Controls search bias toward
            BFS (q>1) or DFS (q<1). q=1 means no preference.
            Default: 1.0 (KEEP paper Table 5).
        window: Word2Vec context window size.
            Default: 10 (KEEP paper Table 5).
        min_count: Minimum word frequency for Word2Vec.
            Default: 1 (include all nodes, KEEP paper Table 5).
        workers: Number of parallel workers. Default: 4.
        seed: Random seed for reproducibility. Default: 42.

    Returns:
        Tuple of:
            - embeddings: np.ndarray of shape ``(num_nodes, embedding_dim)``
              where ``embeddings[i]`` is the vector for ``node_ids[i]``.
            - node_ids: List of integer concept IDs in the same order as
              the embedding rows.

    Raises:
        ImportError: If the ``node2vec`` package is not installed.
            Install with: ``pip install node2vec`` or
            ``pip install pyhealth[keep]``.
        ValueError: If the graph has no nodes.

    Example:
        >>> from pyhealth.medcode.pretrained_embeddings.keep_emb import (
        ...     build_omop_graph, train_node2vec,
        ... )
        >>> G = build_omop_graph.build_hierarchy_graph(
        ...     "data/athena/CONCEPT.csv",
        ...     "data/athena/CONCEPT_RELATIONSHIP.csv",
        ... )
        >>> embeddings, node_ids = train_node2vec.train_node2vec(G)
        >>> embeddings.shape
        (65375, 100)
    """
    try:
        from node2vec import Node2Vec
    except ImportError:
        raise ImportError(
            "node2vec package is required for KEEP Stage 1. "
            "Install with: pip install node2vec  "
            "or: pip install pyhealth[keep]"
        )

    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot train Node2Vec on an empty graph.")

    # Convert to undirected for random walks.
    # The "Is a" hierarchy is directed (child -> parent), but random
    # walks should traverse both directions to capture full structure.
    undirected = graph.to_undirected()

    logger.info(
        "Training Node2Vec: %d nodes, %d edges, dim=%d, "
        "walks=%d, walk_len=%d, p=%.1f, q=%.1f",
        undirected.number_of_nodes(),
        undirected.number_of_edges(),
        embedding_dim,
        num_walks,
        walk_length,
        p,
        q,
    )

    # Initialize Node2Vec model
    # quiet=True suppresses the progress bar from node2vec internals
    model = Node2Vec(
        undirected,
        dimensions=embedding_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
        quiet=True,
    )

    # Train Word2Vec on the random walks
    # batch_words=4096 matches KEEP paper Table 5
    w2v = model.fit(
        window=window,
        min_count=min_count,
        batch_words=4096,
        seed=seed,
    )

    # Extract embeddings in a consistent order
    node_ids = sorted(graph.nodes())
    embeddings = np.zeros((len(node_ids), embedding_dim), dtype=np.float32)

    # Compute mean vector as fallback for any nodes not in the model
    # (can happen if a node is completely isolated)
    all_vectors = [
        w2v.wv[str(nid)]
        for nid in node_ids
        if str(nid) in w2v.wv
    ]
    mean_vector = (
        np.mean(all_vectors, axis=0) if all_vectors
        else np.zeros(embedding_dim, dtype=np.float32)
    )

    missing = 0
    for i, nid in enumerate(node_ids):
        key = str(nid)
        if key in w2v.wv:
            embeddings[i] = w2v.wv[key]
        else:
            embeddings[i] = mean_vector
            missing += 1

    if missing > 0:
        logger.warning(
            "%d nodes not found in Word2Vec model (assigned mean vector)",
            missing,
        )

    logger.info(
        "Node2Vec training complete: %d embeddings of dim %d",
        len(node_ids),
        embedding_dim,
    )
    return embeddings, node_ids


def build_node_id_to_index(node_ids: List[int]) -> Dict[int, int]:
    """Build a lookup from SNOMED concept_id to embedding matrix row index.

    Useful for downstream steps that need to look up a specific concept's
    embedding by its SNOMED ID.

    Args:
        node_ids: List of SNOMED concept IDs (same order as the embedding
            matrix rows returned by ``train_node2vec``).

    Returns:
        Dict mapping concept_id (int) to row index (int).

    Example:
        >>> node_ids = [100, 200, 300]
        >>> idx = build_node_id_to_index(node_ids)
        >>> idx[200]
        1
    """
    return {nid: i for i, nid in enumerate(node_ids)}
