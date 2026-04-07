"""Build SNOMED knowledge graph and ICD-to-SNOMED mappings from Athena CSVs.

Constructs the OMOP knowledge graph described in KEEP Section 4.2:
"We construct the OMOP knowledge graph using only condition concepts
connected through hierarchical 'is-a' relationships." This graph is the
input to KEEP Stage 1 (Node2Vec structural embeddings).

Parses Athena OMOP vocabulary files (downloadable from
https://athena.ohdsi.org/) to produce:

1. A SNOMED "Is a" hierarchy as a NetworkX DiGraph, depth-limited to 5 levels.
   This graph is what Node2Vec walks on (Stage 1 of KEEP).

2. ICD-9-CM -> SNOMED and ICD-10-CM -> SNOMED mapping dictionaries.
   These translate between what MIMIC stores (ICD codes) and what KEEP
   operates on (SNOMED concept IDs).

Why SNOMED and not ICD?
    ICD is a billing taxonomy -- its hierarchy reflects administrative
    categories (e.g., "Diseases of the Circulatory System"). SNOMED is a
    clinical ontology -- its hierarchy encodes medical knowledge (e.g.,
    "Hypertensive heart disease" is linked to both cardiovascular and
    hypertensive subtrees). Node2Vec walking on the SNOMED graph learns
    medically meaningful relationships that ICD's hierarchy cannot capture.

Athena file format:
    All CSVs are tab-separated with a header row. Key files:
    - CONCEPT.csv: concept_id, concept_name, domain_id, vocabulary_id, ...
    - CONCEPT_RELATIONSHIP.csv: concept_id_1, concept_id_2, relationship_id, ...

Authors: Desmond Fung, Colton Loew, Lookman Olowo, Christiana Beard
Paper: Elhussein et al., "KEEP: Integrating Medical Ontologies with Clinical
       Data for Robust Code Embeddings", CHIL 2025.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def load_concepts(
    concept_csv: str | Path,
    vocabulary_ids: Optional[list[str]] = None,
    domain_ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load and filter the Athena CONCEPT.csv file.

    CONCEPT.csv is the master catalog of all medical concepts across
    vocabularies (SNOMED, ICD9CM, ICD10CM, RxNorm, etc.). Each row is
    one concept with a unique ``concept_id``.

    We filter early to avoid loading all 2.2M rows into memory when we
    only need SNOMED Conditions (~150K rows) or ICD codes (~100K rows).

    Args:
        concept_csv: Path to the Athena CONCEPT.csv file.
        vocabulary_ids: Filter to these vocabularies (e.g., ["SNOMED"]).
            If None, loads all vocabularies.
        domain_ids: Filter to these domains (e.g., ["Condition"]).
            If None, loads all domains.

    Returns:
        pd.DataFrame: Filtered concepts with columns: concept_id,
            concept_name, domain_id, vocabulary_id, concept_class_id,
            standard_concept, concept_code.

    Example:
        >>> df = load_concepts("data/athena/CONCEPT.csv",
        ...                    vocabulary_ids=["SNOMED"],
        ...                    domain_ids=["Condition"])
        >>> len(df)  # ~150K SNOMED Condition concepts
        150234
    """
    # Tab-separated, read only the columns we need to save memory
    usecols = [
        "concept_id",
        "concept_name",
        "domain_id",
        "vocabulary_id",
        "concept_class_id",
        "standard_concept",
        "concept_code",
        "invalid_reason",
    ]
    df = pd.read_csv(
        concept_csv,
        sep="\t",
        usecols=usecols,
        dtype={"concept_id": int, "concept_code": str},
        low_memory=False,
    )

    # Filter to valid concepts (no invalid_reason)
    df = df[df["invalid_reason"].isna()]

    if vocabulary_ids is not None:
        df = df[df["vocabulary_id"].isin(vocabulary_ids)]
    if domain_ids is not None:
        df = df[df["domain_id"].isin(domain_ids)]

    logger.info(
        "Loaded %d concepts (vocabularies=%s, domains=%s)",
        len(df),
        vocabulary_ids,
        domain_ids,
    )
    return df


def load_relationships(
    relationship_csv: str | Path,
    relationship_ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load and filter the Athena CONCEPT_RELATIONSHIP.csv file.

    CONCEPT_RELATIONSHIP.csv contains ~14.5M rows encoding relationships
    between concepts. Key relationship types:

    - "Is a": hierarchy edges (child -> parent) within a vocabulary
    - "Maps to": cross-vocabulary mappings (e.g., ICD-9 -> SNOMED)
    - "Subsumes": inverse of "Is a" (parent -> child)

    Args:
        relationship_csv: Path to the Athena CONCEPT_RELATIONSHIP.csv file.
        relationship_ids: Filter to these relationship types
            (e.g., ["Is a", "Maps to"]). If None, loads all.

    Returns:
        pd.DataFrame: Filtered relationships with columns: concept_id_1,
            concept_id_2, relationship_id.

    Example:
        >>> rels = load_relationships("data/athena/CONCEPT_RELATIONSHIP.csv",
        ...                           relationship_ids=["Is a"])
        >>> len(rels)  # ~800K "Is a" edges
        823456
    """
    usecols = [
        "concept_id_1",
        "concept_id_2",
        "relationship_id",
        "invalid_reason",
    ]
    df = pd.read_csv(
        relationship_csv,
        sep="\t",
        usecols=usecols,
        dtype={"concept_id_1": int, "concept_id_2": int},
        low_memory=False,
    )

    # Filter to valid relationships
    df = df[df["invalid_reason"].isna()]

    if relationship_ids is not None:
        df = df[df["relationship_id"].isin(relationship_ids)]

    logger.info(
        "Loaded %d relationships (types=%s)",
        len(df),
        relationship_ids,
    )
    return df


def build_hierarchy_graph(
    concept_csv: str | Path,
    relationship_csv: str | Path,
    max_depth: int = 5,
    vocabulary_id: str = "SNOMED",
    domain_id: str = "Condition",
) -> nx.DiGraph:
    """Build a depth-limited SNOMED "Is a" hierarchy as a NetworkX DiGraph.

    This is the knowledge graph that Node2Vec walks on in KEEP Stage 1.
    The graph structure encodes medical knowledge: nodes that share
    ancestry (e.g., "Type 1 DM" and "Type 2 DM" under "Diabetes mellitus")
    will be visited in similar random walk contexts, producing similar
    Node2Vec embeddings.

    Algorithm:
        1. Load all SNOMED Condition concepts (standard concepts only).
        2. Load all "Is a" relationships between those concepts.
        3. Build a DiGraph with child -> parent edges.
        4. Find root nodes (nodes with no parents within our set).
        5. BFS from roots, keeping only nodes within ``max_depth`` levels.

    Why depth-limit to 5?
        The KEEP paper (Section 4.2) limits to 5 levels from root. Deeper
        levels contain overly specific concepts that few patients have,
        adding noise to Node2Vec walks. At depth 5, the KEEP paper reports
        ~5,686 concepts -- rich enough for meaningful walks, sparse enough
        to avoid noise.

    Args:
        concept_csv: Path to Athena CONCEPT.csv.
        relationship_csv: Path to Athena CONCEPT_RELATIONSHIP.csv.
        max_depth: Maximum hierarchy depth from root nodes. Default: 5.
        vocabulary_id: Vocabulary to build graph from. Default: "SNOMED".
        domain_id: Domain to filter concepts. Default: "Condition".

    Returns:
        nx.DiGraph: Directed graph where each edge is (child, parent).
            Node attributes include "concept_name" and "concept_code".
            The graph contains only nodes reachable within ``max_depth``
            from a root node.

    Example:
        >>> G = build_hierarchy_graph("data/athena/CONCEPT.csv",
        ...                           "data/athena/CONCEPT_RELATIONSHIP.csv")
        >>> G.number_of_nodes()
        5686
        >>> G.number_of_edges()
        7234
    """
    # Step 1: Load SNOMED Condition concepts (standard only)
    concepts = load_concepts(
        concept_csv,
        vocabulary_ids=[vocabulary_id],
        domain_ids=[domain_id],
    )
    # Keep only standard concepts (standard_concept == "S")
    concepts = concepts[concepts["standard_concept"] == "S"]
    valid_ids: Set[int] = set(concepts["concept_id"])
    logger.info("Found %d standard %s %s concepts", len(valid_ids),
                vocabulary_id, domain_id)

    # Step 2: Load "Is a" relationships where BOTH endpoints are in our set
    rels = load_relationships(relationship_csv, relationship_ids=["Is a"])
    rels = rels[
        rels["concept_id_1"].isin(valid_ids)
        & rels["concept_id_2"].isin(valid_ids)
    ]
    logger.info("Found %d 'Is a' edges between valid concepts", len(rels))

    # Step 3: Build directed graph (child -> parent)
    # In Athena's "Is a": concept_id_1 "Is a" concept_id_2
    # meaning concept_id_1 is a child of concept_id_2
    full_graph = nx.DiGraph()

    # Add nodes with attributes
    id_to_name = dict(zip(concepts["concept_id"], concepts["concept_name"]))
    id_to_code = dict(zip(concepts["concept_id"], concepts["concept_code"]))
    for cid in valid_ids:
        full_graph.add_node(
            cid,
            concept_name=id_to_name.get(cid, ""),
            concept_code=id_to_code.get(cid, ""),
        )

    # Add edges: child -> parent
    for _, row in rels.iterrows():
        child = row["concept_id_1"]
        parent = row["concept_id_2"]
        if child != parent:  # skip self-loops
            full_graph.add_edge(child, parent)

    logger.info(
        "Full graph: %d nodes, %d edges",
        full_graph.number_of_nodes(),
        full_graph.number_of_edges(),
    )

    # Step 4: Find root nodes (no outgoing "Is a" edges = no parent)
    roots = [n for n in full_graph.nodes() if full_graph.out_degree(n) == 0]
    logger.info("Found %d root nodes", len(roots))

    # Step 5: BFS from roots, depth-limited
    # We traverse parent -> child direction, so we need the reverse graph
    reverse_graph = full_graph.reverse()
    keep_nodes: Set[int] = set()

    for root in roots:
        # BFS from root in reverse graph (root -> children -> grandchildren)
        queue: deque[Tuple[int, int]] = deque([(root, 0)])
        while queue:
            node, depth = queue.popleft()
            if node in keep_nodes:
                continue
            keep_nodes.add(node)
            if depth < max_depth:
                for child in reverse_graph.successors(node):
                    if child not in keep_nodes:
                        queue.append((child, depth + 1))

    # Build the depth-limited subgraph
    graph = full_graph.subgraph(keep_nodes).copy()
    logger.info(
        "Depth-limited graph (max_depth=%d): %d nodes, %d edges",
        max_depth,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def build_icd_to_snomed(
    concept_csv: str | Path,
    relationship_csv: str | Path,
    source_vocabulary: str = "ICD9CM",
    snomed_concept_ids: Optional[Set[int]] = None,
) -> Dict[str, int]:
    """Build a mapping from ICD codes to SNOMED concept IDs.

    Uses Athena's "Maps to" relationships to translate ICD codes into
    SNOMED concepts. This is the bridge between what MIMIC stores
    (ICD-9/ICD-10 codes) and what KEEP operates on (SNOMED concept IDs).

    The mapping chain:
        ICD concept (e.g., concept_id=44826773, code="428.0")
        --"Maps to"--> SNOMED concept (e.g., concept_id=316139, code="42343007")

    Args:
        concept_csv: Path to Athena CONCEPT.csv.
        relationship_csv: Path to Athena CONCEPT_RELATIONSHIP.csv.
        source_vocabulary: Source vocabulary ID. One of "ICD9CM" or
            "ICD10CM". Default: "ICD9CM".
        snomed_concept_ids: If provided, only keep mappings whose SNOMED
            target is in this set (e.g., nodes in the depth-limited graph).
            If None, keeps all valid SNOMED targets.

    Returns:
        Dict[str, int]: Mapping from ICD code string (dotted format,
            e.g., "428.0") to SNOMED concept_id (int). Codes that don't
            map to SNOMED are excluded.

    Example:
        >>> icd9_to_snomed = build_icd_to_snomed(
        ...     "data/athena/CONCEPT.csv",
        ...     "data/athena/CONCEPT_RELATIONSHIP.csv",
        ...     source_vocabulary="ICD9CM",
        ... )
        >>> icd9_to_snomed["428.0"]  # Congestive heart failure
        316139
    """
    # Load source vocabulary concepts (ICD-9 or ICD-10)
    icd_concepts = load_concepts(
        concept_csv,
        vocabulary_ids=[source_vocabulary],
    )
    icd_ids: Set[int] = set(icd_concepts["concept_id"])
    icd_id_to_code: Dict[int, str] = dict(
        zip(icd_concepts["concept_id"], icd_concepts["concept_code"])
    )

    # Load SNOMED concepts
    snomed_concepts = load_concepts(
        concept_csv,
        vocabulary_ids=["SNOMED"],
    )
    snomed_ids: Set[int] = set(snomed_concepts["concept_id"])

    # Load "Maps to" relationships
    rels = load_relationships(
        relationship_csv, relationship_ids=["Maps to"]
    )

    # Filter: source must be ICD, target must be SNOMED
    rels = rels[
        rels["concept_id_1"].isin(icd_ids)
        & rels["concept_id_2"].isin(snomed_ids)
    ]

    # Optionally filter to only SNOMED concepts in our graph
    if snomed_concept_ids is not None:
        rels = rels[rels["concept_id_2"].isin(snomed_concept_ids)]

    # Build mapping: ICD code string -> SNOMED concept_id
    # If an ICD code maps to multiple SNOMED concepts, keep the first
    # (in practice, most ICD codes map to exactly one SNOMED concept)
    mapping: Dict[str, int] = {}
    for _, row in rels.iterrows():
        icd_id = row["concept_id_1"]
        snomed_id = row["concept_id_2"]
        icd_code = icd_id_to_code.get(icd_id, "")
        if icd_code and icd_code not in mapping:
            mapping[icd_code] = snomed_id

    logger.info(
        "Built %s -> SNOMED mapping: %d codes mapped",
        source_vocabulary,
        len(mapping),
    )
    return mapping


def build_all_mappings(
    concept_csv: str | Path,
    relationship_csv: str | Path,
    snomed_concept_ids: Optional[Set[int]] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build ICD-9 and ICD-10 to SNOMED mappings in one call.

    Convenience function that calls ``build_icd_to_snomed`` twice.
    Use this when processing MIMIC-IV which contains both ICD-9 and
    ICD-10 codes.

    Args:
        concept_csv: Path to Athena CONCEPT.csv.
        relationship_csv: Path to Athena CONCEPT_RELATIONSHIP.csv.
        snomed_concept_ids: If provided, only keep mappings whose SNOMED
            target is in this set.

    Returns:
        Tuple of (icd9_to_snomed, icd10_to_snomed) dictionaries.

    Example:
        >>> icd9_map, icd10_map = build_all_mappings(
        ...     "data/athena/CONCEPT.csv",
        ...     "data/athena/CONCEPT_RELATIONSHIP.csv",
        ... )
        >>> icd9_map["428.0"]   # CHF in ICD-9
        316139
        >>> icd10_map["I50.9"]  # Heart failure, unspecified in ICD-10
        316139
    """
    icd9_to_snomed = build_icd_to_snomed(
        concept_csv,
        relationship_csv,
        source_vocabulary="ICD9CM",
        snomed_concept_ids=snomed_concept_ids,
    )
    icd10_to_snomed = build_icd_to_snomed(
        concept_csv,
        relationship_csv,
        source_vocabulary="ICD10CM",
        snomed_concept_ids=snomed_concept_ids,
    )
    return icd9_to_snomed, icd10_to_snomed
