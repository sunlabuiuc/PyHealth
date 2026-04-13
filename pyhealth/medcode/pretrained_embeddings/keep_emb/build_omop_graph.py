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

Authors: Colton Loew, Desmond Fung, Lookman Olowo, Christiana Beard
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


PAPER_ROOT_CONCEPT_ID = 4274025  # "Disease" concept (paper Appendix A.1.1)


def _rescue_orphan_edges(
    graph: nx.DiGraph,
    ancestor_df: pd.DataFrame,
    root_concept_id: int,
) -> int:
    """Add edges for orphans (nodes with no in-graph parent).

    Some Condition concepts have direct "Is a" parents only in non-Condition
    domains (typically Observation). Example: "DRESS syndrome" has parents
    "Adverse reaction to drug" and "Hypersensitivity reaction", both in the
    Observation domain. When we filter CONCEPT_RELATIONSHIP edges to
    Condition-only endpoints, these edges vanish and the concept has no
    incoming edge in our graph.

    The paper's approach using CONCEPT_ANCESTOR naturally includes these
    concepts (step 2 above). But they still end up with no direct edges in
    CONCEPT_RELATIONSHIP. This function adds a direct edge from each orphan
    to its closest in-graph ancestor, making the graph connected.

    Discovered and documented by Desmond Fung in keep-mimic4.

    Args:
        graph: The current graph. Rescue edges are added in place.
        ancestor_df: CONCEPT_ANCESTOR DataFrame (already loaded).
        root_concept_id: The root concept (excluded from orphan set).

    Returns:
        int: Number of orphans rescued.
    """
    # Orphans = nodes in graph with no outgoing "Is a" edge (no parent),
    # excluding the root (which legitimately has no parent).
    orphans = [
        n for n in graph.nodes()
        if n != root_concept_id and graph.out_degree(n) == 0
    ]

    if not orphans:
        return 0

    logger.info("Found %d orphans (no parent in graph)", len(orphans))

    graph_nodes = set(graph.nodes())
    orphan_set = set(orphans)

    # Pre-filter ancestor_df to rows relevant to our orphans.
    orphan_ancestors = ancestor_df[
        ancestor_df["descendant_concept_id"].isin(orphan_set)
    ]

    rescued = 0
    for orphan in orphans:
        candidates = orphan_ancestors[
            (orphan_ancestors["descendant_concept_id"] == orphan)
            & (orphan_ancestors["ancestor_concept_id"].isin(graph_nodes))
            & (orphan_ancestors["min_levels_of_separation"] > 0)
        ]
        if candidates.empty:
            logger.warning(
                "Orphan %d has no in-graph ancestor — cannot rescue", orphan,
            )
            continue

        # Closest in-graph ancestor; ties broken by smaller concept_id
        # for deterministic output across Athena versions.
        best = candidates.sort_values(
            ["min_levels_of_separation", "ancestor_concept_id"],
            ascending=[True, True],
        ).iloc[0]
        closest_ancestor = int(best["ancestor_concept_id"])
        graph.add_edge(orphan, closest_ancestor)
        rescued += 1

    logger.info("Rescued %d orphans via CONCEPT_ANCESTOR", rescued)
    return rescued


def _bfs_fallback(
    concept_csv: str | Path,
    relationship_csv: str | Path,
    root_concept_id: int,
    max_depth: int,
    valid_condition_ids: Set[int],
) -> Set[int]:
    """BFS-based fallback when CONCEPT_ANCESTOR is unavailable.

    Builds a node set by BFS over "Is a" edges from the root within depth.
    This can miss concepts whose direct parent is in a non-Condition domain
    (see ``_rescue_orphan_edges`` docstring). Not paper-faithful — included
    only for users who lack CONCEPT_ANCESTOR.csv.
    """
    rels = load_relationships(relationship_csv, relationship_ids=["Is a"])
    rels = rels[
        rels["concept_id_1"].isin(valid_condition_ids)
        & rels["concept_id_2"].isin(valid_condition_ids)
    ]
    # Build temporary graph for BFS
    tmp = nx.DiGraph()
    tmp.add_nodes_from(valid_condition_ids)
    for _, row in rels.iterrows():
        if row["concept_id_1"] != row["concept_id_2"]:
            tmp.add_edge(row["concept_id_1"], row["concept_id_2"])

    if root_concept_id not in tmp:
        return set()

    reverse = tmp.reverse()
    keep: Set[int] = set()
    queue: deque[Tuple[int, int]] = deque([(root_concept_id, 0)])
    while queue:
        node, depth = queue.popleft()
        if node in keep:
            continue
        keep.add(node)
        if depth < max_depth:
            for child in reverse.successors(node):
                if child not in keep:
                    queue.append((child, depth + 1))
    return keep


def build_hierarchy_graph(
    concept_csv: str | Path,
    relationship_csv: str | Path,
    max_depth: int = 5,
    vocabulary_id: str = "SNOMED",
    domain_id: str = "Condition",
    root_concept_id: int = PAPER_ROOT_CONCEPT_ID,
    ancestor_csv: Optional[str | Path] = None,
) -> nx.DiGraph:
    """Build a depth-limited SNOMED "Is a" hierarchy as a NetworkX DiGraph.

    This is the knowledge graph that Node2Vec walks on in KEEP Stage 1.
    The graph structure encodes medical knowledge: nodes that share
    ancestry (e.g., "Type 1 DM" and "Type 2 DM" under "Diabetes mellitus")
    will be visited in similar random walk contexts, producing similar
    Node2Vec embeddings.

    Algorithm (paper Appendix A.1.1):
        1. Load CONCEPT_ANCESTOR, get all descendants of root within depth.
        2. Intersect with standard SNOMED Condition concepts.
        3. Load "Is a" edges from CONCEPT_RELATIONSHIP (direct parent only).
        4. Build DiGraph with node attributes (name, code) and edges.
        5. Rescue orphans: nodes with no incoming edge (their direct
           parent was in a non-Condition domain and got filtered out).

    Why a single root (4274025)?
        Paper Appendix A.1.1: "we filter concepts based on their hierarchical
        distance from the root node, 'Disease' (concept ID: 4274025)."
        This excludes non-disease findings like body temperature observations,
        pain findings, family history, etc.

    Why CONCEPT_ANCESTOR?
        The paper uses "the 'CONCEPT_ANCESTOR' table" to calculate distance
        from root. CONCEPT_ANCESTOR is the transitive closure of "Is a"
        relationships with pre-computed distances — one query gives us
        every descendant within depth 5, regardless of intermediate domains.
        Using CONCEPT_RELATIONSHIP alone with BFS would miss concepts whose
        direct parent is in a non-Condition domain.

    Why depth-limit to 5?
        The KEEP paper (Section 4.2) limits to 5 levels from root. Deeper
        levels contain overly specific concepts that few patients have,
        adding noise to Node2Vec walks.

    Args:
        concept_csv: Path to Athena CONCEPT.csv.
        relationship_csv: Path to Athena CONCEPT_RELATIONSHIP.csv.
        max_depth: Maximum hierarchy depth from root. Default: 5.
        vocabulary_id: Vocabulary to build graph from. Default: "SNOMED".
        domain_id: Domain to filter concepts. Default: "Condition".
        root_concept_id: Single root. Default: 4274025 "Disease"
            (KEEP paper Appendix A.1.1). Configurable for tests.
        ancestor_csv: Path to Athena CONCEPT_ANCESTOR.csv. Required for
            paper-faithful reproduction. If None, falls back to a naive
            BFS over CONCEPT_RELATIONSHIP only (misses cross-domain paths).

    Returns:
        nx.DiGraph: Directed graph where each edge is (child, parent).
            Node attributes include "concept_name" and "concept_code".
            All descendants of ``root_concept_id`` within ``max_depth``
            levels are included. Returns empty graph if root is absent.

    Example:
        >>> G = build_hierarchy_graph(
        ...     "data/athena/CONCEPT.csv",
        ...     "data/athena/CONCEPT_RELATIONSHIP.csv",
        ...     ancestor_csv="data/athena/CONCEPT_ANCESTOR.csv",
        ... )
        >>> G.number_of_nodes()
        68396
    """
    # Step 1: Load SNOMED Condition standard concepts
    concepts = load_concepts(
        concept_csv,
        vocabulary_ids=[vocabulary_id],
        domain_ids=[domain_id],
    )
    concepts = concepts[concepts["standard_concept"] == "S"]
    valid_condition_ids: Set[int] = set(concepts["concept_id"])
    id_to_name = dict(zip(concepts["concept_id"], concepts["concept_name"]))
    id_to_code = dict(zip(concepts["concept_id"], concepts["concept_code"]))
    logger.info(
        "Found %d standard %s %s concepts",
        len(valid_condition_ids),
        vocabulary_id,
        domain_id,
    )

    # Step 2: Determine node set via CONCEPT_ANCESTOR (paper-faithful)
    # This gives us descendants of root within depth, regardless of
    # intermediate domains — naturally handles cross-domain paths.
    if ancestor_csv is not None:
        ancestor_df = pd.read_csv(
            ancestor_csv,
            sep="\t",
            usecols=[
                "ancestor_concept_id",
                "descendant_concept_id",
                "min_levels_of_separation",
            ],
            dtype={
                "ancestor_concept_id": int,
                "descendant_concept_id": int,
                "min_levels_of_separation": int,
            },
        )
        descendants = set(
            ancestor_df[
                (ancestor_df["ancestor_concept_id"] == root_concept_id)
                & (ancestor_df["min_levels_of_separation"] <= max_depth)
            ]["descendant_concept_id"]
        )
        # Include root itself (CONCEPT_ANCESTOR typically doesn't have self-loops)
        descendants.add(root_concept_id)
        # Intersect with standard Condition concepts
        node_set = descendants & valid_condition_ids
        logger.info(
            "CONCEPT_ANCESTOR: %d descendants of root %d within depth %d, "
            "intersected with Condition concepts -> %d nodes",
            len(descendants),
            root_concept_id,
            max_depth,
            len(node_set),
        )
    else:
        # Fallback: no CONCEPT_ANCESTOR available.
        # Use naive BFS over "Is a" edges, which misses cross-domain paths.
        # This is not paper-faithful but works when CONCEPT_ANCESTOR is missing.
        logger.warning(
            "CONCEPT_ANCESTOR.csv not provided. Falling back to BFS over "
            "'Is a' edges — may miss cross-domain paths (not paper-faithful)."
        )
        ancestor_df = None
        node_set = _bfs_fallback(
            concept_csv,
            relationship_csv,
            root_concept_id,
            max_depth,
            valid_condition_ids,
        )

    if not node_set:
        logger.warning("Empty node set. Returning empty graph.")
        return nx.DiGraph()

    # Step 3: Build DiGraph with node attributes
    graph = nx.DiGraph()
    for cid in node_set:
        graph.add_node(
            cid,
            concept_name=id_to_name.get(cid, ""),
            concept_code=id_to_code.get(cid, ""),
        )

    # Step 4: Load "Is a" edges and filter to edges between node_set members
    rels = load_relationships(relationship_csv, relationship_ids=["Is a"])
    rels = rels[
        rels["concept_id_1"].isin(node_set)
        & rels["concept_id_2"].isin(node_set)
    ]
    for _, row in rels.iterrows():
        child = row["concept_id_1"]
        parent = row["concept_id_2"]
        if child != parent:  # skip self-loops
            graph.add_edge(child, parent)

    logger.info(
        "Graph (before rescue): %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    # Step 5: Rescue orphans (nodes with no outgoing "Is a" edge)
    # These are Condition concepts whose only direct parents are in
    # non-Condition domains. They got included by CONCEPT_ANCESTOR (which
    # sees transitive paths) but lack direct edges in CONCEPT_RELATIONSHIP
    # (which was filtered to Condition-only endpoints).
    if ancestor_df is not None:
        _rescue_orphan_edges(graph, ancestor_df, root_concept_id)
        logger.info(
            "After orphan rescue: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

    return graph


def build_icd_to_snomed(
    concept_csv: str | Path,
    relationship_csv: str | Path,
    source_vocabulary: str = "ICD9CM",
    snomed_concept_ids: Optional[Set[int]] = None,
) -> Dict[str, List[int]]:
    """Build a mapping from ICD codes to SNOMED concept IDs (multi-target).

    Uses Athena's "Maps to" relationships to translate ICD codes into
    SNOMED concepts. This is the bridge between what MIMIC stores
    (ICD-9/ICD-10 codes) and what KEEP operates on (SNOMED concept IDs).

    Multi-target handling:
        ~24% of ICD-10 codes are "combination codes" that map to multiple
        SNOMED concepts. For example, "A01.04 Typhoid arthritis" maps to
        BOTH "Typhoid fever" AND "Inflammatory arthritis" — these are
        separate atomic concepts in SNOMED. We preserve all in-graph
        targets as a sorted list. Downstream code treats a patient with
        a combination code as having all its SNOMED targets (matching
        the paper's atomic-concept assumption from UK Biobank data).

    The mapping chain:
        ICD concept (e.g., concept_id=44826773, code="428.0")
        --"Maps to"--> SNOMED concept (e.g., concept_id=316139, code="42343007")

    Args:
        concept_csv: Path to Athena CONCEPT.csv.
        relationship_csv: Path to Athena CONCEPT_RELATIONSHIP.csv.
        source_vocabulary: Source vocabulary ID. One of "ICD9CM" or
            "ICD10CM". Default: "ICD9CM".
        snomed_concept_ids: If provided, only keep targets that are in
            this set. Multi-target ICD codes may keep only a subset of
            their SNOMED targets if only some are in the graph. Codes
            with no remaining targets are excluded entirely.

    Returns:
        Dict[str, List[int]]: Mapping from ICD code string (dotted
            format, e.g., "428.0") to a sorted list of SNOMED concept_ids.
            Lists are always non-empty and sorted ascending for
            deterministic output.

    Example:
        >>> icd9_to_snomed = build_icd_to_snomed(
        ...     "data/athena/CONCEPT.csv",
        ...     "data/athena/CONCEPT_RELATIONSHIP.csv",
        ...     source_vocabulary="ICD9CM",
        ... )
        >>> icd9_to_snomed["428.0"]  # CHF — single target
        [316139]
        >>> icd9_to_snomed["250.01"]  # combination code — multi-target
        [201826, 316139]
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

    # Build mapping: ICD code string -> List[SNOMED concept_id]
    # Multi-target codes (~24% of ICD-10) preserve all in-graph targets.
    # Lists sorted ascending for deterministic output across Athena versions.
    mapping_sets: Dict[str, Set[int]] = {}
    for _, row in rels.iterrows():
        icd_id = row["concept_id_1"]
        snomed_id = int(row["concept_id_2"])
        icd_code = icd_id_to_code.get(icd_id, "")
        if icd_code:
            mapping_sets.setdefault(icd_code, set()).add(snomed_id)

    mapping: Dict[str, List[int]] = {
        code: sorted(targets) for code, targets in mapping_sets.items()
    }

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
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
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
