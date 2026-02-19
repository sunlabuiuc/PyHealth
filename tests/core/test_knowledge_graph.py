# Author: Joshua Steier
# Description: Knowledge graph data structure for healthcare code systems.
#   Provides storage for (head, relation, tail) triples and k-hop subgraph
#   extraction for patient-level graph construction. Part of the pyhealth.graph
#   module enabling native PyG support in PyHealth.

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import torch

logger = logging.getLogger(__name__)

# Optional PyG import — only needed for subgraph extraction
try:
    from torch_geometric.data import Data
    from torch_geometric.utils import k_hop_subgraph

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class KnowledgeGraph:
    """A knowledge graph for healthcare code systems.

    Stores (head, relation, tail) triples and provides subgraph
    extraction for patient-level graph construction.

    The user provides the KG — PyHealth does not generate it.

    Supported input formats:
        - List of (head, relation, tail) string tuples
        - Path to a CSV/TSV file with head, relation, tail columns

    Args:
        triples: List of (head, relation, tail) string tuples,
            OR path to a CSV/TSV file with head/relation/tail columns.
        entity2id: Optional pre-built entity-to-ID mapping.
            If None, built automatically from triples.
        relation2id: Optional pre-built relation-to-ID mapping.
            If None, built automatically from triples.
        node_features: Optional tensor of shape (num_entities, feat_dim).
            Pre-computed node embeddings (e.g., from TransE or LLM).

    Attributes:
        entity2id: Dict[str, int] mapping entity names to integer IDs.
        relation2id: Dict[str, int] mapping relation names to integer IDs.
        id2entity: Dict[int, str] reverse mapping.
        id2relation: Dict[int, str] reverse mapping.
        edge_index: Tensor of shape (2, num_triples) in PyG COO format.
        edge_type: Tensor of shape (num_triples,) with relation IDs.
        num_entities: Total number of unique entities.
        num_relations: Total number of unique relation types.
        num_triples: Total number of triples (edges).

    Example:
        >>> from pyhealth.graph import KnowledgeGraph
        >>> triples = [
        ...     ("aspirin", "treats", "headache"),
        ...     ("headache", "symptom_of", "migraine"),
        ...     ("ibuprofen", "treats", "headache"),
        ... ]
        >>> kg = KnowledgeGraph(triples=triples)
        >>> kg.num_entities
        4
        >>> kg.num_relations
        2
        >>> kg.stat()
        KnowledgeGraph: 4 entities, 2 relations, 3 triples
        >>>
        >>> # From a CSV file
        >>> kg = KnowledgeGraph(triples="path/to/triples.csv")
        >>>
        >>> # Extract 2-hop subgraph around seed entities
        >>> subgraph = kg.subgraph(seed_entities=["aspirin", "headache"], num_hops=2)
    """

    def __init__(
        self,
        triples: Union[List[Tuple[str, str, str]], str, Path],
        entity2id: Optional[Dict[str, int]] = None,
        relation2id: Optional[Dict[str, int]] = None,
        node_features: Optional[torch.Tensor] = None,
    ):
        # Load triples from file if path is given
        if isinstance(triples, (str, Path)):
            triples = self._load_triples_from_file(triples)

        if len(triples) == 0:
            raise ValueError("triples must be a non-empty list.")

        # Validate triple format
        for i, t in enumerate(triples):
            if len(t) != 3:
                raise ValueError(
                    f"Triple at index {i} has {len(t)} elements, expected 3: {t}"
                )

        # Build or use provided mappings
        if entity2id is None or relation2id is None:
            entity2id, relation2id = self._build_mappings(triples)

        self.entity2id: Dict[str, int] = entity2id
        self.relation2id: Dict[str, int] = relation2id
        self.id2entity: Dict[int, str] = {v: k for k, v in entity2id.items()}
        self.id2relation: Dict[int, str] = {v: k for k, v in relation2id.items()}

        # Convert string triples to integer triples
        self._int_triples: List[Tuple[int, int, int]] = []
        skipped = 0
        for h, r, t in triples:
            if h not in entity2id or t not in entity2id or r not in relation2id:
                skipped += 1
                continue
            self._int_triples.append(
                (entity2id[h], relation2id[r], entity2id[t])
            )
        if skipped > 0:
            logger.warning(
                f"Skipped {skipped} triples with unknown entities/relations."
            )

        # Build PyG-compatible edge tensors
        if len(self._int_triples) > 0:
            heads = [t[0] for t in self._int_triples]
            tails = [t[2] for t in self._int_triples]
            rels = [t[1] for t in self._int_triples]
            self.edge_index = torch.tensor([heads, tails], dtype=torch.long)
            self.edge_type = torch.tensor(rels, dtype=torch.long)
        else:
            self.edge_index = torch.zeros(2, 0, dtype=torch.long)
            self.edge_type = torch.zeros(0, dtype=torch.long)

        # Optional pre-computed node features
        self.node_features = node_features
        if node_features is not None:
            if node_features.shape[0] != self.num_entities:
                raise ValueError(
                    f"node_features has {node_features.shape[0]} rows but "
                    f"there are {self.num_entities} entities."
                )

        # Build adjacency for fast neighbor lookup
        self._adjacency: Dict[int, Set[int]] = self._build_adjacency()

    @property
    def num_entities(self) -> int:
        """Total number of unique entities."""
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        """Total number of unique relation types."""
        return len(self.relation2id)

    @property
    def num_triples(self) -> int:
        """Total number of triples (edges)."""
        return self.edge_index.shape[1]

    @staticmethod
    def _load_triples_from_file(
        path: Union[str, Path],
    ) -> List[Tuple[str, str, str]]:
        """Load triples from a CSV or TSV file.

        Expects columns named head, relation, tail. If not found,
        uses the first three columns.

        Args:
            path: Path to the CSV/TSV file.

        Returns:
            List of (head, relation, tail) string tuples.
        """
        import pandas as pd

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Triple file not found: {path}")

        sep = "\t" if path.suffix in (".tsv", ".txt") else ","
        df = pd.read_csv(path, sep=sep, dtype=str)

        if {"head", "relation", "tail"}.issubset(df.columns):
            return list(zip(df["head"], df["relation"], df["tail"]))
        else:
            # Use first 3 columns
            cols = df.columns[:3]
            logger.info(
                f"Columns head/relation/tail not found. "
                f"Using columns: {list(cols)}"
            )
            return list(zip(df[cols[0]], df[cols[1]], df[cols[2]]))

    @staticmethod
    def _build_mappings(
        triples: List[Tuple[str, str, str]],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Build entity2id and relation2id mappings from triples.

        Args:
            triples: List of (head, relation, tail) string tuples.

        Returns:
            Tuple of (entity2id, relation2id) dictionaries.
        """
        entities: Set[str] = set()
        relations: Set[str] = set()
        for h, r, t in triples:
            entities.add(str(h))
            entities.add(str(t))
            relations.add(str(r))
        entity2id = {e: i for i, e in enumerate(sorted(entities))}
        relation2id = {r: i for i, r in enumerate(sorted(relations))}
        return entity2id, relation2id

    def _build_adjacency(self) -> Dict[int, Set[int]]:
        """Build undirected adjacency dict for fast neighbor lookup.

        Returns:
            Dict mapping node ID to set of neighbor node IDs.
        """
        adj: Dict[int, Set[int]] = {}
        for h, _, t in self._int_triples:
            adj.setdefault(h, set()).add(t)
            adj.setdefault(t, set()).add(h)
        return adj

    def subgraph(
        self,
        seed_entities: List[str],
        num_hops: int = 2,
    ) -> "Data":
        """Extract a k-hop subgraph around seed entities.

        Uses PyG's k_hop_subgraph to find all nodes within num_hops
        of the seed entities, then returns the induced subgraph.

        Args:
            seed_entities: List of entity names (e.g., medical codes).
                Entities not found in the KG are silently skipped.
            num_hops: Number of hops to expand from seed nodes.
                Default is 2.

        Returns:
            PyG Data object with:
                - x: Node features if available, else zeros (num_nodes, 1).
                - edge_index: Subgraph edges, reindexed to [0, num_nodes).
                - edge_type: Relation type for each edge.
                - node_ids: Original entity IDs for mapping back.
                - seed_mask: Boolean mask, True for seed nodes.

        Raises:
            ImportError: If torch-geometric is not installed.
        """
        if not HAS_PYG:
            raise ImportError(
                "torch-geometric is required for subgraph extraction. "
                "Install with: pip install torch-geometric"
            )

        # Map seed entities to integer IDs, skip unknowns
        seed_ids = [
            self.entity2id[e]
            for e in seed_entities
            if e in self.entity2id
        ]

        if len(seed_ids) == 0:
            # Return empty graph
            return Data(
                x=torch.zeros(0, 1),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_type=torch.zeros(0, dtype=torch.long),
                node_ids=torch.zeros(0, dtype=torch.long),
                seed_mask=torch.zeros(0, dtype=torch.bool),
            )

        seed_tensor = torch.tensor(seed_ids, dtype=torch.long)

        # Use PyG k_hop_subgraph
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=seed_tensor,
            num_hops=num_hops,
            edge_index=self.edge_index,
            relabel_nodes=True,
            num_nodes=self.num_entities,
        )

        # Edge types for subgraph
        sub_edge_type = self.edge_type[edge_mask]

        # Node features
        if self.node_features is not None:
            x = self.node_features[subset]
        else:
            x = torch.zeros(len(subset), 1)

        # Seed mask: which nodes in the subgraph are seeds
        seed_mask = torch.zeros(len(subset), dtype=torch.bool)
        seed_mask[mapping] = True

        return Data(
            x=x,
            edge_index=sub_edge_index,
            edge_type=sub_edge_type,
            node_ids=subset,
            seed_mask=seed_mask,
        )

    def has_entity(self, entity: str) -> bool:
        """Check if an entity exists in the KG.

        Args:
            entity: Entity name string.

        Returns:
            True if entity is in the KG.
        """
        return entity in self.entity2id

    def neighbors(self, entity: str, num_hops: int = 1) -> List[str]:
        """Get neighbor entity names within num_hops.

        Args:
            entity: Entity name string.
            num_hops: Number of hops. Default is 1.

        Returns:
            List of neighbor entity name strings.
        """
        if entity not in self.entity2id:
            return []

        visited: Set[int] = set()
        frontier: Set[int] = {self.entity2id[entity]}

        for _ in range(num_hops):
            next_frontier: Set[int] = set()
            for node in frontier:
                for neighbor in self._adjacency.get(node, set()):
                    if neighbor not in visited and neighbor not in frontier:
                        next_frontier.add(neighbor)
            visited.update(frontier)
            frontier = next_frontier

        visited.update(frontier)
        visited.discard(self.entity2id[entity])
        return [self.id2entity[nid] for nid in sorted(visited)]

    def stat(self):
        """Print statistics of the knowledge graph."""
        print(
            f"KnowledgeGraph: {self.num_entities} entities, "
            f"{self.num_relations} relations, "
            f"{self.num_triples} triples"
        )

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph(entities={self.num_entities}, "
            f"relations={self.num_relations}, "
            f"triples={self.num_triples})"
        )

    def __len__(self) -> int:
        return self.num_triples


if __name__ == "__main__":
    # Smoke test
    print("=== KnowledgeGraph Smoke Test ===\n")

    # Test 1: Basic construction from list
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

    kg = KnowledgeGraph(triples=triples)
    kg.stat()
    print(f"repr: {kg}")
    print(f"len: {len(kg)}")
    print(f"has 'aspirin': {kg.has_entity('aspirin')}")
    print(f"has 'tylenol': {kg.has_entity('tylenol')}")
    print(f"neighbors of 'aspirin' (1-hop): {kg.neighbors('aspirin', 1)}")
    print(f"neighbors of 'aspirin' (2-hop): {kg.neighbors('aspirin', 2)}")

    # Test 2: Subgraph extraction (requires PyG)
    if HAS_PYG:
        print("\n--- Subgraph Extraction ---")
        sub = kg.subgraph(seed_entities=["aspirin", "headache"], num_hops=2)
        print(f"Subgraph nodes: {sub.num_nodes}")
        print(f"Subgraph edges: {sub.num_edges}")
        print(f"Seed mask: {sub.seed_mask}")
        print(f"Node IDs: {sub.node_ids}")
        print(f"Edge index shape: {sub.edge_index.shape}")
        print(f"Edge type shape: {sub.edge_type.shape}")

        # Empty seed test
        sub_empty = kg.subgraph(seed_entities=["unknown_entity"], num_hops=2)
        print(f"\nEmpty subgraph nodes: {sub_empty.num_nodes}")
        print(f"Empty subgraph edges: {sub_empty.num_edges}")
    else:
        print("\n[SKIP] torch-geometric not installed, skipping subgraph test")

    # Test 3: Pre-computed node features
    features = torch.randn(kg.num_entities, 64)
    kg_with_feats = KnowledgeGraph(triples=triples, node_features=features)
    print(f"\nKG with features: {kg_with_feats}")
    if HAS_PYG:
        sub_feat = kg_with_feats.subgraph(["aspirin"], num_hops=1)
        print(f"Subgraph x shape: {sub_feat.x.shape}")

    print("\n=== All smoke tests passed! ===")