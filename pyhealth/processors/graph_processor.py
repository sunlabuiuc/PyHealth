# Author: Joshua Steier
# Description: Graph processor that converts medical codes into patient-level
#   PyG subgraphs using a provided KnowledgeGraph. Registered as "graph" in
#   the PyHealth processor registry. Part of the native PyG support for
#   graph-based EHR models (GraphCare, G-BERT, KAME, etc.).

import logging
from typing import Any, Dict, Iterable, List, Optional

import torch
from . import register_processor
from .base_processor import FeatureProcessor

logger = logging.getLogger(__name__)

# Optional PyG import
try:
    from torch_geometric.data import Data

    HAS_PYG = True
except ImportError:
    HAS_PYG = False

@register_processor("graph")
class GraphProcessor(FeatureProcessor):
    """Processor that converts medical codes into patient-level subgraphs.

    Takes a list of medical codes from a patient visit, looks them up
    in a provided KnowledgeGraph, and extracts the relevant k-hop
    subgraph as a PyG Data object.

    This processor enables graph-based models (GraphCare, G-BERT, KAME)
    to consume standard PyHealth EHR data by bridging medical codes to
    knowledge graph structures.

    Args:
        knowledge_graph: A KnowledgeGraph instance containing the
            medical knowledge graph.
        num_hops: Number of hops for subgraph extraction. Default is 2.
        max_nodes: Maximum number of nodes in the extracted subgraph.
            If exceeded, nodes are pruned by distance from seeds
            (seeds are always kept). Default is None (no limit).

    Example:
        >>> from pyhealth.graph import KnowledgeGraph
        >>> kg = KnowledgeGraph(triples=[
        ...     ("aspirin", "treats", "headache"),
        ...     ("headache", "symptom_of", "migraine"),
        ... ])
        >>> processor = GraphProcessor(knowledge_graph=kg, num_hops=2)
        >>> codes = ["aspirin", "headache"]
        >>> graph = processor.process(codes)
        >>> print(graph.num_nodes, graph.num_edges)

    Example in task schema:
        >>> from pyhealth.graph import KnowledgeGraph
        >>> kg = KnowledgeGraph(triples="path/to/triples.csv")
        >>> input_schema = {
        ...     "conditions": ("graph", {
        ...         "knowledge_graph": kg,
        ...         "num_hops": 2,
        ...         "max_nodes": 500,
        ...     }),
        ... }
    """

    def __init__(
        self,
        knowledge_graph: "KnowledgeGraph",
        num_hops: int = 2,
        max_nodes: Optional[int] = None,
    ):
        if not HAS_PYG:
            raise ImportError(
                "torch-geometric is required for GraphProcessor. "
                "Install with: pip install torch-geometric"
            )
        self.knowledge_graph = knowledge_graph
        self.num_hops = num_hops
        self.max_nodes = max_nodes

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        """No fitting needed — the KG is provided by the user.

        Args:
            samples: List of sample dictionaries (unused).
            field: Field name (unused).
        """
        pass

    def process(self, value: Any) -> "Data":
        """Convert a list of medical codes to a PyG subgraph.

        Args:
            value: A list of medical code strings from a patient visit
                (e.g., ICD codes, ATC codes, CPT codes). Can also be
                a list of list of codes (multi-visit), which will be
                flattened.

        Returns:
            PyG Data object with subgraph around the patient's codes,
            containing: x, edge_index, edge_type, node_ids, seed_mask.
        """
        # Handle list of list of codes (multi-visit)
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
            codes = [code for visit in value for code in visit]
        else:
            codes = list(value)

        # Convert all codes to strings
        codes = [str(c) for c in codes]

        # Extract subgraph from knowledge graph
        subgraph = self.knowledge_graph.subgraph(
            seed_entities=codes,
            num_hops=self.num_hops,
        )

        # Optional: prune to max_nodes
        if (
            self.max_nodes is not None
            and subgraph.num_nodes > self.max_nodes
        ):
            subgraph = self._prune(subgraph)

        return subgraph

    def _prune(self, data: Data) -> Data:
        """Prune subgraph to max_nodes, keeping seeds + closest neighbors.

        Seed nodes are always retained. Remaining slots are filled by
        non-seed nodes in their original order (which reflects BFS
        distance from seeds after k_hop_subgraph).

        Args:
            data: PyG Data object to prune.

        Returns:
            Pruned PyG Data object with at most max_nodes nodes.
        """
        keep = min(self.max_nodes, data.num_nodes)

        # Prioritize seed nodes
        seed_idx = data.seed_mask.nonzero(as_tuple=True)[0]
        non_seed_idx = (~data.seed_mask).nonzero(as_tuple=True)[0]

        remaining = keep - len(seed_idx)
        if remaining > 0:
            keep_idx = torch.cat([seed_idx, non_seed_idx[:remaining]])
        else:
            keep_idx = seed_idx[:keep]

        keep_idx = keep_idx.sort()[0]

        # Build node mask and remapping
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[keep_idx] = True
        node_map = torch.full((data.num_nodes,), -1, dtype=torch.long)
        node_map[keep_idx] = torch.arange(len(keep_idx))

        # Filter edges: both endpoints must be kept
        src, dst = data.edge_index
        edge_mask = mask[src] & mask[dst]
        new_edge_index = node_map[data.edge_index[:, edge_mask]]

        return Data(
            x=data.x[keep_idx],
            edge_index=new_edge_index,
            edge_type=data.edge_type[edge_mask],
            node_ids=data.node_ids[keep_idx],
            seed_mask=data.seed_mask[keep_idx],
        )

    def is_token(self) -> bool:
        """Graph outputs are not discrete token indices.

        Returns:
            False, since graph Data objects are not token-based.
        """
        return False

    def schema(self) -> tuple:
        """Returns the schema of the processed feature.

        Returns:
            Tuple with single element "graph" indicating PyG Data output.
        """
        return ("graph",)

    def dim(self) -> tuple:
        """Graph Data objects don't have a fixed dimensionality.

        Returns:
            Tuple with 0 indicating variable structure.
        """
        return (0,)

    def spatial(self) -> tuple:
        """Graph structures are inherently non-spatial in the grid sense.

        Returns:
            Tuple with False.
        """
        return (False,)

    def __repr__(self) -> str:
        return (
            f"GraphProcessor(num_hops={self.num_hops}, "
            f"max_nodes={self.max_nodes}, "
            f"kg={self.knowledge_graph})"
        )


if __name__ == "__main__":
    # Smoke test
    print("=== GraphProcessor Smoke Test ===\n")

    if not HAS_PYG:
        print("[SKIP] torch-geometric not installed")
        exit(0)

    from pyhealth.graph import KnowledgeGraph

    # Build a small KG
    triples = [
        ("aspirin", "treats", "headache"),
        ("headache", "symptom_of", "migraine"),
        ("ibuprofen", "treats", "headache"),
        ("migraine", "is_a", "neurological_disorder"),
        ("aspirin", "is_a", "nsaid"),
        ("ibuprofen", "is_a", "nsaid"),
    ]
    kg = KnowledgeGraph(triples=triples)

    # Test 1: Basic processing
    processor = GraphProcessor(knowledge_graph=kg, num_hops=2)
    print(f"Processor: {processor}")

    codes = ["aspirin", "headache"]
    graph = processor.process(codes)
    print(f"\nCodes: {codes}")
    print(f"Graph nodes: {graph.num_nodes}")
    print(f"Graph edges: {graph.num_edges}")
    print(f"Seed mask sum: {graph.seed_mask.sum().item()}")

    # Test 2: Multi-visit codes (list of lists)
    multi_visit = [["aspirin"], ["headache", "migraine"]]
    graph2 = processor.process(multi_visit)
    print(f"\nMulti-visit codes: {multi_visit}")
    print(f"Graph nodes: {graph2.num_nodes}")

    # Test 3: Unknown codes (should handle gracefully)
    unknown = ["unknown_drug", "aspirin"]
    graph3 = processor.process(unknown)
    print(f"\nWith unknown code: {unknown}")
    print(f"Graph nodes: {graph3.num_nodes}")

    # Test 4: Pruning
    processor_pruned = GraphProcessor(knowledge_graph=kg, num_hops=2, max_nodes=3)
    graph4 = processor_pruned.process(["aspirin", "headache"])
    print(f"\nPruned (max_nodes=3): {graph4.num_nodes} nodes")

    # Test 5: Schema methods
    print(f"\nis_token: {processor.is_token()}")
    print(f"schema: {processor.schema()}")
    print(f"dim: {processor.dim()}")
    print(f"spatial: {processor.spatial()}")

    print("\n=== All smoke tests passed! ===")