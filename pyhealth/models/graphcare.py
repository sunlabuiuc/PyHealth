# Author: Joshua Steier
# Paper: Jiang et al., "GraphCare: Enhancing Healthcare Predictions
#     with Personalized Knowledge Graphs", ICLR 2024
# Paper link: https://openreview.net/forum?id=tVTN7Zs0ml
# Description: Graph neural network model for EHR prediction using
#     personalized knowledge graphs. Supports BAT (Bi-Attention GNN),
#     GAT, and GIN backbones with configurable patient representation
#     modes (joint, graph-only, node-only). Consumes PyG Data objects
#     produced by pyhealth.processors.GraphProcessor.

from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

try:
    from torch_geometric.nn import global_mean_pool, global_add_pool
    from torch_geometric.nn import GATConv, GINConv
    from torch_geometric.utils import softmax as pyg_softmax

    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from pyhealth.models._graphcare.bat_gnn import BiAttentionGNNConv


def _check_pyg():
    """Raise ImportError with install instructions if PyG is missing."""
    if not HAS_PYG:
        raise ImportError(
            "GraphCare requires torch-geometric. "
            "Install with: pip install torch-geometric"
        )


class GraphCare(BaseModel):
    """GraphCare: Enhancing Healthcare Predictions with Personalized KGs.

    Paper: Jiang, P., Xiao, C., Cross, A., & Sun, J. (2024).
    GraphCare: Enhancing Healthcare Predictions with Personalized
    Knowledge Graphs. ICLR 2024.

    This model takes patient-level knowledge graph subgraphs (produced
    by :class:`~pyhealth.processors.GraphProcessor`) and applies a GNN
    with bi-attention pooling for downstream healthcare prediction tasks.

    Each feature stream receives a PyG ``Batch`` of patient subgraphs.
    Nodes are embedded via a shared entity embedding layer, edges via a
    relation embedding layer. The GNN processes each graph, then pools
    to a fixed-size patient representation. Representations from all
    feature streams are concatenated and projected through a
    classification head.

    Three GNN backbones are supported:

    - **BAT** (default): Bi-Attention augmented GNN from the paper.
      Uses node-level attention (alpha) and optional edge relation
      weights for message passing.
    - **GAT**: Graph Attention Network (Velickovic et al., 2018).
    - **GIN**: Graph Isomorphism Network (Xu et al., 2019).

    Three patient representation modes:

    - **joint** (default): Concatenates graph-level (mean pool) and
      node-level (attention-weighted) representations.
    - **graph**: Graph-level mean pooling only.
    - **node**: Attention-weighted node pooling only.

    Args:
        dataset (SampleDataset): PyHealth dataset with graph-type
            features (produced by GraphProcessor).
        knowledge_graph: A :class:`~pyhealth.graph.KnowledgeGraph`
            instance. Used to determine entity/relation counts and
            optionally initialize embeddings from pre-computed features.
        hidden_dim (int): Hidden dimension for GNN layers. Default 128.
        num_layers (int): Number of GNN layers. Default 3.
        heads (int): Number of attention heads for GAT backbone.
            Ignored for BAT and GIN. Default 4.
        dropout (float): Dropout rate. Default 0.5.
        gnn_type (str): GNN backbone type. One of ``"bat"``, ``"gat"``,
            ``"gin"``. Default ``"bat"``.
        patient_mode (str): Patient representation mode. One of
            ``"joint"``, ``"graph"``, ``"node"``. Default ``"joint"``.
        use_edge_attn (bool): Use edge relation weights in BAT layers.
            Default True.

    Examples:
        >>> from pyhealth.graph import KnowledgeGraph
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import GraphCare
        >>> triples = [
        ...     ("A", "treats", "headache"),
        ...     ("B", "treats", "headache"),
        ...     ("headache", "symptom_of", "migraine"),
        ... ]
        >>> kg = KnowledgeGraph(triples=triples)
        >>> samples = [
        ...     {"patient_id": "p0", "visit_id": "v0",
        ...      "conditions": ["A", "headache"], "label": 1},
        ...     {"patient_id": "p1", "visit_id": "v0",
        ...      "conditions": ["B"], "label": 0},
        ... ]
        >>> input_schema = {
        ...     "conditions": ("graph", {
        ...         "knowledge_graph": kg, "num_hops": 2,
        ...     }),
        ... }
        >>> output_schema = {"label": "binary"}
        >>> dataset = create_sample_dataset(
        ...     samples, input_schema, output_schema,
        ...     dataset_name="test_gc",
        ... )
        >>> model = GraphCare(
        ...     dataset=dataset,
        ...     knowledge_graph=kg,
        ...     hidden_dim=64,
        ...     num_layers=2,
        ... )
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        >>> batch = next(iter(loader))
        >>> output = model(**batch)
        >>> sorted(output.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        knowledge_graph: Any = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5,
        gnn_type: str = "bat",
        patient_mode: str = "joint",
        use_edge_attn: bool = True,
    ):
        _check_pyg()
        super(GraphCare, self).__init__(dataset=dataset)

        assert gnn_type in ("bat", "gat", "gin"), (
            f"gnn_type must be 'bat', 'gat', or 'gin', got '{gnn_type}'"
        )
        assert patient_mode in ("joint", "graph", "node"), (
            f"patient_mode must be 'joint', 'graph', or 'node', "
            f"got '{patient_mode}'"
        )
        assert len(self.label_keys) == 1, (
            "GraphCare supports exactly one label key."
        )

        self.knowledge_graph = knowledge_graph
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_rate = dropout
        self.gnn_type = gnn_type
        self.patient_mode = patient_mode
        self.use_edge_attn = use_edge_attn

        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        # --- Determine entity/relation counts ---
        if knowledge_graph is not None:
            num_entities = knowledge_graph.num_entities
            num_relations = knowledge_graph.num_relations
        else:
            # Fallback: try to get from first graph processor
            for fk in self.feature_keys:
                proc = self.dataset.input_processors.get(fk)
                if proc is not None and hasattr(proc, "knowledge_graph"):
                    kg = proc.knowledge_graph
                    num_entities = kg.num_entities
                    num_relations = kg.num_relations
                    self.knowledge_graph = kg
                    break
            else:
                raise ValueError(
                    "knowledge_graph must be provided or available "
                    "from a GraphProcessor in the dataset."
                )

        # --- Entity and relation embeddings ---
        # Check if KG has pre-computed node features
        if (
            self.knowledge_graph is not None
            and self.knowledge_graph.node_features is not None
        ):
            feat_dim = self.knowledge_graph.node_features.shape[1]
            self.node_projection = nn.Linear(feat_dim, hidden_dim)
            self.register_buffer(
                "pretrained_node_features",
                self.knowledge_graph.node_features,
            )
            self.node_embedding = None
        else:
            self.node_embedding = nn.Embedding(num_entities, hidden_dim)
            self.node_projection = None

        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)

        # --- Alpha attention (node-level) ---
        self.alpha_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # --- GNN layers ---
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == "bat":
                self.gnn_layers.append(
                    BiAttentionGNNConv(
                        hidden_dim=hidden_dim,
                        edge_dim=hidden_dim,
                        edge_attn=use_edge_attn,
                    )
                )
            elif gnn_type == "gat":
                self.gnn_layers.append(
                    GATConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim // heads,
                        heads=heads,
                        concat=True,
                        dropout=dropout,
                    )
                )
            elif gnn_type == "gin":
                gin_nn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.gnn_layers.append(GINConv(nn=gin_nn))

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # --- Classification head ---
        if patient_mode == "joint":
            pool_dim = 2 * hidden_dim
        else:
            pool_dim = hidden_dim

        output_size = self.get_output_size()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            len(self.feature_keys) * pool_dim, output_size
        )

    def _get_node_features(
        self, node_ids: "torch.Tensor"
    ) -> "torch.Tensor":
        """Look up node features from embeddings or pre-computed features.

        Args:
            node_ids: Entity IDs ``[num_nodes]``.

        Returns:
            Node feature tensor ``[num_nodes, hidden_dim]``.
        """
        if self.node_embedding is not None:
            return self.node_embedding(node_ids)
        else:
            raw = self.pretrained_node_features[node_ids]
            return self.node_projection(raw)

    def _encode_graph(
        self, graph_batch: Any
    ) -> "torch.Tensor":
        """Encode a batch of patient graphs to fixed-size representations.

        Args:
            graph_batch: PyG Batch object with attributes:
                ``node_ids``, ``edge_index``, ``edge_type``,
                ``seed_mask``, ``batch``.

        Returns:
            Patient representations ``[batch_size, pool_dim]``.
        """
        # Get node and edge features
        x = self._get_node_features(graph_batch.node_ids.to(self.device))
        edge_index = graph_batch.edge_index.to(self.device)
        edge_attr = self.relation_embedding(
            graph_batch.edge_type.to(self.device)
        )
        batch = graph_batch.batch.to(self.device)

        # Compute alpha attention (node-level)
        alpha_logits = self.alpha_net(x)
        alpha = pyg_softmax(alpha_logits, batch)

        # GNN message passing
        for i, layer in enumerate(self.gnn_layers):
            if self.gnn_type == "bat":
                x_new, _ = layer(x, edge_index, edge_attr, alpha)
            elif self.gnn_type == "gat":
                x_new = layer(x, edge_index)
            elif self.gnn_type == "gin":
                x_new = layer(x, edge_index)

            x_new = self.layer_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(
                x_new, p=self.dropout_rate, training=self.training
            )

            # Residual connection
            x = x + x_new

        # Recompute alpha after GNN (updated features)
        alpha_logits = self.alpha_net(x)
        alpha = pyg_softmax(alpha_logits, batch)

        # Pool to patient representation
        if self.patient_mode == "graph":
            h = global_mean_pool(x, batch)
        elif self.patient_mode == "node":
            h = global_add_pool(x * alpha, batch)
        else:  # joint
            h_g = global_mean_pool(x, batch)
            h_n = global_add_pool(x * alpha, batch)
            h = torch.cat([h_g, h_n], dim=1)

        return h

    def forward(
        self, **kwargs: Any
    ) -> Dict[str, "torch.Tensor"]:
        """Forward propagation.

        Encodes each feature stream's graph batch through the GNN,
        concatenates per-stream patient representations, and projects
        to label space.

        Args:
            **kwargs: Must include all feature keys (PyG Batch objects
                from GraphProcessor) and the label key.

        Returns:
            Dict with keys ``loss``, ``y_prob``, ``y_true``, ``logit``,
            and optionally ``embed`` if ``kwargs["embed"] is True``.
        """
        patient_emb = []

        for feature_key in self.feature_keys:
            graph_batch = kwargs[feature_key]
            emb = self._encode_graph(graph_batch)
            patient_emb.append(emb)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(self.dropout(patient_emb))
        y_prob = self.prepare_y_prob(logits)

        results = {
            "logit": logits,
            "y_prob": y_prob,
        }

        if self.label_key in kwargs:
            y_true = cast(
                torch.Tensor, kwargs[self.label_key]
            ).to(self.device)
            loss = self.get_loss_function()(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        if kwargs.get("embed", False):
            results["embed"] = patient_emb

        return results


if __name__ == "__main__":
    _check_pyg()

    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.graph import KnowledgeGraph

    # Build a small KG
    triples = [
        ("aspirin", "treats", "headache"),
        ("headache", "symptom_of", "migraine"),
        ("ibuprofen", "treats", "headache"),
        ("migraine", "is_a", "neuro"),
        ("aspirin", "is_a", "nsaid"),
        ("ibuprofen", "is_a", "nsaid"),
        ("X", "used_for", "headache"),
        ("Y", "used_for", "migraine"),
    ]
    kg = KnowledgeGraph(triples=triples)
    print(f"KG: {kg}")

    # Create sample dataset with graph processor
    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "conditions": ["aspirin", "headache"],
            "procedures": ["X"],
            "label": 1,
        },
        {
            "patient_id": "p1",
            "visit_id": "v0",
            "conditions": ["ibuprofen"],
            "procedures": ["Y", "X"],
            "label": 0,
        },
    ]
    input_schema = {
        "conditions": ("graph", {
            "knowledge_graph": kg,
            "num_hops": 2,
        }),
        "procedures": ("graph", {
            "knowledge_graph": kg,
            "num_hops": 2,
        }),
    }
    output_schema = {"label": "binary"}

    dataset = create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="test_graphcare",
    )

    # Test BAT (default)
    print("\n=== GraphCare (BAT, joint) ===")
    model = GraphCare(
        dataset=dataset,
        knowledge_graph=kg,
        hidden_dim=64,
        num_layers=2,
    )
    loader = get_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    out = model(**batch)
    print(f"keys: {sorted(out.keys())}")
    print(f"logit shape: {out['logit'].shape}")
    out["loss"].backward()
    print("backward OK")

    # Test GAT
    print("\n=== GraphCare (GAT, graph) ===")
    model_gat = GraphCare(
        dataset=dataset,
        knowledge_graph=kg,
        hidden_dim=64,
        num_layers=2,
        gnn_type="gat",
        heads=4,
        patient_mode="graph",
    )
    out = model_gat(**batch)
    print(f"keys: {sorted(out.keys())}")
    out["loss"].backward()
    print("backward OK")

    # Test GIN
    print("\n=== GraphCare (GIN, node) ===")
    model_gin = GraphCare(
        dataset=dataset,
        knowledge_graph=kg,
        hidden_dim=64,
        num_layers=2,
        gnn_type="gin",
        patient_mode="node",
    )
    out = model_gin(**batch)
    print(f"keys: {sorted(out.keys())}")
    out["loss"].backward()
    print("backward OK")
