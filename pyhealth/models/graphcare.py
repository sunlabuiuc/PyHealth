"""GraphCare model for PyHealth.

Paper: Pengcheng Jiang et al. GraphCare: Enhancing Healthcare Predictions
with Personalized Knowledge Graphs. ICLR 2024.

This model constructs personalized patient-level knowledge graphs from
medical codes (conditions, procedures, drugs) using pre-built code-level
knowledge subgraphs, then applies a GNN (GAT, GIN, or BAT) with
bi-attention pooling for downstream healthcare prediction tasks.

Note:
    This model requires ``torch-geometric`` to be installed::

        pip install torch-geometric

Note:
    This model requires pre-computed knowledge graphs for medical codes.
    See the GraphCare paper and the original implementation at
    https://github.com/pat-jj/GraphCare for graph generation details.
"""

from typing import Optional, Tuple, Union
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for torch_geometric
# ---------------------------------------------------------------------------
try:
    from torch_geometric.nn import GATConv, GINConv
    from torch_geometric.nn.conv import MessagePassing
    from torch_geometric.nn import global_mean_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


def _check_torch_geometric():
    """Raises ImportError with install instructions if torch_geometric is missing."""
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError(
            "GraphCare requires torch-geometric. "
            "Install it with: pip install torch-geometric"
        )


# ===========================================================================
# BiAttentionGNNConv – the BAT message-passing layer from the paper (§3.3)
# ===========================================================================


class BiAttentionGNNConv(MessagePassing if HAS_TORCH_GEOMETRIC else nn.Module):
    r"""Bi-Attention augmented GNN convolution (BAT layer).

    This is a GIN-style message-passing layer augmented with:
    * **node-level attention** (``attn``) injected from the outer model,
    * **edge-level attention** via a learnable projection of relation
      embeddings (``W_R``).

    The message for edge :math:`(j \to i)` is:

    .. math::
        m_{j \to i} = \text{ReLU}\bigl(
            x_j \cdot \alpha_{j} + W_R(e_{ji}) \cdot e_{ji}
        \bigr)

    where :math:`\alpha_j` is the pre-computed bi-attention weight for node
    *j* and :math:`e_{ji}` is the relation embedding on that edge.

    Args:
        nn: A neural network module applied after aggregation (typically
            ``nn.Linear(hidden_dim, hidden_dim)``).
        eps: Initial value of :math:`\varepsilon` for the self-loop weight.
        train_eps: If ``True``, :math:`\varepsilon` is a learnable parameter.
        edge_dim: Dimension of edge (relation) features.
        edge_attn: If ``True``, use edge attention via ``W_R``.
    """

    def __init__(
        self,
        nn: torch.nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        edge_attn: bool = True,
        **kwargs,
    ):
        _check_torch_geometric()
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        self.edge_attn = edge_attn

        if edge_attn:
            self.W_R = torch.nn.Linear(edge_dim, 1)
        else:
            self.W_R = None

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        if self.W_R is not None:
            self.W_R.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, "OptPairTensor"],
        edge_index: "Adj",
        edge_attr: "OptTensor" = None,
        size: "Size" = None,
        attn: Tensor = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate computes messages and aggregates
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, attn=attn)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        if self.W_R is not None:
            w_rel = self.W_R(edge_attr)
        else:
            w_rel = None

        return self.nn(out), w_rel

    def message(self, x_j: Tensor, edge_attr: Tensor, attn: Tensor) -> Tensor:
        if self.edge_attn:
            w_rel = self.W_R(edge_attr)
            out = (x_j * attn + w_rel * edge_attr).relu()
        else:
            out = (x_j * attn).relu()
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"


# ===========================================================================
# GraphCare – main model (PyHealth-compatible)
# ===========================================================================


class GraphCare(nn.Module):
    r"""GraphCare model for EHR-based healthcare predictions.

    This is a **graph-level** model that operates on pre-constructed
    patient knowledge graphs (KGs) via ``torch_geometric``.  It is
    designed to be called **outside** the standard PyHealth ``Trainer``
    loop because the data pipeline requires ``torch_geometric.loader.DataLoader``
    rather than PyHealth's default collation.

    The architecture has three components:

    1. **Embedding layer** – maps node IDs and relation IDs to dense
       vectors using (optionally pre-trained) embedding tables, then
       projects them to ``hidden_dim``.
    2. **GNN encoder** – *L* layers of message passing.  Three GNN
       back-ends are supported:

       * ``"BAT"`` – Bi-Attention augmented GNN (default, from the paper).
         Uses per-node attention weights derived from visit-level
         (alpha) and node-level (beta) attention with temporal decay.
       * ``"GAT"`` – standard Graph Attention Network.
       * ``"GIN"`` – Graph Isomorphism Network.

    3. **Patient representation head** – produces a patient-level vector
       from the node embeddings, with three modes:

       * ``"graph"`` – global mean pool over all graph nodes.
       * ``"node"``  – weighted average of direct EHR-node embeddings.
       * ``"joint"`` – concatenation of both (default).

    Args:
        num_nodes: Total number of nodes in the knowledge graph.
        num_rels: Total number of relation types.
        max_visit: Maximum number of visits per patient.
        embedding_dim: Dimension of pre-trained node/relation embeddings.
        hidden_dim: Hidden dimension used throughout the model.
        out_channels: Number of output classes / labels.
        layers: Number of GNN layers.  Default ``3``.
        dropout: Dropout rate for the final MLP.  Default ``0.5``.
        decay_rate: Temporal decay rate :math:`\gamma` for visit
            weighting: :math:`\lambda_j = e^{\gamma (V - j)}`.
            Default ``0.01``.
        node_emb: Optional pre-trained node embedding tensor of shape
            ``[num_nodes, embedding_dim]``.  If ``None``, embeddings
            are learned from scratch.
        rel_emb: Optional pre-trained relation embedding tensor of shape
            ``[num_rels, embedding_dim]``.  If ``None``, embeddings
            are learned from scratch.
        freeze: If ``True``, freeze pre-trained embeddings.
        patient_mode: Patient representation mode – one of
            ``"joint"`` (default), ``"graph"``, or ``"node"``.
        use_alpha: Use visit-level (alpha) attention.  Default ``True``.
        use_beta: Use node-level (beta) attention with temporal decay.
            Default ``True``.
        use_edge_attn: Use edge (relation) attention in BAT layers.
            Default ``True``.
        gnn: GNN backbone – ``"BAT"`` (default), ``"GAT"``, or ``"GIN"``.
        attn_init: Optional tensor of shape ``[num_nodes]`` for
            initialising the diagonal of alpha attention weights.
        drop_rate: Edge dropout rate during training.  Default ``0.0``.
        self_attn: Initial value of the GIN-style self-loop weight
            :math:`\varepsilon` in the BAT layer.  Default ``0.0``.

    Example:

        The model is instantiated and called directly with
        ``torch_geometric``-style batched data::

            >>> model = GraphCare(
            ...     num_nodes=5000, num_rels=100, max_visit=20,
            ...     embedding_dim=128, hidden_dim=128, out_channels=1,
            ... )
            >>> # node_ids, rel_ids, edge_index, batch from PyG DataLoader
            >>> logits = model(node_ids, rel_ids, edge_index, batch,
            ...                visit_node, ehr_nodes)
    """

    def __init__(
        self,
        num_nodes: int,
        num_rels: int,
        max_visit: int,
        embedding_dim: int,
        hidden_dim: int,
        out_channels: int,
        layers: int = 3,
        dropout: float = 0.5,
        decay_rate: float = 0.01,
        node_emb: Optional[torch.Tensor] = None,
        rel_emb: Optional[torch.Tensor] = None,
        freeze: bool = False,
        patient_mode: str = "joint",
        use_alpha: bool = True,
        use_beta: bool = True,
        use_edge_attn: bool = True,
        gnn: str = "BAT",
        attn_init: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
        self_attn: float = 0.0,
    ):
        super().__init__()
        _check_torch_geometric()

        assert patient_mode in ("joint", "graph", "node"), \
            f"patient_mode must be 'joint', 'graph', or 'node', got '{patient_mode}'"
        assert gnn in ("BAT", "GAT", "GIN"), \
            f"gnn must be 'BAT', 'GAT', or 'GIN', got '{gnn}'"

        self.gnn = gnn
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.decay_rate = decay_rate
        self.patient_mode = patient_mode
        self.use_alpha = use_alpha
        self.use_beta = use_beta
        self.edge_attn = use_edge_attn
        self.drop_rate = drop_rate
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.max_visit = max_visit
        self.num_layers = layers
        self.dropout = dropout

        # --- Temporal decay weights: lambda_j = exp(gamma * (V - j)) ---
        j = torch.arange(max_visit).float()
        lambda_j = (
            torch.exp(self.decay_rate * (max_visit - j))
            .unsqueeze(0)
            .reshape(1, max_visit, 1)
            .float()
        )
        self.register_buffer("lambda_j", lambda_j)

        # --- Embeddings ---
        if node_emb is None:
            self.node_emb = nn.Embedding(num_nodes, embedding_dim)
        else:
            self.node_emb = nn.Embedding.from_pretrained(node_emb.float(), freeze=freeze)

        if rel_emb is None:
            self.rel_emb = nn.Embedding(num_rels, embedding_dim)
        else:
            self.rel_emb = nn.Embedding.from_pretrained(rel_emb.float(), freeze=freeze)

        # --- Projection to hidden_dim ---
        self.lin = nn.Linear(embedding_dim, hidden_dim)

        # --- Per-layer modules ---
        self.alpha_attn = nn.ModuleDict()
        self.beta_attn = nn.ModuleDict()
        self.conv = nn.ModuleDict()

        for layer_idx in range(1, layers + 1):
            k = str(layer_idx)

            # Visit-level attention (alpha)
            if self.use_alpha:
                self.alpha_attn[k] = nn.Linear(num_nodes, num_nodes)
                if attn_init is not None:
                    attn_init_f = attn_init.float()
                    attn_init_matrix = torch.eye(num_nodes).float() * attn_init_f
                    self.alpha_attn[k].weight.data.copy_(attn_init_matrix)
                else:
                    nn.init.xavier_normal_(self.alpha_attn[k].weight)

            # Node-level attention (beta) with temporal decay
            if self.use_beta:
                self.beta_attn[k] = nn.Linear(num_nodes, 1)
                nn.init.xavier_normal_(self.beta_attn[k].weight)

            # GNN convolution
            if self.gnn == "BAT":
                self.conv[k] = BiAttentionGNNConv(
                    nn.Linear(hidden_dim, hidden_dim),
                    edge_dim=hidden_dim,
                    edge_attn=self.edge_attn,
                    eps=self_attn,
                )
            elif self.gnn == "GAT":
                self.conv[k] = GATConv(hidden_dim, hidden_dim)
            elif self.gnn == "GIN":
                self.conv[k] = GINConv(nn.Linear(hidden_dim, hidden_dim))

        # --- Final MLP head ---
        if self.patient_mode == "joint":
            self.MLP = nn.Linear(hidden_dim * 2, out_channels)
        else:
            self.MLP = nn.Linear(hidden_dim, out_channels)

    def forward(
        self,
        node_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        visit_node: torch.Tensor,
        ehr_nodes: Optional[torch.Tensor] = None,
        store_attn: bool = False,
        in_drop: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass.

        Args:
            node_ids: Node ID tensor of shape ``[total_nodes_in_batch]``.
                These are the ``data.y`` values from the PyG batch
                (original node IDs used to look up embeddings).
            rel_ids: Relation ID tensor of shape ``[total_edges_in_batch]``.
                Used to look up relation embeddings.
            edge_index: Edge index tensor of shape ``[2, total_edges_in_batch]``.
            batch: Batch assignment vector of shape ``[total_nodes_in_batch]``,
                mapping each node to its graph index in the batch.
            visit_node: Padded visit-node tensor of shape
                ``[batch_size, max_visit, num_nodes]``.  Binary indicator of
                which KG nodes appear in each patient visit.
            ehr_nodes: Direct EHR node indicator of shape
                ``[batch_size, num_nodes]`` (one-hot).  Required when
                ``patient_mode`` is ``"node"`` or ``"joint"``.
            store_attn: If ``True``, return intermediate attention weights
                for interpretability.
            in_drop: If ``True`` and ``drop_rate > 0``, randomly drop edges
                during training.

        Returns:
            Logits tensor of shape ``[batch_size, out_channels]``.
            If ``store_attn`` is ``True``, also returns alpha, beta,
            attention, and edge weight lists.
        """
        # --- Optional edge dropout ---
        if in_drop and self.drop_rate > 0:
            edge_count = edge_index.size(1)
            edges_to_remove = int(edge_count * self.drop_rate)
            if edges_to_remove > 0:
                indices_to_remove = set(random.sample(range(edge_count), edges_to_remove))
                keep = [i for i in range(edge_count) if i not in indices_to_remove]
                edge_index = edge_index[:, keep].to(edge_index.device)
                rel_ids = rel_ids[keep]

        # --- Embed & project ---
        x = self.lin(self.node_emb(node_ids).float())
        edge_attr = self.lin(self.rel_emb(rel_ids).float())

        batch_size = batch.max().item() + 1

        if store_attn:
            alpha_weights_list, beta_weights_list = [], []
            attention_weights_list, edge_weights_list = [], []

        # --- GNN layers with bi-attention ---
        for layer_idx in range(1, self.num_layers + 1):
            k = str(layer_idx)

            # Compute alpha: visit-level attention  (batch, max_visit, num_nodes)
            if self.use_alpha:
                alpha = torch.softmax(
                    self.alpha_attn[k](visit_node.float()), dim=1
                )

            # Compute beta: node-level attention with temporal decay
            if self.use_beta:
                beta = (
                    torch.tanh(self.beta_attn[k](visit_node.float()))
                    * self.lambda_j
                )

            # Combine alpha and beta
            if self.use_alpha and self.use_beta:
                attn = alpha * beta
            elif self.use_alpha:
                attn = alpha * torch.ones(
                    batch_size, self.max_visit, 1, device=edge_index.device
                )
            elif self.use_beta:
                attn = beta * torch.ones(
                    batch_size, self.max_visit, self.num_nodes,
                    device=edge_index.device,
                )
            else:
                attn = torch.ones(
                    batch_size, self.max_visit, self.num_nodes,
                    device=edge_index.device,
                )

            # Sum over visits → (batch, num_nodes)
            attn = torch.sum(attn, dim=1)

            # Index into per-edge attention: for each edge (j→i),
            # get attn[batch_of_j, node_id_of_j]
            xj_node_ids = node_ids[edge_index[0]]
            xj_batch = batch[edge_index[0]]
            attn_per_edge = attn[xj_batch, xj_node_ids].reshape(-1, 1)

            # Apply GNN conv
            if self.gnn == "BAT":
                x, w_rel = self.conv[k](x, edge_index, edge_attr, attn=attn_per_edge)
            else:
                x = self.conv[k](x, edge_index)
                w_rel = None

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if store_attn:
                alpha_weights_list.append(alpha if self.use_alpha else None)
                beta_weights_list.append(beta if self.use_beta else None)
                attention_weights_list.append(attn_per_edge)
                edge_weights_list.append(w_rel)

        # --- Patient representation ---
        if self.patient_mode in ("joint", "graph"):
            x_graph = global_mean_pool(x, batch)
            x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)

        if self.patient_mode in ("joint", "node"):
            # Weighted average of direct EHR node embeddings
            # ehr_nodes: (batch_size, num_nodes) — binary indicators
            x_node = torch.stack([
                ehr_nodes[i].view(1, -1) @ self.node_emb.weight
                / torch.sum(ehr_nodes[i]).clamp(min=1)
                for i in range(batch_size)
            ])
            x_node = self.lin(x_node).squeeze(1)
            x_node = F.dropout(x_node, p=self.dropout, training=self.training)

        # --- Prediction head ---
        if self.patient_mode == "joint":
            x_concat = torch.cat((x_graph, x_node), dim=1)
            x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
            logits = self.MLP(x_concat)
        elif self.patient_mode == "graph":
            logits = self.MLP(x_graph)
        else:  # "node"
            logits = self.MLP(x_node)

        if store_attn:
            return (
                logits,
                alpha_weights_list,
                beta_weights_list,
                attention_weights_list,
                edge_weights_list,
            )
        return logits