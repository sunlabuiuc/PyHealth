# Author: Joshua Steier
# Paper: Jiang et al., "GraphCare: Enhancing Healthcare Predictions
#     with Personalized Knowledge Graphs", ICLR 2024
# Description: Bi-Attention GNN Convolution layer (BAT) for GraphCare.
#     GIN-style message passing augmented with node-level attention and
#     optional edge relation weights.

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from torch_geometric.nn import MessagePassing

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    MessagePassing = nn.Module  # fallback for class definition


class BiAttentionGNNConv(MessagePassing if HAS_PYG else nn.Module):
    """Bi-attention augmented GNN convolution from GraphCare (ICLR 2024).

    GIN-style message passing with node-level attention (alpha) and
    optional edge relation weights (W_R). Used as the core layer in
    GraphCare's BAT (Bi-Attention augmenTed) GNN backbone.

    The message for edge (j -> i) is:

        m_{j->i} = ReLU(x_j * attn_j + W_R(e_ji) * e_ji)

    where attn_j is the pre-computed attention weight and e_ji is the
    relation embedding on that edge.

    Args:
        hidden_dim: Hidden dimension size.
        edge_dim: Edge attribute dimension. Default: same as hidden_dim.
        edge_attn: Whether to use edge attention weights. Default True.
        eps: Initial epsilon for self-loop weighting. Default 0.0.
        train_eps: Whether epsilon is trainable. Default False.

    Example:
        >>> from pyhealth.models._graphcare.bat_gnn import BiAttentionGNNConv
        >>> layer = BiAttentionGNNConv(hidden_dim=64, edge_attn=True)
        >>> x = torch.randn(10, 64)
        >>> edge_index = torch.tensor([[0,1,2],[1,2,0]])
        >>> edge_attr = torch.randn(3, 64)
        >>> attn = torch.ones(10, 1)
        >>> out, w_rel = layer(x, edge_index, edge_attr, attn)
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: Optional[int] = None,
        edge_attn: bool = True,
        eps: float = 0.0,
        train_eps: bool = False,
    ):
        if HAS_PYG:
            super().__init__(aggr="add")
        else:
            super().__init__()

        self.nn = nn.Linear(hidden_dim, hidden_dim)
        self.edge_attn = edge_attn

        if edge_dim is None:
            edge_dim = hidden_dim

        if edge_attn:
            self.W_R = nn.Linear(edge_dim, 1)
        else:
            self.W_R = None

        self.initial_eps = eps
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.nn.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        if self.W_R is not None:
            self.W_R.reset_parameters()

    def forward(
        self,
        x: "torch.Tensor",
        edge_index: "torch.Tensor",
        edge_attr: "torch.Tensor",
        attn: "torch.Tensor",
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """Forward pass.

        Args:
            x: Node features ``[num_nodes, hidden_dim]``.
            edge_index: Graph connectivity ``[2, num_edges]``.
            edge_attr: Edge features ``[num_edges, edge_dim]``.
            attn: Pre-computed attention weights ``[num_nodes, 1]``.

        Returns:
            Tuple of:
                - Updated node features ``[num_nodes, hidden_dim]``.
                - Edge relation weights ``[num_edges, 1]`` or None.
        """
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, attn=attn)

        # Self-loop with epsilon (GIN-style)
        out = out + (1 + self.eps) * x

        # Compute edge weights for potential use
        w_rel = self.W_R(edge_attr) if self.W_R is not None else None

        return self.nn(out), w_rel

    def message(
        self,
        x_j: "torch.Tensor",
        edge_attr: "torch.Tensor",
        attn_j: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute messages from neighbors.

        Message = ReLU(x_j * attn_j + W_R(edge_attr) * edge_attr) if edge_attn
                  ReLU(x_j * attn_j) otherwise

        The ``_j`` suffix tells PyG to auto-index ``attn`` by source
        nodes, so ``attn_j`` has shape ``[num_edges, 1]``.

        Args:
            x_j: Source node features ``[num_edges, hidden_dim]``.
            edge_attr: Edge features ``[num_edges, edge_dim]``.
            attn_j: Source node attention weights ``[num_edges, 1]``.

        Returns:
            Messages ``[num_edges, hidden_dim]``.
        """
        if self.edge_attn and self.W_R is not None:
            w_rel = self.W_R(edge_attr)
            return (x_j * attn_j + w_rel * edge_attr).relu()
        return (x_j * attn_j).relu()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nn={self.nn}, edge_attn={self.edge_attn})"
        )
