import torch
import torch.nn as nn

from typing import Dict, List, Any
from pyhealth.models import BaseModel


class HierarchicalEmbedding(nn.Module):
    """Concatenates embeddings from multiple hierarchical levels.

    Args:
        num_codes (int): Number of unique medical codes in vocabulary.
        num_levels (int): Number of hierarchical embedding levels.
        emb_dim (int): Embedding dimension per level.
    """

    def __init__(self, num_codes: int, num_levels: int, emb_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_codes, emb_dim) for _ in range(num_levels)]
        )

    def forward(self, code_ids: torch.Tensor) -> torch.Tensor:
        """Forward propagation.

        Args:
            code_ids (torch.Tensor):
                Tensor of shape (num_codes,) representing medical code indices.

        Returns:
            torch.Tensor:
                Concatenated embeddings of shape:
                (num_codes, emb_dim * num_levels)
        """
        embs = [emb(code_ids) for emb in self.embeddings]
        return torch.cat(embs, dim=-1)


class UniGINConv(nn.Module):
    """
    UniGIN hypergraph convolution layer

    This layer implements a simplified version of UniGIN
    (Unified Graph Isomorphism Network for graphs and hypergraphs).

    This layer performs two-stage message passing:
        1. Aggregate node features to hyperedges (mean pooling)
        2. Aggregate hyperedge features back to nodes (sum pooling)
    plus a learnable epsilon to balance self and neighbor information.

    Docs:
        - Huang, W., & Yang, Z. (2021).
            UniGNN: a unified framework for graph and hypergraph neural networks.
            https://arxiv.org/abs/2105.00956

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.eps = nn.Parameter(torch.zeros(1))  # learnable epsilon

        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            X: (num_nodes, in_dim) hierarchical matrix
            H: (num_nodes, num_hyperedges) incidence matrix

        Returns:
            torch.Tensor:
                Updated node representations of shape (num_nodes, out_dim)
        """
        # Node to Hyperedge aggregation
        # Compute degree of each hyperedge (number of nodes)
        edge_deg = H.sum(dim=0, keepdim=True).clamp(min=1e-8)
        # Average node features in each hyperedge
        X_edge = (H.T @ X) / edge_deg.T  # (E, in_dim)

        # Hyperedge to Node aggregation
        node_deg = H.sum(dim=1, keepdim=True).clamp(min=1e-8)
        # Sum of hyperedge embeddings for each node
        X_node_agg = H @ X_edge / node_deg  # (N, in_dim)

        # Combine self and neighbor features with learnable epsilon
        out = (1 + self.eps) * X + X_node_agg

        # Linear projection
        out = self.linear(out)
        out = self.norm(out)
        out = self.activation(out)

        return out


class SHy(BaseModel):
    """Simplified Self-Explaining Hypergraph Model.

    This model performs:
        1. Hierarchical embedding of medical codes
        2. UniGIN-based hypergraph message passing
        3. Temporal modeling using GRU
        4. Phenotype-level attention
        5. Final classification prediction

    Args:
        dataset (Any): PyHealth dataset object.
        embedding_dim (int): Embedding dimension.
        num_levels (int): Number of embedding hierarchy levels.
        hidden_dim (int): Hidden state size.
        num_layers (int): Number of UniGIN layers.
        num_phenotypes (int): Number of phenotype heads.

    Example:
        >>> from pyhealth.models import SHy
        ... from pyhealth.datasets import create_sample_dataset
        >>> CODES = ["1234", "D101", "D102", "D103", "D104"]
        >>> data = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "prev_diag": [[CODES[0], CODES[1]], [CODES[2]]],
        ...         "label": [CODES[0], CODES[2]],
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-1",
        ...         "prev_diag": [[CODES[2]], [CODES[3], CODES[4]]],
        ...         "label": [CODES[1], CODES[3]],
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=data,
        ...     input_schema={"prev_diag": "nested_sequence"},
        ...     output_schema={"label": "multilabel"},
        ...     dataset_name="simple_test",
        ... )
        >>> model = SHy(dataset)
    """

    def __init__(
        self,
        dataset: Any,
        embedding_dim: int = 64,
        num_levels: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_phenotypes: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset=dataset, **kwargs)

        processor = dataset.input_processors[self.feature_keys[0]]
        self.num_codes: int = processor.vocab_size()
        self.output_size: int = self.get_output_size()

        self.embedding_dim = embedding_dim
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_phenotypes = num_phenotypes

        # Embedding
        self.embed = HierarchicalEmbedding(
            num_codes=self.num_codes,
            num_levels=self.num_levels,
            emb_dim=self.embedding_dim // self.num_levels
        )

        # Multiple UniGIN layers with residual connections
        self.convs = nn.ModuleList()
        current_dim = embedding_dim
        for _ in range(num_layers):
            self.convs.append(UniGINConv(current_dim, hidden_dim))
            current_dim = hidden_dim

        # Temporal encoder
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True
        )

        # Phenotype attention
        self.attn = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_phenotypes)]
        )

        # Output layer
        self.classifier = nn.Linear(
            hidden_dim, self.output_size
        )

        # If after convs we still have embedding_dim (num_layers=0),
        # project to hidden_dim
        if current_dim != hidden_dim:
            self.proj = nn.Linear(current_dim, hidden_dim)
        else:
            self.proj = nn.Identity()
        
        self.pheno_attn = nn.Linear(hidden_dim, 1)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self, **batch: Any
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **batch: Keyword arguments from PyHealth dataloader containing:
                - prev_diag (List[List[List[int]]]): visit sequences
                - label (optional): ground truth labels
                - patient_id, visit_id (ignored)

        Returns:
            Dict[str, torch.Tensor]:
                - logits: raw predictions
                - y_true: true diagonsis label
                - y_prob: sigmoid probabilities
                - loss: the penalty of predictions compared to actual value
        """
        # Extract visits (nested sequence) from batch
        prev_diag = batch["prev_diag"]
        y_true = batch["label"]

        code_ids = torch.arange(self.num_codes, device=self.device)
        x = self.embed(code_ids)

        logits_all: List[torch.Tensor] = []

        for visits in prev_diag:
            num_visits = len(visits)

            h = torch.zeros(self.num_codes, num_visits, device=self.device)

            for t, codes in enumerate(visits):
                if not isinstance(codes, torch.Tensor):
                    codes = torch.tensor(codes, device=self.device)
                else:
                    codes = codes.to(self.device)
                h[codes, t] = 1.0

            # Hypergraph message passing with residual connections
            x_h = x
            for conv in self.convs:
                x_h = conv(x_h, h)

            # Project to hidden_dim if needed
            x_h = self.proj(x_h)

            # Visit embeddings
            deg = h.sum(dim=0, keepdim=True).clamp(min=1e-8)
            visit_emb = (h.T @ x_h) / deg.T

            # Temporal GRU
            visit_seq = visit_emb.unsqueeze(0)
            gru_out, _ = self.gru(visit_seq)
            gru_out = gru_out.squeeze(0)

            # Phenotype extraction: attention over time
            phenotypes = []
            alphas = []
            for attn in self.attn:
                score = attn(gru_out).squeeze(-1)
                alpha = torch.softmax(score, dim=0)
                pheno = (alpha.unsqueeze(-1) * gru_out).sum(dim=0)
                phenotypes.append(pheno)
                alphas.append(alpha)
            phenotypes = torch.stack(phenotypes)
            attn = torch.softmax(self.pheno_attn(phenotypes), dim=0)
            patient_repr = (attn * phenotypes).sum(dim=0)

            logits = self.classifier(patient_repr)
            logits_all.append(logits)

        logits_all = torch.stack(logits_all)
        loss = self.criterion(logits_all, y_true)
        y_prob = torch.sigmoid(logits_all)

        return {
            "logits": logits_all,
            "y_true": y_true,
            "y_prob": y_prob,
            "loss": loss,
        }
