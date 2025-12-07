from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.embedding import EmbeddingModel


class UniGCNConv(nn.Module):
    """Unified Graph Convolutional layer for hypergraph convolution."""

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

    def forward(
        self,
        X: torch.Tensor,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        H: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            X: node features [num_nodes, in_channels]
            vertex: vertex indices in hyperedges
            edges: edge indices for each vertex
            H: incidence matrix [num_nodes, num_edges]

        Returns:
            Updated node features [num_nodes, heads * out_channels]
        """
        N = X.shape[0]
        M = H.shape[1]

        degV = torch.sum(H, 1).pow(-0.5)
        degV[torch.isinf(degV)] = 1.0
        degV = degV.unsqueeze(1)

        degE = torch.sum(H, 0).pow(-0.5)
        degE = degE.unsqueeze(1)

        X = self.W(X)

        Xve = X[vertex]
        Xe = torch.zeros(M, Xve.shape[-1], device=X.device, dtype=X.dtype)
        edge_counts = torch.zeros(M, device=X.device)
        for i, e in enumerate(edges):
            Xe[e] += Xve[i]
            edge_counts[e] += 1
        Xe = Xe / (edge_counts.unsqueeze(1) + 1e-8)
        Xe = Xe * degE

        Xev = Xe[edges]
        Xv = torch.zeros(N, Xev.shape[-1], device=X.device, dtype=X.dtype)
        for i, v in enumerate(vertex):
            Xv[v] += Xev[i]
        Xv = Xv * degV

        return Xv


class HypergraphConvLayer(nn.Module):
    """Hypergraph convolution layer stack."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        for i in range(len(channels) - 1):
            self.convs.append(UniGCNConv(channels[i], channels[i + 1], heads=1))

    def forward(
        self,
        X: torch.Tensor,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        H: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through hypergraph convolution layers."""
        for i, conv in enumerate(self.convs):
            X = conv(X, vertex, edges, H)
            if i < len(self.convs) - 1:
                X = self.act(X)
                X = self.dropout(X)
        return F.leaky_relu(X)


class TemporalAggregator(nn.Module):
    """Aggregate node embeddings over time with temporal attention."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.gru = nn.GRU(in_channels, hidden_channels, batch_first=True)
        self.attention = nn.Linear(hidden_channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Aggregate node embeddings using temporal GRU + attention.

        Args:
            X: node embeddings [num_nodes, in_channels]
            H: incidence matrix [num_nodes, num_edges]
            seq_len: actual sequence length (optional, for masking)

        Returns:
            Aggregated patient embedding [hidden_channels]
        """
        edge_emb = torch.matmul(H.T.float(), X)

        hidden_states, _ = self.gru(edge_emb.unsqueeze(0))
        hidden_states = hidden_states.squeeze(0)

        attn_weights = self.attention(hidden_states)
        attn_weights = self.softmax(attn_weights)

        patient_emb = torch.sum(attn_weights * hidden_states, dim=0)
        return patient_emb


class HypergraphBuilder(nn.Module):
    """Build hypergraph incidence matrix from sequence of codes."""

    def __init__(self):
        super().__init__()

    def forward(
        self, X: torch.Tensor, visit_boundaries: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Build incidence matrix where hyperedges = visits, nodes = codes.

        Args:
            X: embeddings [seq_len, embedding_dim]
            visit_boundaries: indices marking visit boundaries (optional)

        Returns:
            Incidence matrix [seq_len, num_visits]
        """
        seq_len = X.shape[0]

        if visit_boundaries is None:
            num_visits = seq_len
            H = torch.eye(seq_len, num_visits, device=X.device)
        else:
            num_visits = len(visit_boundaries) + 1
            H = torch.zeros(seq_len, num_visits, device=X.device)

            prev = 0
            for visit_idx, boundary in enumerate(visit_boundaries):
                H[prev:boundary, visit_idx] = 1.0
                prev = boundary
            H[prev:, num_visits - 1] = 1.0

        return H


class HGNN(BaseModel):
    """Temporal Hypergraph Neural Network model for PyHealth.

    This model leverages hypergraph structure with temporal dynamics to capture
    higher-order relationships in medical event sequences. It treats medical codes
    as nodes and visits/encounters as hyperedges, learning how code co-occurrences
    evolve over time.

    The model combines:
    1. Hypergraph convolution: Captures multi-way code interactions within visits
    2. Temporal aggregation: Uses GRU to model patient state evolution
    3. Attention mechanism: Identifies which visits are most predictive

    Args:
        dataset (SampleDataset): The dataset with fitted processors.
        embedding_dim (int): Embedding dimension for medical codes. Default is 128.
        hidden_dim (int): Hidden dimension for convolution and GRU. Default is 128.
        num_conv_layers (int): Number of hypergraph convolution layers. Default is 2.
        dropout (float): Dropout rate. Default is 0.5.

    Example:
        >>> from pyhealth.datasets import SampleDataset
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "conditions": ["A05B", "A05C"],
        ...         "label": 1,
        ...     },
        ... ]
        >>> dataset = SampleDataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="toy",
        ... )
        >>> model = HGNN(dataset, embedding_dim=64, hidden_dim=64)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_conv_layers: int = 2,
        dropout: float = 0.5,
    ):
        super(HGNN, self).__init__(dataset=dataset)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        self.hypergraph_builder = HypergraphBuilder()

        self.hgnn = nn.ModuleDict()
        self.aggregator = nn.ModuleDict()
        for feature_key in self.feature_keys:
            self.hgnn[feature_key] = HypergraphConvLayer(
                in_channels=embedding_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_layers=num_conv_layers,
                dropout=dropout,
            )
            self.aggregator[feature_key] = TemporalAggregator(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(hidden_dim * len(self.feature_keys), output_size)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: Keyword arguments including all feature keys and label key.

        Returns:
            Dictionary with keys: loss, y_prob, y_true, logit
        """
        embedded = self.embedding_model(kwargs)

        patient_embeddings = []

        for feature_key in self.feature_keys:
            x = embedded[feature_key]
            batch_size = x.shape[0]

            feature_embeddings = []
            for b in range(batch_size):
                X = x[b]

                H = self.hypergraph_builder(X)

                V = torch.nonzero(H)[:, 0]
                E = torch.nonzero(H)[:, 1]

                X_conv = self.hgnn[feature_key](X, V, E, H)

                feat_emb = self.aggregator[feature_key](X_conv, H)
                feature_embeddings.append(feat_emb)

            feature_embeddings = torch.stack(feature_embeddings)
            patient_embeddings.append(feature_embeddings)

        patient_embeddings = torch.cat(patient_embeddings, dim=1)

        logits = self.fc(patient_embeddings)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }

        if kwargs.get("embed", False):
            results["embed"] = patient_embeddings

        return results


if __name__ == "__main__":
    from pyhealth.datasets import SampleDataset, get_dataloader

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": ["cond-33", "cond-86", "cond-80", "cond-12"],
            "procedures": ["proc-1", "proc-2"],
            "label": 1,
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            "conditions": ["cond-33", "cond-86", "cond-80"],
            "procedures": ["proc-1"],
            "label": 0,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-2",
            "conditions": ["cond-80", "cond-12"],
            "procedures": ["proc-2"],
            "label": 1,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-3",
            "conditions": ["cond-33", "cond-12"],
            "procedures": ["proc-1", "proc-2"],
            "label": 0,
        },
    ]

    input_schema = {"conditions": "sequence", "procedures": "sequence"}
    output_schema = {"label": "binary"}

    dataset = SampleDataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="test_hgnn",
    )

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    model = HGNN(dataset, embedding_dim=64, hidden_dim=64, num_conv_layers=2)

    print("Running HGNN model on test data...")
    data_batch = next(iter(train_loader))

    result = model(**data_batch)
    print(result)

    print("\nRunning backward pass...")
    result["loss"].backward()
    print("Backward pass completed successfully!")
