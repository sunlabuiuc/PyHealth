"""Step-wise Embedding model for heterogeneous clinical time-series.

This module implements the step-wise embedding architecture from the paper
"On the Importance of Step-wise Embeddings for Heterogeneous Clinical
Time-Series" (Kuznetsova et al., JMLR 2023).

The key idea is to group heterogeneous clinical features by modality
(e.g., organ system or variable type), apply per-group embedding models
at each timestep, aggregate the group embeddings, and then pass the
result to a sequence backbone (Transformer).

This module provides:
- ``StepwiseEmbedding``: The main PyHealth model class.
- ``StepwiseEmbeddingLayer``: The core grouping + embedding + aggregation
  layer.
- Helper layers: ``LinearEmbedding``, ``MLPEmbedding``,
  ``FTTransformerEmbedding``, ``TransformerBackbone``, etc.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets import SampleDataset
from .base_model import BaseModel


# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Adds sinusoidal positional encodings to input embeddings, following
    the formulation from "Attention Is All You Need" (Vaswani et al.).

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length. Default 1024.
        dropout: Dropout rate. Default 0.0.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape ``(B, T, D)``.

        Returns:
            Tensor of same shape with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single Transformer block with optional causal masking.

    Implements multi-head self-attention followed by a feed-forward
    network, with layer normalization and residual connections.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        ff_hidden_mult: Feed-forward hidden dimension multiplier.
        dropout: Dropout rate. Default 0.0.
        causal: Whether to use causal (autoregressive) masking.
            Default ``True``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_hidden_mult: int = 2,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_hidden_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer block.

        Args:
            x: Input tensor of shape ``(B, T, D)``.

        Returns:
            Output tensor of shape ``(B, T, D)``.
        """
        T = x.size(1)
        attn_mask = None
        if self.causal and T > 1:
            attn_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1,
            )

        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attention(
            normed, normed, normed, attn_mask=attn_mask
        )
        x = x + attn_out

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        return x


class TransformerBackbone(nn.Module):
    """Transformer backbone for temporal sequence modeling.

    Projects input features to the hidden dimension, adds positional
    encoding, and applies a stack of Transformer blocks.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden dimension for the Transformer.
        heads: Number of attention heads.
        depth: Number of Transformer blocks.
        ff_hidden_mult: Feed-forward expansion factor. Default 2.
        dropout: Dropout rate. Default 0.0.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        heads: int = 1,
        depth: int = 1,
        ff_hidden_mult: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=hidden_dim,
                    n_heads=heads,
                    ff_hidden_mult=ff_hidden_mult,
                    dropout=dropout,
                    causal=True,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone.

        Args:
            x: Input tensor of shape ``(B, T, input_dim)``.

        Returns:
            Output tensor of shape ``(B, T, hidden_dim)``.
        """
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.blocks(x)
        return x


# ---------------------------------------------------------------------------
# Per-group embedding layers
# ---------------------------------------------------------------------------


class LinearEmbedding(nn.Module):
    """Simple linear projection per timestep.

    Args:
        input_dim: Number of input features.
        hidden_dim: Output embedding dimension.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, input_dim)``.

        Returns:
            Tensor of shape ``(B, T, hidden_dim)``.
        """
        return self.linear(x)


class MLPEmbedding(nn.Module):
    """Multi-layer perceptron embedding per timestep.

    Args:
        input_dim: Number of input features.
        hidden_dim: Output embedding dimension.
        latent_dim: Hidden layer dimension. Default 64.
        depth: Number of hidden layers. Default 1.
        dropout: Dropout rate. Default 0.0.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int = 64,
        depth: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(latent_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, input_dim)``.

        Returns:
            Tensor of shape ``(B, T, hidden_dim)``.
        """
        return self.mlp(x)


class CLSToken(nn.Module):
    """Learnable [CLS] token appended to a sequence.

    Appends a learned embedding to the end of each sequence in the batch.

    Args:
        d_model: Token embedding dimension.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.xavier_uniform_(self.token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Append CLS token to the sequence.

        Args:
            x: Input tensor of shape ``(B, N, D)``.

        Returns:
            Tensor of shape ``(B, N+1, D)`` with CLS token appended.
        """
        cls = self.token.expand(x.size(0), -1, -1)
        return torch.cat([x, cls], dim=1)


class FTTransformerEmbedding(nn.Module):
    """Feature Tokenizer + Transformer embedding per timestep.

    Implements the FT-Transformer embedding from "Revisiting Deep Learning
    Models for Tabular Data" (Gorishniy et al., NeurIPS 2021). Each input
    feature is projected to a token, a CLS token is appended, and a small
    Transformer processes the token sequence. The CLS token output is
    used as the embedding.

    Args:
        input_dim: Number of input features.
        hidden_dim: Output embedding dimension.
        token_dim: Per-feature token dimension. Default 16.
        n_heads: Number of attention heads in the small Transformer.
            Default 2.
        depth: Number of Transformer layers. Default 1.
        dropout: Dropout rate. Default 0.0.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        token_dim: int = 16,
        n_heads: int = 2,
        depth: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.token_dim = token_dim

        # Per-feature linear tokenization: each feature -> token
        self.feature_weights = nn.Parameter(
            torch.empty(input_dim, token_dim)
        )
        self.feature_biases = nn.Parameter(torch.empty(input_dim, token_dim))
        nn.init.xavier_uniform_(self.feature_weights)
        nn.init.zeros_(self.feature_biases)

        # CLS token
        self.cls_token = CLSToken(token_dim)

        # Small Transformer over feature tokens
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=token_dim,
                    n_heads=n_heads,
                    ff_hidden_mult=2,
                    dropout=dropout,
                    causal=False,
                )
                for _ in range(depth)
            ]
        )

        # Head: CLS token -> hidden_dim
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, input_dim)``.

        Returns:
            Tensor of shape ``(B, T, hidden_dim)``.
        """
        B, T, F = x.shape

        # Reshape for per-timestep processing
        x_flat = x.reshape(B * T, F)  # (B*T, F)

        # Tokenize each feature: x_flat[:, i] * W[i] + b[i]
        # x_flat: (B*T, F) -> (B*T, F, 1) * (F, token_dim) -> (B*T, F, D)
        tokens = (
            x_flat.unsqueeze(-1) * self.feature_weights.unsqueeze(0)
            + self.feature_biases.unsqueeze(0)
        )  # (B*T, F, token_dim)

        # Append CLS token
        tokens = self.cls_token(tokens)  # (B*T, F+1, token_dim)

        # Process with small Transformer
        tokens = self.transformer_blocks(tokens)  # (B*T, F+1, token_dim)

        # Extract CLS token (last position)
        cls_out = tokens[:, -1, :]  # (B*T, token_dim)

        # Project to hidden_dim
        out = self.head(cls_out)  # (B*T, hidden_dim)

        return out.reshape(B, T, -1)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class AttentionAggregation(nn.Module):
    """Aggregate group embeddings via CLS-token attention.

    Stacks per-group embeddings, appends a CLS token, processes with
    a Transformer block, and extracts the CLS output.

    Args:
        hidden_dim: Embedding dimension for each group.
        n_heads: Number of attention heads. Default 2.
        depth: Number of Transformer layers. Default 1.
        dropout: Dropout rate. Default 0.0.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 2,
        depth: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cls_token = CLSToken(hidden_dim)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=hidden_dim,
                    n_heads=n_heads,
                    ff_hidden_mult=2,
                    dropout=dropout,
                    causal=False,
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, group_embeds: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate group embeddings.

        Args:
            group_embeds: List of tensors, each of shape
                ``(B, T, hidden_dim)``.

        Returns:
            Tensor of shape ``(B, T, hidden_dim)``.
        """
        B, T, D = group_embeds[0].shape
        K = len(group_embeds)

        # Reshape: (B, T, D) -> (B*T, D) for each group, then stack
        flat = [g.reshape(B * T, D) for g in group_embeds]
        stacked = torch.stack(flat, dim=1)  # (B*T, K, D)

        # Append CLS token
        stacked = self.cls_token(stacked)  # (B*T, K+1, D)

        # Transformer over groups
        stacked = self.transformer_blocks(stacked)  # (B*T, K+1, D)

        # Extract CLS token (last position)
        cls_out = stacked[:, -1, :]  # (B*T, D)
        cls_out = self.head(cls_out)  # (B*T, D)

        return cls_out.reshape(B, T, D)


# ---------------------------------------------------------------------------
# Core step-wise embedding layer
# ---------------------------------------------------------------------------

# Map from embedding type name to class
_EMBEDDING_REGISTRY = {
    "linear": LinearEmbedding,
    "mlp": MLPEmbedding,
    "ftt": FTTransformerEmbedding,
}


class StepwiseEmbeddingLayer(nn.Module):
    """Core step-wise embedding: group -> embed -> aggregate.

    This is the key contribution of the paper. It:

    1. Splits input features into groups by index lists.
    2. Applies per-group embedding models (Linear, MLP, or
       FT-Transformer) at each timestep.
    3. Aggregates group outputs via concat, mean, or attention.

    Args:
        input_dim: Total number of input features.
        hidden_dim: Output embedding dimension.
        group_indices: List of feature index lists defining the groups.
            If ``None``, all features are treated as a single group.
        embedding_type: Type of per-group embedding. One of
            ``"linear"``, ``"mlp"``, or ``"ftt"``. Default ``"ftt"``.
        aggregation: Method to aggregate group embeddings. One of
            ``"concat"``, ``"mean"``, or ``"attention_cls"``.
            Default ``"mean"``.
        embedding_kwargs: Additional keyword arguments passed to the
            embedding layer constructors.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        group_indices: Optional[List[List[int]]] = None,
        embedding_type: str = "ftt",
        aggregation: str = "mean",
        embedding_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.group_indices = group_indices
        self.aggregation = aggregation
        self.hidden_dim = hidden_dim

        if embedding_kwargs is None:
            embedding_kwargs = {}

        emb_cls = _EMBEDDING_REGISTRY[embedding_type]

        if group_indices is None:
            # Single group: all features
            self.emb_blocks = nn.ModuleList([
                emb_cls(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    **embedding_kwargs,
                )
            ])
        else:
            n_groups = len(group_indices)
            if aggregation == "concat":
                # Divide hidden_dim among groups
                per_group = hidden_dim // n_groups
                dims = [per_group] * n_groups
                for i in range(hidden_dim - sum(dims)):
                    dims[i] += 1
            else:
                dims = [hidden_dim] * n_groups

            self.emb_blocks = nn.ModuleList([
                emb_cls(
                    input_dim=len(group_indices[i]),
                    hidden_dim=dims[i],
                    **embedding_kwargs,
                )
                for i in range(n_groups)
            ])

        # Attention aggregation module
        self.attn_agg = None
        if (
            group_indices is not None
            and aggregation == "attention_cls"
        ):
            self.attn_agg = AttentionAggregation(
                hidden_dim=hidden_dim,
                n_heads=embedding_kwargs.get("agg_heads", 2),
                depth=embedding_kwargs.get("agg_depth", 1),
                dropout=embedding_kwargs.get("dropout", 0.0),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, input_dim)``.

        Returns:
            Tensor of shape ``(B, T, hidden_dim)``.
        """
        if self.group_indices is None:
            return self.emb_blocks[0](x)

        # Split by groups and embed each
        outputs = []
        for i, indices in enumerate(self.group_indices):
            x_group = x[..., indices]
            outputs.append(self.emb_blocks[i](x_group))

        # Aggregate
        if self.aggregation == "concat":
            return torch.cat(outputs, dim=-1)
        elif self.aggregation == "mean":
            return sum(outputs) / len(outputs)
        elif self.aggregation == "attention_cls":
            return self.attn_agg(outputs)
        else:
            raise ValueError(
                f"Unknown aggregation: {self.aggregation}"
            )


# ---------------------------------------------------------------------------
# Main PyHealth model
# ---------------------------------------------------------------------------


class StepwiseEmbedding(BaseModel):
    """Step-wise Embedding model for heterogeneous clinical time-series.

    Implements the architecture from "On the Importance of Step-wise
    Embeddings for Heterogeneous Clinical Time-Series" (Kuznetsova et
    al., JMLR 2023). The model processes dense time-series inputs through:

    1. **Step-wise embedding**: Features are optionally grouped by
       clinical modality, embedded per-group using Linear, MLP, or
       FT-Transformer layers, and aggregated.
    2. **Transformer backbone**: The embedded sequence is processed by
       a stack of causal Transformer blocks.
    3. **Classification head**: The last timestep output is projected
       to the task output dimension.

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` with
            ``"tensor"`` input schema for the time-series feature.
        input_dim: Number of input features per timestep. Default 42
            (MIMIC-III TLS benchmark).
        hidden_dim: Hidden dimension for embeddings and backbone.
            Default 128.
        backbone_depth: Number of Transformer layers in the backbone.
            Default 1.
        backbone_heads: Number of attention heads in the backbone.
            Default 1.
        embedding_type: Per-group embedding type. One of ``"linear"``,
            ``"mlp"``, ``"ftt"``, or ``None`` (backbone only, no
            step-wise embedding). Default ``"ftt"``.
        group_indices: Feature grouping as a list of index lists. Use
            :attr:`~pyhealth.datasets.MIMIC3TLSDataset.ORGAN_GROUPS_INDICES`
            or
            :attr:`~pyhealth.datasets.MIMIC3TLSDataset.TYPE_GROUPS_INDICES`
            for predefined groupings. ``None`` means no grouping.
            Default ``None``.
        aggregation: Group aggregation method. One of ``"concat"``,
            ``"mean"``, or ``"attention_cls"``. Default ``"mean"``.
        dropout: Dropout rate. Default 0.1.
        ff_hidden_mult: Feed-forward hidden dimension multiplier for
            the Transformer. Default 2.
        embedding_kwargs: Additional keyword arguments for the
            embedding layers (e.g., ``token_dim``, ``latent_dim``).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import StepwiseEmbedding
        >>> import numpy as np
        >>>
        >>> # Create synthetic dataset
        >>> samples = [
        ...     {"patient_id": f"p{i}",
        ...      "time_series": np.random.randn(48, 42).tolist(),
        ...      "ihm": i % 2}
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"time_series": "tensor"},
        ...     output_schema={"ihm": "binary"},
        ...     dataset_name="test",
        ... )
        >>>
        >>> # Backbone-only model (no step-wise embedding)
        >>> model = StepwiseEmbedding(
        ...     dataset=dataset, embedding_type=None, input_dim=42,
        ... )
        >>>
        >>> # FTT with organ grouping
        >>> from pyhealth.datasets.mimic3_tls import MIMIC3TLSDataset
        >>> model = StepwiseEmbedding(
        ...     dataset=dataset,
        ...     embedding_type="ftt",
        ...     group_indices=MIMIC3TLSDataset.ORGAN_GROUPS_INDICES,
        ...     aggregation="mean",
        ...     input_dim=42,
        ... )
    """

    def __init__(
        self,
        dataset: SampleDataset,
        input_dim: int = 42,
        hidden_dim: int = 128,
        backbone_depth: int = 1,
        backbone_heads: int = 1,
        embedding_type: Optional[str] = "ftt",
        group_indices: Optional[List[List[int]]] = None,
        aggregation: str = "mean",
        dropout: float = 0.1,
        ff_hidden_mult: int = 2,
        embedding_kwargs: Optional[Dict] = None,
    ):
        super().__init__(dataset=dataset)

        assert len(self.label_keys) == 1, (
            "StepwiseEmbedding supports single-label tasks only"
        )
        self.label_key = self.label_keys[0]
        self.feature_key = self.feature_keys[0]
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        if embedding_kwargs is None:
            embedding_kwargs = {}

        # Step-wise embedding layer (or None for backbone-only)
        if embedding_type is not None:
            self.stepwise = StepwiseEmbeddingLayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                group_indices=group_indices,
                embedding_type=embedding_type,
                aggregation=aggregation,
                embedding_kwargs=embedding_kwargs,
            )
            backbone_input_dim = hidden_dim
        else:
            self.stepwise = None
            backbone_input_dim = input_dim

        # Transformer backbone
        self.backbone = TransformerBackbone(
            input_dim=backbone_input_dim,
            hidden_dim=hidden_dim,
            heads=backbone_heads,
            depth=backbone_depth,
            ff_hidden_mult=ff_hidden_mult,
            dropout=dropout,
        )

        # Classification head
        output_size = self.get_output_size()
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(
        self, **kwargs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            **kwargs: Keyword arguments containing input features and
                labels as tensors. The time-series feature is accessed
                via ``self.feature_key`` and the label via
                ``self.label_key``.

        Returns:
            Dictionary with keys ``"loss"``, ``"y_prob"``, ``"y_true"``,
            and ``"logit"``.
        """
        # Get time-series tensor from TensorProcessor
        feature = kwargs[self.feature_key]
        if isinstance(feature, tuple):
            x = feature[0]
        else:
            x = feature
        x = x.to(self.device).float()

        # Step-wise embedding (optional)
        if self.stepwise is not None:
            x = self.stepwise(x)  # (B, T, hidden_dim)

        # Transformer backbone
        x = self.backbone(x)  # (B, T, hidden_dim)

        # Use last timestep for classification
        x = x[:, -1, :]  # (B, hidden_dim)

        # Classification
        logits = self.fc(x)

        # Compute loss if labels are available
        y_true = kwargs[self.label_key]
        if isinstance(y_true, tuple):
            y_true = y_true[0]
        y_true = y_true.to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
