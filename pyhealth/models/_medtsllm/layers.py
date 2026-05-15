"""Neural network layers for MedTsLLM.

Includes: RevIN, PatchEmbedding, ReprogrammingLayer, LinearProjection,
FlattenHead. These are the lightweight trainable components that wrap
around a frozen LLM backbone.
"""

import math

import torch
from torch import Tensor, nn


class RevIN(nn.Module):
    """Reversible instance normalization for time series.

    Normalizes each feature by subtracting the mean and dividing by the
    standard deviation. Stores statistics so the operation can be
    reversed for reconstruction tasks.

    Args:
        num_features: Number of input features/channels.
        eps: Small value for numerical stability.
        affine: If True, learns per-feature scale and bias.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

        self._mean: Tensor | None = None
        self._stdev: Tensor | None = None

    def forward(self, x: Tensor, mode: str) -> Tensor:
        """Apply normalization or denormalization.

        Args:
            x: (batch, seq_len, num_features).
            mode: ``"norm"`` or ``"denorm"``.
        """
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._stdev = (
                (x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
                .sqrt()
                .detach()
            )
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == "denorm":
            if self._mean is None or self._stdev is None:
                raise RuntimeError("Call forward(x, 'norm') first.")
            if self.affine:
                x = (x - self.affine_bias) / self.affine_weight
            return x * self._stdev + self._mean
        else:
            raise ValueError(f"mode must be 'norm' or 'denorm', got '{mode}'")


class _TokenEmbedding(nn.Module):
    """1D convolution over a single patch (maps patch_len -> d_model).

    Separated from ``PatchEmbedding`` as its own module so the state
    dict keys (``value_embedding.conv.weight``) match the TIME-LLM /
    original-MedTsLLM upstream naming. This lets you load checkpoints
    trained outside of PyHealth without renaming keys.

    Args:
        patch_len: Length of each patch (input channels).
        d_model: Output embedding dimension.
    """

    def __init__(self, patch_len: int, d_model: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=patch_len,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Embed patches via 1D convolution.

        Args:
            x: (batch, seq_len, patch_len).

        Returns:
            Embedded patches: (batch, seq_len, d_model).
        """
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class PatchEmbedding(nn.Module):
    """Unfolds a time series into overlapping patches and embeds via Conv1d.

    Args:
        d_model: Embedding dimension for each patch.
        patch_len: Length of each patch in timesteps.
        stride: Stride between consecutive patches.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        stride: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        self.padding = nn.ReplicationPad1d((0, stride))
        self.value_embedding = _TokenEmbedding(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, int]:
        """Embed time series into patch representations.

        Args:
            x: (batch, n_features, seq_len).

        Returns:
            Tuple of (patch_embeddings, n_features).
            patch_embeddings: (batch * n_features, n_patches, d_model).
        """
        n_vars = x.shape[1]
        x = self.padding(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class ReprogrammingLayer(nn.Module):
    """Cross-attention reprogramming layer.

    Projects patch embeddings (queries) against word prototype
    embeddings (keys/values) using multi-head attention. This is
    the core mechanism of MedTsLLM / Time-LLM.

    Args:
        d_model: Dimension of input patch embeddings.
        n_heads: Number of attention heads.
        d_keys: Dimension per head for keys.
        d_llm: Dimension of LLM embeddings.
        attention_dropout: Dropout on attention weights.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_keys: int,
        d_llm: int,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_keys = d_keys

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        target_embedding: Tensor,
        source_embedding: Tensor,
        value_embedding: Tensor,
    ) -> Tensor:
        """Reprogram patch embeddings via cross-attention.

        Args:
            target_embedding: Queries, (batch, n_patches, d_model).
            source_embedding: Keys, (n_tokens, d_llm).
            value_embedding: Values, (n_tokens, d_llm).

        Returns:
            Reprogrammed embeddings: (batch, n_patches, d_llm).
        """
        b, seq_len, _ = target_embedding.shape
        s, _ = source_embedding.shape
        h = self.n_heads

        queries = self.query_projection(target_embedding).view(
            b, seq_len, h, -1
        )
        keys = self.key_projection(source_embedding).view(s, h, -1)
        values = self.value_projection(value_embedding).view(s, h, -1)

        scale = 1.0 / math.sqrt(queries.shape[-1])
        scores = torch.einsum("blhe,she->bhls", queries, keys)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,she->blhe", attn, values)
        out = out.reshape(b, seq_len, -1)
        return self.out_projection(out)


class LinearProjection(nn.Module):
    """Simple linear projection as ablation alternative.

    Replaces ReprogrammingLayer with a plain linear map from
    d_model to d_llm, ignoring word embeddings.

    Args:
        d_model: Input dimension.
        d_llm: Output dimension.
    """

    def __init__(self, d_model: int, d_llm: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_llm)

    def forward(
        self,
        target_embedding: Tensor,
        source_embedding: Tensor,
        value_embedding: Tensor,
    ) -> Tensor:
        """Project patch embeddings linearly to LLM dimension."""
        return self.linear(target_embedding)


class FlattenHead(nn.Module):
    """Flatten patch dimension and project to output size.

    Args:
        n_features_in: Total input features (d_ff * n_patches).
        n_outputs: Total output size (pred_len * n_outputs_per_step).
        head_dropout: Dropout probability.
    """

    def __init__(
        self,
        n_features_in: int,
        n_outputs: int,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(n_features_in, n_outputs)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x: (batch, d_ff, n_patches). Returns: (batch, n_outputs)."""
        return self.dropout(self.linear(self.flatten(x)))
