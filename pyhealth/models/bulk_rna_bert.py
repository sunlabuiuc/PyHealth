"""BulkRNABert: Transformer-based masked language model for bulk RNA-seq data.

Independent PyTorch implementation of the architecture introduced by
Gelard et al., PMLR 259 (2025):
https://proceedings.mlr.press/v259/gelard25a.html

Written from the paper and its public specifications; no source code has
been copied from any prior implementation. See the PR description for the
full provenance narrative.

The model supports two masked-language-modeling (MLM) regimes:

* ``expression_mode="discrete"``: gene expression values are first binned into
  ``n_bins`` discrete tokens (see :func:`bin_expression_values`), then embedded
  via a learned lookup table. MLM predicts the original bin ID at masked
  positions with cross-entropy loss. This matches the original paper.

* ``expression_mode="continuous"``: raw ``log10(TPM + 1)`` values are projected
  to embeddings via a small MLP and a learned mask embedding is used at masked
  positions. MLM predicts the original continuous value at masked positions
  with mean squared error loss. This mode is an extension to the original
  paper.

Masking (BERT 80/10/10 rule) is applied dynamically inside :meth:`forward` so
that the model is compatible with PyHealth's :class:`~pyhealth.trainer.Trainer`
without any custom collate or loss bookkeeping.

Author: Yohei Shibata (NetID: yoheis2)
Paper: BulkRNABert: Cancer prognosis from bulk RNA-seq based language models
       (Gelard et al., PMLR 259, 2025)
Paper link: https://proceedings.mlr.press/v259/gelard25a.html
Description: PyTorch clean-room port of the BulkRNABert MLM pre-training
    backbone (discrete / continuous expression modes) + the lightweight
    head-only classifier used for the downstream task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# The reference pre-training pipeline uses ``max(log10(TPM + 1))`` over its
# training CSV as the upper bound of the value range before binning. For the
# TCGA pre-training corpus that value is bit-exactly the constant below
# (float64). It is therefore a derivable statistic of the corpus, not a hand-
# tuned hyperparameter — see :func:`compute_normalization_factor`.
DEFAULT_NORMALIZATION_FACTOR: float = 5.547176906585117


def bin_expression_values(
    values: np.ndarray | torch.Tensor,
    n_bins: int = 64,
    normalization_factor: float = DEFAULT_NORMALIZATION_FACTOR,
    already_log_normalized: bool = True,
) -> torch.Tensor:
    """Discretize gene expression values into bin IDs for the discrete MLM mode.

    Mirrors the tokenizer used by the reference BulkRNABert pipeline:

    1. Optionally apply ``log10(x + 1)`` normalization.
    2. Divide by ``normalization_factor`` to bring values roughly into ``[0, 1]``.
    3. Use ``n_bins`` equally spaced thresholds on ``[0, 1]`` and assign each
       value to its bin index via :func:`numpy.digitize`.
    4. Values that were exactly zero in the original (pre-log) space are forced
       into bin 0 to preserve the "not expressed" state even when the log/norm
       step introduces tiny non-zero artifacts.

    Args:
        values: Array/tensor of shape ``(..., n_genes)`` containing gene
            expression values. If ``already_log_normalized`` is ``True`` the
            values are assumed to be ``log10(TPM + 1)``; otherwise raw TPM is
            expected and the log transform is applied here.
        n_bins: Number of discrete bins.
        normalization_factor: Scalar used as the upper bound of the value
            range before binning. The default
            ``DEFAULT_NORMALIZATION_FACTOR = 5.547176906585117`` equals
            ``max(log10(TPM + 1))`` over the TCGA pre-training corpus; see
            :func:`compute_normalization_factor`.
        already_log_normalized: See ``values``.

    Returns:
        A ``torch.LongTensor`` with the same leading shape as ``values`` and
        integer bin IDs in ``[0, n_bins)``.
    """
    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
    else:
        arr = np.asarray(values)
    arr = arr.astype(np.float64)

    if not already_log_normalized:
        zero_mask = arr <= 0.0
        arr = np.log10(arr + 1.0)
    else:
        zero_mask = arr <= 0.0

    scaled = arr / float(normalization_factor)
    # Token 0 is reserved for "zero expression", bins 1..n_bins-1 cover
    # non-zero values, and token n_bins is saturation. np.digitize over
    # np.linspace(0, 1, n_bins) returns values in [0, n_bins]; we then force
    # strict-zero positions to 0 and clip to [0, n_bins - 1] to stay inside
    # the embedding table.
    breakpoints = np.linspace(0.0, 1.0, n_bins)
    bin_ids = np.digitize(scaled, breakpoints)
    bin_ids[zero_mask] = 0
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    return torch.from_numpy(bin_ids).long()


def load_expression_csv(
    path: Union[str, Path],
    *,
    mode: str = "continuous",
    n_bins: int = 64,
    normalization_factor: float = DEFAULT_NORMALIZATION_FACTOR,
    drop_columns: Sequence[str] = ("identifier", "cohort"),
    already_log_normalized: bool = False,
) -> Tuple[torch.Tensor, List[str]]:
    """Load a preprocessed expression CSV into a model-ready tensor.

    The CSV is expected to have one row per sample and one column per gene,
    matching the output of the TCGA preprocessing pipeline in the reference
    repository (columns ``ENSG00000000003, ENSG00000000005, ..., identifier``).
    Non-numeric metadata columns listed in ``drop_columns`` are removed before
    conversion; the remaining columns define the gene order used by the model.

    Args:
        path: Path to the CSV file.
        mode: ``"continuous"`` to return ``log10(TPM + 1)`` floats suitable for
            ``BulkRNABert`` with ``expression_mode="continuous"``; ``"discrete"``
            to return bin-ID tokens for ``expression_mode="discrete"``.
        n_bins: Number of bins used in discrete mode.
        normalization_factor: Upper bound of the log10 range for binning.
            The default ``DEFAULT_NORMALIZATION_FACTOR = 5.547176906585117``
            equals ``max(log10(TPM + 1))`` over the TCGA pre-training corpus;
            see :func:`compute_normalization_factor`.
        drop_columns: Metadata columns to drop if present. Comparison is exact
            on column name.
        already_log_normalized: If ``True``, the CSV values are treated as
            ``log10(TPM + 1)`` already; otherwise raw TPM is assumed and the
            log transform is applied here.

    Returns:
        ``(tensor, gene_names)`` where ``tensor`` has shape ``(N, n_genes)``
        (``float`` for continuous, ``long`` for discrete) and ``gene_names``
        is the ordered list of surviving column names.
    """
    if mode not in {"discrete", "continuous"}:
        raise ValueError(
            f"mode must be 'discrete' or 'continuous', got {mode!r}"
        )

    df = pd.read_csv(path)
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "Non-numeric columns remain after dropping metadata: "
            f"{non_numeric}. Pass them via drop_columns."
        )

    gene_names = list(df.columns)
    values = df.to_numpy(dtype=np.float64)

    if mode == "continuous":
        if not already_log_normalized:
            values = np.log10(values + 1.0)
        values = values / float(normalization_factor)
        return torch.from_numpy(values).float(), gene_names

    tokens = bin_expression_values(
        values,
        n_bins=n_bins,
        normalization_factor=normalization_factor,
        already_log_normalized=already_log_normalized,
    )
    return tokens, gene_names


def compute_normalization_factor(
    csv_path: Union[str, Path],
    *,
    drop_columns: Sequence[str] = ("identifier", "cohort"),
    already_log_normalized: bool = False,
) -> float:
    """Compute ``max(log10(TPM + 1))`` over a bulk RNA-seq expression CSV.

    This is the value the BulkRNABert pre-training pipeline uses as
    ``normalization_factor`` to rescale ``log10(TPM + 1)`` into ``[0, 1]``.
    It is a **derivable statistic of the training corpus**, not a hand-tuned
    hyperparameter: applying this function to the TCGA pre-training corpus
    reproduces the :data:`DEFAULT_NORMALIZATION_FACTOR` constant
    (``5.547176906585117``, bit-exact in float64) used by
    :func:`bin_expression_values`, :func:`load_expression_csv`, and
    :class:`BulkRNABertConfig`.

    Typical usage:

    * **Pre-training on your own corpus**: call this once, pass the result to
      ``BulkRNABertConfig(normalization_factor=...)``, and persist it to the
      checkpoint's ``config.json``.
    * **Inference / downstream on an existing checkpoint**: **do not call
      this here.** Read ``normalization_factor`` from the loaded checkpoint
      config — re-computing it over a smaller inference-time CSV would
      produce a different value and break the input scale the encoder was
      trained on.

    This helper is intentionally *not* wired into :func:`load_expression_csv`
    or the pre-training CLI; it is provided so users can independently verify
    the hardcoded default against their own copy of the corpus.

    Args:
        csv_path: Path to the expression CSV (TPM values; one sample per row,
            one gene per column, plus non-expression columns like
            ``identifier``).
        drop_columns: Metadata columns to drop if present, matching the
            convention used by :func:`load_expression_csv`.
        already_log_normalized: If ``True``, the CSV values are treated as
            ``log10(TPM + 1)`` already; otherwise raw TPM is assumed and the
            log transform is applied here.

    Returns:
        ``max(log10(TPM + 1))`` as a Python ``float`` (float64 precision).
    """
    df = pd.read_csv(csv_path)
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "Non-numeric columns remain after dropping metadata: "
            f"{non_numeric}. Pass them via drop_columns."
        )

    values = df.to_numpy(dtype=np.float64)
    if not already_log_normalized:
        values = np.log10(values + 1.0)
    return float(values.max())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BulkRNABertConfig:
    """Configuration for :class:`BulkRNABert`.

    Defaults follow the reference paper (19,062 genes, 64 bins, 256-dim,
    4 layers, 8 heads, FFN=512). Override ``n_genes`` and dimensions for
    unit tests or smaller experiments.
    """

    n_genes: int = 19_062
    n_bins: int = 64
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_embed_dim: int = 512

    expression_mode: str = "discrete"  # "discrete" or "continuous"
    continuous_hidden_dim: Optional[int] = None

    use_gene_embedding: bool = True
    init_gene_embed_dim: int = 200  # gene2vec dim in the reference paper

    mlm_probability: float = 0.15
    mask_replace_prob: float = 0.8  # within masked: fraction -> <mask>
    mask_random_prob: float = 0.1  # within masked: fraction -> random token
    # remaining 1 - mask_replace_prob - mask_random_prob -> unchanged
    continuous_noise_scale: float = 0.1

    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    ffn_dropout: float = 0.0

    # If set to "bfloat16" or "float16", forward is wrapped in a
    # torch.autocast context when the model is on CUDA. Enables FlashAttention
    # via SDPA for the attention layers. Pure CPU runs ignore this setting.
    autocast_dtype: Optional[str] = None

    def __post_init__(self):
        if self.expression_mode not in {"discrete", "continuous"}:
            raise ValueError(
                f"expression_mode must be 'discrete' or 'continuous', "
                f"got {self.expression_mode!r}"
            )
        if self.autocast_dtype not in (None, "bfloat16", "float16"):
            raise ValueError(
                f"autocast_dtype must be None, 'bfloat16', or 'float16', "
                f"got {self.autocast_dtype!r}"
            )
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if not 0.0 < self.mlm_probability < 1.0:
            raise ValueError("mlm_probability must be in (0, 1)")
        if self.mask_replace_prob + self.mask_random_prob > 1.0:
            raise ValueError(
                "mask_replace_prob + mask_random_prob must be <= 1"
            )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class GeneEmbedding(nn.Module):
    """Learned per-gene positional embedding, optionally projected to ``embed_dim``."""

    def __init__(
        self,
        n_genes: int,
        init_dim: int,
        embed_dim: int,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.embed = nn.Embedding(n_genes, init_dim)
        self.proj: Optional[nn.Linear]
        if init_dim != embed_dim:
            self.proj = nn.Linear(init_dim, embed_dim)
        else:
            self.proj = None
        self.register_buffer(
            "gene_ids", torch.arange(n_genes, dtype=torch.long), persistent=False
        )

    def forward(self) -> torch.Tensor:
        x = self.embed(self.gene_ids)  # (n_genes, init_dim)
        if self.proj is not None:
            x = self.proj(x)
        return x  # (n_genes, embed_dim)


class DiscreteExpressionEmbedding(nn.Module):
    """Embedding table for binned expression tokens.

    Vocabulary layout: ``[0, n_bins)`` standard bins, ``n_bins`` = ``<mask>``.
    """

    def __init__(self, n_bins: int, embed_dim: int):
        super().__init__()
        self.n_bins = n_bins
        self.mask_token_id = n_bins
        self.embed = nn.Embedding(n_bins + 1, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens)


class ContinuousExpressionEmbedding(nn.Module):
    """Projects scalar expression values to ``embed_dim`` vectors.

    A learned ``mask_embedding`` replaces the projected value at positions
    flagged by ``model_mask``.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:
            self.proj = nn.Linear(1, embed_dim)
            nn.init.zeros_(self.proj.bias)
        self.mask_embedding = nn.Parameter(torch.empty(embed_dim))
        nn.init.trunc_normal_(self.mask_embedding, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        model_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.proj(values.unsqueeze(-1))  # (B, L, embed_dim)
        mask_expanded = model_mask.unsqueeze(-1)
        x = torch.where(mask_expanded, self.mask_embedding, x)
        return x


def _he_uniform_(tensor: torch.Tensor) -> None:
    """He uniform: bound = sqrt(6 / fan_in)."""
    fan_in = tensor.shape[-1] if tensor.ndim >= 2 else tensor.shape[0]
    bound = math.sqrt(6.0 / float(fan_in))
    nn.init.uniform_(tensor, -bound, bound)


class MultiHeadSelfAttention(nn.Module):
    """Self-attention with separate Q/K/V/O linears, He-uniform init.

    Uses ``F.scaled_dot_product_attention`` so Flash/mem-efficient kernels
    remain available; the difference vs ``nn.MultiheadAttention`` is only
    in parameterization and initialization.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attention_dropout = attention_dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        for lin in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            _he_uniform_(lin.weight)
            _he_uniform_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        return self.o_proj(attn)


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block.

    Attention Q/K/V/O use He-uniform init; FFN fc1/fc2 use
    TruncatedNormal(stddev=1/sqrt(fan_in)) with zero bias.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_embed_dim: int,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.ln_attn = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
        )
        self.ln_ffn = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.ffn_dropout = nn.Dropout(ffn_dropout)

        for lin in (self.fc1, self.fc2):
            stddev = 1.0 / math.sqrt(float(lin.weight.shape[1]))
            nn.init.trunc_normal_(lin.weight, std=stddev, a=-2.0 * stddev, b=2.0 * stddev)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_attn(x)
        x = x + self.attn(h)
        h = self.ln_ffn(x)
        h = self.fc2(self.act(self.fc1(h)))
        h = self.ffn_dropout(h)
        x = x + h
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class BulkRNABert(BaseModel):
    """Transformer encoder with masked language modeling for bulk RNA-seq.

    Args:
        dataset: A PyHealth :class:`~pyhealth.datasets.SampleDataset`. May be
            ``None`` when the model is used standalone (e.g. for unit tests or
            outside of the standard Trainer pipeline); in that case
            ``feature_key`` must be provided. When ``dataset`` is given, the
            dataset is expected to expose a single input key (the expression
            vector, same key used as the reconstruction target per §5.1 of the
            pre-training plan).
        config: A :class:`BulkRNABertConfig` instance. Defaults to the paper
            configuration for 19,062 genes.
        feature_key: Name of the expression feature in the batch dict. Falls
            back to ``dataset.input_schema`` when ``dataset`` is provided.
        label_key: Name of the target key. Defaults to ``feature_key`` — the
            MLM target is the input itself, so passing the feature tensor as
            both input and label in the Sample dict is the standard usage.

    Input shape per sample:
        * discrete mode: LongTensor ``(n_genes,)`` with bin IDs in
          ``[0, n_bins)``.
        * continuous mode: FloatTensor ``(n_genes,)`` with ``log10(TPM + 1)``
          values.

    Forward returns a dict with ``loss``, ``y_prob`` and ``y_true`` keys, where
    ``y_prob`` and ``y_true`` are gathered at masked positions so that
    PyHealth's accuracy / MSE metrics apply directly.
    """

    def __init__(
        self,
        dataset=None,
        config: Optional[BulkRNABertConfig] = None,
        feature_key: Optional[str] = None,
        label_key: Optional[str] = None,
    ):
        super().__init__(dataset=dataset)
        self.config = config or BulkRNABertConfig()

        # Resolve feature / label keys. When a dataset is provided, we defer to
        # BaseModel's inference; otherwise, the caller must supply keys.
        if dataset is not None:
            if not self.feature_keys:
                raise ValueError(
                    "dataset provided but input_schema is empty; cannot infer "
                    "feature_key"
                )
            self.feature_key = feature_key or self.feature_keys[0]
            self.label_key = label_key or self.feature_key
            if not self.label_keys:
                self.label_keys = [self.label_key]
        else:
            if feature_key is None:
                raise ValueError(
                    "feature_key must be provided when dataset is None"
                )
            self.feature_key = feature_key
            self.label_key = label_key or feature_key
            self.feature_keys = [self.feature_key]
            self.label_keys = [self.label_key]

        # Trainer's .evaluate() reads self.mode to select the metric family.
        self.mode = (
            "multiclass" if self.config.expression_mode == "discrete" else "regression"
        )

        # Build layers.
        cfg = self.config
        if cfg.expression_mode == "discrete":
            self.expression_embedding = DiscreteExpressionEmbedding(
                n_bins=cfg.n_bins, embed_dim=cfg.embed_dim
            )
            self.lm_head = nn.Linear(cfg.embed_dim, cfg.n_bins)
        else:
            self.expression_embedding = ContinuousExpressionEmbedding(
                embed_dim=cfg.embed_dim,
                hidden_dim=cfg.continuous_hidden_dim,
            )
            self.lm_head = nn.Sequential(
                nn.Linear(cfg.embed_dim, cfg.embed_dim),
                nn.GELU(),
                nn.Linear(cfg.embed_dim, 1),
            )
            for lin in (self.lm_head[0], self.lm_head[2]):
                w_bound = math.sqrt(6.0 / lin.weight.shape[1])
                nn.init.uniform_(lin.weight, -w_bound, w_bound)
                b_bound = math.sqrt(6.0 / lin.bias.shape[0])
                nn.init.uniform_(lin.bias, -b_bound, b_bound)

        if cfg.use_gene_embedding:
            self.gene_embedding = GeneEmbedding(
                n_genes=cfg.n_genes,
                init_dim=cfg.init_gene_embed_dim,
                embed_dim=cfg.embed_dim,
            )
        else:
            self.gene_embedding = None

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    ffn_embed_dim=cfg.ffn_embed_dim,
                    attention_dropout=cfg.attention_dropout,
                    ffn_dropout=cfg.ffn_dropout,
                    layer_norm_eps=cfg.layer_norm_eps,
                )
                for _ in range(cfg.num_layers)
            ]
        )

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def _sample_mask_positions(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        return torch.rand(shape, device=device) < self.config.mlm_probability

    def _apply_mask_discrete(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(masked_tokens, mask_positions)`` using the 80/10/10 rule."""
        cfg = self.config
        mask_positions = self._sample_mask_positions(tokens.shape, tokens.device)
        masked_tokens = tokens.clone()

        action = torch.rand(tokens.shape, device=tokens.device)
        replace_with_mask = mask_positions & (action < cfg.mask_replace_prob)
        replace_with_random = (
            mask_positions
            & (action >= cfg.mask_replace_prob)
            & (action < cfg.mask_replace_prob + cfg.mask_random_prob)
        )
        # remaining masked positions are left unchanged.

        masked_tokens[replace_with_mask] = self.expression_embedding.mask_token_id
        if replace_with_random.any():
            random_tokens = torch.randint(
                low=0,
                high=cfg.n_bins,
                size=(int(replace_with_random.sum().item()),),
                device=tokens.device,
                dtype=tokens.dtype,
            )
            masked_tokens[replace_with_random] = random_tokens
        return masked_tokens, mask_positions

    def _apply_mask_continuous(
        self, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(corrupted_values, model_mask, mask_positions)``.

        ``model_mask`` flags the 80% of masked positions where the learned
        mask embedding should be used instead of the projected value.
        ``mask_positions`` flags all 15% selected positions (used for loss).
        """
        cfg = self.config
        mask_positions = self._sample_mask_positions(values.shape, values.device)
        corrupted = values.clone()

        action = torch.rand(values.shape, device=values.device)
        replace_with_mask = mask_positions & (action < cfg.mask_replace_prob)
        replace_with_noise = (
            mask_positions
            & (action >= cfg.mask_replace_prob)
            & (action < cfg.mask_replace_prob + cfg.mask_random_prob)
        )

        corrupted[replace_with_mask] = 0.0
        if replace_with_noise.any():
            noise = torch.randn_like(corrupted) * cfg.continuous_noise_scale
            corrupted = torch.where(replace_with_noise, corrupted + noise, corrupted)
        return corrupted, replace_with_mask, mask_positions

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.gene_embedding is not None:
            gene_embed = self.gene_embedding()  # (L, embed_dim)
            x = x + gene_embed.unsqueeze(0)
        for block in self.encoder_blocks:
            x = block(x)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        inputs = kwargs[self.feature_key]
        if self._autocast_enabled():
            dtype = getattr(torch, self.config.autocast_dtype)
            with torch.autocast(device_type="cuda", dtype=dtype):
                if self.config.expression_mode == "discrete":
                    return self._forward_discrete(inputs)
                return self._forward_continuous(inputs)
        if self.config.expression_mode == "discrete":
            return self._forward_discrete(inputs)
        return self._forward_continuous(inputs)

    def _autocast_enabled(self) -> bool:
        return (
            self.config.autocast_dtype is not None
            and self.device.type == "cuda"
        )

    def _forward_discrete(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.config
        tokens = tokens.to(self.device).long()
        if tokens.dim() != 2:
            raise ValueError(
                f"discrete mode expects tokens of shape (batch, n_genes), got "
                f"{tuple(tokens.shape)}"
            )
        batch_size, seq_len = tokens.shape
        if seq_len != cfg.n_genes:
            raise ValueError(
                f"sequence length {seq_len} does not match config.n_genes "
                f"{cfg.n_genes}"
            )

        masked_tokens, mask_positions = self._apply_mask_discrete(tokens)

        x = self.expression_embedding(masked_tokens)  # (B, L, E)
        x = self._encode(x)
        logits = self.lm_head(x)  # (B, L, n_bins)

        labels = tokens
        flat_logits = logits.reshape(-1, cfg.n_bins)
        flat_labels = labels.reshape(-1).clone()
        flat_mask = mask_positions.reshape(-1)
        flat_labels[~flat_mask] = -100

        if flat_mask.any():
            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
            y_prob = F.softmax(flat_logits[flat_mask], dim=-1)
            y_true = labels.reshape(-1)[flat_mask]
        else:
            loss = logits.new_zeros(())
            y_prob = F.softmax(flat_logits[:0], dim=-1)
            y_true = labels.reshape(-1)[:0]

        return {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logits": logits}

    def _forward_continuous(
        self, values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        cfg = self.config
        values = values.to(self.device).float()
        if values.dim() != 2:
            raise ValueError(
                f"continuous mode expects values of shape (batch, n_genes), "
                f"got {tuple(values.shape)}"
            )
        batch_size, seq_len = values.shape
        if seq_len != cfg.n_genes:
            raise ValueError(
                f"sequence length {seq_len} does not match config.n_genes "
                f"{cfg.n_genes}"
            )

        corrupted, model_mask, mask_positions = self._apply_mask_continuous(
            values
        )

        x = self.expression_embedding(corrupted, model_mask)  # (B, L, E)
        x = self._encode(x)
        predictions = self.lm_head(x).squeeze(-1)  # (B, L)

        flat_pred = predictions.reshape(-1)
        flat_target = values.reshape(-1)
        flat_mask = mask_positions.reshape(-1)

        if flat_mask.any():
            loss = F.mse_loss(flat_pred[flat_mask], flat_target[flat_mask])
            y_prob = flat_pred[flat_mask]
            y_true = flat_target[flat_mask]
        else:
            loss = predictions.new_zeros(())
            y_prob = flat_pred[:0]
            y_true = flat_target[:0]

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "predictions": predictions,
        }

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, expression: torch.Tensor) -> torch.Tensor:
        """Return last-layer encoder output, mean-pooled over genes.

        Runs a no-mask forward pass through the embedding + Transformer stack
        and averages the resulting ``(B, n_genes, embed_dim)`` tensor over the
        gene axis. Used to pre-compute per-sample embeddings for downstream
        tasks (e.g. cancer classification) that train only a classifier head
        on top of frozen BulkRNABert representations.

        Args:
            expression: Expression tensor of shape ``(B, n_genes)``. For
                ``discrete`` mode this must contain bin IDs in ``[0, n_bins)``;
                for ``continuous`` mode it must contain ``log10(TPM + 1)``
                values.

        Returns:
            A tensor of shape ``(B, embed_dim)`` on the model's device.
        """
        cfg = self.config
        expression = expression.to(self.device)
        was_training = self.training
        self.eval()
        try:
            if cfg.expression_mode == "discrete":
                expression = expression.long()
                if expression.dim() != 2 or expression.shape[1] != cfg.n_genes:
                    raise ValueError(
                        f"discrete encode expects shape (B, {cfg.n_genes}), got "
                        f"{tuple(expression.shape)}"
                    )
                x = self.expression_embedding(expression)
            else:
                expression = expression.float()
                if expression.dim() != 2 or expression.shape[1] != cfg.n_genes:
                    raise ValueError(
                        f"continuous encode expects shape (B, {cfg.n_genes}), got "
                        f"{tuple(expression.shape)}"
                    )
                model_mask = torch.zeros_like(expression, dtype=torch.bool)
                x = self.expression_embedding(expression, model_mask)
            x = self._encode(x)  # (B, L, E)
            return x.mean(dim=1)  # (B, E)
        finally:
            self.train(was_training)


# ---------------------------------------------------------------------------
# Downstream: cancer-type classification head
# ---------------------------------------------------------------------------


class BulkRNABertClassifier(BaseModel):
    """MLP classifier head trained on pre-computed BulkRNABert embeddings.

    This model implements the "pattern 2" downstream workflow: per-sample
    embeddings produced by :meth:`BulkRNABert.encode` are saved to disk once,
    and this lightweight head is trained on top of them. The encoder is not
    invoked during training — the input feature is already a fixed
    ``(embed_dim,)`` vector per sample.

    Architecture matches the reference ``RNASeqSurvivalMLP`` used in the
    BulkRNABert paper's downstream experiments: a stack of Linear + SELU
    hidden layers followed by a final Linear projection to ``num_classes``.
    Dropout and layer norm are disabled by default to follow the reference
    checkpoint configuration.

    Args:
        dataset: A PyHealth :class:`~pyhealth.datasets.SampleDataset` whose
            ``input_schema`` exposes a single float-array feature holding the
            pre-computed embedding and whose ``output_schema`` exposes a
            single multiclass label.
        hidden_sizes: Sizes of the hidden Linear layers. Defaults to
            ``(256, 128)`` to match the reference head.
        embed_dim: Dimensionality of the input embedding. Defaults to
            ``256`` (the BulkRNABert encoder output size).
        num_classes: Number of output classes. When ``None`` (default) the
            size is inferred from the dataset's label processor.
        dropout: Dropout probability applied after each hidden activation.
            Defaults to ``0.0`` (disabled) per the reference checkpoint.
        layer_norm: If ``True``, apply :class:`~torch.nn.LayerNorm` before
            the first hidden layer. Defaults to ``False``.
        feature_key: Name of the embedding feature in the batch dict. When
            ``None``, inferred from ``dataset.input_schema``.
        label_key: Name of the label key in the batch dict. When ``None``,
            inferred from ``dataset.output_schema``.

    Forward returns a dict with ``loss``, ``y_prob``, ``y_true`` and
    ``logit`` keys, matching the PyHealth :class:`~pyhealth.trainer.Trainer`
    contract.
    """

    def __init__(
        self,
        dataset,
        hidden_sizes: Sequence[int] = (256, 128),
        embed_dim: int = 256,
        num_classes: Optional[int] = None,
        dropout: float = 0.0,
        layer_norm: bool = False,
        feature_key: Optional[str] = None,
        label_key: Optional[str] = None,
    ):
        super().__init__(dataset=dataset)
        if not self.feature_keys:
            raise ValueError(
                "dataset.input_schema is empty; cannot infer feature_key"
            )
        if not self.label_keys:
            raise ValueError(
                "dataset.output_schema is empty; cannot infer label_key"
            )
        self.feature_key = feature_key or self.feature_keys[0]
        self.label_key = label_key or self.label_keys[0]
        self.embed_dim = embed_dim
        self.mode = "multiclass"
        if num_classes is None:
            num_classes = self.get_output_size()
        self.num_classes = num_classes

        layers: List[nn.Module] = []
        if layer_norm:
            layers.append(nn.LayerNorm(embed_dim))
        in_dim = embed_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        x = kwargs[self.feature_key].to(self.device).float()
        if x.dim() != 2 or x.shape[1] != self.embed_dim:
            raise ValueError(
                f"expected input of shape (B, {self.embed_dim}), got "
                f"{tuple(x.shape)}"
            )
        h = self.backbone(x)
        logits = self.classifier(h)

        y_true = kwargs[self.label_key].to(self.device).long()
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        return {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}


__all__ = [
    "BulkRNABert",
    "BulkRNABertClassifier",
    "BulkRNABertConfig",
    "bin_expression_values",
]
