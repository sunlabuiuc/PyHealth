"""ECG-JEPA encoder with a classification head (Kim, 2024).

ECG-JEPA ("Learning General Representation of 12-Lead Electrocardiogram With a
Joint-Embedding Predictive Architecture", https://arxiv.org/abs/2410.08559)
is a ViT over 8 leads x 50 patches of 50 samples (10 s at 250 Hz), with
cross-pattern attention (CroPA): a patch attends only to patches of its own
lead and to the patches of every lead at the same time step. CLIMB (Dai et
al., ICML 2025) fine-tunes it as its ECG encoder.

The encoder is ported from the authors' code
(https://github.com/sehunfromdaegu/ECG_JEPA, MIT licence, copyright (c) 2024
Sehun Kim), keeping its parameter names so that the published checkpoints
(``ckpt["encoder"]``) load as they are. Only the encoder's representation
path is ported; pretraining (masking and the predictor) is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import SampleDataset
from .base_model import BaseModel


def sincos_pos_embed(dim: int, rows: int, cols: int) -> torch.Tensor:
    """2-D sine-cosine position embedding of a ``rows x cols`` grid.

    Half of the dimensions encode the column and half the row, as in MAE and
    ECG-JEPA; row ``r`` (lead) and column ``c`` (time patch) is position
    ``r * cols + c``.

    Args:
        dim: Embedding size, a multiple of 4.
        rows: Number of leads.
        cols: Number of patches per lead.

    Returns:
        A ``[rows * cols, dim]`` float tensor.

    Examples:
        >>> sincos_pos_embed(8, 2, 3).shape
        torch.Size([6, 8])
    """
    if dim % 4:
        raise ValueError("dim must be a multiple of 4.")

    def one_axis(pos: np.ndarray) -> np.ndarray:
        omega = 1.0 / 10000 ** (np.arange(dim // 4, dtype=float) / (dim / 4))
        out = np.outer(pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    col, row = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
    return torch.from_numpy(np.concatenate([one_axis(col), one_axis(row)], axis=1)).float()


def cross_pattern_mask(leads: int, patches: int) -> torch.Tensor:
    """CroPA: which of the ``leads * patches`` tokens each token attends to.

    A token attends to the tokens of its own lead and to those at its own
    time patch on every lead.

    Args:
        leads: Number of leads.
        patches: Number of patches per lead.

    Returns:
        A ``[n, n]`` boolean tensor, ``n = leads * patches``; ``True`` where
        attention is allowed.

    Examples:
        >>> cross_pattern_mask(2, 2).int()
        tensor([[1, 1, 1, 0],
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [0, 1, 1, 1]], dtype=torch.int32)
    """
    lead = torch.arange(leads).repeat_interleave(patches)
    patch = torch.arange(patches).repeat(leads)
    return (lead[:, None] == lead[None, :]) | (patch[:, None] == patch[None, :])


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.num_heads, d // self.num_heads).permute(2, 0, 3, 1, 4)
        drop = self.dropout if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        return F.dropout(self.proj(x.transpose(1, 2).reshape(b, n, d)), drop)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


class _Blocks(nn.Module):
    """Holds the blocks under ``blocks``, the checkpoints' key layout."""

    def __init__(self, blocks: list[_Block]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)


class ECGJEPAEncoder(nn.Module):
    """The ECG-JEPA encoder: ``[B, leads, 2500]`` signal to ``[B, dim]``.

    Each lead is cut into patches of ``patch_size`` samples, linearly
    embedded, and passed through ``depth`` pre-norm transformer blocks with
    the CroPA mask (:func:`cross_pattern_mask`); the fixed sine-cosine
    position embedding is added before every block, as in the original. The
    output is the mean of the final, layer-normed tokens.

    Args:
        num_leads: Leads the position grid has (8 in the checkpoints).
        num_patches: Patches per lead (50).
        patch_size: Samples per patch (50).
        dim: Token size (768).
        depth: Number of blocks (12).
        num_heads: Attention heads (16).
        mlp_ratio: Hidden size of the MLPs relative to ``dim`` (4).
        dropout: Dropout in attention and MLPs (0, as in the original).

    Examples:
        >>> import torch
        >>> encoder = ECGJEPAEncoder(num_leads=2, num_patches=4, patch_size=5,
        ...                          dim=16, depth=1, num_heads=2)
        >>> encoder(torch.randn(3, 2, 20)).shape
        torch.Size([3, 16])
    """

    def __init__(
        self,
        num_leads: int = 8,
        num_patches: int = 50,
        patch_size: int = 50,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_leads, self.num_patches, self.patch_size = num_leads, num_patches, patch_size
        self.W_P = nn.Linear(patch_size, dim)
        self.pos_embed = nn.Parameter(
            sincos_pos_embed(dim, num_leads, num_patches), requires_grad=False
        )
        self.encoder_blocks = _Blocks(
            [_Block(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.register_buffer(
            "attention_mask", cross_pattern_mask(num_leads, num_patches), persistent=False
        )
        self.dim = dim

    def forward(self, x: torch.Tensor, leads: Sequence[int] | None = None) -> torch.Tensor:
        """Encodes ``x`` (``[B, len(leads), num_patches * patch_size]``).

        Args:
            x: The signals.
            leads: Which rows of the position grid the leads of ``x`` are
                (all, in order, by default).

        Returns:
            The ``[B, dim]`` representation.
        """
        b, n_leads, length = x.shape
        if length != self.num_patches * self.patch_size:
            raise ValueError(
                f"Expected {self.num_patches * self.patch_size} samples per lead, got {length}."
            )
        index = torch.arange(self.num_leads, device=x.device) if leads is None else torch.as_tensor(
            list(leads), device=x.device
        )
        if len(index) != n_leads:
            raise ValueError(f"x has {n_leads} leads but {len(index)} were named.")
        if index.min() < 0 or index.max() >= self.num_leads:
            raise ValueError(f"leads must be rows 0..{self.num_leads - 1} of the grid, got {list(leads)}.")
        tokens = (index[:, None] * self.num_patches + torch.arange(self.num_patches, device=x.device)).reshape(-1)
        pos = self.pos_embed[tokens]
        mask = self.attention_mask[tokens][:, tokens]
        h = self.W_P(x.reshape(b, n_leads * self.num_patches, self.patch_size))
        for block in self.encoder_blocks.blocks:
            h = block(h + pos, mask)
        return self.norm(h).mean(dim=1)


class ECGJEPA(BaseModel):
    """ECG-JEPA with a linear classification head.

    Reads one signal field of shape ``[B, leads, 2500]``, such as
    :class:`~pyhealth.tasks.CLIMBECGClassification` produces with its default
    eight leads (I, II, V1-V6, the order of the checkpoints' position grid),
    and predicts the task's labels with a linear layer on the mean token,
    as the authors' fine-tuning does.

    Args:
        dataset: The samples; gives the signal field and the label type.
        checkpoint: Path of a published ECG-JEPA checkpoint (a dict whose
            ``"encoder"`` entry is the encoder's state dict), or ``None`` for
            a randomly initialised encoder.
        freeze_encoder: Train only the head (linear probing).
        num_leads: Rows of the position grid (8 in the checkpoints).
        lead_indices: Rows of the position grid the signal's leads are;
            defaults to ``0..leads-1``.
        dim: Token size (768 in the checkpoints).
        depth: Number of blocks (12).
        num_heads: Attention heads (16).
        num_patches: Patches per lead (50).
        patch_size: Samples per patch (50).
        dropout: Dropout inside the encoder.
        feature_key: The signal field; defaults to the only input field.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": str(i), "signal": torch.randn(2, 20), "label": ["a"] if i % 2 else ["b"]}
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(samples, {"signal": "tensor"}, {"label": "multilabel"})
        >>> model = ECGJEPA(dataset, num_leads=2, dim=16, depth=1, num_heads=2,
        ...                 num_patches=4, patch_size=5)
        >>> out = model(**next(iter(get_dataloader(dataset, batch_size=4))))
        >>> out["y_prob"].shape
        torch.Size([4, 2])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        checkpoint: str | Path | None = None,
        freeze_encoder: bool = False,
        num_leads: int = 8,
        lead_indices: Sequence[int] | None = None,
        dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        num_patches: int = 50,
        patch_size: int = 50,
        dropout: float = 0.0,
        feature_key: str | None = None,
    ) -> None:
        super().__init__(dataset)
        if feature_key is None:
            if len(self.feature_keys) != 1:
                raise ValueError("Pass feature_key: the dataset has several input fields.")
            feature_key = self.feature_keys[0]
        self.feature_key = feature_key
        self.label_key = self.label_keys[0]
        n_leads = int(torch.as_tensor(dataset[0][feature_key]).shape[0])
        self.lead_indices = list(range(n_leads)) if lead_indices is None else list(lead_indices)
        if (
            len(self.lead_indices) != n_leads
            or min(self.lead_indices) < 0
            or max(self.lead_indices) >= num_leads
        ):
            raise ValueError(
                f"The signal has {n_leads} leads; lead_indices must name {n_leads} "
                f"rows of the {num_leads}-lead grid, got {self.lead_indices}."
            )
        self.encoder = ECGJEPAEncoder(
            num_leads=num_leads,
            num_patches=num_patches,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(state.get("encoder", state))
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)
        self.fc = nn.Linear(dim, self.label_processor.output_size())

    def embed(self, signal: torch.Tensor) -> torch.Tensor:
        """The encoder's ``[B, dim]`` representation of ``signal``."""
        signal = signal.to(self.device).float()
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                return self.encoder(signal, self.lead_indices)
        return self.encoder(signal, self.lead_indices)

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Classifies the batch.

        Returns:
            A dict with ``logit``, ``y_prob`` and the representation
            ``embed``; with labels also ``loss`` and ``y_true``.
        """
        embed = self.embed(kwargs[self.feature_key])
        logit = self.fc(embed)
        out = {"logit": logit, "y_prob": self.label_processor.activation(logit), "embed": embed}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            out["loss"] = self.label_processor.loss(logit, y_true)
            out["y_true"] = y_true
        return out
