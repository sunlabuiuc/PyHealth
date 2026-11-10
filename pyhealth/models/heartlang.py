"""HeartLang: an ECG language model of heartbeat "words" and "sentences".

Jin, Wang, Li, Li, Pan and Hong, "Reading Your Heart: Learning ECG Words and
Sentences via Pre-training ECG Language Model", ICLR 2025
(https://openreview.net/forum?id=6Hz1Ko087B).

An ECG is cut into heartbeats around the QRS complexes detected on lead I.
Each heartbeat of each lead is a "word" (a fixed-length segment), and the
words of all leads form a "sentence" together with a lead index and a
one-second time index per word. A transformer pre-trained on MIMIC-IV-ECG by
masked heartbeat modelling encodes the sentence; downstream tasks use its
[CLS] token.

This module ports the QRS tokenizer and the transformer from the authors'
code (https://github.com/PKUDigitalHealth/HeartLang). The feed-forward
blocks reuse PyHealth's transformer components; the other parameter names
are the original ones, and :func:`heartlang_state_dict` renames the rest, so
the published checkpoint (https://huggingface.co/PKUDigitalHealth/HeartLang)
loads with every weight matched and identical outputs. The VQ-HBR
tokenizer is only needed for pre-training (it produces the masked-modelling
targets) and is not ported.

The original code is released under the MIT License:

    Copyright (c) 2025 PKUDigitalHealth

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions: The above copyright notice and this permission
    notice shall be included in all copies or substantial portions of the
    Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from einops import pack, rearrange, repeat
from torch import nn

from pyhealth.datasets import SampleDataset

from .base_model import BaseModel
from .transformer import PositionwiseFeedForward, SublayerConnection

HF_REPO_ID = "PKUDigitalHealth/HeartLang"
"""Hugging Face repository with the published HeartLang checkpoints."""

HF_CHECKPOINT = "checkpoints/pretrain/MIMIC-IV/checkpoint-200.pth"
"""The pre-trained checkpoint (MIMIC-IV-ECG, 200 epochs) in :data:`HF_REPO_ID`."""

HF_REVISION = "b8118c1bc34a43cc2202b6d30267244ac7fc52c1"
"""The commit of :data:`HF_REPO_ID` that :data:`HF_CHECKPOINT` is read from."""

MAX_INDEX = 15
"""Largest lead or time index the model embeds (its index tables have 16 rows,
row 0 being padding): at most 15 leads, and QRS complexes in the first 15 s."""


# ----------------------------------------------------------------- tokenizer
def _qrs_segments(
    signal: np.ndarray, qrs_inds: Sequence[int], token_len: int
) -> list[list[np.ndarray]]:
    """Cuts every lead into one zero-padded segment per detected QRS complex.

    A segment spans from halfway to the previous QRS to halfway to the next
    one, is capped at ``token_len`` samples around the QRS, and is placed so
    that the QRS sits at the centre of the ``token_len`` window.
    """
    n = signal.shape[1]
    channels = []
    for lead in signal:
        segments = []
        for i, center in enumerate(qrs_inds):
            if i == 0:
                start = max(center - token_len // 2, 0)
                end = (
                    (center + qrs_inds[i + 1]) // 2
                    if i + 1 < len(qrs_inds)
                    else min(start + token_len, n)
                )
            elif i == len(qrs_inds) - 1:
                start = (center + qrs_inds[i - 1]) // 2
                end = min(center + token_len // 2, n)
            else:
                start = (center + qrs_inds[i - 1]) // 2
                end = (center + qrs_inds[i + 1]) // 2
            start, end = max(start, 0), min(end, n)
            if end - start > token_len:
                start = max(center - token_len // 2, 0)
                end = min(start + token_len, n)
            segment = np.zeros(token_len, dtype=np.float32)
            seg_start = max(token_len // 2 - (center - start), 0)
            seg_end = seg_start + (end - start)
            if seg_end > token_len:
                end -= seg_end - token_len
                seg_end = token_len
            segment[seg_start:seg_end] = lead[start:end]
            segments.append(segment)
        channels.append(segments)
    return channels


def ecg_to_heartbeat_sentence(
    signal: np.ndarray,
    sampling_rate: int = 100,
    max_len: int = 256,
    token_len: int = 96,
    leads: Sequence[int] | None = None,
    qrs_inds: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turns an ECG into a HeartLang "sentence" of heartbeat "words".

    QRS complexes are detected on the first selected lead (lead I with the
    default ``leads``) with ``wfdb.processing.xqrs_detect`` (the
    ``pyhealth[ptbxl]`` extra), unless ``qrs_inds`` is given. Every lead is
    cut into one ``token_len`` segment per QRS; the segments of lead 1, then
    lead 2, and so on form the sentence, padded with zeros or truncated to
    ``max_len`` words.

    If no QRS complex is found (for example a flat or very noisy first
    lead), the sentence is all padding and a ``UserWarning`` is issued; the
    model still accepts it, but its prediction carries no ECG information.

    The model's time embedding covers whole seconds 1 to 15
    (:data:`MAX_INDEX`), so a QRS at 15 s or later raises ``ValueError``:
    crop longer recordings (HeartLang was trained on 10 s ECGs).

    Args:
        signal: ECG of shape ``(leads, samples)``, already standardized the
            way the model was trained (HeartLang's PTB-XL protocol uses one
            mean and standard deviation over the training set).
        sampling_rate: Sampling rate in Hz. HeartLang was trained at 100 Hz.
        max_len: Words per sentence (256 for the published model).
        token_len: Samples per word (96 for the published model).
        leads: Indices of the leads to use, in PTB-XL order (I, II, III,
            aVR, aVL, aVF, V1-V6). Defaults to all leads.
        qrs_inds: QRS sample indices, to skip detection.

    Returns:
        ``(words, lead_ids, time_ids)``: float32 ``(max_len, token_len)``,
        and int64 ``(max_len,)`` arrays. ``lead_ids`` is the lead index plus
        one and ``time_ids`` the second of the QRS plus one; both are 0 on
        padding.

    Raises:
        ImportError: If ``qrs_inds`` is not given and ``wfdb`` is missing.
        ValueError: If a lead index is above 14 or a QRS lies at 15 s or
            later, beyond the model's lead and time embeddings.

    Examples:
        >>> import numpy as np
        >>> ecg = np.random.randn(12, 1000).astype("float32")
        >>> words, lead_ids, time_ids = ecg_to_heartbeat_sentence(
        ...     ecg, qrs_inds=[50, 150, 250])
        >>> words.shape, lead_ids[:4].tolist(), time_ids[:4].tolist()
        ((256, 96), [1, 1, 1, 2], [1, 2, 3, 1])
    """
    signal = np.asarray(signal, dtype=np.float32)
    leads = list(range(signal.shape[0])) if leads is None else list(leads)
    if any(not 0 <= lead < MAX_INDEX for lead in leads):
        raise ValueError(f"lead indices must be in [0, {MAX_INDEX - 1}]: {leads}")
    signal = signal[leads]
    if qrs_inds is None:
        try:
            from wfdb import processing
        except ImportError as e:  # pragma: no cover - depends on the extra
            raise ImportError(
                "QRS detection needs wfdb: pip install 'pyhealth[ptbxl]'"
            ) from e
        qrs_inds = processing.xqrs_detect(
            sig=signal[0].astype(np.float64), fs=sampling_rate, verbose=False
        )
    qrs_inds = [int(i) for i in qrs_inds]
    if not qrs_inds:
        warnings.warn(
            "no QRS complex found; the HeartLang sentence is all padding",
            stacklevel=2,
        )
    words, lead_ids, time_ids = [], [], []
    times = [i // sampling_rate + 1 for i in qrs_inds]
    if times and max(times) > MAX_INDEX:
        raise ValueError(
            f"a QRS at {max(qrs_inds) / sampling_rate:.1f} s is beyond the "
            f"{MAX_INDEX} s the HeartLang time embedding covers; crop the "
            "recording (the model was trained on 10 s ECGs)"
        )
    for lead, segments in zip(leads, _qrs_segments(signal, qrs_inds, token_len)):
        words.extend(segments)
        lead_ids.extend([lead + 1] * len(segments))
        time_ids.extend(times)
    words, lead_ids, time_ids = (
        words[:max_len], lead_ids[:max_len], time_ids[:max_len]
    )
    pad = max_len - len(words)
    out = np.zeros((max_len, token_len), dtype=np.float32)
    if words:
        out[: len(words)] = np.stack(words)
    return (
        out,
        np.asarray(lead_ids + [0] * pad, dtype=np.int64),
        np.asarray(time_ids + [0] * pad, dtype=np.int64),
    )


# ----------------------------------------------------------------- backbone
class _FeedForward(nn.Module):
    """HeartLang's pre-norm feed-forward block, residual included.

    ``x + Dropout(Linear(Dropout(GELU(Linear(LayerNorm(x))))))``, which is
    exactly :class:`~pyhealth.models.transformer.SublayerConnection` around
    :class:`~pyhealth.models.transformer.PositionwiseFeedForward` (same
    exact GELU, LayerNorm eps, biases and dropout positions). The original
    code names these weights ``net.0`` (norm), ``net.1`` and ``net.4``;
    :func:`heartlang_state_dict` renames them.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.sublayer = SublayerConnection(dim, dropout)
        self.ff = PositionwiseFeedForward(dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sublayer(x, self.ff)


class _Attention(nn.Module):
    # Kept separate from transformer.MultiHeadedAttention: the published
    # checkpoint stores a fused ``to_qkv`` projection and a pre-norm inside
    # the attention block, which that class doesn't have.
    def __init__(self, embed_dim: int, heads: int, head_dim: int, dropout: float):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim**-0.5
        self.norm = nn.LayerNorm(embed_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)
        project = not (heads == 1 and head_dim == embed_dim)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embed_dim), nn.Dropout(dropout))
            if project else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.heads)
            for t in self.to_qkv(self.norm(x)).chunk(3, dim=-1)
        )
        attn = self.dropout(self.attend(q @ k.transpose(-1, -2) * self.scale))
        return self.to_out(rearrange(attn @ v, "b h n d -> b n (h d)"))


class _Transformer(nn.Module):
    def __init__(self, embed_dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                _Attention(embed_dim, heads, dim_head, dropout),
                _FeedForward(embed_dim, mlp_dim, dropout),
            ])
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x)  # the residual is inside _FeedForward
        return x


class _TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.tokenConv = nn.Conv1d(
            c_in, d_model, kernel_size=3, padding=1, padding_mode="circular",
            bias=False,
        )
        nn.init.kaiming_normal_(
            self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class _IndexEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return x + self.embed(index)


class HeartLangBackbone(nn.Module):
    """The HeartLang sentence encoder (``ST_ECGFormer`` in the original code).

    Each word is embedded with a 1-D convolution over the word axis, plus a
    lead (spatial) and a one-second time (temporal) embedding; a [CLS] token
    and a learned position embedding are added, and the sequence goes
    through a pre-norm transformer. Parameter names match the published
    checkpoint after :func:`heartlang_state_dict`.

    Args:
        seq_len: Words per sentence.
        token_len: Samples per word.
        embed_dim: Transformer width.
        depth: Transformer layers.
        heads: Attention heads.
        dim_head: Width of each head.
        mlp_dim: Hidden width of the feed-forward blocks.
        dropout: Dropout in the transformer.
        emb_dropout: Dropout on the embeddings.

    Examples:
        >>> import torch
        >>> backbone = HeartLangBackbone(seq_len=8, token_len=16, embed_dim=32,
        ...                              depth=1, heads=2, dim_head=8, mlp_dim=64)
        >>> words = torch.randn(2, 8, 16)
        >>> ids = torch.ones(2, 8, dtype=torch.long)
        >>> backbone(words, ids, ids).shape
        torch.Size([2, 9, 32])
    """

    def __init__(
        self,
        seq_len: int = 256,
        token_len: int = 96,
        embed_dim: int = 768,
        depth: int = 12,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embed = _TokenEmbedding(token_len, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))
        self.spa_embed = _IndexEmbedding(MAX_INDEX + 1, embed_dim)
        self.tem_embed = _IndexEmbedding(MAX_INDEX + 1, embed_dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = _Transformer(
            embed_dim, depth, heads, dim_head, mlp_dim, dropout
        )
        self.norm_layer = nn.LayerNorm(embed_dim)

    def forward(
        self,
        words: torch.Tensor,
        lead_ids: torch.Tensor | None = None,
        time_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encodes sentences.

        Args:
            words: ``(batch, seq_len, token_len)`` heartbeat words.
            lead_ids: ``(batch, seq_len)`` lead indices (0 = padding).
            time_ids: ``(batch, seq_len)`` time indices (0 = padding).

        Returns:
            ``(batch, seq_len + 1, embed_dim)``; position 0 is [CLS].
        """
        x = self.token_embed(words)
        if lead_ids is not None:
            x = self.spa_embed(x, lead_ids.long())
        if time_ids is not None:
            x = self.tem_embed(x, time_ids.long())
        cls = repeat(self.cls_token, "d -> b d", b=x.shape[0])
        x, _ = pack([cls, x], "b * d")
        x = self.dropout(x + self.pos_embed)
        return self.norm_layer(self.transformer(x))


# Original feed-forward weight names (``layers.<i>.1.net.<j>.``) to ours.
_FF_NAMES = {"0": "sublayer.norm.", "1": "ff.w_1.", "4": "ff.w_2."}
_FF_RENAMES = re.compile(r"(?<=layers\.)(\d+\.1\.)net\.([014])\.")


def heartlang_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Extracts the backbone weights from a HeartLang checkpoint.

    Accepts a training checkpoint (weights under ``"model"`` or
    ``"module"``) or a plain state dict, strips ``module.`` and
    ``backbone.`` prefixes, drops the pre-training ``lm_head``, and renames
    the feed-forward weights to PyHealth's transformer components
    (``net.0`` to ``sublayer.norm``, ``net.1`` to ``ff.w_1``, ``net.4`` to
    ``ff.w_2``).

    Args:
        checkpoint: The loaded checkpoint.

    Returns:
        A state dict for :class:`HeartLangBackbone`.

    Examples:
        >>> import torch
        >>> ckpt = {"model": {"module.cls_token": torch.zeros(4),
        ...                   "lm_head.weight": torch.zeros(2, 4)}}
        >>> list(heartlang_state_dict(ckpt))
        ['cls_token']
    """
    state = checkpoint
    for key in ("model", "module"):
        if key in checkpoint and isinstance(checkpoint[key], Mapping):
            state = checkpoint[key]
            break
    out = {}
    for name, value in state.items():
        for prefix in ("module.", "backbone."):
            name = name.removeprefix(prefix)
        if name.startswith("lm_head."):
            continue
        name = _FF_RENAMES.sub(lambda m: m.group(1) + _FF_NAMES[m.group(2)], name)
        out[name] = value
    return out



def load_heartlang_checkpoint(path: str | None = None) -> dict[str, torch.Tensor]:
    """Loads the published HeartLang weights as a backbone state dict.

    Args:
        path: A local checkpoint file. Defaults to downloading
            :data:`HF_CHECKPOINT` from :data:`HF_REPO_ID` at the pinned
            commit :data:`HF_REVISION` with ``huggingface_hub`` (about
            535 MB, cached).

    Returns:
        A state dict for :class:`HeartLangBackbone`.

    Examples:
        >>> state = load_heartlang_checkpoint()  # doctest: +SKIP
        >>> state["pos_embed"].shape  # doctest: +SKIP
        torch.Size([1, 257, 768])
    """
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=HF_CHECKPOINT, revision=HF_REVISION
        )
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # The training checkpoint also stores its argparse arguments, which hold
    # numpy scalars; allow exactly those types and keep weights_only loading.
    from numpy._core.multiarray import scalar

    allowed = [
        argparse.Namespace, scalar, (scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        *(t for t in vars(np.dtypes).values() if isinstance(t, type)),
    ]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return heartlang_state_dict(checkpoint)


# --------------------------------------------------------------------- model
class HeartLang(BaseModel):
    """HeartLang classifier: the sentence encoder and a linear head on [CLS].

    Expects the three inputs produced by
    :class:`~pyhealth.tasks.PTBXLHeartLangClassification` (or any task built
    with :func:`ecg_to_heartbeat_sentence`): the heartbeat words and their
    lead and time indices. With ``freeze_backbone=True`` only the head is
    trained, the linear probe of the paper.

    Args:
        dataset: The sample dataset; one label (e.g. multilabel).
        pretrained: ``True`` loads the published checkpoint from Hugging
            Face, a string loads a local checkpoint file, ``False`` starts
            from random weights.
        freeze_backbone: Train only the linear head. The frozen backbone
            also stays in eval mode (no dropout) when ``train()`` is called.
            This is the usual linear-probe setup; the authors' script
            instead keeps the frozen encoder in training mode, with dropout
            on.
        words_key: Input key of the heartbeat words.
        lead_key: Input key of the lead indices.
        time_key: Input key of the time indices.
        seq_len: Words per sentence.
        token_len: Samples per word.
        embed_dim: Transformer width.
        depth: Transformer layers.
        heads: Attention heads.
        dim_head: Width of each head.
        mlp_dim: Hidden width of the feed-forward blocks.
        dropout: Dropout in the transformer.
        emb_dropout: Dropout on the embeddings.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": str(i), "ecg_sentence": np.random.randn(8, 16),
        ...      "lead_ids": np.ones(8), "time_ids": np.ones(8),
        ...      "labels": ["NORM"] if i % 2 else ["MI"]}
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     input_schema={"ecg_sentence": "tensor", "lead_ids": "tensor",
        ...                   "time_ids": "tensor"},
        ...     output_schema={"labels": "multilabel"},
        ... )
        >>> model = HeartLang(dataset, pretrained=False, seq_len=8, token_len=16,
        ...                   embed_dim=32, depth=1, heads=2, dim_head=8,
        ...                   mlp_dim=64)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> model(**batch)["y_prob"].shape
        torch.Size([4, 2])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        pretrained: bool | str = True,
        freeze_backbone: bool = False,
        words_key: str = "ecg_sentence",
        lead_key: str = "lead_ids",
        time_key: str = "time_ids",
        seq_len: int = 256,
        token_len: int = 96,
        embed_dim: int = 768,
        depth: int = 12,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ):
        super().__init__(dataset)
        for key in (words_key, lead_key, time_key):
            if key not in self.feature_keys:
                raise ValueError(
                    f"HeartLang needs input {key!r}; the dataset has "
                    f"{self.feature_keys}"
                )
        if len(self.label_keys) != 1:
            raise ValueError("HeartLang supports exactly one label")
        self.words_key, self.lead_key, self.time_key = words_key, lead_key, time_key
        self.label_key = self.label_keys[0]
        self.backbone = HeartLangBackbone(
            seq_len, token_len, embed_dim, depth, heads, dim_head, mlp_dim,
            dropout, emb_dropout,
        )
        if pretrained:
            state = load_heartlang_checkpoint(
                pretrained if isinstance(pretrained, str) else None
            )
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ValueError(
                    "checkpoint does not match the backbone: missing "
                    f"{missing}, unexpected {unexpected}"
                )
        self.mlp_head = nn.Linear(embed_dim, self.label_processor.output_size())
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> HeartLang:
        """Sets training mode, keeping a frozen backbone in eval mode.

        Args:
            mode: ``True`` for training mode, ``False`` for eval mode.

        Returns:
            The model.

        Examples:
            >>> model = HeartLang(dataset, pretrained=False,  # doctest: +SKIP
            ...                   freeze_backbone=True)
            >>> model.train().backbone.training  # doctest: +SKIP
            False
        """
        super().train(mode)
        if mode and getattr(self, "freeze_backbone", False):
            self.backbone.eval()
        return self

    def embed(self, **kwargs: Any) -> torch.Tensor:
        """The [CLS] embedding of each sample in a batch.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`.

        Returns:
            ``(batch, embed_dim)`` features.
        """
        words = kwargs[self.words_key].to(self.device).float()
        lead_ids = kwargs[self.lead_key].to(self.device).long()
        time_ids = kwargs[self.time_key].to(self.device).long()
        return self.backbone(words, lead_ids, time_ids)[:, 0]

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Predicts the label from the [CLS] token.

        Args:
            **kwargs: A batch with the three inputs and, optionally, the label.

        Returns:
            ``logit``, ``y_prob`` and ``embed``, plus ``loss`` and ``y_true``
            when the label is in the batch.
        """
        features = self.embed(**kwargs)
        logits = self.mlp_head(features)
        result = {
            "logit": logits, "y_prob": self.label_processor.activation(logits), "embed": features,
        }
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            result["loss"] = self.label_processor.loss(logits, y_true)
            result["y_true"] = y_true
        return result
