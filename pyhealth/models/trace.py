"""TRACE: a time-aware Transformer for intra-visit clinical event nowcasting.

Paper:
    Yuyang Liang, Yankai Chen, Yixiang Fang, Laks V. S. Lakshmanan and
    Chenhao Ma. "TRACE: Intra-visit Clinical Event Nowcasting via Effective
    Patient Trajectory Encoding." Companion Proceedings of the ACM Web
    Conference 2025 (WWW '25 Companion), short paper.
    https://doi.org/10.1145/3701716.3715545, https://arxiv.org/abs/2503.23072

This is a reimplementation from the paper. The authors' code
(https://github.com/Amehi/TRACE) has no license, so none of it is copied; it
was read only to confirm details the short paper leaves open (sizes, the
100-event window, the time scales, and where the time embedding and the
denoising mask enter the attention). Where the paper and the code differ,
the default follows the code, which produced the reported numbers, and the
paper's form is an option; see :class:`TRACE`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import SampleDataset
from .base_model import BaseModel


class TRACETimeEmbedding(nn.Module):
    """Timestamp embedding with a decay and a periodic part (Eqs. 2-4).

    ``decay(t) = W_d (1 - tanh((W_t t / s + b_t)^2)) + b_d`` is largest for
    recent events (``t`` near 0) and fades with a learned rate;
    ``periodic(t) = W_p [sin(2 pi t / w), cos(2 pi t / w)] + b_p`` captures
    cycles such as daily routines (``w = 24`` hours). The embedding is their
    sum.

    Args:
        dim: Output size.
        hidden_dim: Size of the decay layer (64 in the authors' code).
        period: Period ``w`` of the periodic part, in the units of ``t``.
        decay_scale: ``s``, which divides ``t`` before the decay layer (180
            hours in the authors' code).

    Examples:
        >>> import torch
        >>> TRACETimeEmbedding(16)(torch.rand(2, 5) * 48).shape
        torch.Size([2, 5, 16])
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        period: float = 24.0,
        decay_scale: float = 180.0,
    ) -> None:
        super().__init__()
        self.period = period
        self.decay_scale = decay_scale
        self.decay_in = nn.Linear(1, hidden_dim)
        self.decay_out = nn.Linear(hidden_dim, dim)
        self.periodic = nn.Linear(2, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embeds times ``[B, L]`` as ``[B, L, dim]``."""
        t = t.float()
        decay = 1 - torch.tanh(self.decay_in(t.unsqueeze(-1) / self.decay_scale) ** 2)
        angle = 2 * math.pi * t / self.period
        cycle = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        return self.decay_out(decay) + self.periodic(cycle)


class TRACEAttention(nn.Module):
    """Multi-head self-attention with a time term and a smoothed mask.

    Scores are ``Q K^T / sqrt(d_k)``, plus ``Q K_t^T / sqrt(d_k)`` when a
    time embedding is given (``K_t`` is its own projection of the time
    embedding). After the softmax, the attention weights are multiplied by
    the denoising mask ``sigmoid(Z)``, a learnable ``max_len x max_len``
    matrix of logits initialized with high values (uniform in [2, 5]); a
    sequence of length ``N`` uses its bottom-right ``N x N`` block, so the
    mask is indexed by distance from the end of the sequence.

    Args:
        hidden_dim: Model size.
        num_heads: Attention heads (must divide ``hidden_dim``).
        max_len: Longest sequence, mask token included.
        dropout: Dropout on the attention weights.

    Examples:
        >>> import torch
        >>> attention = TRACEAttention(8, 2, max_len=6)
        >>> x = torch.randn(3, 4, 8)
        >>> attention(x, torch.ones(3, 4, dtype=torch.bool)).shape
        torch.Size([3, 4, 8])
    """

    def __init__(
        self, hidden_dim: int, num_heads: int, max_len: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("num_heads must divide hidden_dim.")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.time_key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.mask_logits = nn.Parameter(torch.empty(max_len, max_len).uniform_(2, 5))
        self.dropout = nn.Dropout(dropout)

    def smoothed_mask(self, length: int | None = None) -> torch.Tensor:
        """``sigmoid(Z)``, or its bottom-right ``length x length`` block."""
        mask = torch.sigmoid(self.mask_logits)
        return mask if length is None else mask[-length:, -length:]

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attends over ``x`` ``[B, N, D]``.

        Args:
            x: Inputs.
            key_mask: ``[B, N]``, True for real (non-padding) positions.
            time: Optional time embedding ``[B, N, D]`` for the time term.

        Returns:
            ``[B, N, D]``.
        """
        batch, length, _ = x.shape
        q, k, v = self._heads(self.query(x)), self._heads(self.key(x)), self._heads(
            self.value(x)
        )
        scale = math.sqrt(self.head_dim)
        scores = q @ k.transpose(-2, -1) / scale
        scores = scores.masked_fill(~key_mask[:, None, None, :], -1e9)
        if time is not None:
            scores = scores + q @ self._heads(self.time_key(time)).transpose(-2, -1) / (
                scale
            )
        weights = torch.softmax(scores, dim=-1) * self.smoothed_mask(length)
        out = self.dropout(weights) @ v
        return self.output(out.transpose(1, 2).reshape(batch, length, -1))


class _FeedForward(nn.Module):
    """Position-wise feed-forward block: linear, GELU (tanh), dropout, linear.

    TRACE keeps its own copy rather than reusing the Transformer model's
    block: the authors' code uses the tanh approximation of GELU.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TRACELayer(nn.Module):
    """One pre-norm Transformer block of TRACE (attention, then feed-forward).

    Args:
        hidden_dim: Model size.
        num_heads: Attention heads.
        max_len: Longest sequence, mask token included.
        dropout: Dropout rate.

    Examples:
        >>> import torch
        >>> layer = TRACELayer(8, 2, max_len=6)
        >>> layer(torch.randn(3, 4, 8), torch.ones(3, 4, dtype=torch.bool)).shape
        torch.Size([3, 4, 8])
    """

    def __init__(
        self, hidden_dim: int, num_heads: int, max_len: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.attention = TRACEAttention(hidden_dim, num_heads, max_len, dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.ffn = _FeedForward(hidden_dim, 4 * hidden_dim, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Applies the block to ``x`` ``[B, N, D]``; see :class:`TRACEAttention`."""
        x = x + self.dropout(self.attention(self.attention_norm(x), key_mask, time))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return self.dropout(x)


class TRACE(BaseModel):
    """TRACE for intra-visit laboratory event nowcasting (WWW 2025).

    TRACE reads a visit trajectory, a sequence of event codes with times,
    appends a mask token as a placeholder for the patient's next state, and
    encodes the sequence with a stack of time-aware Transformer layers. The
    final hidden state of the mask token goes through a linear layer and a
    sigmoid to give one probability per label (the laboratory events of the
    next draw). Each event is a code embedding plus a learnable position
    (counted from the end of the sequence); times enter through
    :class:`TRACETimeEmbedding` (decay plus periodic parts). Each layer's
    attention weights are multiplied by a learnable smoothed mask
    ``sigmoid(Z)`` (see :class:`TRACEAttention`), and the loss adds
    ``denoise_weight`` times the sum over layers of the Frobenius norm of the
    mask to the binary cross-entropy (Eq. 7-8 of the paper).

    The model reads, from the dataset:

    - ``feature_key``: event codes, a ``"sequence"`` field ``[B, L]``
      (0-padded on the right);
    - ``time_key``: the hours from each event to the prediction time, a
      ``"tensor"`` field ``[B, L]``; the mask token gets time 0;
    - one ``"multilabel"`` label (optional when calling the model
      directly: ``forward`` then returns ``logit`` and ``y_prob`` without
      ``loss``; :meth:`~pyhealth.trainer.Trainer.inference` reads
      ``output["loss"]``, so it still needs the label).

    :class:`~pyhealth.tasks.LabNowcastingMIMIC4` and
    :class:`~pyhealth.tasks.LabNowcastingMIMIC3` produce these fields.
    Sequences longer than ``max_len - 1`` keep their most recent events.

    Paper and code differ in two places; the defaults follow the code:

    - ``time_embedding="attention"``: the time embedding is projected to an
      extra key in every layer (scores ``+= Q K_t^T``), as in the code.
      ``"input"`` adds it to the event embedding instead (``h_i = c_i +
      f_i``, Eq. 5 of the paper); ``"none"`` drops it (the paper's "w/o DP"
      ablation).
    - The smoothed mask scales the attention weights after the softmax, as
      in the code; the paper writes it inside the softmax.

    The loss is the paper's binary cross-entropy (the code applies a
    softmax cross-entropy to the flattened batch).

    Args:
        dataset: The samples; used for the vocabulary and label sizes.
        embedding_dim: Model size ``d`` (256).
        num_layers: Transformer layers (3).
        num_heads: Attention heads (4).
        dropout: Dropout rate (0.1).
        max_len: Longest input, mask token included (100).
        period: Period of the periodic time feature, in hours (24).
        decay_scale: Time scale of the decay feature, in hours (180).
        denoise_weight: Weight ``lambda`` of the mask penalty (0.01). Set 0
            to train without it.
        time_embedding: ``"attention"``, ``"input"`` or ``"none"``.
        feature_key: Field with the event codes. Defaults to the only
            ``"sequence"`` input.
        time_key: Field with the event times. Defaults to the other input.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": f"p{i}", "events": ["LAB:1:N", "RX:a", "LAB:2:H"],
        ...      "event_times": [30.0, 12.0, 4.0], "labs": ["LAB:1:N", "LAB:2:H"]}
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     {"events": "sequence", "event_times": "tensor"},
        ...     {"labs": "multilabel"},
        ... )
        >>> model = TRACE(dataset, embedding_dim=16, num_layers=1, num_heads=2)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> out = model(**batch)
        >>> out["y_prob"].shape
        torch.Size([4, 2])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_len: int = 100,
        period: float = 24.0,
        decay_scale: float = 180.0,
        denoise_weight: float = 0.01,
        time_embedding: str = "attention",
        feature_key: str | None = None,
        time_key: str | None = None,
    ) -> None:
        super().__init__(dataset)
        if len(self.label_keys) != 1:
            raise ValueError("TRACE expects exactly one label field.")
        if time_embedding not in ("attention", "input", "none"):
            raise ValueError("time_embedding must be 'attention', 'input' or 'none'.")
        if max_len < 2:
            raise ValueError("max_len must be at least 2 (one event and the mask).")
        self.label_key = self.label_keys[0]
        self.feature_key, self.time_key = self._input_keys(feature_key, time_key)
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.denoise_weight = denoise_weight
        self.time_embedding = time_embedding

        vocab_size = dataset.input_processors[self.feature_key].size()
        self.mask_index = vocab_size  # one extra row for the mask token
        self.code_embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        self.time_encoder = TRACETimeEmbedding(
            embedding_dim, period=period, decay_scale=decay_scale
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            TRACELayer(embedding_dim, num_heads, max_len, dropout)
            for _ in range(num_layers)
        )
        self.fc = nn.Linear(embedding_dim, self.label_processor.output_size())

    def _input_keys(
        self, feature_key: str | None, time_key: str | None
    ) -> tuple[str, str]:
        keys = list(self.feature_keys)
        if feature_key is None:
            sequences = [
                k for k in keys if getattr(self.dataset.input_processors[k],
                                           "is_token", lambda: False)()
            ]
            if len(sequences) != 1:
                raise ValueError("Pass feature_key: expected one code-sequence input.")
            feature_key = sequences[0]
        if time_key is None:
            others = [k for k in keys if k != feature_key]
            if len(others) != 1:
                raise ValueError("Pass time_key: expected one time input.")
            time_key = others[0]
        for key in (feature_key, time_key):
            if key not in keys:
                raise ValueError(f"{key!r} is not an input of the dataset.")
        return feature_key, time_key

    def _left_align(
        self, codes: torch.Tensor, times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keeps each sequence's last ``max_len - 1`` events and appends the mask.

        Right-padded ``[B, L]`` inputs become left-padded ``[B, N]`` ones that
        end with the mask token (time 0), so the mask token is always the
        last position.
        """
        # A zero column keeps gather valid when every sequence is empty.
        codes, times = F.pad(codes, (0, 1)), F.pad(times, (0, 1))
        lengths = (codes != 0).sum(dim=1)
        width = int(lengths.clamp(max=self.max_len - 1).max()) if len(codes) else 0
        slots = torch.arange(width, device=codes.device)
        source = lengths[:, None] - width + slots[None, :]
        valid = source >= 0
        source = source.clamp(min=0)
        history_codes = codes.gather(1, source) * valid
        history_times = times.gather(1, source) * valid
        mask_token = codes.new_full((len(codes), 1), self.mask_index)
        aligned_codes = torch.cat([history_codes, mask_token], dim=1)
        aligned_times = torch.cat([history_times, times.new_zeros(len(times), 1)], 1)
        return aligned_codes, aligned_times

    def encode(self, codes: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Hidden state of the mask token, ``[B, embedding_dim]``.

        Args:
            codes: Event codes ``[B, L]``, 0-padded on the right.
            times: Hours from each event to the prediction time ``[B, L]``.
        """
        codes, times = self._left_align(codes, times.float())
        length = codes.shape[1]
        positions = torch.arange(
            self.max_len - length, self.max_len, device=codes.device
        )
        x = self.code_embedding(codes) + self.position_embedding(positions)[None]
        time = self.time_encoder(times)
        if self.time_embedding == "input":
            x = x + time
        x = self.embedding_dropout(x)
        time = self.embedding_dropout(time) if self.time_embedding == "attention" else None
        key_mask = codes != 0
        for layer in self.layers:
            x = layer(x, key_mask, time)
        return x[:, -1]

    def denoise_loss(self) -> torch.Tensor:
        """Sum over layers of the Frobenius norm of ``sigmoid(Z)`` (Eq. 7).

        Returns:
            A scalar tensor.

        Examples:
            >>> model.denoise_loss()  # doctest: +SKIP
            tensor(290.1, grad_fn=<AddBackward0>)
        """
        return sum(
            torch.linalg.matrix_norm(layer.attention.smoothed_mask())
            for layer in self.layers
        )

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Scores the next draw's laboratory events.

        Returns:
            A dict with ``logit`` and ``y_prob`` (``[B, num_labels]``) and,
            when the label is given, ``loss`` (BCE plus the weighted mask
            penalty) and ``y_true``. With ``embed=True``, also ``embed``,
            the mask token's hidden state. A direct call runs without the
            label; ``Trainer.inference`` needs it, as it reads
            ``output["loss"]``.
        """
        codes = kwargs[self.feature_key].to(self.device).long()
        times = kwargs[self.time_key].to(self.device).float()
        if codes.dim() == 1:
            codes, times = codes[None], times[None]
        hidden = self.encode(codes, times)
        logit = self.fc(hidden)
        out = {"logit": logit, "y_prob": self.label_processor.activation(logit)}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device).float()
            loss = self.label_processor.loss(logit, y_true)
            if self.denoise_weight:
                loss = loss + self.denoise_weight * self.denoise_loss()
            out["loss"] = loss
            out["y_true"] = y_true
        if kwargs.get("embed", False):
            out["embed"] = hidden
        return out

