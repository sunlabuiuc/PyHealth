"""APRICOT-M: a state-space model for real-time ICU acuity prediction.

Implements APRICOT-M from Contreras et al., "Real-time prediction of
intensive care unit patient acuity and therapy requirements using
state-space modelling", Nature Communications 16, 7315 (2025),
https://doi.org/10.1038/s41467-025-62121-1.

APRICOT-M reads the clinical events of the last few hours as a sequence of
(time, variable, value) triplets, so irregular, sparse measurements need no
resampling or imputation. Time and value pass through two position-wise
layers (the paper's 1D convolutions with kernel size 1), the variable code
through an embedding, and the two are added. A stack of Mamba (selective
state-space) blocks contextualizes the events; top-k pooling over time keeps
the ``k`` largest activations of each channel, an MLP maps them to one
vector, and a static-feature embedding is added to it. One small head per
outcome gives the probability of each label (12 in the paper: four acuity
states, two transitions and the stop and onset of three therapies).

Clean reimplementation from the paper. The authors' code
(https://github.com/iheallab/apricotM) is GPL-3.0, so none of it is copied
here; it was only read to confirm details (the default sizes below are the
ones it trains with). Differences:

- The Mamba blocks are written in pure PyTorch (the selective scan of Gu
  and Dao, 2023, computed in parallel in log space), so the CUDA
  ``mamba_ssm`` package is not needed and the model runs on CPU. It uses
  more memory than the fused kernel on long sequences.
- Padding is masked: padded positions do not update the state and are
  excluded from top-k pooling, so a sample's prediction does not depend on
  the other samples in its batch.
- No positional encoding is added (the authors' code has it disabled; the
  scan is order-aware).
- Class imbalance: the paper weights each label's loss term; here
  :meth:`~pyhealth.models.BaseModel.set_pos_weight` (e.g. ``"balanced"``)
  weights the positive examples of each label.
- The isotonic recalibration and the overall-acuity logic of the paper are
  evaluation steps and are not part of the model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel

__all__ = ["APRICOTM", "SelectiveSSM", "selective_scan"]


def selective_scan(log_decay: Tensor, drive: Tensor, chunk_size: int = 32) -> Tensor:
    """Parallel scan of ``h_t = exp(log_decay_t) * h_{t-1} + drive_t``, ``h_0 = 0``.

    Computed along dimension 1 in log space: within a chunk, with ``S_t``
    the cumulative sum of ``log_decay`` since the chunk start,
    ``h_t = exp(S_t) h_start + sum_{s <= t} exp(S_t - S_s) drive_s``. The
    positive and negative parts of ``drive`` are summed separately with
    ``logcumsumexp``, so no ``exp(-S)`` can overflow; chunking keeps ``S``
    small enough for float32 precision.

    Args:
        log_decay: Non-positive log decay factors, shape ``[B, L, ...]``.
        drive: Inputs, same shape.
        chunk_size: Steps per chunk.

    Returns:
        The states ``h``, same shape.

    Examples:
        >>> import torch
        >>> log_decay = torch.log(torch.tensor([[0.5, 0.5, 0.5]]))
        >>> drive = torch.tensor([[1.0, -2.0, 4.0]])
        >>> selective_scan(log_decay, drive)  # h = 1, 0.5 - 2, -0.75 + 4
        tensor([[ 1.0000, -1.5000,  3.2500]])
    """
    very_negative = -1e30

    def part(values: Tensor, cumulative: Tensor) -> Tensor:
        positive = values > 0
        log_values = torch.where(
            positive, torch.log(torch.where(positive, values, 1.0)), very_negative
        )
        return torch.exp(
            torch.logcumsumexp(log_values - cumulative, dim=1) + cumulative
        )

    chunks = []
    carry = None
    for begin in range(0, log_decay.shape[1], chunk_size):
        decay = log_decay[:, begin : begin + chunk_size]
        inputs = drive[:, begin : begin + chunk_size]
        cumulative = torch.cumsum(decay, dim=1)
        states = part(inputs, cumulative) - part(-inputs, cumulative)
        if carry is not None:
            states = states + torch.exp(cumulative) * carry.unsqueeze(1)
        carry = states[:, -1]
        chunks.append(states)
    if not chunks:
        return drive.clone()
    return torch.cat(chunks, dim=1)


class SelectiveSSM(nn.Module):
    """A Mamba (selective state-space) mixer, in pure PyTorch.

    Follows Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective
    State Spaces" (2023): an input projection into a gated branch and an SSM
    branch, a causal depthwise convolution, input-dependent step size
    ``dt`` and matrices ``B`` and ``C``, a diagonal state matrix ``A``, and
    the scan ``h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t``,
    ``y_t = C_t h_t + D x_t``, computed in parallel by
    :func:`selective_scan`.

    Args:
        d_model: Input and output width.
        d_state: State size ``N`` per channel.
        d_conv: Kernel size of the causal depthwise convolution.
        expand: Inner width is ``expand * d_model``.
        dt_rank: Rank of the ``dt`` projection; ``ceil(d_model / 16)`` when
            None.

    Examples:
        >>> import torch
        >>> mixer = SelectiveSSM(d_model=8, d_state=4)
        >>> x = torch.randn(2, 5, 8)
        >>> mixer(x).shape
        torch.Size([2, 5, 8])
        >>> mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]).bool()
        >>> y = mixer(x, mask)
        >>> torch.allclose(y[0, :3], mixer(x[:1, :3])[0], atol=1e-6)
        True
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # Causal depthwise convolution, weight [D, d_conv] (last tap = now).
        bound = 1.0 / math.sqrt(d_conv)
        self.conv_weight = nn.Parameter(
            torch.empty(self.d_inner, d_conv).uniform_(-bound, bound)
        )
        self.conv_bias = nn.Parameter(
            torch.empty(self.d_inner).uniform_(-bound, bound)
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        # dt initialized log-uniformly in [1e-3, 1e-1] (Mamba's default).
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3))
            + math.log(1e-3)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _causal_conv(self, u: Tensor) -> Tensor:
        """Depthwise causal convolution over time of ``u`` ``[B, L, D]``.

        Written as shifted sums: on CPU this is much faster to train than a
        grouped ``nn.Conv1d``.
        """
        taps = self.conv_weight.shape[1]
        padded = F.pad(u, (0, 0, taps - 1, 0))
        length = u.shape[1]
        out = self.conv_bias.expand_as(u)
        for k in range(taps):
            out = out + padded[:, k : k + length] * self.conv_weight[:, k]
        return out

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Mixes a sequence.

        Args:
            x: Shape ``[B, L, d_model]``.
            mask: Optional bool ``[B, L]``, True for real positions. Padded
                positions do not change the state; use right padding.

        Returns:
            Shape ``[B, L, d_model]``; padded positions are zero.
        """
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        if mask is not None:
            u = u * mask.unsqueeze(-1)
        u = F.silu(self._causal_conv(u))
        dt, b, c = self.x_proj(u).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # [B, L, D]
        if mask is not None:
            dt = dt * mask.unsqueeze(-1)
        a = -torch.exp(self.A_log.float())  # [D, N]
        log_decay = dt.unsqueeze(-1) * a  # [B, L, D, N]
        drive = (dt * u).unsqueeze(-1) * b.unsqueeze(2)  # [B, L, D, N]
        states = selective_scan(log_decay, drive)
        y = torch.einsum("bldn,bln->bld", states, c) + u * self.D
        y = self.out_proj(y * F.silu(gate))
        if mask is not None:
            y = y * mask.unsqueeze(-1)
        return y


class _MambaStack(nn.Module):
    """Pre-norm residual Mamba blocks with a final LayerNorm."""

    def __init__(self, d_model: int, n_layers: int, **ssm_kwargs) -> None:
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.mixers = nn.ModuleList(
            SelectiveSSM(d_model, **ssm_kwargs) for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for norm, mixer in zip(self.norms, self.mixers):
            x = x + mixer(norm(x), mask)
        return self.final_norm(x)


class APRICOTM(BaseModel):
    """APRICOT-M: Mamba over clinical event triplets, for ICU acuity.

    The input is a ``sequence_v2`` series of events with one token (the
    variable) and one value per event, such as the ``events`` field of
    :class:`~pyhealth.tasks.ICUAcuityMIMIC4`; its ``time`` is the event time
    (hours since ICU admission in that task), divided by ``time_scale``.
    Only the last ``max_seq_len`` events of a sample are read, as in the
    paper. ``static_key`` names an optional static tensor field.

    The label is a tensor field of ``K`` binary labels (the ``acuity``
    field of :class:`~pyhealth.tasks.ICUAcuityMIMIC4`, ``K = 12``). The
    model is a multilabel classifier: one two-layer head per label,
    ``mode`` is ``"multilabel"``, and the loss is the binary cross-entropy
    averaged over labels. Use :meth:`set_pos_weight` (e.g. ``"balanced"``
    with the training samples) for the class weighting of the paper.

    Args:
        dataset: The dataset (schemas, vocabulary and static width).
        feature_key: The event series field.
        label_key: The label field; None uses the only output field.
        static_key: Optional static tensor field.
        embedding_dim: Model width ``d_model`` (92 in the authors' code).
        hidden_dim: Hidden width of the time-value layers (28).
        num_layers: Mamba blocks (2).
        state_size: SSM state size per channel.
        conv_kernel: Kernel size of the Mamba convolution.
        expand: Mamba inner width factor.
        top_k: Time steps kept per channel by top-k pooling (5).
        dropout: Dropout after the pooling MLP (0.2).
        max_seq_len: Events read per sample, the most recent ones (512).
        time_scale: Divides the event times (720 hours = 30 days, the
            longest stay in the paper's cohort, maps them to [0, 1]).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> levels = [{"ordering": "series", "tokens": 1, "values": [1]}]
        >>> samples = [
        ...     {"patient_id": f"p{i}",
        ...      "events": {"time": [0.5, 1.0 + i], "tokens": ["hr", "sbp"],
        ...                 "values": [[80.0 + i], [120.0]]},
        ...      "static": [60.0, 1.0], "acuity": [float(i % 2), 1.0, 0.0]}
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"events": ("sequence_v2", {"levels": levels}),
        ...                   "static": "tensor"},
        ...     output_schema={"acuity": "tensor"},
        ...     dataset_name="toy",
        ... )
        >>> model = APRICOTM(dataset, static_key="static", embedding_dim=16)
        >>> model.mode, model.num_labels
        ('multilabel', 3)

    References:
        Contreras, M., et al. Real-time prediction of intensive care unit
        patient acuity and therapy requirements using state-space
        modelling. Nature Communications 16, 7315 (2025).
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str = "events",
        label_key: str | None = None,
        static_key: str | None = None,
        embedding_dim: int = 92,
        hidden_dim: int = 28,
        num_layers: int = 2,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        top_k: int = 5,
        dropout: float = 0.2,
        max_seq_len: int = 512,
        time_scale: float = 720.0,
    ) -> None:
        super().__init__(dataset)
        if feature_key not in dataset.input_schema:
            raise ValueError(f"{feature_key!r} is not in dataset.input_schema")
        if static_key is not None and static_key not in dataset.input_schema:
            raise ValueError(f"{static_key!r} is not in dataset.input_schema")
        if label_key is None:
            if len(self.label_keys) != 1:
                raise ValueError("label_key is required with several output fields")
            label_key = self.label_keys[0]
        if label_key not in dataset.output_schema:
            raise ValueError(f"{label_key!r} is not in dataset.output_schema")
        if top_k < 1 or max_seq_len < 1:
            raise ValueError("top_k and max_seq_len must be positive")
        self.feature_key = feature_key
        self.label_key = label_key
        self.label_keys = [label_key]
        self.static_key = static_key
        self.top_k = top_k
        self.max_seq_len = max_seq_len
        self.time_scale = float(time_scale)
        self.mode = "multilabel"

        processor = dataset.input_processors[feature_key]
        self._roles = self._event_roles(processor, feature_key)
        sample = dataset[0]
        self.num_labels = int(torch.as_tensor(sample[label_key]).numel())
        static_dim = 0
        if static_key is not None:
            static_dim = int(torch.as_tensor(sample[static_key]).numel())

        self.value_time = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim)
        )
        self.variable_embedding = nn.Embedding(
            processor.vocab_size()[0][0], embedding_dim, padding_idx=0
        )
        self.encoder = _MambaStack(
            embedding_dim,
            num_layers,
            d_state=state_size,
            d_conv=conv_kernel,
            expand=expand,
        )
        self.pool_mlp = nn.Sequential(
            nn.Linear(embedding_dim * top_k, embedding_dim), nn.Dropout(dropout)
        )
        self.static_ffn = (
            nn.Sequential(
                nn.Linear(static_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            if static_key is not None
            else None
        )
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 1),
            )
            for _ in range(self.num_labels)
        )

    # ------------------------------------------------------------ input
    @staticmethod
    def _event_roles(processor, key: str) -> dict[str, int]:
        schema = getattr(processor, "schema", None)
        entries = schema() if callable(schema) else ()
        roles: dict[str, int] = {}
        for index, entry in enumerate(entries):
            if not isinstance(entry, tuple) or entry[0] != 0:
                raise ValueError(
                    f"{key!r} must be a single-level sequence_v2 series of events"
                )
            roles[entry[1]] = index
        if not {"tokens", "values", "time"} <= set(roles):
            raise ValueError(
                f"{key!r} must be a sequence_v2 'series' level with one token "
                "namespace and values"
            )
        # SequenceProcessorV2's collate appends the record offsets.
        roles["offsets"] = len(entries)
        return roles

    def _dense(self, value) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Packed events to right-padded ``tokens, values, times, mask``.

        Keeps the last ``max_seq_len`` events of each sample.
        """
        tokens = value[self._roles["tokens"]].to(self.device).reshape(-1)
        values = value[self._roles["values"]].to(self.device, torch.float32)
        values = values.reshape(values.shape[0], -1)[:, 0]
        times = value[self._roles["time"]].to(self.device, torch.float32).reshape(-1)
        offsets = value[self._roles["offsets"]].to(self.device)
        lengths = (offsets[1:] - offsets[:-1]).clamp(max=self.max_seq_len)
        starts = offsets[1:] - lengths
        steps = max(int(lengths.max().item()) if len(lengths) else 0, 1)
        position = torch.arange(steps, device=self.device)
        mask = position[None, :] < lengths[:, None]
        if tokens.shape[0] == 0:
            zeros = torch.zeros_like(mask, dtype=torch.float32)
            return zeros.long(), zeros, zeros, mask
        index = (starts[:, None] + position[None, :]).clamp(max=tokens.shape[0] - 1)
        dense_tokens = torch.where(mask, tokens[index], 0)
        dense_values = torch.where(mask, torch.nan_to_num(values[index]), 0.0)
        dense_times = torch.where(mask, times[index] / self.time_scale, 0.0)
        return dense_tokens, dense_values, dense_times, mask

    # ------------------------------------------------------------ model
    def encode(self, **kwargs) -> Tensor:
        """The fused patient representation, ``[B, embedding_dim]``.

        Args:
            **kwargs: Batch fields from the PyHealth dataloader.

        Returns:
            The pooled event encoding plus the static embedding.
        """
        tokens, values, times, mask = self._dense(kwargs[self.feature_key])
        x = self.value_time(torch.stack([times, values], dim=-1))
        x = x + self.variable_embedding(tokens)
        x = self.encoder(x * mask.unsqueeze(-1), mask)  # [B, L, d]
        x = x.masked_fill(~mask.unsqueeze(-1), float("-inf")).transpose(1, 2)
        if x.shape[-1] < self.top_k:
            x = F.pad(x, (0, self.top_k - x.shape[-1]), value=float("-inf"))
        pooled = torch.topk(x, k=self.top_k, dim=-1).values  # [B, d, k]
        pooled = torch.nan_to_num(pooled, neginf=0.0).flatten(1)
        h = self.pool_mlp(pooled)
        if self.static_ffn is not None:
            static = torch.as_tensor(kwargs[self.static_key]).to(
                self.device, torch.float32
            )
            h = h + self.static_ffn(static.reshape(static.shape[0], -1))
        return h

    def forward(self, **kwargs) -> dict[str, Tensor]:
        """Runs APRICOT-M on a batch.

        Args:
            **kwargs: Batch fields from the PyHealth dataloader; the label
                is optional (inference without labels).

        Returns:
            ``logit`` ``[B, K]``, ``y_prob`` (sigmoid of the logits), and
            with labels ``loss`` (mean binary cross-entropy, weighted by
            :attr:`pos_weight` when set) and ``y_true``. ``embed=True`` adds
            ``embed``, the fused representation.
        """
        h = self.encode(**kwargs)
        logits = torch.cat([head(h) for head in self.heads], dim=-1)
        out = {"logit": logits, "y_prob": torch.sigmoid(logits)}
        if self.label_key in kwargs:
            y_true = torch.as_tensor(kwargs[self.label_key]).to(
                self.device, torch.float32
            ).reshape(logits.shape)
            pos_weight = self.pos_weight
            out["loss"] = F.binary_cross_entropy_with_logits(
                logits,
                y_true,
                pos_weight=None if pos_weight is None else pos_weight.to(self.device),
            )
            out["y_true"] = y_true
        if kwargs.get("embed", False):
            out["embed"] = h
        return out
