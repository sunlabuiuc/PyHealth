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

- The Mamba blocks are :class:`~pyhealth.models.SelectiveSSM`, the pure
  PyTorch block shared with ARMR, with its parallel log-space scan (Gu and
  Dao, 2023), so the CUDA ``mamba_ssm`` package is not needed and the
  model runs on CPU. It uses more memory than the fused kernel on long
  sequences.
- Padding is masked: padded positions do not update the state and are
  excluded from top-k pooling, so a sample's prediction does not depend on
  the other samples in its batch.
- No positional encoding is added (the authors' code has it disabled; the
  scan is order-aware).
- Class imbalance: the paper weights each label's loss term; here the
  ``pos_weight`` argument (e.g. ``"balanced"``) weights the positive
  examples of each label.
- The isotonic recalibration and the overall-acuity logic of the paper are
  evaluation steps and are not part of the model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.selective_ssm import SelectiveSSM
from pyhealth.processors import LabelProcessor, MultiLabelProcessor

__all__ = ["APRICOTM"]


class _MambaStack(nn.Module):
    """Pre-norm residual Mamba blocks with a final LayerNorm."""

    def __init__(self, d_model: int, n_layers: int, **ssm_kwargs) -> None:
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.mixers = nn.ModuleList(
            SelectiveSSM(d_model, scan="parallel", **ssm_kwargs)
            for _ in range(n_layers)
        )
        for mixer in self.mixers:
            # dt initialized log-uniformly in [1e-3, 1e-1], as mamba_ssm does.
            dt = torch.exp(
                torch.rand(mixer.dt_proj.bias.shape)
                * (math.log(0.1) - math.log(1e-3))
                + math.log(1e-3)
            )
            with torch.no_grad():
                mixer.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
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
    averaged over labels. Pass ``pos_weight="balanced"`` with the training
    split for the class weighting of the paper.

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
        pos_weight: Weight of the positive examples of each label in the
            loss: a number, one number per label, ``"balanced"``
            (negatives / positives of each label in ``dataset``, so pass the
            training split; this reads every sample of ``dataset`` once) or
            None. With ``"balanced"`` a label that is positive in every
            sample gets weight 0 and one with no positive gets weight 1.

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
        pos_weight: float | Sequence[float] | str | None = None,
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

        processor = dataset.input_processors[feature_key]
        self._roles = self._event_roles(processor, feature_key)
        sample = dataset[0]
        self.num_labels = int(torch.as_tensor(sample[label_key]).numel())
        # The label is a tensor field, which is not a label processor, so the
        # model supplies one: K fixed binary labels with BCE (see
        # label_processors). "balanced" is learned from ``dataset``.
        self._label = MultiLabelProcessor(labels=self.num_labels, pos_weight=pos_weight)
        if pos_weight == "balanced":
            self._label._fit_pos_weight(
                [
                    torch.as_tensor(dataset[i][label_key], dtype=torch.float32)
                    for i in range(len(dataset))
                ]
            )
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
            state_size=state_size,
            conv_size=conv_kernel,
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
        # Times stay absolute (hours since ICU admission), as in the authors'
        # code, which min-max scales them over the training stays (<= 30 days).
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

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """The ``K`` binary labels, scored as a multilabel classification.

        The label is a ``"tensor"`` field, which is not a label processor, so
        APRICOT-M supplies a fixed :class:`~pyhealth.processors.MultiLabelProcessor`
        with ``K`` labels and the model's ``pos_weight``.

        Examples:
            >>> model.label_processor.size()  # doctest: +SKIP
            12
        """
        return {self.label_key: self._label}

    def forward(self, **kwargs) -> dict[str, Tensor]:
        """Runs APRICOT-M on a batch.

        Args:
            **kwargs: Batch fields from the PyHealth dataloader; the label
                is optional (inference without labels).

        Returns:
            ``logit`` ``[B, K]``, ``y_prob`` (sigmoid of the logits), and
            with labels ``loss`` (mean binary cross-entropy, weighted by
            ``pos_weight`` when given) and ``y_true``. ``embed=True`` adds
            ``embed``, the fused representation.
        """
        h = self.encode(**kwargs)
        logits = torch.cat([head(h) for head in self.heads], dim=-1)
        out = {"logit": logits, "y_prob": self.label_processor.activation(logits)}
        if self.label_key in kwargs:
            y_true = torch.as_tensor(kwargs[self.label_key]).to(
                self.device, torch.float32
            ).reshape(logits.shape)
            out["loss"] = self.label_processor.loss(logits, y_true)
            out["y_true"] = y_true
        if kwargs.get("embed", False):
            out["embed"] = h
        return out
