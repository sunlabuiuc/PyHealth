"""CLMBR-T, the EHRSHOT foundation model, in plain PyTorch with a linear head.

CLMBR-T-base (Wornow et al., "EHRSHOT: An EHR Benchmark for Few-Shot
Evaluation of Foundation Models", NeurIPS 2023 Datasets and Benchmarks;
https://arxiv.org/abs/2307.02028) is a 141M-parameter autoregressive
transformer pretrained with next-code prediction on 2.57 million Stanford
Medicine EHRs, following CLMBR (Steinberg et al., "Language models are an
effective representation learning technique for electronic health record
data", JBI 2021) with the GRU replaced by a transformer. Its weights are the
gated ``StanfordShahLab/clmbr-t-base`` Hugging Face repo, published for
femr 0.2.x (https://github.com/som-shahlab/femr, Apache-2.0).

This module reimplements femr 0.2.3's ``FEMRTransformer`` and
``FEMRTokenizer`` without femr, xformers or their pinned torch versions:

- tokens are looked up in femr's ``dictionary.msgpack``, with numeric values
  binned into half-open ranges and a code dropped when it repeats on the
  same day;
- rotary position embeddings use each token's age in days, not its index;
- attention is causal and local (each token sees itself and the previous
  ``attention_width - 1`` tokens of the same patient), done here with a
  banded boolean mask and ``scaled_dot_product_attention`` so it runs on any
  device;
- RMSNorm, a fused QKV + feed-forward input projection and a parallel
  attention / feed-forward block, as in femr.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, timedelta
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import SampleDataset
from .ehr_fm import EHRFoundationModelBase, TokenizedTimeline, as_value

logger = logging.getLogger(__name__)


def _require(module: str) -> Any:
    """Imports an optional dependency, with an install hint if missing."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"{module} is not installed; CLMBR needs it to load checkpoints: "
            "`pip install huggingface-hub safetensors msgpack`"
        ) from e

CLMBR_T_BASE = "StanfordShahLab/clmbr-t-base"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class CLMBRConfig:
    """The ``transformer_config`` of a femr 0.2 ``FEMRModel``.

    Defaults are femr 0.2.3's ``FEMRTransformerConfig`` defaults; a
    checkpoint's ``config.json`` sets the real values (CLMBR-T-base's card
    lists 12 layers, hidden size 768 and context window 496).

    Examples:
        >>> config = CLMBRConfig.from_dict({"transformer_config": {
        ...     "vocab_size": 100, "n_layers": 2, "hidden_act": "swiglu"}})
        >>> config.vocab_size, config.n_layers, config.hidden_act
        (100, 2, 'swiglu')
    """

    vocab_size: int = 32768
    is_hierarchical: bool = False
    hidden_size: int = 768
    intermediate_size: int = 3072
    n_heads: int = 12
    n_layers: int = 6
    attention_width: int = 496
    use_normed_ages: bool = False
    use_bias: bool = True
    hidden_act: str = "gelu"

    #: Keys a checkpoint's ``config.json`` must set: a wrong default for
    #: most of them changes no weight shape, so loading would not catch it.
    REQUIRED_KEYS: ClassVar[tuple[str, ...]] = (
        "vocab_size", "hidden_size", "n_heads", "n_layers",
        "attention_width", "use_normed_ages", "hidden_act", "use_bias",
    )

    @classmethod
    def from_dict(cls, config: Mapping[str, Any], strict: bool = False
                  ) -> CLMBRConfig:
        """Reads a ``FEMRModel`` ``config.json`` (or its transformer part).

        Unknown keys (``transformers`` bookkeeping) are ignored.

        Args:
            config: The parsed config.
            strict: Raise ``ValueError`` if any of :attr:`REQUIRED_KEYS` is
                missing, instead of using the default. Used for
                checkpoint configs.
        """
        config = config.get("transformer_config", config)
        if strict:
            missing = [k for k in cls.REQUIRED_KEYS if k not in config]
            if missing:
                raise ValueError(
                    f"the checkpoint's transformer_config lacks {missing}; "
                    "refusing to guess them, since a wrong value changes the "
                    "features without any shape error"
                )
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in names})


# ---------------------------------------------------------------------------
# backbone
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMS normalization with a learned scale (femr's, as Llama's).

    Examples:
        >>> RMSNorm(4)(torch.ones(2, 4)).shape
        torch.Size([2, 4])
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


def age_rotary_embedding(ages: torch.Tensor, dim: int, dtype: torch.dtype
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """femr's rotary ``sin``/``cos`` tables for token ages (in days).

    As femr's ``fixed_pos_embedding``: the inverse frequencies are
    ``10000 ** -linspace(0, 2, dim // 2)`` (note: up to 2, not 1), and each
    frequency is repeated for the two channels of a rotated pair.

    Args:
        ages: ``[..., L]`` float32 ages in days.
        dim: Head size.
        dtype: Output dtype.

    Returns:
        ``sin`` and ``cos``, each ``[..., L, 1, dim]``.

    Examples:
        >>> sin, cos = age_rotary_embedding(torch.zeros(3), 8, torch.float32)
        >>> sin.shape, bool(cos.eq(1).all())
        (torch.Size([3, 1, 8]), True)
    """
    ages = ages.to(torch.float32)
    inv_freq = 1.0 / (10000 ** torch.linspace(0, 2, steps=dim // 2,
                                              device=ages.device))
    t = ages.unsqueeze(-1) * inv_freq  # [..., L, dim // 2]
    sin = torch.stack((t.sin(), t.sin()), dim=-1).flatten(-2).unsqueeze(-2)
    cos = torch.stack((t.cos(), t.cos()), dim=-1).flatten(-2).unsqueeze(-2)
    return sin.to(dtype), cos.to(dtype)


def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
                  ) -> torch.Tensor:
    return x * cos.to(x.dtype) + _rotate_every_two(x) * sin.to(x.dtype)


def local_causal_mask(lengths: torch.Tensor, max_len: int, width: int
                      ) -> torch.Tensor:
    """The banded attention mask of femr's local causal attention.

    Query ``i`` of a patient sees key ``j`` of the same patient when
    ``0 <= i - j < width`` (itself and the ``width - 1`` tokens before it),
    as xformers' ``BlockDiagonalMask(...).make_local_attention(width)``.
    Padding queries see only themselves, so no row is empty.

    Args:
        lengths: ``[B]`` real tokens per patient (right padding).
        max_len: Padded length ``L``.
        width: ``attention_width``.

    Returns:
        A ``[B, 1, L, L]`` boolean mask, ``True`` where attention is allowed.

    Examples:
        >>> local_causal_mask(torch.tensor([3]), 3, 2)[0, 0].int()
        tensor([[1, 0, 0],
                [1, 1, 0],
                [0, 1, 1]], dtype=torch.int32)
    """
    i = torch.arange(max_len, device=lengths.device)
    distance = i[:, None] - i[None, :]
    band = (distance >= 0) & (distance < width)
    valid = i[None, :] < lengths[:, None]  # [B, L]
    mask = band[None] & valid[:, None, :] & valid[:, :, None]
    mask = mask | torch.eye(max_len, dtype=torch.bool, device=lengths.device)
    return mask.unsqueeze(1)


class CLMBRLayer(nn.Module):
    """One femr encoder layer: parallel attention and feed-forward blocks.

    Parameter names match the published ``transformer.layers.{i}.*`` keys.

    Examples:
        >>> config = CLMBRConfig(hidden_size=8, intermediate_size=16, n_heads=2)
        >>> layer = CLMBRLayer(config)
        >>> x, ages = torch.randn(1, 3, 8), torch.zeros(1, 3)
        >>> sin, cos = age_rotary_embedding(ages, 4, x.dtype)
        >>> mask = local_causal_mask(torch.tensor([3]), 3, 2)
        >>> layer(x, ages, sin, cos, mask).shape
        torch.Size([1, 3, 8])
    """

    def __init__(self, config: CLMBRConfig) -> None:
        super().__init__()
        if config.hidden_act not in ("gelu", "swiglu"):
            raise ValueError(f"unsupported hidden_act {config.hidden_act!r}")
        self.config = config
        hidden, inner = config.hidden_size, config.intermediate_size
        mult = 2 if config.hidden_act == "swiglu" else 1
        self.norm = RMSNorm(hidden)
        self.input_proj = nn.Linear(hidden, hidden * 3 + mult * inner,
                                    bias=config.use_bias)
        self.output_proj = nn.Linear(hidden + inner, hidden, bias=config.use_bias)

    def forward(self, x: torch.Tensor, normed_ages: torch.Tensor,
                sin: torch.Tensor, cos: torch.Tensor, mask: torch.Tensor
                ) -> torch.Tensor:
        config = self.config
        batch, length, hidden = x.shape
        x = self.norm(x)
        if config.use_normed_ages:
            x = x.clone()
            x[..., -2] = normed_ages.to(x.dtype)
            x[..., -1] = (normed_ages ** 2).to(x.dtype)
        transformed = self.input_proj(x)
        ff = transformed[..., : -hidden * 3]
        qkv = transformed[..., -hidden * 3:]
        head = hidden // config.n_heads
        qkv = qkv.reshape(batch, length, 3, config.n_heads, head)
        q = _apply_rotary(qkv[:, :, 0], sin, cos)
        k = _apply_rotary(qkv[:, :, 1], sin, cos)
        v = qkv[:, :, 2]
        # [B, heads, L, head]; femr's scale is 1 / sqrt(head)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(batch, length, hidden)
        if config.hidden_act == "gelu":
            ff = F.gelu(ff)
        else:
            x1, x2 = ff.chunk(2, dim=-1)
            ff = F.silu(x1) * x2
        return self.output_proj(torch.cat((attn, ff), dim=-1))


class CLMBRTransformer(nn.Module):
    """femr 0.2's ``FEMRTransformer`` on right-padded ``[B, L]`` batches.

    femr packs patients into one sequence with a block-diagonal local mask;
    a padded batch with a per-patient banded mask
    (:func:`local_causal_mask`) computes the same thing. Parameter names
    match the published keys without their ``transformer.`` prefix (see
    :func:`load_clmbr_weights`).

    Args:
        config: The transformer config. Hierarchical (``EmbeddingBag``)
            vocabularies are not supported.

    Examples:
        >>> config = CLMBRConfig(vocab_size=10, hidden_size=8,
        ...                      intermediate_size=16, n_heads=2, n_layers=1)
        >>> model = CLMBRTransformer(config)
        >>> tokens = torch.tensor([[1, 2, 3]])
        >>> ages = torch.tensor([[0.0, 1.0, 1.5]])
        >>> model(tokens, ages, ages, torch.tensor([3])).shape
        torch.Size([1, 3, 8])
    """

    def __init__(self, config: CLMBRConfig) -> None:
        super().__init__()
        if config.is_hierarchical:
            raise NotImplementedError(
                "hierarchical femr vocabularies are not supported; "
                "CLMBR-T-base is not hierarchical")
        if config.hidden_size % config.n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        self.config = config
        self.in_norm = RMSNorm(config.hidden_size)
        self.out_norm = RMSNorm(config.hidden_size)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(CLMBRLayer(config)
                                    for _ in range(config.n_layers))

    def forward(self, tokens: torch.Tensor, ages: torch.Tensor,
                normed_ages: torch.Tensor, lengths: torch.Tensor
                ) -> torch.Tensor:
        """Final hidden states.

        Args:
            tokens: ``[B, L]`` token ids, right-padded.
            ages: ``[B, L]`` ages in days (rotary positions).
            normed_ages: ``[B, L]`` normalized ages (used only with
                ``use_normed_ages``).
            lengths: ``[B]`` real tokens per row.

        Returns:
            ``[B, L, hidden_size]``; padding rows are meaningless.
        """
        x = self.in_norm(self.embed(tokens))
        head = self.config.hidden_size // self.config.n_heads
        sin, cos = age_rotary_embedding(ages, head, x.dtype)
        mask = local_causal_mask(lengths, tokens.shape[1],
                                 self.config.attention_width)
        for layer in self.layers:
            x = x + layer(x, normed_ages, sin, cos, mask)
        return self.out_norm(x)


def load_clmbr_weights(model: CLMBRTransformer,
                       state_dict: Mapping[str, torch.Tensor]) -> list[str]:
    """Loads a published femr ``FEMRModel`` state dict into ``model``.

    The key mapping (``i`` is the layer, ``H`` the hidden size, ``I`` the
    intermediate size, ``m`` 2 for swiglu else 1)::

        transformer.embed.weight                   [V, H]        -> embed.weight
        transformer.in_norm.weight                 [H]           -> in_norm.weight
        transformer.out_norm.weight                [H]           -> out_norm.weight
        transformer.layers.i.norm.weight           [H]           -> layers.i.norm.weight
        transformer.layers.i.input_proj.weight     [3H + mI, H]  -> layers.i.input_proj.weight
        transformer.layers.i.input_proj.bias       [3H + mI]     -> (if use_bias)
        transformer.layers.i.output_proj.weight    [H, H + I]    -> layers.i.output_proj.weight
        transformer.layers.i.output_proj.bias      [H]           -> (if use_bias)
        task_model.*  (the CLMBR next-code head)                 -> not loaded

    Every backbone key must be present with the right shape.

    Args:
        model: The backbone to fill.
        state_dict: The published tensors (e.g. from ``model.safetensors``).

    Returns:
        The published keys that were not loaded (the pretraining head).

    Examples:
        >>> config = CLMBRConfig(vocab_size=10, hidden_size=8,
        ...                      intermediate_size=16, n_heads=2, n_layers=1)
        >>> model = CLMBRTransformer(config)
        >>> published = {f"transformer.{k}": v
        ...              for k, v in model.state_dict().items()}
        >>> published["task_model.final_layer.weight"] = torch.zeros(10, 8)
        >>> load_clmbr_weights(model, published)
        ['task_model.final_layer.weight']
    """
    ours: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state_dict.items():
        if key.startswith("transformer."):
            ours[key[len("transformer."):]] = value
        else:
            skipped.append(key)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(ours))
    unexpected = sorted(set(ours) - set(expected))
    if missing or unexpected:
        raise ValueError(f"CLMBR weights do not match the config: missing "
                         f"{missing[:5]}, unexpected {unexpected[:5]}")
    for key, value in ours.items():
        if tuple(value.shape) != tuple(expected[key].shape):
            raise ValueError(f"{key}: published shape {tuple(value.shape)}, "
                             f"config shape {tuple(expected[key].shape)}")
    model.load_state_dict(ours, strict=True)
    return sorted(skipped)


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------
def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    raise TypeError(f"event times must be datetimes, not {type(value).__name__}")


class CLMBRTokenizer:
    """A reimplementation of femr 0.2's (non-hierarchical) ``FEMRTokenizer``.

    The dictionary (``dictionary.msgpack`` in the model repo) is
    ``{"vocab": [entry, ...], "is_hierarchical": bool, "age_stats": {"mean",
    "std"}}``. An entry's index is its token id; entries are one of:

    - ``{"type": "code", "code_string"}``: the code without a value;
    - ``{"type": "numeric", "code_string", "val_start", "val_end"}``: a
      numeric value in ``[val_start, val_end)`` (half-open, unlike hf_ehr);
    - ``{"type": "text", "code_string", "text_string"}``: a text value;
    - any other type (e.g. ``"unused"``): never produced, but still counted
      in the ids.

    As in femr, an event with a numeric value maps only to a numeric entry,
    one with a text value only to a text entry, and one without a value only
    to a code entry; anything else is dropped. A token already produced on
    the same calendar day is dropped too.

    Args:
        dictionary: The parsed ``dictionary.msgpack``.

    Examples:
        >>> tokenizer = CLMBRTokenizer({"is_hierarchical": False,
        ...     "age_stats": {"mean": 0.0, "std": 1.0}, "vocab": [
        ...     {"type": "code", "code_string": "SNOMED/3950001"},
        ...     {"type": "numeric", "code_string": "LOINC/2236-8",
        ...      "val_start": float("-inf"), "val_end": 4.0},
        ...     {"type": "text", "code_string": "SNOMED/228490006",
        ...      "text_string": "N"}]})
        >>> tokenizer.event_to_id("LOINC/2236-8", "3.9"), tokenizer.event_to_id(
        ...     "LOINC/2236-8", 4.0)
        (1, None)
    """

    def __init__(self, dictionary: Mapping[str, Any]) -> None:
        if dictionary.get("is_hierarchical"):
            raise NotImplementedError("hierarchical femr dictionaries are not "
                                      "supported")
        vocab = dictionary["vocab"]
        self.dictionary = dictionary
        self.vocab_size = len(vocab)
        self._codes: dict[str, int] = {}
        self._numeric: dict[str, list[tuple[float, float, int]]] = {}
        self._text: dict[tuple[str, str], int] = {}
        for i, entry in enumerate(vocab):
            kind = entry.get("type")
            if kind == "code":
                self._codes[entry["code_string"]] = i
            elif kind == "numeric":
                self._numeric.setdefault(entry["code_string"], []).append(
                    (float(entry["val_start"]), float(entry["val_end"]), i))
            elif kind == "text":
                self._text[(entry["code_string"], entry["text_string"])] = i
            elif kind != "unused":
                raise ValueError(f"unknown femr dictionary entry type {kind!r} "
                                 f"at index {i}")
        stats = dictionary.get("age_stats") or {}
        #: Whether the dictionary sets the age normalization (required when
        #: the model uses normalized ages).
        self.has_age_stats = "mean" in stats and "std" in stats
        self.age_mean = float(stats.get("mean", 0.0))
        self.age_std = float(stats.get("std", 1.0)) or 1.0

    @classmethod
    def from_file(cls, path: str) -> CLMBRTokenizer:
        """Loads a ``dictionary.msgpack`` file (needs ``msgpack``).

        Examples:
            >>> CLMBRTokenizer.from_file("dictionary.msgpack")  # doctest: +SKIP
        """
        msgpack = _require("msgpack")
        with open(path, "rb") as f:
            return cls(msgpack.load(f))

    def __len__(self) -> int:
        return self.vocab_size

    def event_to_id(self, code: str, value: Any = None) -> int | None:
        """The token id of one event, or ``None`` (read with ``as_value``)."""
        value = as_value(value)
        if isinstance(value, float):
            for start, end, index in self._numeric.get(code, ()):
                if start <= value < end:
                    return index
            return None
        if isinstance(value, str):
            return self._text.get((code, value))
        return self._codes.get(code)

    def normalize_age(self, age: timedelta) -> float:
        """femr's normalized age: ``(seconds - mean) / std``."""
        return (age.total_seconds() - self.age_mean) / self.age_std

    def encode(self, events: Sequence[Sequence[Any]], birth: Any = None
               ) -> tuple[list[int], list[float], list[float], list[str]]:
        """Tokens, ages (days) and normalized ages of a timeline.

        Args:
            events: ``(timestamp, code, value, unit)`` tuples in time order.
            birth: The birth time; defaults to the first event's time.

        Returns:
            The token ids, their ages in days and normalized ages, and the
            codes of events that mapped to no token.
        """
        ids: list[int] = []
        ages: list[float] = []
        normed: list[float] = []
        unknown: list[str] = []
        if not events:
            return ids, ages, normed, unknown
        birth = _as_datetime(birth if birth is not None else events[0][0])
        day = None
        seen: set[int] = set()
        for event in events:
            when = _as_datetime(event[0])
            if when.date() != day:
                day, seen = when.date(), set()
            index = self.event_to_id(event[1], event[2] if len(event) > 2 else None)
            if index is None:
                unknown.append(event[1])
                continue
            if index in seen:
                continue
            seen.add(index)
            age = when - birth
            ids.append(index)
            ages.append(age / timedelta(days=1))
            normed.append(self.normalize_age(age))
        return ids, ages, normed, unknown


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class CLMBR(EHRFoundationModelBase):
    """CLMBR-T (EHRSHOT's foundation model) with a linear head.

    Pairs with :class:`~pyhealth.tasks.EHRFoundationModelTask` on
    :class:`~pyhealth.datasets.EHRShotDataset`. Each sample's events are
    tokenized with :class:`CLMBRTokenizer` (femr 0.2 rules), the most recent
    ``max_length`` tokens are kept, and the backbone
    (:class:`CLMBRTransformer`) embeds them with age-based rotary positions
    and local causal attention. The final hidden state of the last token, at
    or before the prediction time, is the patient representation, as femr's
    ``compute_features`` returns for a label.

    Access: ``StanfordShahLab/clmbr-t-base`` is gated on the Hugging Face Hub
    under CC-BY-NC-4.0. Request access on the model page (a current CITI
    "Data or Specimens Only Research" certificate, whose name must match the
    application, and the EHRSHOT Credentialed Health Data License), then log
    in with ``huggingface-cli login`` or pass ``token``. EHRSHOT itself is on
    Redivis under the EHRSHOT data-use agreement.

    Citation: Wornow et al., "EHRSHOT: An EHR Benchmark for Few-Shot
    Evaluation of Foundation Models", NeurIPS 2023 Datasets and Benchmarks.

    Args:
        dataset: The dataset; it needs the ``events`` (and ideally
            ``birth_datetime``) inputs of
            :class:`~pyhealth.tasks.EHRFoundationModelTask` and one label.
        model_name: Hugging Face repo id or local directory with
            ``config.json``, ``dictionary.msgpack`` and ``model.safetensors``.
        mode: ``"frozen"`` (linear probe on cached embeddings) or
            ``"finetune"``.
        max_length: Tokens read per timeline. Default 1024, the length femr
            0.2's ``compute_features`` uses (``tokens_per_batch=1024`` with
            one patient per batch); attention stays local within
            ``attention_width``.
        revision: Hub revision of ``model_name``.
        token: Hugging Face token for the gated repo.
        dtype: Backbone dtype, e.g. ``torch.bfloat16`` on a GPU.
        config: A :class:`CLMBRConfig` (or ``config.json`` contents) to use
            instead of the repo's.
        dictionary: A :class:`CLMBRTokenizer`, or the dictionary contents or
            path, to use instead of the repo's.
        state_dict: Published-format weights to load instead of the repo's
            ``model.safetensors``.
        random_init: Skip loading weights (tests and smoke runs).
        cache_dir: Directory for an on-disk embedding cache (frozen mode).
        cache_name: The backbone's identity in the cache key; defaults to
            ``model_name`` and ``revision``. Required with ``cache_dir`` when
            ``state_dict`` or ``random_init`` is given.
        memoize: Keep frozen embeddings in memory.
        events_key: The events input (default ``"events"``).

    Examples:
        >>> from datetime import datetime
        >>> from pyhealth.datasets import create_sample_dataset
        >>> t = datetime(2020, 1, 1)
        >>> samples = [
        ...     {"patient_id": "p0", "label": 1, "birth_datetime": t,
        ...      "events": [(t, "SNOMED/3950001", None, None),
        ...                 (t, "LOINC/2236-8", "3.0", "mg/dL")]},
        ...     {"patient_id": "p1", "label": 0, "birth_datetime": t,
        ...      "events": [(t, "SNOMED/3950001", None, None)]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, input_schema={"events": "raw", "birth_datetime": "raw"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> dictionary = {"is_hierarchical": False,
        ...     "age_stats": {"mean": 0.0, "std": 1.0}, "vocab": [
        ...     {"type": "code", "code_string": "SNOMED/3950001"},
        ...     {"type": "numeric", "code_string": "LOINC/2236-8",
        ...      "val_start": float("-inf"), "val_end": 4.0}]}
        >>> config = CLMBRConfig(vocab_size=2, hidden_size=8,
        ...                      intermediate_size=16, n_heads=2, n_layers=1)
        >>> model = CLMBR(dataset, config=config, dictionary=dictionary,
        ...               random_init=True)  # tiny and random; no download
        >>> out = model(events=[s["events"] for s in samples],
        ...             birth_datetime=[t, t], label=torch.tensor([[1.0], [0.0]]))
        >>> out["y_prob"].shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str = CLMBR_T_BASE,
        mode: str = "frozen",
        max_length: int = 1024,
        revision: str | None = None,
        token: str | bool | None = None,
        dtype: torch.dtype | None = None,
        config: CLMBRConfig | Mapping[str, Any] | None = None,
        dictionary: CLMBRTokenizer | Mapping[str, Any] | str | None = None,
        state_dict: Mapping[str, torch.Tensor] | None = None,
        random_init: bool = False,
        cache_dir: str | None = None,
        cache_name: str | None = None,
        memoize: bool = True,
        events_key: str | None = None,
    ) -> None:
        if cache_dir and cache_name is None and (state_dict is not None
                                                 or random_init):
            raise ValueError("pass cache_name with cache_dir when state_dict "
                             "or random_init is given")
        super().__init__(
            dataset, mode=mode, cache_dir=cache_dir,
            cache_name=cache_name or f"{model_name}@{revision}",
            memoize=memoize, events_key=events_key,
        )
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.model_name = model_name
        self.revision = revision
        self.max_length = int(max_length)

        def fetch(filename: str) -> str:
            if os.path.isdir(model_name):
                return os.path.join(model_name, filename)
            hub = _require("huggingface_hub")
            return hub.hf_hub_download(model_name, filename, revision=revision,
                                       token=token)

        if config is None:
            with open(fetch("config.json")) as f:
                config = CLMBRConfig.from_dict(json.load(f), strict=True)
        elif isinstance(config, Mapping):
            config = CLMBRConfig.from_dict(config)
        if dictionary is None:
            dictionary = CLMBRTokenizer.from_file(fetch("dictionary.msgpack"))
        elif isinstance(dictionary, str):
            dictionary = CLMBRTokenizer.from_file(dictionary)
        elif isinstance(dictionary, Mapping):
            dictionary = CLMBRTokenizer(dictionary)
        if config.use_normed_ages and not dictionary.has_age_stats:
            raise ValueError("the model uses normalized ages but the "
                             "dictionary has no age_stats {mean, std}")
        if len(dictionary) > config.vocab_size:
            raise ValueError(f"the dictionary has {len(dictionary)} entries but "
                             f"the model only {config.vocab_size} embeddings")
        self.config = config
        self.tokenizer = dictionary
        backbone = CLMBRTransformer(config)
        if not random_init:
            if state_dict is None:
                safetensors = _require("safetensors.torch")
                state_dict = safetensors.load_file(fetch("model.safetensors"))
            skipped = load_clmbr_weights(backbone, state_dict)
            logger.info("CLMBR: loaded %s; not loaded: %s", model_name, skipped)
        if dtype is not None:
            backbone.to(dtype)
        self.embedding_dim = config.hidden_size
        if mode == "frozen":
            backbone.eval()
            for p in backbone.parameters():
                p.requires_grad_(False)
            # not a submodule: not trained, not saved; follows the device
            self._frozen: list[nn.Module] = [backbone]
        else:
            self._frozen = []
            self.backbone = backbone
        self.fc = nn.Linear(self.embedding_dim, self.label_processor.output_size())

    @property
    def encoder(self) -> CLMBRTransformer:
        """The backbone, whether frozen or fine-tuned."""
        return self._frozen[0] if self._frozen else self.backbone

    def cache_settings(self) -> str:
        config = json.dumps(asdict(self.config), sort_keys=True)
        return (f"{self.max_length}|{config}|"
                f"{next(self.encoder.parameters()).dtype}")

    def tokenize(self, events: Sequence[Sequence[Any]], birth: Any = None
                 ) -> TokenizedTimeline:
        """femr tokens of one timeline, truncated to the most recent ones.

        Examples:
            >>> model.tokenize([(t, "SNOMED/3950001", None, None)]).ids  # doctest: +SKIP
            [0]
        """
        ids, ages, normed, unknown = self.tokenizer.encode(events, birth)
        start = max(0, len(ids) - self.max_length)
        return TokenizedTimeline(ids=ids[start:], ages=ages[start:],
                                 extra=normed[start:], n_events=len(events),
                                 n_unknown=len(unknown), unknown_codes=unknown)

    def encode(self, timelines: Sequence[TokenizedTimeline]) -> torch.Tensor:
        """The last real token's final hidden state of each timeline.

        A timeline with no tokens gets a zero vector.
        """
        encoder = self.encoder
        device = self.device
        if next(encoder.parameters()).device != device:
            encoder.to(device)
        lengths = torch.tensor([len(t.ids) for t in timelines], dtype=torch.long)
        width = max(1, int(lengths.max()))
        tokens = torch.zeros((len(timelines), width), dtype=torch.long)
        ages = torch.zeros((len(timelines), width), dtype=torch.float32)
        normed = torch.zeros((len(timelines), width), dtype=torch.float32)
        for row, timeline in enumerate(timelines):
            n = len(timeline.ids)
            if n:
                tokens[row, :n] = torch.tensor(timeline.ids)
                ages[row, :n] = torch.tensor(timeline.ages, dtype=torch.float32)
                normed[row, :n] = torch.tensor(timeline.extra, dtype=torch.float32)
        hidden = encoder(tokens.to(device), ages.to(device), normed.to(device),
                         lengths.to(device))
        last = (lengths - 1).clamp(min=0).to(device)
        out = hidden[torch.arange(len(timelines), device=device), last]
        empty = (lengths == 0).to(device)
        return torch.where(empty[:, None], torch.zeros_like(out), out)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Predicts the label of each sample from its raw event timeline.

        The batch is what :func:`~pyhealth.datasets.get_dataloader` yields
        for a :class:`~pyhealth.tasks.EHRFoundationModelTask` dataset, so
        :class:`~pyhealth.trainer.Trainer` can call it directly. All
        FM-specific processing happens here, not in the task:

        1. each timeline's events are mapped to femr dictionary ids and ages
           since birth (:class:`CLMBRTokenizer`), keeping the most recent
           ``max_length`` tokens;
        2. the timelines run through :class:`CLMBRTransformer` (without
           gradients and through the cache in ``"frozen"`` mode);
        3. the last token's final hidden state goes through the linear head.

        Args:
            **kwargs: The batch. ``events`` (one list of ``(timestamp, code,
                value, unit)`` per sample) is required, and
                ``birth_datetime`` sets the ages (else the first event's
                time). The label (e.g. ``label``) is optional, so the model
                also predicts on unlabeled timelines.

        Returns:
            A dict with ``logit``, ``y_prob`` and ``embed``, plus ``loss`` and
            ``y_true`` when the label is in the batch.

        Examples:
            >>> loader = get_dataloader(dataset, batch_size=32)  # doctest: +SKIP
            >>> Trainer(model=model, metrics=["roc_auc"]).train(  # doctest: +SKIP
            ...     train_dataloader=loader, epochs=10)
        """
        births = kwargs.get(self.birth_key) if self.birth_key else None
        embeddings = self.embed_events(kwargs[self.events_key], births)
        return self.apply_head(embeddings, kwargs.get(self.label_key))
