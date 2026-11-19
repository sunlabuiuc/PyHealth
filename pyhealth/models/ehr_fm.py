"""Pretrained EHR foundation models (hf_ehr checkpoints) with a linear head.

This wraps the autoregressive EHR foundation models of Wornow et al.,
"Context Clues: Evaluating Long Context Models for Clinical Prediction Tasks
on EHRs" (2024; https://arxiv.org/abs/2412.16178), released by the Stanford
Shah Lab as Hugging Face checkpoints (``StanfordShahLab/gpt-base-1024-clmbr``
and its gpt / llama / mamba / hyena siblings). They were pretrained on 2.57
million Stanford Medicine EHRs with the CLMBR vocabulary of CLMBR-T (Wornow
et al., "EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation
Models", NeurIPS 2023; https://arxiv.org/abs/2307.02028).

Reference code: https://github.com/som-shahlab/hf_ehr (Apache-2.0). The
tokenizer here is a reimplementation of its ``CLMBRTokenizer``, so the
``hf_ehr`` package (which pins many heavy dependencies) is not needed.

All FM-specific processing happens in ``forward()``: the samples carry raw
``(timestamp, code, value, unit)`` timelines from
:class:`~pyhealth.tasks.EHRFoundationModelTask`, and the model maps them to
its own vocabulary, truncates, pads, runs the backbone and pools the last
token, as hf_ehr's EHRSHOT evaluation does.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from ..datasets import SampleDataset
from .base_model import BaseModel
from .utils import save_array_atomically

logger = logging.getLogger(__name__)

FM_MODES = ("frozen", "finetune")

_FM_EXTRA_HINT = (
    "it ships with PyHealth's transformers dependency; reinstall it with "
    "`pip install transformers huggingface-hub safetensors`"
)


def _require(module: str) -> Any:
    """Imports an optional dependency, with an install hint if missing."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(f"{module} is not installed: {_FM_EXTRA_HINT}") from e


def as_value(value: Any) -> float | str | None:
    """Reads an event value as a number, a text value or ``None``.

    Numbers (and strings that parse as finite numbers, since CSV exports such
    as EHRSHOT's write every value as text) become ``float``; other non-empty
    strings stay text. ``None``, ``NaN``, empty strings and booleans are
    ``None``.

    Examples:
        >>> as_value("26.3"), as_value(4), as_value("N"), as_value("")
        (26.3, 4.0, 'N', None)
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return None if math.isnan(value) else value
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return number if math.isfinite(number) else text


@dataclass
class TokenizedTimeline:
    """One sample's FM input.

    Attributes:
        ids: Token ids, oldest first, already truncated.
        ages: Optional per-token ages (models with age embeddings).
        extra: Optional per-token features, e.g. normalized ages.
        n_events: Events read.
        n_unknown: Events that mapped to no token.
        unknown_codes: The codes of those events.

    Examples:
        >>> timeline = TokenizedTimeline(ids=[7, 9], n_events=3, n_unknown=1)
        >>> timeline.key() == TokenizedTimeline(ids=[7, 9]).key()
        True
    """

    ids: list[int]
    ages: list[float] | None = None
    extra: list[float] | None = None
    n_events: int = 0
    n_unknown: int = 0
    unknown_codes: list[str] = field(default_factory=list)

    def key(self) -> bytes:
        """The bytes that identify this input in the embedding cache."""
        parts = [np.asarray(self.ids, dtype=np.int64).tobytes()]
        for values in (self.ages, self.extra):
            parts.append(b"|" if values is None
                         else np.asarray(values, dtype=np.float64).tobytes())
        return b"#".join(parts)


class EHRFoundationModelBase(BaseModel):
    """Shared plumbing for pretrained EHR foundation models.

    Subclasses implement :meth:`tokenize` (one sample's events to a
    :class:`TokenizedTimeline`) and :meth:`encode` (a batch of them to one
    embedding each). This class reads the ``events`` and ``birth_datetime``
    inputs of :class:`~pyhealth.tasks.EHRFoundationModelTask`, keeps the
    vocabulary coverage counts, caches frozen embeddings and applies the
    linear head and the label's loss.

    In ``"frozen"`` mode the backbone runs under ``torch.no_grad()`` and only
    the head trains (a linear probe). The backbone is then not a submodule,
    so checkpoints hold the head only, and embeddings are memoized in memory
    (``memoize``) and, with ``cache_dir``, on disk, keyed by the model's
    identity and the sample's token ids, so a cached vector is exactly the
    one the backbone would return. In ``"finetune"`` mode the backbone is a
    submodule and trains with the head; nothing is cached.

    Args:
        dataset: The dataset; it needs a ``"raw"`` events input and one
            label.
        mode: ``"frozen"`` or ``"finetune"``. Stored as :attr:`fm_mode`
            (``BaseModel.mode`` is the label kind).
        cache_dir: Directory for an on-disk embedding cache (frozen mode).
        cache_name: The backbone's identity in the cache key.
        memoize: Keep frozen embeddings in memory across calls and epochs.
        events_key: The events input. Defaults to ``"events"``.
        birth_key: The birth time input. Defaults to ``"birth_datetime"`` if
            the dataset has it.

    Examples:
        >>> class CountModel(EHRFoundationModelBase):  # a toy "backbone"
        ...     def __init__(self, dataset):
        ...         super().__init__(dataset)
        ...         self.fc = torch.nn.Linear(1, 1)
        ...     def tokenize(self, events, birth=None):
        ...         return TokenizedTimeline(ids=[0] * len(events),
        ...                                  n_events=len(events))
        ...     def encode(self, timelines):
        ...         return torch.tensor([[float(len(t.ids))] for t in timelines])
        >>> from pyhealth.datasets import create_sample_dataset
        >>> dataset = create_sample_dataset(
        ...     [{"patient_id": "p0", "events": [(0, "A/1", None, None)],
        ...       "label": 1},
        ...      {"patient_id": "p1", "events": [], "label": 0}],
        ...     input_schema={"events": "raw"}, output_schema={"label": "binary"})
        >>> CountModel(dataset)(events=[[(0, "A/1", None, None)]])["embed"]
        tensor([[1.]])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        mode: str = "frozen",
        cache_dir: str | None = None,
        cache_name: str | None = None,
        memoize: bool = True,
        events_key: str | None = None,
        birth_key: str | None = None,
    ) -> None:
        super().__init__(dataset)
        if mode not in FM_MODES:
            raise ValueError(f"mode must be one of {FM_MODES}, not {mode!r}")
        if len(self.label_keys) != 1:
            raise ValueError(f"{type(self).__name__} supports exactly one label")
        events_key = events_key or "events"
        if events_key not in self.feature_keys:
            raise ValueError(
                f"{events_key!r} is not an input of the dataset; build it with "
                "EHRFoundationModelTask"
            )
        if birth_key is None and "birth_datetime" in self.feature_keys:
            birth_key = "birth_datetime"
        self.events_key = events_key
        self.birth_key = birth_key
        self.label_key = self.label_keys[0]
        self.fm_mode = mode
        self.cache_dir = cache_dir if mode == "frozen" else None
        self.cache_name = cache_name
        self.memoize = memoize and mode == "frozen"
        self._memo: dict[str, torch.Tensor] = {}
        self.coverage: Counter[str] = Counter()
        self.unknown_codes: Counter[str] = Counter()
        self._coverage_logged = False

    # ------------------------------------------------------- subclass hooks
    def tokenize(self, events: Sequence[Sequence[Any]], birth: Any = None
                 ) -> TokenizedTimeline:
        """Maps one sample's events to the FM's input (subclasses)."""
        raise NotImplementedError

    def encode(self, timelines: Sequence[TokenizedTimeline]) -> torch.Tensor:
        """Embeds tokenized timelines, one vector each (subclasses)."""
        raise NotImplementedError

    def cache_settings(self) -> str:
        """Settings that change the embeddings, for the cache key."""
        return ""

    # ------------------------------------------------------------ coverage
    def _count(self, timeline: TokenizedTimeline) -> None:
        self.coverage["samples"] += 1
        self.coverage["events"] += timeline.n_events
        self.coverage["unknown_events"] += timeline.n_unknown
        self.coverage["tokens"] += len(timeline.ids)
        self.unknown_codes.update(timeline.unknown_codes)

    def coverage_report(self, top: int = 10) -> dict[str, Any]:
        """How many events mapped to the FM's vocabulary so far.

        Args:
            top: How many of the most frequent unknown codes to list.

        Returns:
            A dict with the ``samples``, ``events``, ``unknown_events`` and
            ``tokens`` (after truncation) seen by ``forward``, the
            ``coverage`` fraction of events that mapped to a token, and the
            ``top_unknown`` codes with their counts.

        Examples:
            >>> model.coverage_report()["coverage"]  # doctest: +SKIP
            0.97
        """
        events = self.coverage["events"]
        unknown = self.coverage["unknown_events"]
        return {
            **{k: self.coverage[k] for k in
               ("samples", "events", "unknown_events", "tokens")},
            "coverage": (events - unknown) / events if events else None,
            "top_unknown": self.unknown_codes.most_common(top),
        }

    # ------------------------------------------------------------- caching
    def _cache_key(self, timeline: TokenizedTimeline) -> str:
        settings = f"{type(self).__name__}|{self.cache_name}|{self.cache_settings()}|"
        return hashlib.sha256(settings.encode() + timeline.key()).hexdigest()

    def _cache_path(self, key: str) -> str | None:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, key[:2], f"{key}.npy")

    @torch.no_grad()
    def _embed_frozen(self, timelines: Sequence[TokenizedTimeline]) -> torch.Tensor:
        keys = [self._cache_key(t) for t in timelines]
        found: dict[str, torch.Tensor] = {}
        todo: dict[str, TokenizedTimeline] = {}
        for key, timeline in zip(keys, timelines):
            if key in found or key in todo:
                continue
            if key in self._memo:
                found[key] = self._memo[key]
                continue
            path = self._cache_path(key)
            if path and os.path.exists(path):
                found[key] = torch.from_numpy(np.load(path))
            else:
                todo[key] = timeline
        if todo:
            vectors = self.encode(list(todo.values())).float().cpu()
            for key, vector in zip(todo, vectors):
                found[key] = vector
                path = self._cache_path(key)
                if path:
                    save_array_atomically(path, vector.numpy(), mode=0o600)
        if self.memoize:
            self._memo.update(found)
        return torch.stack([found[k] for k in keys]).to(self.device)

    # ------------------------------------------------------------- forward
    def embed_events(
        self,
        events: Sequence[Sequence[Sequence[Any]] | None],
        births: Sequence[Any] | None = None,
    ) -> torch.Tensor:
        """Tokenizes a batch of timelines and embeds them, one row each.

        In ``"frozen"`` mode this runs without gradients and reads and fills
        the embedding cache; in ``"finetune"`` mode gradients flow into the
        backbone. Also counts vocabulary coverage (:meth:`coverage_report`).

        Args:
            events: One list of ``(timestamp, code, value, unit)`` per sample.
            births: One birth time per sample, or ``None``.

        Returns:
            A float tensor of shape ``(batch, embedding_dim)``.
        """
        if births is None:
            births = [None] * len(events)
        timelines = [self.tokenize(sample or [], birth)
                     for sample, birth in zip(events, births)]
        for timeline in timelines:
            self._count(timeline)
        if not self._coverage_logged:
            report = self.coverage_report(top=5)
            logger.info("%s vocabulary coverage of the first batch: %s",
                        type(self).__name__, report)
            self._coverage_logged = True
        if self.fm_mode == "frozen":
            return self._embed_frozen(timelines)
        return self.encode(timelines).float()

    def apply_head(self, embeddings: torch.Tensor,
                y_true: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Applies the linear head, and the label's loss if ``y_true`` is given.

        Returns:
            ``logit``, ``y_prob`` and ``embed``, plus ``loss`` and ``y_true``
            when ``y_true`` is given.
        """
        logits = self.fc(embeddings)
        out = {"logit": logits, "y_prob": self.label_processor.activation(logits),
               "embed": embeddings}
        if y_true is not None:
            y_true = y_true.to(self.device)
            out["loss"] = self.label_processor.loss(logits, y_true)
            out["y_true"] = y_true
        return out

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Embeds the batch's timelines and applies the linear head.

        Returns:
            ``logit``, ``y_prob`` and ``embed``, plus ``loss`` and ``y_true``
            when the label is in the batch.
        """
        births = kwargs.get(self.birth_key) if self.birth_key else None
        embeddings = self.embed_events(kwargs[self.events_key], births)
        return self.apply_head(embeddings, kwargs.get(self.label_key))


# ---------------------------------------------------------------------------
# hf_ehr CLMBR tokenizer
# ---------------------------------------------------------------------------
HF_EHR_SPECIAL_TOKENS: tuple[str, ...] = (
    "[BOS]", "[EOS]", "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]",
)


class HFEHRTokenizer:
    """A reimplementation of hf_ehr's ``CLMBRTokenizer``.

    The hf_ehr checkpoints ship a ``tokenizer_config.json`` of the form
    ``{"timestamp": ..., "metadata": {...}, "tokens": [entry, ...]}``, each
    entry ``{"code", "type", "description", "tokenization", "stats"}`` with
    ``type`` one of:

    - ``"code"``: the code alone, token string ``"LOINC/2236-8"``;
    - ``"numerical_range"``: ``tokenization = {"unit", "range_start",
      "range_end"}``, token string ``"CODE || unit || start - end"``;
    - ``"categorical"``: ``tokenization = {"categories": [...]}``, token
      string ``"CODE || cat1,cat2"``.

    Token ids are the position in ``[BOS] [EOS] [UNK] [SEP] [PAD] [CLS]
    [MASK]`` followed by the entries' token strings (``[PAD]`` is 4). An event
    becomes (as in hf_ehr, which ignores units for CLMBR):

    1. nothing if its code is not in the vocabulary;
    2. if the code has numeric ranges and the value is a number, the first
       range with ``range_start <= value <= range_end``, or nothing;
    3. if the code has categories and the value is non-empty text, the first
       entry listing it, or nothing;
    4. else the code's ``"code"`` token, or nothing.

    Unmapped events are dropped, not mapped to ``[UNK]``.

    Args:
        config: The parsed ``tokenizer_config.json`` (or just its
            ``tokens`` list).

    Examples:
        >>> tokenizer = HFEHRTokenizer({"tokens": [
        ...     {"code": "SNOMED/3950001", "type": "code", "tokenization": {}},
        ...     {"code": "LOINC/2236-8", "type": "numerical_range",
        ...      "tokenization": {"unit": "None", "range_start": -1e308,
        ...                       "range_end": 4.0}},
        ... ]})
        >>> tokenizer.event_to_id("LOINC/2236-8", -3.0), tokenizer.pad_token_id
        (8, 4)
        >>> tokenizer.event_to_id("LOINC/2236-8", 9.0) is None
        True
    """

    def __init__(self, config: Mapping[str, Any] | Sequence[Mapping[str, Any]]):
        entries = config["tokens"] if isinstance(config, Mapping) else config
        vocab = list(HF_EHR_SPECIAL_TOKENS)
        self._codes: dict[str, int] = {}
        self._ranges: dict[str, list[tuple[float, float, int]]] = {}
        self._categories: dict[str, list[tuple[frozenset[str], int]]] = {}
        strings = []
        for entry in entries:
            strings.append(self._token_string(entry))
        vocab.extend(strings)
        # as hf_ehr's token_2_idx: a repeated token string keeps its last id
        token_to_id = {token: i for i, token in enumerate(vocab)}
        for entry, token in zip(entries, strings):
            code, kind = entry["code"], entry["type"]
            index = token_to_id[token]
            spec = entry.get("tokenization") or {}
            if kind == "code":
                self._codes.setdefault(code, index)
            elif kind == "numerical_range":
                self._ranges.setdefault(code, []).append(
                    (float(spec["range_start"]), float(spec["range_end"]), index))
            elif kind == "categorical":
                self._categories.setdefault(code, []).append(
                    (frozenset(spec.get("categories", [])), index))
            else:
                raise ValueError(f"unknown hf_ehr token type {kind!r} for {code}")
        self.vocab = vocab
        self.token_to_id = token_to_id
        self.pad_token_id = token_to_id["[PAD]"]
        self.bos_token_id = token_to_id["[BOS]"]
        self.eos_token_id = token_to_id["[EOS]"]
        self.cls_token_id = token_to_id["[CLS]"]

    @staticmethod
    def _token_string(entry: Mapping[str, Any]) -> str:
        code, kind = entry["code"], entry["type"]
        spec = entry.get("tokenization") or {}
        if kind == "numerical_range":
            return (f"{code} || {spec['unit']} || "
                    f"{spec['range_start']} - {spec['range_end']}")
        if kind == "categorical":
            return f"{code} || {','.join(spec['categories'])}"
        return f"{code}"

    @classmethod
    def from_file(cls, path: str) -> HFEHRTokenizer:
        """Loads a ``tokenizer_config.json`` file.

        Examples:
            >>> HFEHRTokenizer.from_file("tokenizer_config.json")  # doctest: +SKIP
        """
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: str | None = None,
                        token: str | bool | None = None) -> HFEHRTokenizer:
        """Downloads ``tokenizer_config.json`` from a Hugging Face repo.

        The StanfordShahLab repos are gated: accept their terms on the Hub
        and log in (``huggingface-cli login``) or pass ``token``.

        Examples:
            >>> HFEHRTokenizer.from_pretrained(  # doctest: +SKIP
            ...     "StanfordShahLab/gpt-base-1024-clmbr")
        """
        if os.path.isdir(repo_id):
            return cls.from_file(os.path.join(repo_id, "tokenizer_config.json"))
        hub = _require("huggingface_hub")
        path = hub.hf_hub_download(repo_id, "tokenizer_config.json",
                                   revision=revision, token=token)
        return cls.from_file(path)

    def __len__(self) -> int:
        return len(self.vocab)

    def event_to_id(self, code: str, value: Any = None) -> int | None:
        """The token id of one event, or ``None`` if it has none.

        ``value`` is read with :func:`as_value`.
        """
        value = as_value(value)
        if isinstance(value, float) and code in self._ranges:
            for start, end, index in self._ranges[code]:
                if start <= value <= end:
                    return index
            return None
        if isinstance(value, str) and code in self._categories:
            for categories, index in self._categories[code]:
                if value in categories:
                    return index
            return None
        return self._codes.get(code)

    def encode(self, events: Sequence[Sequence[Any]]
               ) -> tuple[list[int], list[str]]:
        """Token ids of ``(timestamp, code, value, unit)`` events.

        Returns:
            The ids, in event order, and the codes of unmapped events.
        """
        ids: list[int] = []
        unknown: list[str] = []
        for event in events:
            code = event[1]
            value = event[2] if len(event) > 2 else None
            index = self.event_to_id(code, value)
            if index is None:
                unknown.append(code)
            else:
                ids.append(index)
        return ids, unknown


def _context_length(config: Any, model_name: str | None) -> int | None:
    """The context length from the config, else from an hf_ehr repo name."""
    for attr in ("n_positions", "max_position_embeddings"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    match = re.search(r"-(\d+)-clmbr", model_name or "")
    return int(match.group(1)) if match else None


class HFEHRFoundationModel(EHRFoundationModelBase):
    """An hf_ehr EHR foundation model (gpt / llama / mamba / hyena) with a
    linear head.

    Pairs with :class:`~pyhealth.tasks.EHRFoundationModelTask`, e.g. on
    :class:`~pyhealth.datasets.EHRShotDataset`, whose OMOP codes are the
    ones the models were trained on. Each sample's events are tokenized with
    :class:`HFEHRTokenizer`, truncated to the most recent ``max_length``
    tokens, left-padded with ``[PAD]`` and embedded; the last token's final
    hidden state is the patient representation, as in hf_ehr's EHRSHOT
    evaluation (``hf_ehr/eval/ehrshot.py``: no special tokens, the last
    ``max_length`` tokens, left padding to ``max_length``, last-token
    embedding).

    Positions: hf_ehr pads every timeline on the left to ``max_length``
    without position ids, so a timeline of ``n`` tokens sits at positions
    ``max_length - n`` to ``max_length - 1``. For backbones that take
    ``position_ids`` (gpt2, llama) those positions are passed explicitly and
    the batch is padded only to its longest timeline, which gives the same
    embeddings with less compute. Other backbones (mamba, hyena) are padded
    to ``max_length``.

    Values: EHRSHOT stores every value as text, so a value that parses as a
    number is read as one (:func:`as_value`). A numeric-looking category,
    e.g. ``"1"`` on a code with only categorical entries, therefore matches
    no category and becomes the code's plain token (or nothing).

    Training: ``forward()`` returns ``loss``, ``y_prob``, ``y_true`` and
    ``logit`` like every PyHealth model, so :class:`~pyhealth.trainer.Trainer`
    trains and evaluates it directly. In ``"frozen"`` mode only the linear
    head has gradients; in ``"finetune"`` mode the backbone trains too (use
    a small learning rate, e.g. ``optimizer_params={"lr": 1e-5}``). The
    model classifies or regresses from a patient embedding; it does not
    generate events.

    Access: the checkpoints are gated on the Hugging Face Hub under
    CC-BY-NC-4.0. Request access on the model page (it requires a current
    CITI "Data or Specimens Only Research" certificate and agreeing to the
    EHRSHOT Credentialed Health Data License), then log in with
    ``huggingface-cli login`` or pass ``token``. EHRSHOT itself is on
    Redivis under the EHRSHOT data-use agreement (see
    :class:`~pyhealth.datasets.EHRShotDataset`).

    Citation: Wornow et al., "Context Clues: Evaluating Long Context Models
    for Clinical Prediction Tasks on EHRs", arXiv:2412.16178, 2024.

    Args:
        dataset: The dataset; it needs the ``events`` input of
            :class:`~pyhealth.tasks.EHRFoundationModelTask` and one label.
        model_name: Hugging Face repo id or local directory, e.g.
            ``"StanfordShahLab/gpt-base-1024-clmbr"``,
            ``"StanfordShahLab/llama-base-4096-clmbr"``,
            ``"StanfordShahLab/mamba-tiny-16384-clmbr"`` or
            ``"StanfordShahLab/hyena-large-1024-clmbr"``.
        mode: ``"frozen"`` (linear probe on cached embeddings) or
            ``"finetune"`` (train the backbone too).
        max_length: Tokens read per timeline. Defaults to the checkpoint's
            context length (``n_positions`` or ``max_position_embeddings``
            of its config, else the number in its name, e.g. 16384 for
            ``mamba-tiny-16384-clmbr``).
        add_special_tokens: Wrap timelines in ``[CLS] [BOS] ... [EOS]`` as in
            hf_ehr's pretraining. Default ``False``, as in its EHRSHOT
            evaluation.
        revision: Hub revision (commit) of ``model_name``.
        token: Hugging Face token for the gated repos.
        trust_remote_code: Needed for the hyena checkpoints, whose code
            lives in ``LongSafari/hyenadna-*`` repos.
        model_kwargs: Extra ``from_pretrained`` arguments, e.g.
            ``{"dtype": torch.bfloat16}``.
        backbone: A loaded ``transformers`` model to use instead of
            ``model_name`` (with ``tokenizer``), e.g. a small one in tests.
        tokenizer: An :class:`HFEHRTokenizer` (or the path to, or contents
            of, a ``tokenizer_config.json``).
        cache_dir: Directory for an on-disk embedding cache (frozen mode).
        cache_name: The backbone's identity in the cache key; defaults to
            ``model_name`` and ``revision``. Required with ``cache_dir`` when
            ``backbone`` is given.
        memoize: Keep frozen embeddings in memory.
        events_key: The events input (default ``"events"``).

    Examples:
        >>> from datetime import datetime
        >>> from transformers import GPT2Config, GPT2Model
        >>> from pyhealth.datasets import create_sample_dataset
        >>> t = datetime(2020, 1, 1)
        >>> samples = [
        ...     {"patient_id": "p0", "label": 1, "birth_datetime": t,
        ...      "events": [(t, "SNOMED/3950001", None, None),
        ...                 (t, "LOINC/2236-8", "-3.0", "mg/dL")]},
        ...     {"patient_id": "p1", "label": 0, "birth_datetime": t,
        ...      "events": [(t, "SNOMED/3950001", None, None)]},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, input_schema={"events": "raw", "birth_datetime": "raw"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> tokenizer = HFEHRTokenizer({"tokens": [
        ...     {"code": "SNOMED/3950001", "type": "code", "tokenization": {}},
        ...     {"code": "LOINC/2236-8", "type": "numerical_range",
        ...      "tokenization": {"unit": "None", "range_start": -1e308,
        ...                       "range_end": 4.0}},
        ... ]})
        >>> backbone = GPT2Model(GPT2Config(  # a tiny random model; no download
        ...     vocab_size=len(tokenizer), n_positions=16, n_embd=8,
        ...     n_layer=1, n_head=2))
        >>> model = HFEHRFoundationModel(dataset, backbone=backbone,
        ...                              tokenizer=tokenizer, max_length=16)
        >>> out = model(events=[s["events"] for s in samples],
        ...             birth_datetime=[t, t], label=torch.tensor([[1.0], [0.0]]))
        >>> out["y_prob"].shape, model.coverage_report()["coverage"]
        (torch.Size([2, 1]), 1.0)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        model_name: str = "StanfordShahLab/gpt-base-1024-clmbr",
        mode: str = "frozen",
        max_length: int | None = None,
        add_special_tokens: bool = False,
        revision: str | None = None,
        token: str | bool | None = None,
        trust_remote_code: bool = False,
        model_kwargs: Mapping[str, Any] | None = None,
        backbone: nn.Module | None = None,
        tokenizer: HFEHRTokenizer | str | Mapping[str, Any] | None = None,
        cache_dir: str | None = None,
        cache_name: str | None = None,
        memoize: bool = True,
        events_key: str | None = None,
    ) -> None:
        if cache_dir and cache_name is None and backbone is not None:
            raise ValueError(
                "pass cache_name with cache_dir when backbone is given: "
                "model_name does not identify a custom backbone"
            )
        super().__init__(
            dataset, mode=mode, cache_dir=cache_dir,
            cache_name=cache_name or f"{model_name}@{revision}",
            memoize=memoize, events_key=events_key,
        )
        self.model_name = model_name
        self.revision = revision
        self.add_special_tokens = add_special_tokens
        if tokenizer is None:
            tokenizer = HFEHRTokenizer.from_pretrained(
                model_name, revision=revision, token=token)
        elif isinstance(tokenizer, str):
            tokenizer = HFEHRTokenizer.from_file(tokenizer)
        elif isinstance(tokenizer, Mapping):
            tokenizer = HFEHRTokenizer(tokenizer)
        self.tokenizer = tokenizer
        custom = backbone is not None
        if backbone is None:
            transformers = _require("transformers")
            backbone = transformers.AutoModel.from_pretrained(
                model_name, revision=revision, token=token,
                trust_remote_code=trust_remote_code, **dict(model_kwargs or {}))
        config = getattr(backbone, "config", None)
        if max_length is None:
            max_length = _context_length(
                config, None if custom else model_name)
            if max_length is None:
                raise ValueError("pass max_length: the backbone's context "
                                 "length could not be read")
        self.max_length = int(max_length)
        limit = _context_length(config, None)
        if limit is not None and self.max_length > limit:
            raise ValueError(f"max_length {self.max_length} is longer than the "
                             f"backbone's {limit} positions")
        reserved = 3 if add_special_tokens else 0
        if self.max_length <= reserved:
            raise ValueError("max_length is too short")
        params = inspect.signature(backbone.forward).parameters
        self._takes_positions = "position_ids" in params
        self._takes_mask = "attention_mask" in params
        hidden = (getattr(config, "hidden_size", None)
                  or getattr(config, "n_embd", None)
                  or getattr(config, "d_model", None))
        if hidden is None:
            raise ValueError("the backbone's config has no hidden size")
        self.embedding_dim = int(hidden)
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
    def encoder(self) -> nn.Module:
        """The backbone, whether frozen or fine-tuned."""
        return self._frozen[0] if self._frozen else self.backbone

    def cache_settings(self) -> str:
        return (f"{self.max_length}|{self.add_special_tokens}|"
                f"{next(self.encoder.parameters()).dtype}")

    def tokenize(self, events: Sequence[Sequence[Any]], birth: Any = None
                 ) -> TokenizedTimeline:
        """Token ids of one timeline, truncated to the most recent tokens.

        ``birth`` is unused: the hf_ehr models take no ages. It is part of the
        :class:`EHRFoundationModelBase` interface for models that do (e.g.
        CLMBR-T's age positions).

        Examples:
            >>> model.tokenize([(t, "SNOMED/3950001", None, None)]).ids  # doctest: +SKIP
            [7]
        """
        ids, unknown = self.tokenizer.encode(events)
        tok = self.tokenizer
        if self.add_special_tokens:
            ids = ids[-(self.max_length - 3):]
            ids = [tok.cls_token_id, tok.bos_token_id, *ids, tok.eos_token_id]
        else:
            ids = ids[-self.max_length:]
        if not ids:
            # hf_ehr turns an empty timeline into a single [PAD]
            ids = [tok.pad_token_id]
        return TokenizedTimeline(ids=ids, n_events=len(events),
                                 n_unknown=len(unknown), unknown_codes=unknown)

    def encode(self, timelines: Sequence[TokenizedTimeline]) -> torch.Tensor:
        """Last-token final hidden states of left-padded timelines."""
        encoder = self.encoder
        device = self.device
        if next(encoder.parameters()).device != device:
            encoder.to(device)
        pad = self.tokenizer.pad_token_id
        width = (max(len(t.ids) for t in timelines) if self._takes_positions
                 else self.max_length)
        input_ids = torch.full((len(timelines), width), pad, dtype=torch.long)
        mask = torch.zeros((len(timelines), width), dtype=torch.long)
        positions = torch.zeros((len(timelines), width), dtype=torch.long)
        for row, timeline in enumerate(timelines):
            n = len(timeline.ids)
            input_ids[row, width - n:] = torch.tensor(timeline.ids)
            mask[row, width - n:] = 1
            # as if left-padded to max_length (see the class docstring)
            positions[row, width - n:] = torch.arange(self.max_length - n,
                                                      self.max_length)
        inputs: dict[str, torch.Tensor] = {"input_ids": input_ids.to(device)}
        if self._takes_mask:
            inputs["attention_mask"] = mask.to(device)
        if self._takes_positions:
            inputs["position_ids"] = positions.to(device)
        out = encoder(**inputs, output_hidden_states=True)
        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None and isinstance(out, (tuple, list)):
            hidden_states = out[1] if len(out) > 1 else None
        if hidden_states:
            hidden = hidden_states[-1]
        else:
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return hidden[:, -1, :]

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Predicts the label of each sample from its raw event timeline.

        The batch is what :func:`~pyhealth.datasets.get_dataloader` yields
        for a :class:`~pyhealth.tasks.EHRFoundationModelTask` dataset, so
        :class:`~pyhealth.trainer.Trainer` can call it directly. All
        FM-specific processing happens here, not in the task:

        1. each timeline's events are mapped to hf_ehr token ids
           (:class:`HFEHRTokenizer`), keeping the most recent ``max_length``;
        2. the timelines are left-padded and run through the backbone
           (without gradients and through the cache in ``"frozen"`` mode);
        3. the last token's final hidden state goes through the linear head.

        Args:
            **kwargs: The batch. ``events`` (one list of ``(timestamp, code,
                value, unit)`` per sample) is required; the label (e.g.
                ``label``) is optional, so the model also predicts on
                unlabeled timelines. Other keys, such as ``patient_id`` or
                ``birth_datetime``, are ignored.

        Returns:
            A dict with ``logit`` and ``y_prob`` (``(batch, output_size)``)
            and ``embed`` (``(batch, embedding_dim)``, the patient
            representation), plus ``loss`` and ``y_true`` when the label is
            in the batch.

        Examples:
            >>> loader = get_dataloader(dataset, batch_size=32)  # doctest: +SKIP
            >>> Trainer(model=model, metrics=["roc_auc"]).train(  # doctest: +SKIP
            ...     train_dataloader=loader, epochs=10)
            >>> model(events=[timeline])["y_prob"]  # unlabeled  # doctest: +SKIP
            tensor([[0.31]])
        """
        embeddings = self.embed_events(kwargs[self.events_key])
        return self.apply_head(embeddings, kwargs.get(self.label_key))
