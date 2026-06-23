"""Concept vocabulary and CEHR feature processor for FHIR timelines.

Public API
----------
ConceptVocab
    Token-to-dense-id mapping with PAD/UNK reserved at 0 and 1. JSON-serialisable.
ensure_special_tokens(vocab)
    Add CEHR/MPF specials (``<cls>``, ``<reg>``, ``<mor>``, ``<readm>``) and
    return their ids.
CehrProcessor
    Standard :class:`~pyhealth.processors.FeatureProcessor` that maps a sample's
    list of concept-key strings (already boundary-padded by the task) to a 1-D
    ``torch.long`` tensor of token ids. Vocab growth happens inside the standard
    ``SampleBuilder.fit(samples)`` loop -- no warm-up or freeze flag needed.

The per-patient timeline-extraction helpers (`collect_cehr_timeline_events`,
`build_cehr_sequences`, `infer_mortality_label`, etc.) live with the task that
owns that logic: :mod:`pyhealth.tasks.mpf_clinical_prediction`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import orjson
import torch

from . import register_processor
from .base_processor import FeatureProcessor

DEFAULT_PAD = 0
DEFAULT_UNK = 1
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

__all__ = [
    "DEFAULT_PAD",
    "DEFAULT_UNK",
    "PAD_TOKEN",
    "UNK_TOKEN",
    "ConceptVocab",
    "ensure_special_tokens",
    "CehrProcessor",
]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


@dataclass
class ConceptVocab:
    """Maps concept keys to dense ids with PAD/UNK reserved at 0 and 1."""

    token_to_id: Dict[str, int] = field(default_factory=dict)
    pad_id: int = DEFAULT_PAD
    unk_id: int = DEFAULT_UNK
    _next_id: int = 2

    def __post_init__(self) -> None:
        if not self.token_to_id:
            self.token_to_id = {PAD_TOKEN: self.pad_id, UNK_TOKEN: self.unk_id}
            self._next_id = 2

    def add_token(self, key: str) -> int:
        if key in self.token_to_id:
            return self.token_to_id[key]
        tid = self._next_id
        self.token_to_id[key] = tid
        self._next_id += 1
        return tid

    def __getitem__(self, key: str) -> int:
        return self.token_to_id.get(key, self.unk_id)

    @property
    def vocab_size(self) -> int:
        return self._next_id

    def to_json(self) -> Dict[str, Any]:
        return {
            "token_to_id": self.token_to_id,
            "next_id": self._next_id,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ConceptVocab":
        pad_id = int(data.get("pad_id", DEFAULT_PAD))
        unk_id = int(data.get("unk_id", DEFAULT_UNK))
        vocab = cls(pad_id=pad_id, unk_id=unk_id)
        loaded = dict(data.get("token_to_id") or {})
        if loaded:
            vocab.token_to_id = loaded
            vocab._next_id = int(data.get("next_id", max(loaded.values()) + 1))
        else:
            vocab._next_id = int(data.get("next_id", 2))
        return vocab

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(orjson.dumps(self.to_json(), option=orjson.OPT_SORT_KEYS))

    @classmethod
    def load(cls, path: str) -> "ConceptVocab":
        return cls.from_json(orjson.loads(Path(path).read_bytes()))


def ensure_special_tokens(vocab: ConceptVocab) -> Dict[str, int]:
    """Add EHRMamba/CEHR special tokens and return their ids."""
    return {name: vocab.add_token(name) for name in ("<cls>", "<reg>", "<mor>", "<readm>")}


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------


@register_processor("cehr")
class CehrProcessor(FeatureProcessor):
    """Map a sample's list of concept-key strings to a 1-D LongTensor of ids.

    The task is expected to have already done all boundary-token insertion
    (``<mor>`` / ``<cls>`` / ``<reg>``) and left-padding with ``<pad>``. This
    processor's only state is a :class:`ConceptVocab`, grown during the
    standard :meth:`~pyhealth.datasets.sample_dataset.SampleBuilder.fit`
    pass over cached samples.
    """

    def __init__(self, max_len: int = 512) -> None:
        self.vocab = ConceptVocab()
        ensure_special_tokens(self.vocab)
        self.max_len = max_len

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> "CehrProcessor":
        for sample in samples:
            keys = sample.get(field)
            if not keys:
                continue
            for key in keys:
                if isinstance(key, str):
                    self.vocab.add_token(key)
        return self

    def process(self, value: List[Any]) -> torch.Tensor:
        ids = [
            self.vocab[k] if isinstance(k, str) else int(k)
            for k in value
        ]
        return torch.tensor(ids, dtype=torch.long)

    def save(self, path: str) -> None:
        self.vocab.save(path)

    def load(self, path: str) -> None:
        self.vocab = ConceptVocab.load(path)

    def is_token(self) -> bool:
        return True

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (True,)

    def __repr__(self) -> str:
        return f"CehrProcessor(max_len={self.max_len})"
