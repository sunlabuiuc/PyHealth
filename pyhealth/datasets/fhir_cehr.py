"""CEHR-style tokenization, vocabulary, and sequence building for FHIR timelines.

Key public API
--------------
ConceptVocab
    Token-to-dense-id mapping with PAD/UNK reserved at 0 and 1. JSON-serializable.

build_cehr_sequences(patient, vocab, max_len)
    Flatten a Patient's tabular FHIR rows into CEHR-aligned feature lists.

infer_mortality_label(patient)
    Heuristic binary mortality label from flattened patient rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orjson

from pyhealth.data import Patient

from .fhir_ingest import as_naive, parse_dt

DEFAULT_PAD = 0
DEFAULT_UNK = 1

__all__ = [
    # Constants
    "DEFAULT_PAD",
    "DEFAULT_UNK",
    "EVENT_TYPE_TO_TOKEN_TYPE",
    # Vocabulary
    "ConceptVocab",
    "ensure_special_tokens",
    # Sequence building
    "collect_cehr_timeline_events",
    "warm_mpf_vocab_from_patient",
    "build_cehr_sequences",
    # Labels
    "infer_mortality_label",
]

EVENT_TYPE_TO_TOKEN_TYPE = {
    "encounter": 1,
    "condition": 2,
    "medication_request": 3,
    "observation": 4,
    "procedure": 5,
}

# Table-driven lookups for flattened event-row column access.
_CONCEPT_KEY_COL: Dict[str, str] = {
    "condition": "condition/concept_key",
    "observation": "observation/concept_key",
    "medication_request": "medication_request/concept_key",
    "procedure": "procedure/concept_key",
}

_ENCOUNTER_ID_COL: Dict[str, str] = {
    "condition": "condition/encounter_id",
    "observation": "observation/encounter_id",
    "medication_request": "medication_request/encounter_id",
    "procedure": "procedure/encounter_id",
    "encounter": "encounter/encounter_id",
}

# ---------------------------------------------------------------------------
# ConceptVocab
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
            self.token_to_id = {"<pad>": self.pad_id, "<unk>": self.unk_id}
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
        return {"token_to_id": self.token_to_id, "next_id": self._next_id}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ConceptVocab":
        vocab = cls()
        loaded = dict(data.get("token_to_id") or {})
        if not loaded:
            vocab._next_id = int(data.get("next_id", 2))
            return vocab
        vocab.token_to_id = loaded
        vocab._next_id = int(data.get("next_id", max(loaded.values()) + 1))
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
# Row utilities for flattened event stream
# ---------------------------------------------------------------------------


def _clean_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return str(value)


def _deceased_boolean_column_means_dead(value: Any) -> bool:
    """True only for an explicit affirmative stored flag (not Python truthiness)."""
    s = _clean_string(value)
    return s is not None and s.lower() == "true"


def _row_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return as_naive(value)
    try:
        return parse_dt(str(value))
    except Exception:
        return None


def _concept_key_from_row(row: Dict[str, Any]) -> str:
    event_type = row.get("event_type")
    col = _CONCEPT_KEY_COL.get(event_type)
    if col:
        return _clean_string(row.get(col)) or f"{event_type}|unknown"
    if event_type == "encounter":
        enc_class = _clean_string(row.get("encounter/encounter_class"))
        return f"encounter|{enc_class}" if enc_class else "encounter|unknown"
    return f"{event_type or 'event'}|unknown"


def _linked_encounter_id_from_row(row: Dict[str, Any]) -> Optional[str]:
    col = _ENCOUNTER_ID_COL.get(row.get("event_type"))
    return _clean_string(row.get(col)) if col else None


def _birth_datetime_from_patient(patient: Patient) -> Optional[datetime]:
    for row in patient.data_source.iter_rows(named=True):
        if row.get("event_type") != "patient":
            continue
        birth = _row_datetime(row.get("timestamp"))
        if birth is not None:
            return birth
        raw = _clean_string(row.get("patient/birth_date"))
        if raw:
            return parse_dt(raw)
    return None


def _sequential_visit_idx_for_time(
    event_time: Optional[datetime],
    visit_encounters: List[Tuple[datetime, int]],
) -> int:
    if not visit_encounters:
        return 0
    if event_time is None:
        return visit_encounters[-1][1]
    event_time = as_naive(event_time)
    chosen = visit_encounters[0][1]
    for encounter_start, visit_idx in visit_encounters:
        if encounter_start <= event_time:
            chosen = visit_idx
        else:
            break
    return chosen


# ---------------------------------------------------------------------------
# CEHR timeline and sequence building
# ---------------------------------------------------------------------------


def collect_cehr_timeline_events(
    patient: Patient,
) -> List[Tuple[datetime, str, str, int]]:
    """Collect (time, concept_key, event_type, visit_idx) tuples from a patient's rows."""
    rows = list(
        patient.data_source.sort(["timestamp", "event_type"], nulls_last=True).iter_rows(named=True)
    )

    encounter_rows: List[Tuple[datetime, str]] = []
    for row in rows:
        if row.get("event_type") != "encounter":
            continue
        enc_id = _linked_encounter_id_from_row(row)
        enc_start = _row_datetime(row.get("timestamp"))
        if enc_id is not None and enc_start is not None:
            encounter_rows.append((enc_start, enc_id))

    encounter_rows.sort(key=lambda pair: pair[0])
    encounter_visit_idx = {enc_id: idx for idx, (_, enc_id) in enumerate(encounter_rows)}
    encounter_start_by_id = {enc_id: enc_start for enc_start, enc_id in encounter_rows}
    visit_encounters = [(enc_start, idx) for idx, (enc_start, _) in enumerate(encounter_rows)]

    events: List[Tuple[datetime, str, str, int]] = []
    unlinked: List[Tuple[Optional[datetime], str, str]] = []

    for row in rows:
        event_type = row.get("event_type")
        if event_type not in EVENT_TYPE_TO_TOKEN_TYPE:
            continue
        event_time = _row_datetime(row.get("timestamp"))
        concept_key = _concept_key_from_row(row)

        if event_type == "encounter":
            enc_id = _linked_encounter_id_from_row(row)
            if enc_id is None or event_time is None:
                continue
            visit_idx = encounter_visit_idx.get(enc_id)
            if visit_idx is None:
                continue
            events.append((event_time, concept_key, event_type, visit_idx))
            continue

        enc_id = _linked_encounter_id_from_row(row)
        if enc_id and enc_id in encounter_visit_idx:
            visit_idx = encounter_visit_idx[enc_id]
            if event_time is None:
                event_time = encounter_start_by_id.get(enc_id)
            if event_time is None:
                continue
            events.append((event_time, concept_key, event_type, visit_idx))
        else:
            unlinked.append((event_time, concept_key, event_type))

    for event_time, concept_key, event_type in unlinked:
        visit_idx = _sequential_visit_idx_for_time(event_time, visit_encounters)
        if event_time is None:
            if not visit_encounters:
                continue
            for enc_start, enc_visit_idx in visit_encounters:
                if enc_visit_idx == visit_idx:
                    event_time = enc_start
                    break
            else:
                event_time = visit_encounters[-1][0]
        if event_time is None:
            continue
        events.append((event_time, concept_key, event_type, visit_idx))

    events.sort(key=lambda item: item[0])
    return events


def warm_mpf_vocab_from_patient(
    vocab: ConceptVocab,
    patient: Patient,
    clinical_cap: int,
) -> None:
    """Add concept keys from the last clinical_cap events of a patient to vocab."""
    tail = collect_cehr_timeline_events(patient)[-clinical_cap:] if clinical_cap > 0 else []
    for _, concept_key, _, _ in tail:
        vocab.add_token(concept_key)


def build_cehr_sequences(
    patient: Patient,
    vocab: ConceptVocab,
    max_len: int,
    *,
    base_time: Optional[datetime] = None,
    grow_vocab: bool = True,
) -> Tuple[List[int], List[int], List[float], List[float], List[int], List[int]]:
    """Flatten a patient's tabular FHIR rows into CEHR-aligned feature lists."""
    events = collect_cehr_timeline_events(patient)
    birth = _birth_datetime_from_patient(patient)

    if base_time is None:
        base_time = events[0][0] if events else datetime.now()
    base_time = as_naive(base_time)
    birth = as_naive(birth)

    concept_ids: List[int] = []
    token_types: List[int] = []
    time_stamps: List[float] = []
    ages: List[float] = []
    visit_orders: List[int] = []
    visit_segments: List[int] = []

    for event_time, concept_key, event_type, visit_idx in (events[-max_len:] if max_len > 0 else []):
        event_time = as_naive(event_time)
        concept_id = vocab.add_token(concept_key) if grow_vocab else vocab[concept_key]
        token_type = EVENT_TYPE_TO_TOKEN_TYPE.get(event_type, 0)
        time_delta = (
            float((event_time - base_time).total_seconds())
            if base_time is not None and event_time is not None
            else 0.0
        )
        age_years = (
            (event_time - birth).days / 365.25
            if birth is not None and event_time is not None
            else 0.0
        )
        concept_ids.append(concept_id)
        token_types.append(token_type)
        time_stamps.append(time_delta)
        ages.append(age_years)
        visit_orders.append(min(visit_idx, 511))
        visit_segments.append(visit_idx % 2)

    return concept_ids, token_types, time_stamps, ages, visit_orders, visit_segments


# ---------------------------------------------------------------------------
# Label inference
# ---------------------------------------------------------------------------


def infer_mortality_label(patient: Patient) -> int:
    """Heuristic binary mortality label from flattened patient rows."""
    for row in patient.data_source.iter_rows(named=True):
        if row.get("event_type") == "patient":
            if _deceased_boolean_column_means_dead(row.get("patient/deceased_boolean")):
                return 1
            if _clean_string(row.get("patient/deceased_datetime")):
                return 1
    for row in patient.data_source.iter_rows(named=True):
        if row.get("event_type") == "condition":
            key = (_clean_string(row.get("condition/concept_key")) or "").lower()
            if any(token in key for token in ("death", "deceased", "mortality")):
                return 1
    return 0
