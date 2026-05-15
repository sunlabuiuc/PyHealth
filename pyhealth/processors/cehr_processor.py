"""CEHR-style tokenization, vocabulary, and sequence building for FHIR timelines.

Key public API
--------------
ConceptVocab
    Token-to-dense-id mapping with PAD/UNK reserved at 0 and 1. JSON-serializable.

CehrProcessor
    FeatureProcessor subclass that owns a ConceptVocab, can be warmed over a patient
    stream, and converts a Patient's tabular FHIR rows into CEHR-aligned sequences.

build_cehr_sequences(patient, vocab, max_len)
    Flatten a Patient's tabular FHIR rows into CEHR-aligned feature lists.

infer_mortality_label(patient)
    Heuristic binary mortality label from flattened patient rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import orjson

from pyhealth.data import Patient

from .base_processor import FeatureProcessor
from . import register_processor

DEFAULT_PAD = 0
DEFAULT_UNK = 1

# ---------------------------------------------------------------------------
# Datetime helpers
#
# These are intentional copies of the identically-named functions in
# pyhealth.datasets.fhir_utils.  They exist here to avoid a circular import:
# importing pyhealth.datasets.fhir_utils (even as a submodule) triggers
# pyhealth/datasets/__init__.py, which imports MIMIC4FHIRDataset, which in turn
# imports from this module — creating a cycle.  The implementations are pure
# stdlib (no pyhealth deps), so keeping them in sync is straightforward;
# any change to the canonical copy in fhir_utils must be mirrored here.
# ---------------------------------------------------------------------------


def parse_dt(s: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 or YYYY-MM-DD date string to a naive datetime."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        dt = None
    if dt is None and len(s) >= 10:
        try:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def as_naive(dt: Optional[datetime]) -> Optional[datetime]:
    """Strip timezone info from a datetime, or return None unchanged."""
    if dt is None:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


__all__ = [
    # Constants
    "DEFAULT_PAD",
    "DEFAULT_UNK",
    "EVENT_TYPE_TO_TOKEN_TYPE",
    # Vocabulary
    "ConceptVocab",
    "ensure_special_tokens",
    # Processor
    "CehrProcessor",
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

    # Build encounter list — rows are already timestamp-sorted so the loop
    # preserves chronological order without an explicit sort.
    encounter_rows: List[Tuple[datetime, str]] = []
    for row in rows:
        if row.get("event_type") != "encounter":
            continue
        enc_id = _linked_encounter_id_from_row(row)
        enc_start = _row_datetime(row.get("timestamp"))
        if enc_id is not None and enc_start is not None:
            encounter_rows.append((enc_start, enc_id))

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


# ---------------------------------------------------------------------------
# CehrProcessor
# ---------------------------------------------------------------------------


@register_processor("cehr")
class CehrProcessor(FeatureProcessor):
    """CEHR concept sequence processor for FHIR timelines.

    Owns a :class:`ConceptVocab` and converts a
    :class:`~pyhealth.data.Patient`'s tabular FHIR event rows into
    CEHR-aligned integer sequence lists ready for downstream models.

    This processor departs from the standard ``FeatureProcessor`` contract
    in two ways that are intentional for this domain:

    * ``process(patient)`` takes a full :class:`~pyhealth.data.Patient` rather
      than a single scalar field value, because building CEHR sequences
      requires access to all event rows simultaneously.
    * ``fit(patients, clinical_cap)`` takes an iterable of
      :class:`~pyhealth.data.Patient` objects instead of the base-class
      ``fit(samples, field)`` signature, because vocabulary warming is driven
      by the Patient timeline, not a pre-aggregated sample dict.

    Typical usage::

        processor = CehrProcessor(max_len=512)
        processor.fit(dataset.iter_patients())      # warm vocabulary
        sequences = processor.process(some_patient) # build sequences
        processor.save("vocab.json")                # persist vocab

    Attributes:
        vocab: Concept-to-id mapping (PAD=0, UNK=1).
        max_len: Maximum number of clinical tokens per patient (boundary tokens
            not counted; see :class:`~pyhealth.tasks.MPFClinicalPredictionTask`).
        frozen_vocab: When True, unknown concepts map to UNK instead of adding
            new ids — used for multi-worker safety after vocab warm-up.
    """

    def __init__(
        self,
        vocab: Optional[ConceptVocab] = None,
        max_len: int = 512,
        frozen_vocab: bool = False,
    ) -> None:
        self.vocab = vocab or ConceptVocab()
        self.max_len = max_len
        self.frozen_vocab = frozen_vocab

    def fit(  # type: ignore[override]
        self,
        patients: Iterable[Patient],
        clinical_cap: Optional[int] = None,
    ) -> "CehrProcessor":
        """Warm vocabulary from a stream of patients.

        Note: this method intentionally overrides ``FeatureProcessor.fit(samples,
        field)`` with a different signature, because CEHR vocabulary warming
        operates on :class:`~pyhealth.data.Patient` timelines rather than
        pre-aggregated sample dicts.

        Special tokens are *not* inserted here; they are added lazily by
        :meth:`~pyhealth.tasks.MPFClinicalPredictionTask._ensure_processor`
        on the first call to the task.  This keeps ``fit`` focused on concept
        key discovery.

        Args:
            patients: Iterable of :class:`~pyhealth.data.Patient` objects.
            clinical_cap: Maximum number of tail events to scan per patient.
                Defaults to ``max_len - 2`` (room for two boundary tokens).

        Returns:
            self (for chaining).
        """
        cap = clinical_cap if clinical_cap is not None else max(0, self.max_len - 2)
        for patient in patients:
            warm_mpf_vocab_from_patient(self.vocab, patient, cap)
        return self

    def process(
        self,
        patient: Patient,
    ) -> Tuple[List[int], List[int], List[float], List[float], List[int], List[int]]:
        """Build CEHR sequences from a patient's FHIR event rows.

        Args:
            patient: A tabular :class:`~pyhealth.data.Patient`.

        Returns:
            Six equal-length lists ``(concept_ids, token_type_ids, time_stamps,
            ages, visit_orders, visit_segments)`` ready for boundary-token
            insertion and left-padding by the task.
        """
        clinical_cap = max(0, self.max_len - 2)
        return build_cehr_sequences(
            patient, self.vocab, clinical_cap, grow_vocab=not self.frozen_vocab
        )

    def save(self, path: str) -> None:
        """Persist the vocabulary to a JSON file at *path*."""
        self.vocab.save(path)

    def load(self, path: str) -> None:
        """Load a previously saved vocabulary from *path*."""
        self.vocab = ConceptVocab.load(path)

    def is_token(self) -> bool:
        """All six output lists contain discrete token/index values."""
        return True

    def schema(self) -> Tuple[str, ...]:
        return (
            "concept_ids",
            "token_type_ids",
            "time_stamps",
            "ages",
            "visit_orders",
            "visit_segments",
        )

    def dim(self) -> Tuple[int, ...]:
        """Each of the six output lists becomes a 1-D tensor."""
        return (1, 1, 1, 1, 1, 1)

    def spatial(self) -> Tuple[bool, ...]:
        """All six outputs are along the sequence (temporal) axis."""
        return (True, True, True, True, True, True)

    def __repr__(self) -> str:
        # frozen_vocab is a runtime flag, not a constructor parameter, so it
        # must be excluded here.  This repr is used by BaseDataset.set_task to
        # compute the LitData task-cache UUID via vars(task); including
        # frozen_vocab would produce different UUIDs for single- vs
        # multi-worker runs of the same pipeline, defeating caching.
        return f"CehrProcessor(max_len={self.max_len})"
