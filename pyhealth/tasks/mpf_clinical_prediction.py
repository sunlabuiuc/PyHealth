"""Multitask Prompted Fine-tuning (MPF) clinical prediction on FHIR timelines.

The task reads per-patient events via :meth:`pyhealth.data.Patient.get_events`
and :class:`~pyhealth.data.Event` attribute access (the standard PyHealth
idiom). It builds six aligned CEHR feature sequences, inserts MPF boundary
specials, and left-pads to ``max_len``.

Concept-key → integer-id mapping happens later, inside the standard pipeline:
``SampleBuilder.fit`` walks the cached ``task_df.ld`` and fits a
:class:`~pyhealth.processors.CehrProcessor` on the ``concept_ids`` field;
that processor's vocab is then applied per sample by ``_proc_transform``.
The other five sequences are plain numeric lists handled by the standard
tensor processor.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

import polars as pl

from pyhealth.data import Event, Patient
from pyhealth.processors.cehr_processor import PAD_TOKEN

from .base_task import BaseTask

__all__ = [
    "EVENT_TYPE_TO_TOKEN_TYPE",
    "MPFClinicalPredictionTask",
    "collect_cehr_timeline_events",
    "infer_mortality_label",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVENT_TYPE_TO_TOKEN_TYPE: Dict[str, int] = {
    "encounter": 1,
    "condition": 2,
    "medication_request": 3,
    "observation": 4,
    "procedure": 5,
}

_CLINICAL_EVENT_TYPES: Tuple[str, ...] = (
    "condition",
    "observation",
    "medication_request",
    "procedure",
)


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _deceased_boolean_column_means_dead(value: Any) -> bool:
    """True only for an explicit ``"true"`` flag (not Python truthiness)."""
    if value is None:
        return False
    return str(value).strip().lower() == "true"


def _encounter_concept_key(event: Any) -> str:
    enc_class = getattr(event, "encounter_class", None)
    if enc_class:
        return f"encounter|{enc_class}"
    return "encounter|unknown"


def _sequential_visit_idx_for_time(
    event_time: Optional[datetime],
    visit_encounters: List[Tuple[datetime, int]],
) -> int:
    """Bucket an unlinked event into the nearest preceding encounter's index."""
    if not visit_encounters:
        return 0
    if event_time is None:
        return visit_encounters[-1][1]
    chosen = visit_encounters[0][1]
    for encounter_start, visit_idx in visit_encounters:
        if encounter_start <= event_time:
            chosen = visit_idx
        else:
            break
    return chosen


def _birth_datetime_from_patient(patient: Patient) -> Optional[datetime]:
    """Patient's birth date.

    The ``patient`` table's yaml entry declares ``timestamp: birth_date``, so
    the Event's ``timestamp`` field is the birth date itself.
    """
    events = patient.get_events(event_type="patient")
    return events[0].timestamp if events else None


# ---------------------------------------------------------------------------
# Timeline extraction
# ---------------------------------------------------------------------------


def collect_cehr_timeline_events(
    patient: Patient,
) -> List[Tuple[datetime, str, str, int]]:
    """Collect ``(time, concept_key, event_type, visit_idx)`` tuples for one patient.

    Encounters define the visit boundaries. Clinical events that reference a
    known encounter id are linked directly; events without a matching
    encounter reference are bucketed into the chronologically nearest
    preceding visit.
    """
    # Only well-formed encounters (real id + non-null timestamp) define visit
    # indices. We have to inspect the raw polars frame here:
    # ``Event.__init__`` silently coerces ``timestamp=None`` to
    # ``datetime.now()`` (data.py:43-45), so by the time we get back an Event
    # we can no longer tell which encounters were timestamp-less.
    encounters_df = patient.get_events(event_type="encounter", return_df=True)
    valid_encounters = [
        Event.from_dict(row)
        for row in encounters_df.filter(
            pl.col("timestamp").is_not_null()
            & pl.col("encounter/encounter_id").is_not_null()
        ).iter_rows(named=True)
    ]

    encounter_visit_idx: Dict[str, int] = {}
    encounter_start_by_id: Dict[str, datetime] = {}
    visit_encounters: List[Tuple[datetime, int]] = []
    for idx, enc in enumerate(valid_encounters):
        enc_id = enc.encounter_id
        encounter_visit_idx[enc_id] = idx
        encounter_start_by_id[enc_id] = enc.timestamp
        visit_encounters.append((enc.timestamp, idx))

    events: List[Tuple[datetime, str, str, int]] = []
    unlinked: List[Tuple[Optional[datetime], str, str]] = []

    for enc in valid_encounters:
        events.append(
            (
                enc.timestamp,
                _encounter_concept_key(enc),
                "encounter",
                encounter_visit_idx[enc.encounter_id],
            )
        )

    for et in _CLINICAL_EVENT_TYPES:
        for ev in patient.get_events(event_type=et):
            concept_key = getattr(ev, "concept_key", None) or f"{et}|unknown"
            enc_id = getattr(ev, "encounter_id", None)
            # ``ev.timestamp`` is coerced (``None`` -> ``datetime.now()`` by
            # ``Event.__init__``), so it can't reveal a missing time. The raw,
            # null-aware signal is the ``event_time`` attribute (the yaml
            # surfaces it for every clinical table); use it as the null sentinel
            # and ``ev.timestamp`` for the already-parsed value.
            t = ev.timestamp if getattr(ev, "event_time", None) is not None else None
            if enc_id and enc_id in encounter_visit_idx:
                if t is None:
                    t = encounter_start_by_id.get(enc_id)
                if t is None:
                    continue
                events.append((t, concept_key, et, encounter_visit_idx[enc_id]))
            else:
                unlinked.append((t, concept_key, et))

    for t, concept_key, et in unlinked:
        idx = _sequential_visit_idx_for_time(t, visit_encounters)
        if t is None:
            if not visit_encounters:
                continue
            # Use the start of the chosen visit; fall back to the latest encounter.
            t = next(
                (start for start, v_idx in visit_encounters if v_idx == idx),
                visit_encounters[-1][0],
            )
        events.append((t, concept_key, et, idx))

    events.sort(key=lambda item: item[0])
    return events


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------


def infer_mortality_label(patient: Patient) -> int:
    """Heuristic binary mortality label from flattened patient rows."""
    for ev in patient.get_events(event_type="patient"):
        if _deceased_boolean_column_means_dead(getattr(ev, "deceased_boolean", None)):
            return 1
        if getattr(ev, "deceased_datetime", None):
            return 1
    for ev in patient.get_events(event_type="condition"):
        ck = (getattr(ev, "concept_key", None) or "").lower()
        if any(token in ck for token in ("death", "deceased", "mortality")):
            return 1
    return 0


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class MPFClinicalPredictionTask(BaseTask):
    """Binary mortality prediction from FHIR CEHR sequences with optional MPF tokens.

    The task does timeline extraction and emits **raw** per-event lists,
    including concept keys as strings. Tokenization is the
    :class:`~pyhealth.processors.CehrProcessor`'s job, fit during the
    standard ``SampleBuilder.fit(dataset)`` pass.

    Attributes:
        max_len: Output sequence length (must be >= 2 for boundary tokens).
        use_mpf: If True, prepend ``<mor>`` to the sequence; else ``<cls>``.
            The closing ``<reg>`` is always emitted.
    """

    task_name: str = "MPFClinicalPredictionFHIR"
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, max_len: int = 512, use_mpf: bool = True) -> None:
        if max_len < 2:
            raise ValueError("max_len must be >= 2 for MPF boundary tokens")
        self.max_len = max_len
        self.use_mpf = use_mpf
        self.boundary_start = "<mor>" if use_mpf else "<cls>"
        self.boundary_end = "<reg>"
        self.input_schema: Dict[str, Any] = {
            "concept_ids":    ("cehr",   {"max_len": max_len}),
            "token_type_ids": ("tensor", {"dtype": torch.long}),
            "time_stamps":    ("tensor", {"dtype": torch.float32}),
            "ages":           ("tensor", {"dtype": torch.float32}),
            "visit_orders":   ("tensor", {"dtype": torch.long}),
            "visit_segments": ("tensor", {"dtype": torch.long}),
        }

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Build one labeled sample dict per patient."""
        timeline = collect_cehr_timeline_events(patient)
        birth = _birth_datetime_from_patient(patient)

        clinical_cap = self.max_len - 2
        tail = timeline[-clinical_cap:] if clinical_cap > 0 else []
        base_time = tail[0][0] if tail else None

        # Build the six aligned sequences in a single pass.
        keys:        List[str]   = [self.boundary_start]
        token_types: List[int]   = [0]
        time_stamps: List[float] = [0.0]
        ages:        List[float] = [0.0]
        vis_o:       List[int]   = [0]
        vis_s:       List[int]   = [0]

        for event_time, concept_key, event_type, visit_idx in tail:
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
            keys.append(concept_key)
            token_types.append(EVENT_TYPE_TO_TOKEN_TYPE.get(event_type, 0))
            time_stamps.append(time_delta)
            ages.append(age_years)
            vis_o.append(min(visit_idx, 511))
            vis_s.append(visit_idx % 2)

        keys.append(self.boundary_end)
        token_types.append(0)
        time_stamps.append(0.0)
        ages.append(0.0)
        vis_o.append(0)
        vis_s.append(0)

        ml = self.max_len
        keys        = _left_pad(keys,        ml, PAD_TOKEN)
        token_types = _left_pad(token_types, ml, 0)
        time_stamps = _left_pad(time_stamps, ml, 0.0)
        ages        = _left_pad(ages,        ml, 0.0)
        vis_o       = _left_pad(vis_o,       ml, 0)
        vis_s       = _left_pad(vis_s,       ml, 0)

        return [
            {
                "patient_id":     patient.patient_id,
                "concept_ids":    keys,
                "token_type_ids": token_types,
                "time_stamps":    time_stamps,
                "ages":           ages,
                "visit_orders":   vis_o,
                "visit_segments": vis_s,
                "label":          infer_mortality_label(patient),
            }
        ]


def _left_pad(seq: List[Any], max_len: int, pad: Any) -> List[Any]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [pad] * (max_len - len(seq)) + seq
