
"""Diagnosis-focused evidence retrieval from longitudinal MIMIC-III notes."""

from datetime import timedelta
from typing import Any, Dict, List, Optional

from pyhealth.data import Patient

from .base_task import BaseTask
from ._diagnosis_evidence_utils import (
    clean_id,
    parse_time,
    make_samples,
    choose_history,
)


class DiagnosisEvidenceRetrievalMIMIC3(BaseTask):
    """Construct diagnosis-specific historical clinical note samples.

    Builds retrieval inputs from reviewed diagnoses associated with
    radiology reports in MIMIC-III. Historical note selection follows
    the temporal eligibility and deduplication rules developed for
    reproducing "Retrieving Evidence from EHRs with LLMs:
    Possibilities and Challenges".

    The task prepares retrieval inputs without inventing evidence
    relevance labels.

    Args:
        reviewed_queries: Reviewed diagnosis records containing
            patient_id and index_note_id, or the original research
            fields subject_id and report_id, together with diagnosis.
        min_prior_notes: Minimum eligible notes before history limits.
        max_notes: Maximum historical notes. None retains all notes.
        window_hours: Maximum ED-registration-to-imaging interval.
        strict_availability: Exclude notes stored at or after imaging.
        unknown_availability: Retain or exclude notes with unknown
            storage timestamps.
        deduplication: One of "none", "exact", or "report_versions".
        history_order: Select "recent" or "earliest" historical notes.
        evidence_type: Evidence retrieval target, "risk" or "sign".

    Examples:
        >>> task = DiagnosisEvidenceRetrievalMIMIC3(
        ...     reviewed_queries=[
        ...         {
        ...             "patient_id": "1",
        ...             "index_note_id": "105",
        ...             "diagnosis": "stroke",
        ...         }
        ...     ],
        ...     min_prior_notes=8,
        ...     max_notes=40,
        ... )
    """

    task_name = "diagnosis_evidence_retrieval_mimic3"

    input_schema = {
        "diagnosis": "text",
        "evidence_type": "raw",
        "historical_notes": "raw",
        "note_ids": "raw",
        "note_metadata": "raw",
        "index_note_id": "raw",
        "index_time": "raw",
        "history_audit": "raw",
        "prior_note_count": "raw",
    }

    output_schema = {}

    def __init__(
        self,
        reviewed_queries: List[Dict[str, Any]],
        min_prior_notes: int = 10,
        max_notes: Optional[int] = None,
        window_hours: int = 48,
        strict_availability: bool = True,
        unknown_availability: str = "exclude",
        deduplication: str = "exact",
        history_order: str = "recent",
        evidence_type: str = "risk",
    ) -> None:
        super().__init__()

        if min_prior_notes < 1 or window_hours <= 0:
            raise ValueError("Invalid cohort thresholds.")

        if max_notes is not None and max_notes < min_prior_notes:
            raise ValueError(
                "max_notes cannot be smaller than min_prior_notes."
            )

        if unknown_availability not in {"retain", "exclude"}:
            raise ValueError("Invalid unknown availability policy.")

        if deduplication not in {
            "none", "exact", "report_versions"
        }:
            raise ValueError("Invalid deduplication policy.")

        if history_order not in {"recent", "earliest"}:
            raise ValueError("Invalid history selection policy.")

        if evidence_type not in {"risk", "sign"}:
            raise ValueError("Evidence type must be risk or sign.")

        self.min_prior_notes = min_prior_notes
        self.max_notes = max_notes
        self.window_hours = window_hours
        self.strict_availability = strict_availability
        self.unknown_availability = unknown_availability
        self.deduplication = deduplication
        self.history_order = history_order
        self.evidence_type = evidence_type

        # Accept both PyHealth-style and original research query fields.
        self.queries_by_patient: Dict[str, List[Dict]] = {}

        for query in reviewed_queries:
            subject = clean_id(
                query.get("patient_id", query.get("subject_id"))
            )
            report_id = clean_id(
                query.get("index_note_id", query.get("report_id"))
            )
            diagnosis = str(
                query.get("diagnosis") or ""
            ).strip()

            if not subject or not report_id or not diagnosis:
                raise ValueError(
                    "Each query requires a patient, report and diagnosis."
                )

            record = dict(query)
            record.update(
                subject_id=subject,
                report_id=report_id,
                diagnosis=diagnosis,
            )

            self.queries_by_patient.setdefault(
                subject, []
            ).append(record)

    def _patient_notes(self, patient: Patient) -> List[Dict]:
        """Convert PyHealth note events to the research note schema."""
        notes = []

        required = {
            "row_id",
            "hadm_id",
            "chartdate",
            "charttime_missing",
            "storetime",
            "iserror",
            "text",
            "category",
            "description",
        }

        for event in patient.get_events(event_type="noteevents"):
            attrs = event.attr_dict

            if not required.issubset(attrs):
                raise ValueError(
                    "Required clinical note metadata is missing. "
                    "Check the MIMIC-III configuration and rebuild "
                    "any outdated dataset cache."
                )

            missing = attrs["charttime_missing"]

            if missing is not True and missing is not False:
                raise ValueError(
                    "Missing or invalid chart-time provenance."
                )

            # Restore the original timestamp before midnight imputation.
            charttime = None if missing else event.timestamp

            note = {
                key: attrs.get(key)
                for key in required
                if key != "charttime_missing"
            }

            note.update(
                subject_id=clean_id(patient.patient_id),
                row_id=clean_id(attrs["row_id"]),
                hadm_id=clean_id(attrs["hadm_id"]) or None,
                charttime=(
                    charttime.isoformat()
                    if charttime is not None
                    else None
                ),
            )

            # Normalize boolean error flags for the original utilities.
            if note["iserror"] is True:
                note["iserror"] = "1"

            notes.append(note)

        return notes

    def _accepted_queries(
        self,
        queries: List[Dict],
        notes: List[Dict],
        admissions: Dict,
    ) -> List[Dict]:
        """Validate reviewed reports and ED timing eligibility."""
        notes_by_id = {}

        for note in notes:
            note_id = note["row_id"]

            if note_id in notes_by_id:
                if notes_by_id[note_id] != note:
                    raise ValueError("Conflicting duplicate note ROW_ID.")

            notes_by_id[note_id] = note

        accepted = []

        for query in queries:
            report = notes_by_id.get(query["report_id"])

            if report is None:
                continue

            if str(report["category"]).strip().lower() != "radiology":
                continue

            if str(report["iserror"]).strip() in {"1", "1.0"}:
                continue

            if not isinstance(report["text"], str):
                continue

            if not report["text"].strip():
                continue

            target = parse_time(report["charttime"])
            hadm_id = clean_id(report["hadm_id"])
            admission = admissions.get(hadm_id)

            if target is None or admission is None:
                continue

            admission_type = str(
                admission.attr_dict.get("admission_type") or ""
            ).strip().upper()

            if admission_type not in {"EMERGENCY", "URGENT"}:
                continue

            anchor = parse_time(
                admission.attr_dict.get("edregtime")
            )

            if anchor is None:
                continue

            if not (
                timedelta(0)
                <= target - anchor
                <= timedelta(hours=self.window_hours)
            ):
                continue

            # Validate supplied research metadata when available.
            if query.get("hadm_id") is not None:
                if clean_id(query["hadm_id"]) != hadm_id:
                    raise ValueError("Reviewed admission ID mismatch.")

            if query.get("report_time") is not None:
                if parse_time(query["report_time"]) != target:
                    raise ValueError("Reviewed report timestamp mismatch.")

            accepted.append(
                {
                    "subject_id": query["subject_id"],
                    "hadm_id": hadm_id,
                    "report_id": query["report_id"],
                    "report_time": target.isoformat(),
                    "diagnosis": query["diagnosis"],
                }
            )

        return accepted

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Generate retrieval samples using the original research rules."""
        subject = clean_id(patient.patient_id)
        queries = self.queries_by_patient.get(subject, [])

        if not queries:
            return []

        notes = self._patient_notes(patient)

        admissions = {
            clean_id(event.attr_dict.get("hadm_id")): event
            for event in patient.get_events(event_type="admissions")
        }

        accepted = self._accepted_queries(
            queries, notes, admissions
        )

        # Reuse the original research implementation.
        research_samples = make_samples(
            reviews=accepted,
            notes=notes,
            min_prior_notes=self.min_prior_notes,
            strict_availability=self.strict_availability,
            unknown_availability=self.unknown_availability,
            deduplication=self.deduplication,
        )

        samples = []

        for sample in research_samples:

            # Apply inference history limits after cohort eligibility.
            history = choose_history(
                sample["notes"],
                max_notes=self.max_notes or 0,
                order=self.history_order,
            )

            record_id = (
                f"{subject}_{sample['report_id']}_"
                f"{sample['diagnosis']}_{self.evidence_type}"
            )

            samples.append(
                {
                    "patient_id": subject,
                    "record_id": record_id,
                    "diagnosis": sample["diagnosis"],
                    "evidence_type": self.evidence_type,
                    "index_note_id": sample["report_id"],
                    "index_time": sample["report_time"],
                    "historical_notes": [
                        note["text"] for note in history
                    ],
                    "note_ids": [
                        clean_id(note["row_id"]) for note in history
                    ],
                    "note_metadata": history,
                    "prior_note_count": sample["prior_note_count"],
                    "history_audit": sample["history_audit"],
                }
            )

        return samples
