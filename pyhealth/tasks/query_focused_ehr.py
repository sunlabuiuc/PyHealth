"""Query-focused longitudinal EHR task for MIMIC-III.

Implements distant-supervision sample construction inspired by
McInerney et al. (2020), "Query-Focused EHR Summarization
to Aid Imaging Diagnosis".

This task prepares historical clinical sentences, diagnosis
queries, and future-diagnosis labels. It does not train a model.
"""

import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import ClassVar

from pyhealth.data import Patient
from pyhealth.processors import BinaryLabelProcessor, TextProcessor
from pyhealth.tasks.base_task import BaseTask


def parse_clinical_datetime(value) -> datetime | None:
    """Parse a clinical timestamp without inventing missing values.

    Returns None for missing or invalid timestamps.
    Examples:
        >>> from pyhealth.tasks.query_focused_ehr import parse_clinical_datetime
        >>> parse_clinical_datetime("2025-01-10T08:00:00").year
        2025
        >>> parse_clinical_datetime("") is None
        True
    """
    if isinstance(value, datetime):
        return value

    if value is None:
        return None

    if not isinstance(value, str):
        return None

    value = value.strip()

    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
    
def normalize_icd9(code: str) -> str:
    """Normalize an ICD-9 code for consistent matching.

    Handles dotted and undotted representations, for example
    '415.19' and '41519'.

    Examples:
        >>> from pyhealth.tasks.query_focused_ehr import normalize_icd9
        >>> normalize_icd9("415.19")
        '41519'
    """
    if code is None:
        raise ValueError("ICD-9 code cannot be None.")

    normalized = str(code).strip().upper().replace(".", "")

    if not normalized:
        raise ValueError("ICD-9 code cannot be empty.")

    return normalized

def split_clinical_sentences(text: str) -> list[str]:
    """Split clinical note text into candidate sentences.

    Args:
        text: Raw clinical note text.

    Returns:
        Nonempty candidate sentences in original order.

    Notes:
        This lightweight rule-based splitter is suitable for
        baseline preprocessing. Clinical abbreviations and
        templated reports may require more advanced segmentation.
    
        Examples:
        >>> from pyhealth.tasks.query_focused_ehr import split_clinical_sentences
        >>> split_clinical_sentences("Fever noted. Chest X-ray clear.")
        ['Fever noted.', 'Chest X-ray clear.']
    """
    if not isinstance(text, str):
        return []

    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

class QueryFocusedEHRTask(BaseTask):
    """Construct query-focused EHR samples using future diagnoses.

    Each sample contains clinical sentences from previous
    admissions, a diagnosis-description query, and a binary
    label indicating whether the diagnosis occurs in the
    target admission.

    Diagnosis queries may represent individual ICD-9 codes
    or groups of clinically related ICD-9 codes.

    Args:
        diagnosis_descriptions: Optional mapping of individual
            ICD-9 codes to diagnosis descriptions. Mutually
            exclusive with diagnosis_groups.
        history_days: Maximum historical lookback in days.
            None includes all eligible previous admissions.
        max_sentences: Optional maximum number of historical
            sentences, retaining the most recent candidates.
        note_categories: Optional list of clinical note
            categories to include.
        min_sentences: Minimum required historical sentences.
        diagnosis_groups: Optional mapping of group identifiers
            to dictionaries containing a "description" and
            a nonempty collection of ICD-9 "codes". Mutually
            exclusive with diagnosis_descriptions.
        negative_sampling: Strategy for negative query selection.
            "never_observed" (default) labels a diagnosis group
            as negative only when none of its ICD-9 codes appear
            anywhere in the patient's observed record.
            "target_absent" labels a diagnosis group as negative
            whenever none of its codes appear in the target
            admission, even if present elsewhere.

    Notes:
        Positive queries correspond to diagnosis groups
        recorded in the target admission.

        With "never_observed", diagnosis groups found elsewhere
        in the patient's record but absent from the target
        admission are excluded as ambiguous.

        Negative labels indicate that a diagnosis was not
        recorded under the selected strategy; they do not
        establish clinical absence.

        Labels provide distant supervision at the diagnosis
        level, not ground-truth sentence relevance judgments.

        The prediction cutoff is the target admission time.
        Only clinical notes from earlier admissions that were
        available before this cutoff are included.

        "never_observed" uses the complete observed patient
        record to select negative examples and is intended
        for retrospective sample construction.

    Examples:
        >>> from pyhealth.tasks import QueryFocusedEHRTask
        >>> task = QueryFocusedEHRTask(
        ...     diagnosis_descriptions={"41519": "Pulmonary embolism"},
        ...     history_days=90,
        ... )
        >>> task.task_name
        'query_focused_ehr'
    """

    task_name: str = "query_focused_ehr"

    input_schema: ClassVar[dict[str, type]] = {
        "sentences": TextProcessor,
        "query": TextProcessor,
    }

    output_schema: ClassVar[dict[str, type]] = {
        "label": BinaryLabelProcessor,
    }

    def __init__(
        self,
        diagnosis_descriptions: Mapping[str, str] | None = None,
        history_days: int | None = None,
        max_sentences: int | None = None,
        note_categories: list[str] | None = None,
        min_sentences: int = 1,
        diagnosis_groups: Mapping[str, dict] | None = None,
        negative_sampling: str = "never_observed",
    ) -> None:
        super().__init__()

        # Support conservative retrospective negative selection
        # and standard target-admission absence labeling.
        # Reject unsupported strategies at initialization.
        valid_negative_sampling = {
            "never_observed",
            "target_absent",
        }

        if negative_sampling not in valid_negative_sampling:
            raise ValueError(
                f"Unsupported negative_sampling: {negative_sampling!r}. "
                f"Choose from {sorted(valid_negative_sampling)}."
            )

        self.negative_sampling = negative_sampling

        if (diagnosis_descriptions is None) == (diagnosis_groups is None):
            raise ValueError(
                "Provide exactly one of diagnosis_descriptions "
                "or diagnosis_groups."
            )

        if history_days is not None and history_days <= 0:
            raise ValueError("history_days must be positive.")

        if max_sentences is not None and max_sentences <= 0:
            raise ValueError("max_sentences must be positive.")

        if min_sentences < 1:
            raise ValueError("min_sentences must be at least 1.")

        if (
            max_sentences is not None
            and min_sentences > max_sentences
        ):
            raise ValueError(
                "min_sentences cannot exceed max_sentences."
            )

        # Canonical representation for both configuration styles.
        # group_id -> {"description": str, "codes": set[str]}
        self.diagnosis_groups = {}

        if diagnosis_descriptions is not None:
            if not diagnosis_descriptions:
                raise ValueError(
                    "diagnosis_descriptions cannot be empty."
                )

            for code, description in diagnosis_descriptions.items():
                normalized = normalize_icd9(code)
                description = str(description).strip()

                if not description:
                    raise ValueError(
                        "Diagnosis descriptions cannot be empty."
                    )

                if normalized in self.diagnosis_groups:
                    raise ValueError(
                        f"Duplicate normalized ICD-9 code: {normalized}"
                    )

                self.diagnosis_groups[normalized] = {
                    "description": description,
                    "codes": {normalized},
                }

        elif diagnosis_groups is not None:
            if not diagnosis_groups:
                raise ValueError(
                    "diagnosis_groups cannot be empty."
                )

            for group_id, group in diagnosis_groups.items():
                group_id = str(group_id).strip()

                if not group_id:
                    raise ValueError(
                        "Diagnosis group IDs cannot be empty."
                    )

                if group_id in self.diagnosis_groups:
                    raise ValueError(
                        f"Duplicate diagnosis group: {group_id}"
                    )

                if not isinstance(group, Mapping):
                    raise TypeError(
                        f"Group '{group_id}' must be a mapping."
                    )

                description = str(
                    group.get("description") or ""
                ).strip()

                raw_codes = group.get("codes")

                if not description:
                    raise ValueError(
                        f"Missing description for '{group_id}'."
                    )

                if (
                    not isinstance(raw_codes, (list, tuple, set))
                    or not raw_codes
                ):
                    raise ValueError(
                        f"Group '{group_id}' requires nonempty codes."
                    )

                normalized_codes = {
                    normalize_icd9(code)
                    for code in raw_codes
                }

                self.diagnosis_groups[group_id] = {
                    "description": description,
                    "codes": normalized_codes,
                }

        self.history_days = history_days
        self.max_sentences = max_sentences
        self.min_sentences = min_sentences

        self.note_categories = (
            {category.lower().strip() for category in note_categories}
            if note_categories is not None
            else None
        )

    def __call__(self, patient: Patient) -> list[dict]:
        """Generate training examples for a single patient.

        Args:
            patient: PyHealth Patient containing admissions,
                clinical notes, and ICD-9 diagnoses.

        Returns:
            A list of diagnosis-conditioned samples with
            historical sentences and binary weak labels.
        """
        admissions = [
            event
            for event in patient.get_events("admissions")
            if isinstance(event.timestamp, datetime)
            and event.attr_dict.get("hadm_id") is not None
        ]

        admissions.sort(key=lambda event: event.timestamp)

        if len(admissions) < 2:
            return []

        notes_by_visit = defaultdict(list)
        diagnoses_by_visit = defaultdict(set)

        for event in patient.get_events("noteevents"):
            visit_id = event.attr_dict.get("hadm_id")
            text = event.attr_dict.get("text")
            timestamp = event.timestamp

            if visit_id is None or not isinstance(text, str):
                continue

            if not text.strip():
                continue

            # Do not silently accept notes without timestamps.
            if not isinstance(timestamp, datetime):
                continue

            category = str(
                event.attr_dict.get("category") or ""
            ).lower().strip()

            if (
                self.note_categories is not None
                and category not in self.note_categories
            ):
                continue

            # A note cannot be used before its charted or stored time.
            raw_storetime = event.attr_dict.get("storetime")
            storetime = parse_clinical_datetime(raw_storetime)

            # If a supplied storage timestamp cannot be interpreted,
            # exclude the note rather than risk temporal leakage.
            if raw_storetime not in (None, "") and storetime is None:
                continue

            available_at = timestamp

            if storetime is not None:
                available_at = max(timestamp, storetime)

            notes_by_visit[str(visit_id)].append(
                (available_at, text)
            )

        for event in patient.get_events("diagnoses_icd"):
            visit_id = event.attr_dict.get("hadm_id")
            code = event.attr_dict.get("icd9_code")

            if visit_id is None or code is None:
                continue

            diagnoses_by_visit[str(visit_id)].add(
                normalize_icd9(code)
            )

        # Collect ICD-9 codes from the complete observed patient
        # record. These codes are used only to determine negative
        # query eligibility in "never_observed" mode.
        #
        # This is retrospective supervision: later diagnoses may
        # influence whether a candidate negative is excluded,
        # but must never enter the historical sentence inputs.
        patient_codes = set()

        for visit_codes in diagnoses_by_visit.values():
            patient_codes.update(visit_codes)

        samples = []

        for target_index in range(1, len(admissions)):
            target = admissions[target_index]
            cutoff = target.timestamp
            target_id = str(target.attr_dict["hadm_id"])

            earliest_time = (
                cutoff - timedelta(days=self.history_days)
                if self.history_days is not None
                else None
            )

            history = []

            for previous in admissions[:target_index]:
                previous_id = str(
                    previous.attr_dict["hadm_id"]
                )

                for available_at, note_text in (
                    notes_by_visit.get(previous_id, [])
                ):
                    # Strictly prevent post-cutoff leakage.
                    if available_at >= cutoff:
                        continue

                    if (
                        earliest_time is not None
                        and available_at < earliest_time
                    ):
                        continue

                    for sentence_index, sentence in enumerate(
                        split_clinical_sentences(note_text)
                    ):
                        history.append(
                            (
                                available_at,
                                previous_id,
                                sentence_index,
                                sentence,
                            )
                        )

            history.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2],
                )
            )

            if self.max_sentences is not None:
                history = history[-self.max_sentences:]

            sentences = [
                item[3] for item in history
            ]

            if len(sentences) < self.min_sentences:
                continue

            positive_codes = diagnoses_by_visit.get(
                target_id, set()
            )

            for group_id, group in self.diagnosis_groups.items():

                # A query is positive when any ICD-9 code belonging
                # to its diagnosis group occurs in the target admission.
                is_positive = bool(
                    group["codes"].intersection(positive_codes)
                )

                if self.negative_sampling == "never_observed":
                    # Conservative strategy: select a negative only if
                    # the diagnosis group is never recorded for this patient.
                    # Groups observed at other admissions are excluded.
                    is_valid_negative = not bool(
                        group["codes"].intersection(patient_codes)
                    )
                else:
                    # Target-based strategy: any group missing from the
                    # target admission qualifies as a negative.
                    is_valid_negative = not is_positive

                # Keep positives and eligible negatives only.
                if not is_positive and not is_valid_negative:
                    continue

                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": target_id,
                        "sentences": list(sentences),
                        "query": group["description"],
                        "label": int(is_positive),
                        "diagnosis_code": group_id,
                    }
                )


        return samples
