"""PyHealth task for LLM-based evidence retrieval from EHR clinical notes.

Paper:
    M. Ahsan et al. "Retrieving Evidence from EHRs with LLMs:
    Possibilities and Challenges." Proceedings of Machine Learning
    Research, 2024.

Paper link:
    https://proceedings.mlr.press/v248/ahsan24a.html

Author:
    Arnab Karmakar (arnabk3@illinois.edu)
"""
import logging
import re
from typing import Any, Dict, List, Optional

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


# Lightweight, paper-inspired keyword mappings used to derive weak
# ground-truth "is positive" labels for the synthetic corpus and, when
# MIMIC-III is available, for MIMIC note events. Kept deliberately small
# so the task remains fast and readable — downstream users are expected
# to customize these or pass their own overrides.
_DEFAULT_CONDITION_KEYWORDS: Dict[str, List[str]] = {
    "intracranial hemorrhage": [
        "intracranial hemorrhage",
        "intraparenchymal hemorrhage",
        "subdural hematoma",
        "hyperdense focus",
        "craniotomy",
    ],
    "stroke": [
        "stroke",
        "ischemic stroke",
        "mca infarct",
        "middle cerebral artery",
        "restricted diffusion",
    ],
    "pneumonia": [
        "pneumonia",
        "lobar consolidation",
        "lower lobe consolidation",
        "productive cough",
    ],
}


class EvidenceRetrievalMIMIC3(BaseTask):
    """Task for evidence retrieval from EHR clinical notes.

    Given a patient and a clinical-note event, this task emits one
    sample per (note, condition) pair. The sample carries the raw note
    text and the diagnosis condition query, together with a weak
    ground-truth "is_positive" label derived from keyword matches.

    Downstream retriever models consume these samples to produce either
    (a) a yes/no note-level decision for the condition, or (b) a short
    natural-language explanation of risk factors or signs. The binary
    output makes the task directly compatible with PyHealth's standard
    binary metrics for the Pass-1 note classification sub-task.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): Schema for the task input.
        output_schema (Dict[str, str]): Schema for the task output.
        conditions (List[str]): Diagnosis conditions to query notes for.
        keywords (Dict[str, List[str]]): Per-condition keyword list used
            to derive the weak ``is_positive`` label. Users can override
            this to use stricter regex patterns or dictionary-based
            phenotyping rules.

    Examples:
        >>> from pyhealth.datasets import SyntheticEHRNotesDataset
        >>> from pyhealth.tasks import EvidenceRetrievalMIMIC3
        >>> dataset = SyntheticEHRNotesDataset(root="./synthetic_notes")
        >>> task = EvidenceRetrievalMIMIC3(
        ...     conditions=["stroke", "intracranial hemorrhage"],
        ... )
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "EvidenceRetrievalMIMIC3"
    input_schema: Dict[str, str] = {
        "note_text": "text",
        "condition": "text",
    }
    output_schema: Dict[str, str] = {"is_positive": "binary"}

    def __init__(
        self,
        conditions: Optional[List[str]] = None,
        keywords: Optional[Dict[str, List[str]]] = None,
        event_type: str = "notes",
        text_attribute: str = "text",
    ) -> None:
        """Initialize the evidence retrieval task.

        Args:
            conditions (Optional[List[str]]): Diagnosis conditions to
                query notes for. When ``None``, the condition attached
                to each event (if present) is used, which keeps the
                behavior natural for the synthetic corpus.
            keywords (Optional[Dict[str, List[str]]]): Per-condition
                keyword overrides used to derive the weak
                ``is_positive`` label. When ``None``, a small paper-
                inspired default is used.
            event_type (str): Name of the event type holding the
                note text. Defaults to ``"notes"`` (synthetic
                corpus). For MIMIC-III, use ``"noteevents"``.
            text_attribute (str): Attribute name on the note event that
                stores the free-text content. Defaults to ``"text"``.

        Raises:
            ValueError: If ``conditions`` is an empty list.
        """
        if conditions is not None and len(conditions) == 0:
            raise ValueError("conditions, if provided, must be non-empty.")
        self.conditions = conditions
        self.keywords = keywords or _DEFAULT_CONDITION_KEYWORDS
        self.event_type = event_type
        self.text_attribute = text_attribute
        # No code_mapping required for this text-centric task.
        super().__init__()

    def _derive_is_positive(self, note_text: str, condition: str) -> int:
        """Return a binary weak label for ``(note_text, condition)``.

        The label is ``1`` if any keyword for ``condition`` (falling
        back to the condition string itself) appears in the
        lower-cased note text, and ``0`` otherwise.

        Args:
            note_text (str): Raw note text.
            condition (str): Target diagnosis condition.

        Returns:
            int: ``1`` for a positive match, ``0`` otherwise.
        """
        haystack = (note_text or "").lower()
        patterns = self.keywords.get(condition.lower(), [condition])
        for p in patterns:
            if p.lower() in haystack:
                return 1
        return 0

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Produce one sample per (note, condition) pair for ``patient``.

        Args:
            patient (Any): A PyHealth ``Patient`` object.

        Returns:
            List[Dict[str, Any]]: Samples with keys
                ``patient_id``, ``note_id``, ``note_text``,
                ``condition``, and ``is_positive``.
        """
        events = patient.get_events(event_type=self.event_type)
        samples: List[Dict[str, Any]] = []

        for event in events:
            # Tolerate both attribute-style and dict-style event access.
            note_text = _safe_get(event, self.text_attribute, default="")
            note_id = _safe_get(event, "note_id", default="")
            if not note_text:
                continue

            if self.conditions is not None:
                condition_candidates = list(self.conditions)
            else:
                event_condition = _safe_get(event, "condition", default="")
                if not event_condition:
                    continue
                condition_candidates = [event_condition]

            for condition in condition_candidates:
                provided_label = _safe_get(event, "is_positive", default=None)
                if provided_label is not None and self.conditions is None:
                    # Respect the ground-truth label shipped with the
                    # synthetic corpus when the task is run condition-
                    # agnostic on that event.
                    is_positive = int(str(provided_label))
                else:
                    is_positive = self._derive_is_positive(note_text, condition)

                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "note_id": str(note_id),
                        "note_text": str(note_text),
                        "condition": str(condition),
                        "is_positive": int(is_positive),
                    }
                )

        return samples


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Look up ``key`` on ``obj`` using attribute or item access.

    Args:
        obj (Any): Source object (PyHealth event, dict, or similar).
        key (str): Attribute or item key.
        default (Any): Value to return when ``key`` is missing.

    Returns:
        Any: The resolved value or ``default``.
    """
    if obj is None:
        return default
    try:
        value = getattr(obj, key)
        if value is not None:
            return value
    except AttributeError:
        pass
    try:
        return obj[key]
    except (KeyError, TypeError):
        return default


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentence-like fragments.

    The tokenizer is intentionally simple (no NLTK dependency) so that
    this utility works out-of-the-box inside unit tests.

    Args:
        text (str): Raw note text.

    Returns:
        List[str]: Non-empty trimmed sentence fragments.
    """
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
