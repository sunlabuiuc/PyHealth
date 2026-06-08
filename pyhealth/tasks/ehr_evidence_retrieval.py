"""EHR evidence retrieval task for zero-shot LLM-based clinical NLP.
Contributor: Abhisek Sinha (abhisek5@illinois.edu)
Paper: `Ahsan et al. (2024) <https://arxiv.org/abs/2309.04550>`
Implements the task proposed in:
    Ahsan et al. (2024) "Retrieving Evidence from EHRs with LLMs:
    Possibilities and Challenges." CHIL 2024, PMLR 248:489-505.
    arXiv: 2309.04550
"""
from typing import Any, Dict, List, Optional, Set

from .base_task import BaseTask
from ..data import Patient


class EHREvidenceRetrievalTask(BaseTask):
    """Binary task: does a patient's notes support a given query diagnosis?

    Each sample pairs a patient's concatenated clinical notes with a free-text
    query diagnosis string. The binary label indicates whether the patient has
    been assigned any of the specified ICD-9 codes, serving as a computable
    proxy for the radiologist-provided ground-truth used in the original paper.

    This task is designed to be used with :class:`~pyhealth.datasets.MIMIC3NoteDataset`
    (or :class:`~pyhealth.datasets.MIMIC4NoteDataset`) and the
    :class:`~pyhealth.models.ZeroShotEvidenceLLM` model.

    Args:
        query_diagnosis (str): Free-text description of the clinical condition
            to query (e.g. ``"small vessel disease"``).
        condition_icd_codes (List[str]): ICD-9 codes that define a positive
            label. A patient is labelled ``1`` if any of their
            ``diagnoses_icd`` events match at least one code in this set.
        note_categories (Optional[List[str]]): If provided, only notes whose
            ``category`` attribute is in this list are included (e.g.
            ``["Discharge summary", "Radiology"]``). When ``None`` all note
            types are included.
        max_notes (int): Maximum number of notes to include per sample.
            Notes are ordered chronologically and the most recent
            ``max_notes`` are kept. Defaults to ``10``.
        note_separator (str): String used to join multiple note texts into a
            single ``notes`` string. Defaults to ``"\\n\\n---\\n\\n"``.

    Attributes:
        task_name (str): ``"EHREvidenceRetrieval"``
        input_schema (Dict[str, str]): ``{"notes": "text"}``
        output_schema (Dict[str, str]): ``{"label": "binary"}``

    Examples:
        >>> from pyhealth.datasets import MIMIC3NoteDataset
        >>> from pyhealth.tasks import EHREvidenceRetrievalTask
        >>> dataset = MIMIC3NoteDataset(root="/path/to/mimic-iii/1.4")
        >>> task = EHREvidenceRetrievalTask(
        ...     query_diagnosis="small vessel disease",
        ...     condition_icd_codes=["437.3", "437.30", "437.31"],
        ... )
        >>> samples = dataset.set_task(task)
        >>> print(samples[0])
        {'patient_id': ..., 'notes': '...', 'query_diagnosis': 'small vessel disease', 'label': 0}
    """

    task_name: str = "EHREvidenceRetrieval"
    input_schema: Dict[str, str] = {"notes": "text"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        query_diagnosis: str,
        condition_icd_codes: List[str],
        note_categories: Optional[List[str]] = None,
        max_notes: int = 10,
        note_separator: str = "\n\n---\n\n",
    ) -> None:
        """Initialise the EHREvidenceRetrievalTask.

        Args:
            query_diagnosis (str): Free-text clinical condition to query.
            condition_icd_codes (List[str]): ICD-9 codes for positive label.
            note_categories (Optional[List[str]]): Note categories to include.
            max_notes (int): Max notes per sample. Defaults to ``10``.
            note_separator (str): Separator between notes.
        """
        super().__init__()
        self.query_diagnosis = query_diagnosis
        self._condition_codes: Set[str] = set(condition_icd_codes)
        self.note_categories: Optional[Set[str]] = (
            set(note_categories) if note_categories is not None else None
        )
        self.max_notes = max_notes
        self.note_separator = note_separator

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Transform a patient record into EHR evidence retrieval samples.

        One sample is produced per patient. If the patient has no usable note
        text the function returns an empty list so that the patient is skipped.

        Args:
            patient (Patient): PyHealth patient record. Must expose events of
                type ``"noteevents"`` (text, category, iserror) and
                ``"diagnoses_icd"`` (icd9_code).

        Returns:
            List[Dict[str, Any]]: A list with at most one sample dict
            containing:

            - ``"patient_id"`` (str): patient identifier.
            - ``"notes"`` (str): concatenated clinical note text.
            - ``"query_diagnosis"`` (str): the configured query string.
            - ``"label"`` (int): ``1`` if any ICD code matches, else ``0``.
        """
        note_events = patient.get_events(event_type="noteevents")

        # Filter to requested note categories if specified
        if self.note_categories is not None:
            note_events = [
                e for e in note_events
                if getattr(e, "category", None) in self.note_categories
            ]

        # Filter erroneous notes (iserror == "1" or True in MIMIC-III)
        note_events = [
            e for e in note_events
            if str(getattr(e, "iserror", "0")).strip() not in {"1", "1.0"}
        ]

        # Extract non-empty text strings
        texts: List[str] = [
            e.text
            for e in note_events
            if isinstance(getattr(e, "text", None), str) and e.text.strip()
        ]

        if not texts:
            return []

        # Keep the most recent max_notes
        texts = texts[-self.max_notes :]

        notes_text = self.note_separator.join(texts)

        # Derive binary label from ICD-9 diagnoses
        diag_events = patient.get_events(event_type="diagnoses_icd")
        patient_codes: Set[str] = {
            str(getattr(e, "icd9_code", "")).strip()
            for e in diag_events
        }
        label = int(bool(patient_codes & self._condition_codes))

        return [
            {
                "patient_id": patient.patient_id,
                "notes": notes_text,
                "query_diagnosis": self.query_diagnosis,
                "label": label,
            }
        ]
