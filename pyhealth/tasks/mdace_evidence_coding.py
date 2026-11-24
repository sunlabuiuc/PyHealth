"""
Evidence-grounded ICD coding task for the MDACE dataset.

Contributors:
    Juri Moon (jmoon326)
"""
import json
from typing import Any, ClassVar

from ..data import Patient
from .base_task import BaseTask


class MDACEEvidenceCoding(BaseTask):
    """Predict the ICD codes of an MDACE note, keeping the evidence spans.

    One sample per annotated note. ``icd_codes`` is the multilabel target;
    ``evidence_spans`` is aligned with it (``evidence_spans[i]`` is the list
    of ``[begin, end]`` character spans supporting ``icd_codes[i]``) and is
    passed through untouched for explainability evaluation, together with
    ``note_id``, ``hadm_id``, ``split`` and ``category``.

    Args:
        discharge_summaries_only: Keep only ``Discharge summary`` notes, as
            evaluated by Edin et al. (2024). Defaults to True.

    Examples:
        >>> from pyhealth.datasets import MDACEDataset
        >>> from pyhealth.tasks import MDACEEvidenceCoding
        >>> dataset = MDACEDataset(root="...", mimic3_root="...")
        >>> samples = dataset.set_task(MDACEEvidenceCoding())
    """

    task_name: str = "MDACEEvidenceCoding"
    input_schema: ClassVar[dict[str, str]] = {"text": "text"}
    output_schema: ClassVar[dict[str, str]] = {"icd_codes": "multilabel"}

    def __init__(self, discharge_summaries_only: bool = True) -> None:
        super().__init__()
        self.discharge_summaries_only = discharge_summaries_only

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Builds one sample per annotated note of the patient.

        Args:
            patient: Patient with ``mdace`` events.

        Returns:
            List of sample dicts.
        """
        samples = []
        for event in patient.get_events(event_type="mdace"):
            if (
                self.discharge_summaries_only
                and event.category != "Discharge summary"
            ):
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "note_id": event.note_id,
                    "hadm_id": event.hadm_id,
                    "split": event.split,
                    "category": event.category,
                    "text": event.text,
                    "icd_codes": json.loads(event.codes),
                    "evidence_spans": json.loads(event.spans),
                }
            )
        return samples
