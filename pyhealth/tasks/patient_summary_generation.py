"""Patient summary generation task.

This task extracts clinical text and patient-facing discharge instruction
pairs for text summarization. It is designed to work with the
MimicIVNoteExtDIDataset but can be used with any dataset that provides
events containing ``text`` (source clinical context) and ``summary``
(target patient summary) attributes.

Reference:
    Hegselmann, S., et al. (2024). A Data-Centric Approach To Generate
    Faithful and High Quality Patient Summaries with Large Language Models.
    Proceedings of Machine Learning Research, 248, 339-379.
"""

from typing import Any, Dict, List

from ..data import Patient
from .base_task import BaseTask


class PatientSummaryGeneration(BaseTask):
    """Task for generating patient-facing summaries from clinical notes.

    This task maps clinical context text (e.g., Brief Hospital Course) to
    patient-facing Discharge Instructions. Each patient record produces a
    single sample consisting of the source text and target summary.

    The task supports research on:
        - Clinical text summarization
        - Hallucination reduction via data-centric approaches
        - Faithfulness evaluation of generated patient summaries

    Attributes:
        task_name: Name of the task.
        input_schema: Schema defining input features. Contains ``"text"``
            mapped to type ``"text"``.
        output_schema: Schema defining output features. Contains
            ``"summary"`` mapped to type ``"text"``.

    Examples:
        >>> from pyhealth.datasets import MimicIVNoteExtDIDataset
        >>> from pyhealth.tasks import PatientSummaryGeneration
        >>> dataset = MimicIVNoteExtDIDataset(
        ...     root="/path/to/data",
        ...     variant="bhc_train",
        ... )
        >>> task = PatientSummaryGeneration()
        >>> samples = dataset.set_task(task)
        >>> print(samples[0].keys())
    """

    task_name: str = "PatientSummaryGeneration"
    input_schema: Dict[str, str] = {"text": "text"}
    output_schema: Dict[str, str] = {"summary": "text"}

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Process a patient record to extract a summarization sample.

        Each patient in the MimicIVNoteExtDIDataset corresponds to a
        single discharge note. This method extracts the clinical context
        text and the target patient summary.

        Args:
            patient: Patient record containing a ``summaries`` event with
                ``text`` and ``summary`` attributes.

        Returns:
            A list containing a single sample dict with keys ``"id"``,
            ``"text"``, and ``"summary"``. Returns an empty list if
            either field is missing or invalid.
        """
        events = patient.get_events(event_type="summaries")
        if len(events) == 0:
            return []

        event = events[0]

        text_valid = isinstance(event.text, str) and len(event.text) > 0
        summary_valid = (
            isinstance(event.summary, str) and len(event.summary) > 0
        )

        if text_valid and summary_valid:
            sample = {
                "id": patient.patient_id,
                "text": event.text,
                "summary": event.summary,
            }
            return [sample]
        return []
