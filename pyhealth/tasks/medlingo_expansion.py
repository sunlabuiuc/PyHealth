from typing import Any, Dict, List

from pyhealth.data import Patient
from pyhealth.tasks.base_task import BaseTask
from pyhealth.processors import TextProcessor, MultiClassLabelProcessor

class MedLingoExpansionTask(BaseTask):
    """
    Clinical jargon expansion task for the MedLingo dataset.

    This task uses a single clinical snippet and jargon term as input and predicts
    the correct expansion as a single multiclass label.

    Assumption:
        Each patient/sample unit contains exactly one event of type "medlingo"
        with the following fields:
            - context: The clinical snippet containing the jargon term.
            - term: The jargon term that needs to be expanded.
            - expansion: The correct expansion for the jargon term (the label).
    """

    task_name: str = "medlingo_expansion"
    input_schema: Dict[str, Any] = {"text": TextProcessor}
    output_schema: Dict[str, Any] = {"label": MultiClassLabelProcessor}

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """
        Convert one MedLingo patient/sample unit into a PyHealth task sample.

        Args:
            patient: A PyHealth Patient object representing one MedLingo sample.

        Returns:
            A list containing one dictionary with:
                - patient_id: sample identifier
                - text: formatted input text with clinical snippet and jargon term
                - label: correct expansion for the jargon term

        Raises:
            ValueError: If the patient does not contain exactly one medlingo event.
        """
        events = patient.get_events(event_type="medlingo")
        if len(events) != 1:
            raise ValueError("Expected exactly one medlingo event per patient.")
        event = events[0]

        text = f"Clinical snippet: {event.context}\nJargon term: {event.term}"

        return [
            {
                "patient_id": patient.patient_id,
                "text": text,
                "label": event.expansion,
            }
        ]