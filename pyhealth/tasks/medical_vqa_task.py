from typing import Any, Dict, List

from ..data import Patient
from .base_task import BaseTask


class MedicalVQATask(BaseTask):
    """Task for medical visual question answering."""

    task_name: str = "MedicalVQA"
    input_schema: Dict[str, str] = {"image": "image", "question": "text"}
    output_schema: Dict[str, str] = {"answer": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Converts VQA-RAD patient events into image-question-answer samples."""
        samples = []
        events = patient.get_events(event_type="vqarad")
        for event in events:
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "image": event.image_path,
                    "question": event.question,
                    "answer": event.answer,
                }
            )
        return samples
