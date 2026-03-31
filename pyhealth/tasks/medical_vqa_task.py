from typing import Any, Dict, List

from ..data import Patient
from .base_task import BaseTask


class MedicalVQATask(BaseTask):
    """Task for medical visual question answering.

    This task takes a medical image and a natural-language question as input
    and predicts the corresponding answer. It processes patient records
    containing ``vqarad`` events and extracts image-question-answer triples.

    Attributes:
        task_name (str): Name of the task.
        input_schema (Dict[str, str]): Schema defining input features.
        output_schema (Dict[str, str]): Schema defining output features.

    Examples:
        >>> from pyhealth.datasets import VQARADDataset
        >>> from pyhealth.tasks import MedicalVQATask
        >>> dataset = VQARADDataset(root="/path/to/vqarad")
        >>> task = MedicalVQATask()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "MedicalVQA"
    input_schema: Dict[str, str] = {"image": "image", "question": "text"}
    output_schema: Dict[str, str] = {"answer": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Process a patient record into medical VQA samples.

        Args:
            patient (Patient): Patient record containing VQA-RAD events.

        Returns:
            List[Dict[str, Any]]: List of samples containing patient ID,
                image path, question, and answer.
        """
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
