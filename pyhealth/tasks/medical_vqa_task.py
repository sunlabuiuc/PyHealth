"""Medical Visual Question Answering (VQA) task."""

from typing import Any, Dict, List

from ..data import Patient
from .base_task import BaseTask


class MedicalVQATask(BaseTask):
    """Task for medical Visual Question Answering (VQA).

    Expects a dataset with medical images, questions, and answers. Each
    sample maps an (image, question) pair to a single answer string,
    treated as a multiclass classification label.

    Attributes:
        task_name: ``"MedicalVQA"``.
        input_schema: ``{"image": "image", "question": "text"}``.
        output_schema: ``{"answer": "multiclass"}``.

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

        Each event in the ``"vqarad"`` table becomes one (image, question,
        answer) sample.

        Args:
            patient: Patient record containing VQA-RAD events.

        Returns:
            A list of sample dicts with patient ID, image, question,
            and answer.
        """
        events = patient.get_events(event_type="vqarad")
        samples: List[Dict[str, Any]] = []
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
