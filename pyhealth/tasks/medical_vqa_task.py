"""Medical Visual Question Answering (VQA) task.

This module defines the task for medical VQA, where the model receives a
medical image and a natural-language question and must predict the correct
answer. The primary benchmark is VQA-RAD (Lau et al., 2018).

The task frames VQA as **multiclass classification** over a closed answer
vocabulary extracted from the training set. This is the standard evaluation
protocol used by MedFlamingo (Moor et al., 2023) and other medical VQA
models on VQA-RAD.
"""

from typing import Any, Dict, List

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

    Note:
        The ``"text"`` processor for ``"question"`` will tokenize the
        question string. If your model needs raw strings instead, you
        can override the processor in ``dataset.set_task()``. The assumed
        schema here is a reasonable default -- adjust once Teammate A
        confirms the final field names and processor types.

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

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient's VQA data into samples.

        Each event in the ``"vqarad"`` table becomes one (image, question,
        answer) sample.

        Args:
            patient: A patient object from :class:`~pyhealth.datasets.VQARADDataset`.

        Returns:
            A list of sample dicts, each with keys ``"image"``,
            ``"question"``, and ``"answer"``.
        """
        events = patient.get_events(event_type="vqarad")
        samples: List[Dict[str, Any]] = []
        for event in events:
            samples.append(
                {
                    "image": event.image_path,
                    "question": event.question,
                    "answer": event.answer,
                }
            )
        return samples
