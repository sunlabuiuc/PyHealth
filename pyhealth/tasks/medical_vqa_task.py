"""Medical Visual Question Answering task for the VQA-RAD dataset.

This module defines :class:`MedicalVQATask`, which converts raw VQA-RAD
patient events (each consisting of a radiology image, a clinical question,
and a free-text answer) into image-question-answer samples suitable for
multiclass classification.

The task frames VQA as **closed-set multiclass classification** over the
vocabulary of all answers seen during training.  At inference time the model
selects the most probable answer from this fixed vocabulary.  Open-ended
generation is supported separately via :meth:`~pyhealth.models.MedFlamingo.generate`.

Paper:
    Lau et al. "A dataset of clinically generated visual questions and
    answers about radiology images." Scientific Data 5, 180251 (2018).
    https://doi.org/10.1038/sdata.2018.251
"""

from typing import Any, Dict, List

from ..data import Patient
from .base_task import BaseTask


class MedicalVQATask(BaseTask):
    """Task for medical visual question answering on the VQA-RAD dataset.

    Each sample pairs a radiology image with a clinical question and maps
    the corresponding free-text answer to a class index.  The full answer
    vocabulary is inferred from the training split by the PyHealth processor
    pipeline.

    Input schema:
        - ``image`` (``"image"``): A radiology image path, processed by
          :class:`~pyhealth.processors.ImageProcessor` into a
          ``(3, 224, 224)`` float tensor.
        - ``question`` (``"text"``): A free-text clinical question string
          (returned as-is by :class:`~pyhealth.processors.TextProcessor`).

    Output schema:
        - ``answer`` (``"multiclass"``): The free-text answer string, encoded
          as an integer class index by
          :class:`~pyhealth.processors.MulticlassProcessor`.

    Attributes:
        task_name: Unique identifier used for cache-key generation.
        input_schema: Maps feature names to their processor type strings.
        output_schema: Maps label names to their processor type strings.

    Examples:
        >>> from pyhealth.tasks import MedicalVQATask
        >>> task = MedicalVQATask()
        >>> task.task_name
        'MedicalVQA'
        >>> task.input_schema
        {'image': 'image', 'question': 'text'}
        >>> task.output_schema
        {'answer': 'multiclass'}
    """

    task_name: str = "MedicalVQA"
    input_schema: Dict[str, str] = {"image": "image", "question": "text"}
    output_schema: Dict[str, str] = {"answer": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Convert a VQA-RAD patient's events into image-question-answer samples.

        Iterates over all events of type ``"vqarad"`` attached to ``patient``
        and emits one sample dict per event.  Events without a valid
        ``image_path`` are included; the downstream
        :class:`~pyhealth.processors.ImageProcessor` will raise an error if
        the path does not point to a readable image file.

        Args:
            patient: A :class:`~pyhealth.data.Patient` object whose events
                were populated by :class:`~pyhealth.datasets.VQARADDataset`.

        Returns:
            A list of sample dicts, each with the keys:

            - ``"patient_id"`` (:class:`str`): The patient identifier.
            - ``"image"`` (:class:`str`): Absolute path to the radiology image.
            - ``"question"`` (:class:`str`): The clinical question text.
            - ``"answer"`` (:class:`str`): The free-text answer string (will be
              encoded as an integer by the multiclass processor).

        Example:
            >>> # Typically called internally by BaseDataset.set_task()
            >>> samples = dataset.set_task(MedicalVQATask())
            >>> samples[0].keys()
            dict_keys(['patient_id', 'image', 'question', 'answer'])
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
