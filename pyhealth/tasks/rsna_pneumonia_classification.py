"""RSNA Pneumonia binary classification task."""

from typing import Any, ClassVar

from .base_task import BaseTask


class RSNAPneumoniaClassification(BaseTask):
    """Classify RSNA chest X-rays as lung-opacity positive or negative.

    The task reads one DICOM image per sample and uses the binary
    ``pneumonia`` label from the RSNA mapping metadata.

    Examples:
        >>> from pyhealth.datasets import RSNAPneumoniaDataset
        >>> from pyhealth.tasks import RSNAPneumoniaClassification
        >>> dataset = RSNAPneumoniaDataset(root="/path/to/rsna")
        >>> samples = dataset.set_task(RSNAPneumoniaClassification())
    """

    task_name: str = "RSNAPneumoniaClassification"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": ("dicom", {"lazy": True, "mode": "L"}),
    }
    output_schema: ClassVar[dict[str, str]] = {"pneumonia": "binary"}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per RSNA image event.

        Args:
            patient: A PyHealth ``Patient`` with ``images`` events.

        Returns:
            A list of sample dictionaries with ``image`` and ``pneumonia``.
        """
        samples = []
        for event in patient.get_events(event_type="images"):
            if not getattr(event, "image_path", None):
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": event.record_id,
                    "image": event.image_path,
                    "pneumonia": int(event.label),
                }
            )
        return samples
