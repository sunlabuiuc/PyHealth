"""RSNA Pneumonia binary classification task."""

from typing import Any, ClassVar

from .base_task import BaseTask


class RSNAPneumoniaClassification(BaseTask):
    """Classify RSNA chest X-rays as lung-opacity positive or negative.

    The task reads one DICOM image per sample, applies grayscale ImageNet-style
    normalisation, and uses the binary ``pneumonia`` label from the RSNA
    metadata (adjudicated annotations when available).

    Examples:
        >>> from pyhealth.datasets import RSNAPneumoniaDataset
        >>> from pyhealth.tasks import RSNAPneumoniaClassification
        >>> dataset = RSNAPneumoniaDataset(
        ...     root="/path/to/rsna"
        ... )  # doctest: +SKIP
        >>> samples = dataset.set_task(
        ...     RSNAPneumoniaClassification()
        ... )  # doctest: +SKIP
    """

    task_name: str = "RSNAPneumoniaClassification"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": (
            "dicom",
            {
                "lazy": True,
                "mode": "L",
                "normalize": True,
                "mean": [0.449],
                "std": [0.226],
            },
        ),
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
