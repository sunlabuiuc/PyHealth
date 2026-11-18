"""Benign vs. malignant classification task for ISIC 2020."""

from typing import Any, ClassVar

from .base_task import BaseTask


class ISIC2020MelanomaClassification(BaseTask):
    """Classify ISIC 2020 dermoscopy images as benign or malignant.

    Works with :class:`~pyhealth.datasets.ISIC2020Dataset`.

    One sample per image. The input is the RGB image resized to a square of
    ``image_size`` pixels (the 3:2 photos are stretched, not padded); the
    binary ``malignant`` label is the challenge ``target``. ``patient_id``
    is the ISIC patient, so a :class:`~pyhealth.datasets.PatientSplit` keeps
    each patient's images in one part. Only 1.8% of images are malignant:
    report AUROC or PR-AUC rather than accuracy.

    Cache size: the image processor stores each image as a float32 tensor,
    about 580 KB per sample at 224 px, or about 19 GB for the 32,701 images
    left after dropping duplicates. The source JPEGs are up to 6000x4000, so
    building the cache is also slow. Resize the JPEGs once beforehand (for
    example to 256 px on the short side) to speed it up, and pass a smaller
    ``image_size`` to shrink the cache (128 px is about 190 KB per sample).

    Args:
        image_size: Side length of the square input image. Defaults to 224.

    Examples:
        >>> from pyhealth.datasets import ISIC2020Dataset
        >>> from pyhealth.tasks import ISIC2020MelanomaClassification
        >>> dataset = ISIC2020Dataset(root="/path/to/isic2020")  # doctest: +SKIP
        >>> samples = dataset.set_task(
        ...     ISIC2020MelanomaClassification()
        ... )  # doctest: +SKIP
        >>> ISIC2020MelanomaClassification(image_size=128).input_schema["image"]
        ('image', {'image_size': 128, 'mode': 'RGB'})
    """

    task_name: str = "ISIC2020MelanomaClassification"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": ("image", {"image_size": 224, "mode": "RGB"})
    }
    output_schema: ClassVar[dict[str, str]] = {"malignant": "binary"}

    def __init__(self, image_size: int = 224) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.input_schema = {
            "image": ("image", {"image_size": self.image_size, "mode": "RGB"})
        }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per ``isic2020`` image event of a patient.

        Args:
            patient: A PyHealth ``Patient`` with ``isic2020`` events.

        Returns:
            Samples with ``patient_id``, ``image_id``, ``image`` (the JPEG
            path) and ``malignant`` (0 or 1).
        """
        return [
            {
                "patient_id": patient.patient_id,
                "image_id": event.image_id,
                "image": event.path,
                "malignant": int(float(event.target)),
            }
            for event in patient.get_events(event_type="isic2020")
        ]
