"""Image-level classification tasks for the CBIS-DDSM mammography dataset."""

from typing import Any, ClassVar

from .base_task import BaseTask

_SPLITS = ("train", "test")

# DicomProcessor resizes to a square, so the roughly 1.6:1 mammograms are
# squashed to image_size x image_size (no padding).
_IMAGE_SCHEMA = (
    "dicom",
    {
        "lazy": True,
        "image_size": 512,
        "mode": "L",
        "normalize": True,
        "mean": [0.5],
        "std": [0.5],
    },
)


def _check_split(split: str | None) -> str | None:
    """Validate the official-split filter of a CBIS-DDSM task."""
    if split is not None and split not in _SPLITS:
        raise ValueError(f"split must be None, 'train' or 'test', got {split!r}.")
    return split


def _events_by_image(patient: Any, split: str | None) -> dict:
    """Group a patient's lesion events by (image_id, official split)."""
    images: dict = {}
    for event in patient.get_events(event_type="lesions"):
        if split is not None and event.split != split:
            continue
        images.setdefault((event.image_id, event.split), []).append(event)
    return images


class CBISDDSMPathologyClassification(BaseTask):
    """Image-level malignant vs benign classification on CBIS-DDSM.

    Works with :class:`~pyhealth.datasets.CBISDDSMDataset`.

    One sample per full-field mammogram (patient, side, view) and official
    split. The image is malignant (1) when any of its lesions has pathology
    ``MALIGNANT``; ``BENIGN`` and ``BENIGN_WITHOUT_CALLBACK`` count as
    benign (0). Each sample keeps the official ``split`` ("train" or
    "test").

    The image is read lazily with the ``dicom`` processor at 512 x 512,
    single channel. The processor resizes to a square, so the roughly 1.6:1
    mammograms are squashed rather than padded.

    Args:
        split: Keep only images from this official split (``"train"`` or
            ``"test"``). Default None keeps both.

    Examples:
        >>> from pyhealth.datasets import CBISDDSMDataset
        >>> from pyhealth.tasks import CBISDDSMPathologyClassification
        >>> CBISDDSMPathologyClassification().output_schema
        {'malignant': 'binary'}
        >>> dataset = CBISDDSMDataset(root="/path/to/cbis-ddsm")  # doctest: +SKIP
        >>> samples = dataset.set_task(  # doctest: +SKIP
        ...     CBISDDSMPathologyClassification()
        ... )
    """

    task_name: str = "CBISDDSMPathologyClassification"
    input_schema: ClassVar[dict[str, Any]] = {"image": _IMAGE_SCHEMA}
    output_schema: ClassVar[dict[str, Any]] = {"malignant": "binary"}

    def __init__(self, split: str | None = None) -> None:
        self.split = _check_split(split)
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per mammogram of the patient.

        Args:
            patient: A PyHealth ``Patient`` with ``lesions`` events.

        Returns:
            Sample dicts with ``image``, ``malignant`` and ``split``.
        """
        samples = []
        for (image_id, split), events in _events_by_image(patient, self.split).items():
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": f"{image_id}_{split}",
                    "split": split,
                    "image": events[0].image_path,
                    "malignant": int(
                        any(str(e.pathology) == "MALIGNANT" for e in events)
                    ),
                }
            )
        return samples


class CBISDDSMBIRADSClassification(BaseTask):
    """Image-level BI-RADS assessment (1-5) classification on CBIS-DDSM.

    Works with :class:`~pyhealth.datasets.CBISDDSMDataset`.

    Lesions with BI-RADS 0 (incomplete assessment) are dropped and each image
    takes the highest remaining assessment, so there are 5 classes with
    labels 1 to 5 (class indices 0 to 4). This follows CLIMB's code
    (``NUM_CLASSES=5``); CLIMB's documentation says 6 classes (BI-RADS 0-5),
    but BI-RADS 0 is not a diagnostic category and CLIMB's code drops it too.
    The classes are very imbalanced: BI-RADS 1 has 3 images in the whole
    collection and BI-RADS 4 about half.

    Uses the same lazy 512 x 512 ``dicom`` input as
    :class:`CBISDDSMPathologyClassification` (square resize, no padding).

    Args:
        split: Keep only images from this official split (``"train"`` or
            ``"test"``). Default None keeps both.

    Examples:
        >>> from pyhealth.datasets import CBISDDSMDataset
        >>> from pyhealth.tasks import CBISDDSMBIRADSClassification
        >>> CBISDDSMBIRADSClassification().output_schema["birads"]
        ('multiclass', {'labels': [1, 2, 3, 4, 5]})
        >>> dataset = CBISDDSMDataset(root="/path/to/cbis-ddsm")  # doctest: +SKIP
        >>> samples = dataset.set_task(  # doctest: +SKIP
        ...     CBISDDSMBIRADSClassification(split="train")
        ... )
    """

    task_name: str = "CBISDDSMBIRADSClassification"
    input_schema: ClassVar[dict[str, Any]] = {"image": _IMAGE_SCHEMA}
    output_schema: ClassVar[dict[str, Any]] = {
        "birads": ("multiclass", {"labels": [1, 2, 3, 4, 5]})
    }

    def __init__(self, split: str | None = None) -> None:
        self.split = _check_split(split)
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per mammogram with a BI-RADS 1-5 lesion.

        Args:
            patient: A PyHealth ``Patient`` with ``lesions`` events.

        Returns:
            Sample dicts with ``image``, ``birads`` (1-5) and ``split``.
        """
        samples = []
        for (image_id, split), events in _events_by_image(patient, self.split).items():
            scores = [int(float(e.assessment)) for e in events]
            scores = [s for s in scores if s > 0]
            if not scores:
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": f"{image_id}_{split}",
                    "split": split,
                    "image": events[0].image_path,
                    "birads": max(scores),
                }
            )
        return samples
