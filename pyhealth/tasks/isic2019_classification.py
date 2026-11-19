"""Eight-class (or CLIMB five-class) diagnosis task for ISIC 2019."""

from typing import Any, ClassVar

from .base_task import BaseTask

_ISIC8 = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")
_CLIMB5 = ("MEL", "NV", "BCC", "AKIEC", "OTHER")
_TO_CLIMB5 = {
    "AK": "AKIEC",
    "SCC": "AKIEC",
    "BKL": "OTHER",
    "DF": "OTHER",
    "VASC": "OTHER",
}


class ISIC2019Classification(BaseTask):
    """Classify ISIC 2019 dermoscopy images by diagnosis.

    Works with :class:`~pyhealth.datasets.ISIC2019Dataset`.

    One sample per image. The input is the RGB image resized to a square of
    ``image_size`` pixels. The ``diagnosis`` label uses one of two fixed
    class lists, so every class gets an output even if a split lacks it:

    - ``"isic8"`` (default): the eight official classes MEL, NV, BCC, AK,
      BKL, DF, VASC, SCC.
    - ``"climb5"``: the five classes used by CLIMB, MEL, NV, BCC, AKIEC (AK
      and SCC) and OTHER (BKL, DF and VASC).

    Test images labelled UNK (no training class) are skipped. ``patient_id``
    is the lesion where known, so a :class:`~pyhealth.datasets.PatientSplit`
    keeps every image of a lesion in one part. The classes are imbalanced
    (NV is 51% of training images, DF under 1%), so report balanced accuracy
    or macro F1 as well as accuracy.

    Cache size: the image processor stores each image as a float32 tensor,
    about 580 KB per sample at 224 px, or about 15 GB for the 25,331
    training images. Pass a smaller ``image_size`` to shrink it (128 px is
    about 190 KB per sample), or resize the JPEGs (up to 1024x1024)
    beforehand to speed up building the cache.

    Args:
        label_scheme: ``"isic8"`` or ``"climb5"``. Defaults to ``"isic8"``.
        exclude_ham10000: Skip images that are also in HAM10000 / ISIC 2018
            Task 3 (``is_ham10000``), e.g. to test a model trained on
            HAM10000 without leakage.
            Defaults to False.
        image_size: Side length of the square input image. Defaults to 224.
        split: Keep only images of this split (``"train"`` or ``"test"``).
            Defaults to None, all indexed images.

    Examples:
        >>> from pyhealth.datasets import ISIC2019Dataset
        >>> from pyhealth.tasks import ISIC2019Classification
        >>> dataset = ISIC2019Dataset(root="/path/to/isic2019")  # doctest: +SKIP
        >>> samples = dataset.set_task(
        ...     ISIC2019Classification(exclude_ham10000=True)
        ... )  # doctest: +SKIP
        >>> ISIC2019Classification(label_scheme="climb5").output_schema
        {'diagnosis': ('multiclass', {'labels': ['MEL', 'NV', 'BCC', 'AKIEC', 'OTHER']})}
    """

    task_name: str = "ISIC2019Classification"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": ("image", {"image_size": 224, "mode": "RGB"})
    }
    output_schema: ClassVar[dict[str, Any]] = {
        "diagnosis": ("multiclass", {"labels": list(_ISIC8)})
    }

    def __init__(
        self,
        label_scheme: str = "isic8",
        exclude_ham10000: bool = False,
        image_size: int = 224,
        split: str | None = None,
    ) -> None:
        super().__init__()
        if label_scheme not in ("isic8", "climb5"):
            raise ValueError(
                f"label_scheme must be 'isic8' or 'climb5', got {label_scheme!r}."
            )
        if split not in (None, "train", "test"):
            raise ValueError(f"split must be train, test or None, got {split!r}.")
        self.label_scheme = label_scheme
        self.exclude_ham10000 = bool(exclude_ham10000)
        self.image_size = int(image_size)
        self.split = split
        labels = _ISIC8 if label_scheme == "isic8" else _CLIMB5
        self.input_schema = {
            "image": ("image", {"image_size": self.image_size, "mode": "RGB"})
        }
        self.output_schema = {"diagnosis": ("multiclass", {"labels": list(labels)})}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per ``isic2019`` image event of a patient.

        Args:
            patient: A PyHealth ``Patient`` with ``isic2019`` events.

        Returns:
            Samples with ``patient_id``, ``image_id``, ``split``, ``image``
            (the JPEG path) and ``diagnosis``.
        """
        samples = []
        for event in patient.get_events(event_type="isic2019"):
            if event.diagnosis == "UNK":
                continue
            if self.split is not None and event.split != self.split:
                continue
            if self.exclude_ham10000 and int(float(event.is_ham10000)):
                continue
            diagnosis = event.diagnosis
            if self.label_scheme == "climb5":
                diagnosis = _TO_CLIMB5.get(diagnosis, diagnosis)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "image_id": event.image_id,
                    "split": event.split,
                    "image": event.path,
                    "diagnosis": diagnosis,
                }
            )
        return samples
