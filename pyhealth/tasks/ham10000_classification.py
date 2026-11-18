"""Seven-class (or CLIMB five-class) diagnosis task for HAM10000."""

from typing import Any, ClassVar

from .base_task import BaseTask

_HAM7 = ("MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC")
_CLIMB5 = ("MEL", "NV", "BCC", "AKIEC", "OTHER")
_TO_CLIMB5 = {"BKL": "OTHER", "DF": "OTHER", "VASC": "OTHER"}


class HAM10000Classification(BaseTask):
    """Classify HAM10000 dermoscopy images by diagnosis.

    Works with :class:`~pyhealth.datasets.HAM10000Dataset`.

    One sample per image. The input is the RGB image resized to a square of
    ``image_size`` pixels. The ``diagnosis`` label uses one of two fixed
    class lists, so every class gets an output even if a split lacks it:

    - ``"ham7"`` (default): the official seven classes MEL, NV, BCC, AKIEC,
      BKL, DF, VASC.
    - ``"climb5"``: the five classes used by CLIMB, with BKL, DF and VASC
      merged into OTHER: MEL, NV, BCC, AKIEC, OTHER.

    ``patient_id`` is the lesion, so a :class:`~pyhealth.datasets.PatientSplit`
    keeps every image of a lesion in one part. NV is 67% of the training
    images, so report balanced accuracy or macro F1 as well as accuracy.

    Cache size: the image processor stores each image as a float32 tensor,
    about 580 KB per sample at 224 px, or about 6 GB for the 10,015 training
    images. Pass a smaller ``image_size`` to shrink it (128 px is about
    190 KB per sample), or resize the JPEGs beforehand to speed up building
    the cache.

    Args:
        label_scheme: ``"ham7"`` or ``"climb5"``. Defaults to ``"ham7"``.
        image_size: Side length of the square input image. Defaults to 224.
        split: Keep only images of this official split (``"train"``,
            ``"val"`` or ``"test"``). Defaults to None, all indexed images.

    Examples:
        >>> from pyhealth.datasets import HAM10000Dataset
        >>> from pyhealth.tasks import HAM10000Classification
        >>> dataset = HAM10000Dataset(root="/path/to/isic2018")  # doctest: +SKIP
        >>> samples = dataset.set_task(HAM10000Classification())  # doctest: +SKIP
        >>> HAM10000Classification(label_scheme="climb5").output_schema
        {'diagnosis': ('multiclass', {'labels': ['MEL', 'NV', 'BCC', 'AKIEC', 'OTHER']})}
    """

    task_name: str = "HAM10000Classification"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": ("image", {"image_size": 224, "mode": "RGB"})
    }
    output_schema: ClassVar[dict[str, Any]] = {
        "diagnosis": ("multiclass", {"labels": list(_HAM7)})
    }

    def __init__(
        self,
        label_scheme: str = "ham7",
        image_size: int = 224,
        split: str | None = None,
    ) -> None:
        super().__init__()
        if label_scheme not in ("ham7", "climb5"):
            raise ValueError(
                f"label_scheme must be 'ham7' or 'climb5', got {label_scheme!r}."
            )
        if split not in (None, "train", "val", "test"):
            raise ValueError(f"split must be train, val, test or None, got {split!r}.")
        self.label_scheme = label_scheme
        self.image_size = int(image_size)
        self.split = split
        labels = _HAM7 if label_scheme == "ham7" else _CLIMB5
        self.input_schema = {
            "image": ("image", {"image_size": self.image_size, "mode": "RGB"})
        }
        self.output_schema = {"diagnosis": ("multiclass", {"labels": list(labels)})}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per ``ham10000`` image event of a patient.

        Args:
            patient: A PyHealth ``Patient`` with ``ham10000`` events.

        Returns:
            Samples with ``patient_id``, ``image_id``, ``split``, ``image``
            (the JPEG path) and ``diagnosis``.
        """
        samples = []
        for event in patient.get_events(event_type="ham10000"):
            if self.split is not None and event.split != self.split:
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
