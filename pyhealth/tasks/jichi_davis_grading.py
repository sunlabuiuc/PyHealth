"""Davis diabetic-retinopathy grading task for the Jichi fundus dataset."""

from __future__ import annotations

from typing import Any

from .base_task import BaseTask

DAVIS_GRADES: tuple[str, ...] = ("ndr", "sdr", "ppdr", "pdr")
"""Modified Davis grades in order of severity; the label is the index."""

_LABEL_COLUMNS = ("davis_one", "davis_concatenated")


class JichiDavisGrading(BaseTask):
    """Four-grade Davis diabetic-retinopathy staging from one fundus image.

    Works with :class:`~pyhealth.datasets.JichiDRDataset`.

    One sample per image. The label is the ordinal Davis grade as an integer:
    ``0`` no DR (ndr), ``1`` simple DR (sdr), ``2`` pre-proliferative DR
    (ppdr), ``3`` proliferative DR (pdr). Integers keep the clinical order;
    the raw strings would sort as ndr, pdr, ppdr, sdr. The class count is
    pinned to 4.

    CLIMB's code also uses these four classes, although its dataset table
    lists three (SDR, PPDR, PDR). A three-class variant can be built by
    dropping label ``0`` and subtracting 1.

    Args:
        label_column: ``"davis_one"`` (default, the grade of the single
            image, as in CLIMB) or ``"davis_concatenated"`` (the grade given
            with all of the patient's images in view).
        image_size: Side length of the resized square image.

    Raises:
        ValueError: If ``label_column`` is unknown.

    Examples:
        >>> from pyhealth.tasks import JichiDavisGrading
        >>> task = JichiDavisGrading(label_column="davis_concatenated")
        >>> task.output_schema["label"]
        ('multiclass', {'labels': 4})
        >>> from pyhealth.datasets import JichiDRDataset
        >>> dataset = JichiDRDataset(root="/data/jichi/dmr")  # doctest: +SKIP
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "JichiDavisGrading"

    def __init__(self, label_column: str = "davis_one", image_size: int = 224) -> None:
        if label_column not in _LABEL_COLUMNS:
            raise ValueError(
                f"label_column must be one of {list(_LABEL_COLUMNS)}, "
                f"got {label_column!r}."
            )
        super().__init__()
        self.label_column = label_column
        self.image_size = int(image_size)
        self.input_schema = {
            "image": ("image", {"image_size": self.image_size, "mode": "RGB"})
        }
        self.output_schema = {"label": ("multiclass", {"labels": len(DAVIS_GRADES)})}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per fundus image of the patient.

        Args:
            patient: A PyHealth ``Patient`` with ``jichi`` events.

        Returns:
            Samples with ``patient_id``, ``image_id``, ``eye``, ``image``
            (the JPEG path) and ``label`` (``0..3``). Images with an unknown
            grade are skipped.
        """
        samples = []
        for event in patient.get_events(event_type="jichi"):
            grade = str(getattr(event, self.label_column)).strip().lower()
            if grade not in DAVIS_GRADES:
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "image_id": event.image_id,
                    "eye": event.eye,
                    "image": event.path,
                    "label": DAVIS_GRADES.index(grade),
                }
            )
        return samples
