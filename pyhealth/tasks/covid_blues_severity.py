"""Lung ultrasound severity scoring (0-3) on COVID-BLUES frames."""

from typing import Any, ClassVar

from .base_task import BaseTask

#: Number of severity classes (Soldati et al., 2020 scores 0, 1, 2, 3).
COVID_BLUES_SEVERITY_LEVELS = 4


class COVIDBLUESSeverityClassification(BaseTask):
    """Predict the expert LUS severity score (0-3) of a COVID-BLUES frame.

    Works with :class:`~pyhealth.datasets.COVIDBLUESDataset`.

    Two medical experts scored every video with the lung ultrasound scheme
    of Soldati et al. (2020), from 0 (continuous pleural line, A-lines) to
    3 (dense, extended white lung). The scores are in
    the dataset's ``severity.csv``, one per video; this task gives every
    frame its video's score as the ``severity`` label (``multiclass``, 4
    classes) and skips frames whose video has no score. In the released
    ``severity.csv`` the 362 videos split 120 / 111 / 83 / 48 over scores
    0-3. As for :class:`~pyhealth.tasks.COVIDBLUESFrameClassification`,
    split by ``patient_id`` and consider aggregating per video.

    Frames are loaded with the ``image`` processor, converted to single
    channel and resized to 224 x 224.

    Examples:
        >>> from pyhealth.tasks import COVIDBLUESSeverityClassification
        >>> COVIDBLUESSeverityClassification().output_schema
        {'severity': ('multiclass', {'labels': 4})}
        >>> from pyhealth.datasets import COVIDBLUESDataset
        >>> dataset = COVIDBLUESDataset(  # doctest: +SKIP
        ...     root="/path/to/COVID-BLUES", frames_dir="/path/to/frames"
        ... )
        >>> samples = dataset.set_task(  # doctest: +SKIP
        ...     COVIDBLUESSeverityClassification()
        ... )
    """

    task_name: str = "COVIDBLUESSeverityClassification"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": ("image", {"image_size": 224, "mode": "L"})
    }
    output_schema: ClassVar[dict[str, Any]] = {
        "severity": ("multiclass", {"labels": COVID_BLUES_SEVERITY_LEVELS})
    }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per frame whose video has a severity score.

        Args:
            patient: A PyHealth ``Patient`` with ``frames`` events.

        Returns:
            Sample dicts with ``image``, ``severity``, ``video_id`` and
            ``blue_point``.
        """
        samples = []
        for event in patient.get_events(event_type="frames"):
            score = event.severity_score
            if score is None or str(score).strip() in ("", "nan", "None"):
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": event.record_id,
                    "video_id": event.video_id,
                    "blue_point": event.blue_point,
                    "image": event.image_path,
                    "severity": round(float(score)),
                }
            )
        return samples
