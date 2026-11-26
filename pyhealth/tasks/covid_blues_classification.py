"""Frame-level COVID-19 classification task for COVID-BLUES lung ultrasound."""

from typing import Any, ClassVar

from .base_task import BaseTask


class COVIDBLUESFrameClassification(BaseTask):
    """Classify COVID-BLUES lung ultrasound frames as COVID-19 positive or not.

    Works with :class:`~pyhealth.datasets.COVIDBLUESDataset`.

    One sample per frame. The label ``covid`` is the patient's PCR result
    (``cov_test``), so every frame of a patient has the same label. Frames
    of one patient are strongly correlated: split by ``patient_id`` and
    consider aggregating predictions per video (``video_id``) or patient.

    Frames are loaded with the ``image`` processor, converted to single
    channel and resized to 224 x 224.

    Examples:
        >>> from pyhealth.datasets import COVIDBLUESDataset
        >>> from pyhealth.tasks import COVIDBLUESFrameClassification
        >>> COVIDBLUESFrameClassification().output_schema
        {'covid': 'binary'}
        >>> dataset = COVIDBLUESDataset(  # doctest: +SKIP
        ...     root="/path/to/COVID-BLUES", frames_dir="/path/to/frames"
        ... )
        >>> samples = dataset.set_task(  # doctest: +SKIP
        ...     COVIDBLUESFrameClassification()
        ... )
    """

    task_name: str = "COVIDBLUESFrameClassification"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": ("image", {"image_size": 224, "mode": "L"})
    }
    output_schema: ClassVar[dict[str, Any]] = {"covid": "binary"}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per frame of the patient.

        Args:
            patient: A PyHealth ``Patient`` with ``frames`` events.

        Returns:
            Sample dicts with ``image``, ``covid``, ``video_id`` and
            ``blue_point``.
        """
        samples = []
        for event in patient.get_events(event_type="frames"):
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": event.record_id,
                    "video_id": event.video_id,
                    "blue_point": event.blue_point,
                    "image": event.image_path,
                    "covid": int(float(event.covid)),
                }
            )
        return samples
