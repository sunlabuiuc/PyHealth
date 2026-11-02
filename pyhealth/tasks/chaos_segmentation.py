"""Segmentation task for the CHAOS CT/MRI dataset."""

from typing import Any, ClassVar

from .base_task import BaseTask


class CHAOSSegmentationTask(BaseTask):
    """Build DICOM-series and image-mask samples from CHAOS.

    The image is read lazily by ``DicomSeriesProcessor``. The ground-truth
    image slices are recorded lazily by ``ImageMaskDirectoryProcessor`` and
    stacked into integer labels when a sample is read.

    Examples:
        >>> from pyhealth.datasets import CHAOSDataset
        >>> from pyhealth.tasks import CHAOSSegmentationTask
        >>> dataset = CHAOSDataset(root="/path/to/CHAOS")
        >>> samples = dataset.set_task(CHAOSSegmentationTask())
    """

    task_name: str = "CHAOSSegmentation"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": ("dicom_series", {"lazy": True}),
    }
    output_schema: ClassVar[dict[str, Any]] = {
        "mask": ("image_mask_dir", {"lazy": True}),
    }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per labelled CHAOS series.

        Args:
            patient: A PyHealth ``Patient`` with ``series`` events.

        Returns:
            A list of sample dictionaries. ``image`` is the DICOM series path
            and ``mask`` is a ``(mask_dir, series_path)`` tuple used to align
            image slices with the DICOM slice order.
        """
        samples = []
        for event in patient.get_events(event_type="series"):
            if not getattr(event, "mask_dir", None):
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": event.record_id,
                    "image": event.series_path,
                    "mask": (event.mask_dir, event.series_path),
                }
            )
        return samples
