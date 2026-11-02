"""Segmentation task for the Medical Segmentation Decathlon datasets."""

from typing import Any, ClassVar

from .base_task import BaseTask


class MSDSegmentationTask(BaseTask):
    """Build image-mask samples from an MSD dataset.

    The task uses NIfTI processors for both the image and the mask. The mask
    is an output target and is processed as an integer, nearest-neighbour
    volume.

    Examples:
        >>> from pyhealth.datasets import MSDHippocampusDataset
        >>> from pyhealth.tasks import MSDSegmentationTask
        >>> dataset = MSDHippocampusDataset(root="/path/to/Task04_Hippocampus")
        >>> samples = dataset.set_task(MSDSegmentationTask())
    """

    task_name: str = "MSDSegmentation"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": ("nifti", {"lazy": True}),
    }
    output_schema: ClassVar[dict[str, Any]] = {
        "mask": ("nifti", {"lazy": True, "is_mask": True}),
    }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per labelled MSD case.

        Args:
            patient: A PyHealth ``Patient`` with ``cases`` events.

        Returns:
            A list of sample dictionaries with ``image`` and ``mask`` paths.
        """
        samples = []
        for event in patient.get_events(event_type="cases"):
            if str(getattr(event, "split", "")) != "train":
                continue
            if not getattr(event, "label_path", None):
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": event.record_id,
                    "image": event.image_path,
                    "mask": event.label_path,
                }
            )
        return samples
