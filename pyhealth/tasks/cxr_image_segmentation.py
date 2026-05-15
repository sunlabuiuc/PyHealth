"""
Chest X-Ray Lung Segmentation Task implementation.
"""

from typing import Any

from pyhealth.tasks import BaseTask


class CXRSegmentationTask(BaseTask):
    """
    Task for CXR Lung Segmentation.

    This task transforms patient events into segmentation samples.
    It expects each event to have an image path and a mask path.

    Args:
        image_config (dict[str, Any] | None, optional): Optional dictionary of
            kwargs for the image ImageProcessor. Defaults to {"mode": "L"}.
        mask_config (dict[str, Any] | None, optional): Optional dictionary of
            kwargs for the mask ImageProcessor. Defaults to {"mode": "L"}.
    """

    def __init__(
        self,
        image_config: dict[str, Any] | None = None,
        mask_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_attr = "image_path"
        self.mask_attr = "mask_path"

        # Default both image and mask to grayscale ('L' mode)
        image_config = image_config or {"mode": "L"}
        mask_config = mask_config or {"mode": "L"}

        # input_schema and output_schema are used by SampleBuilder
        # to instantiate the correct processors.
        # We use a tuple (name, config) to pass arguments to the processor.
        self.input_schema = {"image": ("image", image_config)}
        self.output_schema = {"mask": ("image", mask_config)}
        self.task_name = "CXRSegmentationTask"

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """
        Process a patient's events into segmentation samples.

        Args:
            patient (Any): A Patient object.

        Returns:
            list[dict[str, Any]]: A list of samples, where each sample is a
                dictionary containing image path, mask path, and patient_id.
        """
        # Get all events (for CXR segmentation, usually one event per patient)
        events = patient.get_events()
        samples = []
        for event in events:
            samples.append(
                {
                    "image": getattr(event, self.image_attr),
                    "mask": getattr(event, self.mask_attr),
                    "patient_id": patient.patient_id,
                }
            )
        return samples
