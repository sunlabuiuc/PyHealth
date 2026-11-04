"""Image generation task for the COVID-19 Radiography Database.

Author: Sameh Aissa
Description:
    Turns every chest X-ray in :class:`~pyhealth.datasets.COVID19CXRDataset`
    into an unlabeled sample for generative models such as
    :class:`~pyhealth.models.VAE`.
"""

from typing import Any, ClassVar

from .base_task import BaseTask


class COVID19CXRImageGeneration(BaseTask):
    """A task for generating chest X-ray images.

    Every X-ray becomes one sample holding only the image, so the output
    schema is empty. Images are loaded as single-channel grayscale, since the
    X-rays carry no color, and resized to 128 x 128.

    Attributes:
        task_name (str): The name of the task, set to
            "COVID19CXRImageGeneration".
        input_schema (dict[str, Any]): A single key "image", processed as a
            grayscale image of 128 x 128 pixels.
        output_schema (dict[str, Any]): Empty, since generation has no label.

    Examples:
        >>> from pyhealth.datasets import COVID19CXRDataset
        >>> from pyhealth.tasks import COVID19CXRImageGeneration
        >>> dataset = COVID19CXRDataset(root="/path/to/covid19_cxr")
        >>> samples = dataset.set_task(COVID19CXRImageGeneration())
    """

    task_name: str = "COVID19CXRImageGeneration"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": ("image", {"image_size": 128, "mode": "L"})
    }
    output_schema: ClassVar[dict[str, Any]] = {}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Returns one sample per chest X-ray of the patient.

        Args:
            patient: A patient object holding "covid19_cxr" events.

        Returns:
            list[dict[str, Any]]: One dictionary per X-ray, with the image path
                under "image".
        """
        events = patient.get_events(event_type="covid19_cxr")
        return [{"image": event.path} for event in events]
