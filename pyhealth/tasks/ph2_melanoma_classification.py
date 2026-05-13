"""
PyHealth task for multiclass melanoma classification using the PH2 dataset.

Dataset source:
    https://www.kaggle.com/datasets/spacesurfer/ph2-dataset
"""

import logging
from typing import Dict, List

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


class PH2MelanomaClassification(BaseTask):
    """Multiclass lesion classification task for the PH2 dataset.

    Each dermoscopic image is classified into one of three categories:
    ``"common_nevus"``, ``"atypical_nevus"``, or ``"melanoma"``.

    This task is designed for use with
    :class:`~pyhealth.datasets.PH2Dataset`.

    Attributes:
        task_name (str): Unique task identifier.
        input_schema (Dict[str, str]): Maps ``"image"`` to the ``"image"``
            processor type.
        output_schema (Dict[str, str]): Maps ``"label"`` to ``"multiclass"``.

    Examples:
        >>> from pyhealth.datasets import PH2Dataset
        >>> from pyhealth.tasks import PH2MelanomaClassification
        >>> dataset = PH2Dataset(root="/path/to/ph2")
        >>> samples = dataset.set_task(PH2MelanomaClassification())
    """

    task_name: str = "PH2MelanomaClassification"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict]:
        """Generate multiclass classification samples for a single patient.

        Args:
            patient: A :class:`~pyhealth.data.Patient` object containing at
                least one ``"ph2"`` event.

        Returns:
            A list with one dict per image::

                [{"image": "/abs/path/img.bmp", "label": "melanoma"}, ...]
        """
        events: List[Event] = patient.get_events(event_type="ph2")

        samples = []
        for event in events:
            path = event["path"]
            diagnosis = event["diagnosis"]
            if not path or not diagnosis:
                continue
            samples.append({"image": path, "label": diagnosis})
        return samples
