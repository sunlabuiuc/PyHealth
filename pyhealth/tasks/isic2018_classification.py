"""
PyHealth task for multiclass classification using the ISIC 2018 dataset.

Dataset link:
    https://challenge.isic-archive.com/data/#2018

License:
    CC-BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)

Dataset paper: (please cite if you use this dataset)
    [1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, et al. "Skin Lesion
    Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the
    International Skin Imaging Collaboration (ISIC)", 2018;
    https://arxiv.org/abs/1902.03368

    [2] Tschandl, P., Rosendahl, C. & Kittler, H. "The HAM10000 dataset, a large
    collection of multi-source dermatoscopic images of common pigmented skin
    lesions." Sci. Data 5, 180161 (2018).

Dataset paper link:
    https://doi.org/10.1038/sdata.2018.161
"""

import logging
from typing import Dict, List

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


class ISIC2018Classification(BaseTask):
    """
    A PyHealth task class for multiclass skin lesion classification using the
    ISIC 2018 Task 3 dataset.

    The task maps each dermoscopy image to one of seven skin lesion categories.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.

    Examples:
        >>> from pyhealth.datasets import ISIC2018Dataset
        >>> from pyhealth.tasks import ISIC2018Classification
        >>> dataset = ISIC2018Dataset(root="/path/to/isic2018")
        >>> task = ISIC2018Classification()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "ISIC2018Classification"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __call__(self, patient: Patient) -> List[Dict]:
        """
        Generates multiclass classification data samples for a single patient.

        Args:
            patient (Patient): A patient object containing at least one
                               'isic2018' event.

        Returns:
            List[Dict]: A list containing a dictionary for each image with:
                - 'image': path to the dermoscopy image.
                - 'label': the skin lesion class label (str) from
                  ISIC2018Dataset.classes.
        """
        events: List[Event] = patient.get_events(event_type="isic2018")

        samples = []
        from pyhealth.datasets import ISIC2018Dataset  # Avoid circular import

        for event in events:
            label = next(
                (cls for cls in ISIC2018Dataset.classes if float(event[cls])),
                None,
            )
            if label is not None:
                samples.append(
                    {
                        "image": event["path"],
                        "label": label,
                    }
                )

        return samples
