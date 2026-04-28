"""
PyHealth task for binary melanoma classification using the ISIC 2018 artifact
annotation dataset (Bissoto et al. 2020).

Dataset link:
    https://challenge.isic-archive.com/data/#2018

Annotation source:
    Bissoto et al. "Debiasing Skin Lesion Datasets and Models? Not So Fast"
    ISIC Skin Image Analysis Workshop @ CVPR 2020
    https://github.com/alceubissoto/debiasing-skin

License:
    CC-0 (Public Domain) — https://creativecommons.org/public-domain/cc0/
"""

import logging
from typing import Dict, List

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


class ISIC2018ArtifactsBinaryClassification(BaseTask):
    """Binary melanoma classification task for the ISIC 2018 artifact dataset.

    Each dermoscopy image is mapped to a binary label (1 = malignant,
    0 = benign) as provided by the Bissoto et al. ``isic_bias.csv``
    annotation file.

    This task is designed for use with
    :class:`~pyhealth.datasets.ISIC2018ArtifactsDataset`.  The dataset's
    ``set_task`` method automatically injects a
    :class:`~pyhealth.processors.DermoscopicImageProcessor`, so the
    ``mode`` (e.g. ``"whole"``, ``"lesion"``) is controlled at the dataset
    level, not here.

    Attributes:
        task_name (str): Unique task identifier.
        input_schema (Dict[str, str]): Maps ``"image"`` to the ``"image"``
            processor type.
        output_schema (Dict[str, str]): Maps ``"label"`` to ``"binary"``.

    Examples:
        >>> from pyhealth.datasets import ISIC2018ArtifactsDataset
        >>> from pyhealth.tasks import ISIC2018ArtifactsBinaryClassification
        >>> dataset = ISIC2018ArtifactsDataset(
        ...     root="/path/to/data",
        ...     image_dir="ISIC2018_Task1-2_Training_Input",
        ...     mask_dir="ISIC2018_Task1_Training_GroundTruth",
        ...     mode="whole",
        ... )
        >>> task = ISIC2018ArtifactsBinaryClassification()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "ISIC2018ArtifactsBinaryClassification"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __call__(self, patient: Patient) -> List[Dict]:
        """Generate binary classification samples for a single patient.

        Args:
            patient: A :class:`~pyhealth.data.Patient` object containing at
                least one ``"isic_artifacts"`` event.

        Returns:
            A list with one dict per image::

                [{"image": "/abs/path/to/ISIC_XXXXXXX.png", "label": 0}, ...]
        """
        events: List[Event] = patient.get_events(event_type="isic_artifacts")

        samples = []
        for event in events:
            samples.append(
                {
                    "image": event["path"],
                    "label": int(event["label"]),
                }
            )
        return samples
