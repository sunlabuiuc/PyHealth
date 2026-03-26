"""Task for binary melanoma classification from dermoscopic images.

Works with the composite DermoscopyDataset which combines ISIC 2018,
HAM10000, and PH2 datasets. Each sample yields an (image_path, mask_path)
tuple as input, enabling the DermoscopyImageProcessor to apply mode-based
processing (whole, lesion, or background).

Author:
    Generated for PyHealth integration of dermoscopic_artifacts datasets.
"""

from typing import Any, Dict, List, Optional

from .base_task import BaseTask


class DermoscopyMelanomaClassification(BaseTask):
    """Binary melanoma classification task for dermoscopic images.

    Each patient event contains a dermoscopic image path, a segmentation
    mask path, and a binary label (0 = benign, 1 = melanoma). The task
    returns the image and mask paths as a tuple so that the
    DermoscopyImageProcessor can apply mode-based masking.

    Args:
        source_dataset: If provided, only samples whose ``source_dataset``
            field matches this string are returned. Pass one of
            ``"isic2018"``, ``"ham10000"``, or ``"ph2"`` to restrict
            evaluation/training to a single sub-dataset. Defaults to
            ``None`` (all sub-datasets included).

    Attributes:
        task_name: "DermoscopyMelanomaClassification"
        input_schema: {"image": "dermoscopy_image"} — maps to DermoscopyImageProcessor
        output_schema: {"melanoma": "binary"} — binary classification

    Examples:
        >>> from pyhealth.datasets import DermoscopyDataset
        >>> from pyhealth.tasks import DermoscopyMelanomaClassification
        >>> dataset = DermoscopyDataset(root="/path/to/data")
        >>> # All sub-datasets
        >>> samples = dataset.set_task(DermoscopyMelanomaClassification())
        >>> # ISIC 2018 only
        >>> samples = dataset.set_task(DermoscopyMelanomaClassification(source_dataset="isic2018"))
    """

    task_name: str = "DermoscopyMelanomaClassification"
    input_schema: Dict[str, str] = {"image": "dermoscopy_image"}
    output_schema: Dict[str, str] = {"melanoma": "binary"}

    def __init__(self, source_dataset: Optional[str] = None) -> None:
        self.source_dataset = source_dataset
        # Make task instances with different filters hash to different cache keys.
        if source_dataset is not None:
            self.task_name = f"DermoscopyMelanomaClassification_{source_dataset}"

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Extract melanoma classification samples from a patient record.

        Args:
            patient: A Patient object containing dermoscopy events.

        Returns:
            List of sample dicts, each with:
                - "image": tuple of (image_path, mask_path)
                - "melanoma": integer label (0 or 1)
            Events whose ``source_dataset`` does not match ``self.source_dataset``
            are skipped when a filter is active.
        """
        events = patient.get_events(event_type="dermoscopy")
        samples = []
        for event in events:
            if (
                self.source_dataset is not None
                and event.source_dataset != self.source_dataset
            ):
                continue
            samples.append(
                {
                    "image": (event.image_path, event.mask_path),
                    "melanoma": int(event.label),
                }
            )
        return samples
