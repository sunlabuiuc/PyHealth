"""Task for binary melanoma classification from dermoscopic images.

Works with the composite DermoscopyDataset which combines ISIC 2018,
HAM10000, and PH2 datasets. Each sample yields an (image_path, mask_path)
tuple as input, enabling the DermoscopyImageProcessor to apply mode-based
processing (whole, lesion, or background).

Author:
    Generated for PyHealth integration of dermoscopic_artifacts datasets.
"""

from typing import Any, Dict, List

from .base_task import BaseTask


class DermoscopyMelanomaClassification(BaseTask):
    """Binary melanoma classification task for dermoscopic images.

    Each patient event contains a dermoscopic image path, a segmentation
    mask path, and a binary label (0 = benign, 1 = melanoma). The task
    returns the image and mask paths as a tuple so that the
    DermoscopyImageProcessor can apply mode-based masking.

    Attributes:
        task_name: "DermoscopyMelanomaClassification"
        input_schema: {"image": "dermoscopy_image"} — maps to DermoscopyImageProcessor
        output_schema: {"melanoma": "binary"} — binary classification

    Examples:
        >>> from pyhealth.datasets import DermoscopyDataset
        >>> from pyhealth.tasks import DermoscopyMelanomaClassification
        >>> dataset = DermoscopyDataset(root="/path/to/data")
        >>> task = DermoscopyMelanomaClassification()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "DermoscopyMelanomaClassification"
    input_schema: Dict[str, str] = {"image": "dermoscopy_image"}
    output_schema: Dict[str, str] = {"melanoma": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Extract melanoma classification samples from a patient record.

        Args:
            patient: A Patient object containing dermoscopy events.

        Returns:
            List of sample dicts, each with:
                - "image": tuple of (image_path, mask_path)
                - "melanoma": integer label (0 or 1)
        """
        events = patient.get_events(event_type="dermoscopy")
        samples = []
        for event in events:
            image_path = event.image_path
            mask_path = event.mask_path
            label = int(event.label)
            samples.append({
                "image": (image_path, mask_path),
                "melanoma": label,
            })
        return samples
