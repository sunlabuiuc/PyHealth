"""Task for binary melanoma classification from dermoscopic images.

Works with the composite DermoscopyDataset which combines ISIC 2018,
HAM10000, and PH2 datasets. Each sample yields an (image_path, mask_path)
tuple as input, enabling the DermoscopyImageProcessor to apply mode-based
processing (whole, lesion, or background).

Author:
    Generated for PyHealth integration of dermoscopic_artifacts datasets.
"""

from typing import Any, Dict, List, Optional, Union

from pyhealth.tasks import BaseTask

class DermoscopyMelanomaClassification(BaseTask):
    """Binary melanoma classification task for dermoscopic images.

    Each patient event contains a dermoscopic image path, a segmentation
    mask path, and a binary label (0 = benign, 1 = melanoma). The task
    returns the image and mask paths as a tuple so that the
    DermoscopyImageProcessor can apply mode-based masking.

    Args:
        source_datasets: If provided, only samples whose ``source_dataset``
            field matches one of these strings are returned. Pass a string 
            (e.g., ``"isic2018"``) or a list of strings (e.g., 
            ``["isic2018", "ham10000"]``) to restrict evaluation/training 
            to specific sub-datasets. Defaults to ``None`` (all sub-datasets included).

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
        >>> samples = dataset.set_task(DermoscopyMelanomaClassification(source_datasets="isic2018"))
        >>> # Mega-ResNet (ISIC + HAM)
        >>> samples = dataset.set_task(DermoscopyMelanomaClassification(source_datasets=["isic2018", "ham10000"]))
    """

    task_name: str = "DermoscopyMelanomaClassification"
    input_schema: Dict[str, str] = {"image": "dermoscopy_image"}
    output_schema: Dict[str, str] = {"melanoma": "binary"}

    # PATCH: Allow either a single string or a list of strings
    def __init__(self, source_datasets: Optional[Union[str, List[str]]] = None) -> None:
        # Normalize input: If it's a string, wrap it in a list so the rest of the code works perfectly
        if isinstance(source_datasets, str):
            self.source_datasets = [source_datasets]
        else:
            self.source_datasets = source_datasets
            
        # Make task instances with different filters hash to different cache keys.
        if self.source_datasets is not None:
            name_suffix = "_".join(self.source_datasets)
            self.task_name = f"DermoscopyMelanomaClassification_{name_suffix}"

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Extracts melanoma classification samples from a patient record.
        
        Parses a PyHealth Patient object and retrieves all associated 'dermoscopy' 
        events. Safely extracts the required file paths and labels from the event's 
        attr_dict (PyHealth 2.0 standard).
        
        Args:
            patient: A PyHealth Patient object containing dermoscopy events.

        Returns:
            List of sample dictionaries, where each dictionary contains:
                - "patient_id" (str): Unique patient identifier for cross-validation splitting.
                - "visit_id" (str): Unique visit identifier.
                - "image" (tuple): A tuple of (image_path, mask_path) for the processor.
                - "melanoma" (int): Binary label (0 for benign, 1 for melanoma).
                
            Note: Events whose `source_dataset` does not match the filters provided 
            in `self.source_datasets` will be excluded from the returned list.
        """
        events = patient.get_events(event_type="dermoscopy")
        samples = []
        for event in events:
            # PATCH 1: Safely fetch from PyHealth 2.0 attr_dict instead of direct attributes
            source_ds = event.attr_dict.get("source_dataset", "")

            # PATCH 2: Allow multiple datasets for Mega-ResNet (list check instead of strict string equality)
            if self.source_datasets is not None and source_ds not in self.source_datasets:
                continue

            image_path = event.attr_dict.get("image_path", "")
            mask_path = event.attr_dict.get("mask_path", "")
            label = event.attr_dict.get("label", 0)

            samples.append(
                {
                    # PATCH 3: Include patient_id and visit_id to prevent cross-validation data leakage
                    # Dermoscopy datasets typically feature a 1:1 patient-to-visit ratio. 
                    # We default visit_id to patient_id to satisfy PyHealth's strict EHR schema. 
                    # If a user explicitly parsed a longitudinal visit_id during dataset 
                    # initialization, getattr() will safely fetch that instead.
                    "patient_id": patient.patient_id,
                    "visit_id": getattr(patient, 'visit_id', patient.patient_id),
                    "image": (image_path, mask_path),
                    # PATCH 4: Protect against Polars string-floats (e.g., "0.0" -> 0.0 -> 0)
                    "melanoma": int(float(label)), 
                }
            )
        return samples