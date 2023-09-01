
from typing import List
import numpy as np

from pyhealth.datasets import BaseEHRDataset

def sensitive_attributes_from_patient_ids(dataset: BaseEHRDataset,
                                          patient_ids: List[str],
                                          sensitive_attribute: str,
                                          protected_group: str) -> np.ndarray:
    """
    Returns the desired sensitive attribute array from patient_ids.

    Args:
        dataset: Dataset object.
        patient_ids: List of patient IDs.
        sensitive_attribute: Sensitive attribute to extract.
        protected_group: Value of the protected group.
    
    Returns:
        Sensitive attribute array of shape (n_samples,).
    """

    sensitive_attribute_array = np.zeros(len(patient_ids))
    for idx, patient_id in enumerate(patient_ids):
        sensitive_attribute_value = getattr(dataset.patients[patient_id], sensitive_attribute)
        if sensitive_attribute_value == protected_group:
            sensitive_attribute_array[idx] = 1
    return sensitive_attribute_array

    