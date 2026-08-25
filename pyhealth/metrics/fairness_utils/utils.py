
from typing import List
import numpy as np

from pyhealth.datasets import BaseDataset

def sensitive_attributes_from_patient_ids(dataset: BaseDataset,
                                          patient_ids: List[str],
                                          sensitive_attribute: str,
                                          protected_group: str) -> np.ndarray:
    """
    Returns the desired sensitive attribute array from patient_ids.

    Args:
        dataset: Dataset object (must implement ``get_patient(patient_id)``
            returning a :class:`~pyhealth.data.data.Patient` with a
            ``"patients"``-typed demographic event).
        patient_ids: List of patient IDs.
        sensitive_attribute: Sensitive attribute to extract.
        protected_group: Value of the protected group.

    Returns:
        Sensitive attribute array of shape (n_samples,).

    Examples:
        >>> import polars as pl
        >>> from datetime import datetime
        >>> from pyhealth.data import Patient
        >>> event_df = pl.DataFrame({
        ...     "patient_id": ["patient-0", "patient-1"],
        ...     "event_type": ["patients", "patients"],
        ...     "timestamp": [datetime(2020, 1, 1), datetime(2020, 1, 1)],
        ...     "patients/gender": ["F", "M"],
        ... })
        >>> class ToyDataset:
        ...     def get_patient(self, patient_id):
        ...         return Patient(
        ...             patient_id=patient_id,
        ...             data_source=event_df.filter(pl.col("patient_id") == patient_id),
        ...         )
        >>> sensitive_attributes_from_patient_ids(
        ...     ToyDataset(), ["patient-0", "patient-1"], "gender", "F"
        ... )
        array([1., 0.])
    """

    sensitive_attribute_array = np.zeros(len(patient_ids))
    for idx, patient_id in enumerate(patient_ids):
        patient = dataset.get_patient(patient_id)
        demographic_events = patient.get_events(event_type="patients")
        sensitive_attribute_value = (
            demographic_events[0].attr_dict.get(sensitive_attribute)
            if demographic_events
            else None
        )
        if sensitive_attribute_value == protected_group:
            sensitive_attribute_array[idx] = 1
    return sensitive_attribute_array

    