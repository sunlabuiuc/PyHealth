"""
COVID-RED Classification Task Function for PyHealth

This module provides task functions for COVID-19 detection and prediction
using the COVID-RED wearable device dataset.
"""

from typing import Dict, List, Any
import torch


def covidred_detection_fn(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task function for COVID-19 detection from wearable device data.
    
    This function processes a sample from the COVID-RED dataset and formats it
    for PyHealth's standard pipeline. The task is to classify whether a participant
    has COVID-19 based on their wearable device measurements (heart rate, steps, sleep).
    
    Parameters
    ----------
    sample : Dict[str, Any]
        A sample dictionary from COVIDREDDataset containing:
        - participant_id: Participant identifier
        - window_start_date: Start date of the measurement window
        - window_end_date: End date of the measurement window
        - features: Feature tensor of shape (window_days, n_features)
        - label: Binary label (1=COVID-19 positive, 0=negative)
    
    Returns
    -------
    Dict[str, Any]
        Processed sample dictionary in PyHealth format:
        - patient_id: Participant identifier (str)
        - visit_id: Unique identifier for this window (str)
        - signal: Time series tensor of shape (n_features, window_days)
        - label: Binary classification label (int, 0 or 1)
        - metadata: Additional information (dict)
    
    Examples
    --------
    >>> from pyhealth.datasets import COVIDREDDataset
    >>> from pyhealth.tasks import covidred_detection_fn
    >>> 
    >>> dataset = COVIDREDDataset(root="/path/to/covidred", split="train")
    >>> sample = dataset[0]
    >>> processed_sample = covidred_detection_fn(sample)
    >>> print(processed_sample.keys())
    dict_keys(['patient_id', 'visit_id', 'signal', 'label', 'metadata'])
    
    Notes
    -----
    The signal tensor is transposed to shape (n_features, window_days) to match
    PyHealth's expected format for time series data, where the first dimension
    represents different feature channels and the second represents time steps.
    """
    # Extract patient and visit identifiers
    patient_id = str(sample["participant_id"])
    visit_id = f"{patient_id}_{sample['window_start_date'].strftime('%Y%m%d')}"
    
    # Transpose features from (window_days, n_features) to (n_features, window_days)
    # This matches PyHealth's expected signal format
    signal = sample["features"].transpose(0, 1)
    
    # Extract label
    label = int(sample["label"])
    
    # Create metadata
    metadata = {
        "window_start_date": sample["window_start_date"],
        "window_end_date": sample["window_end_date"],
        "window_days": signal.shape[1],
        "n_features": signal.shape[0],
    }
    
    return {
        "patient_id": patient_id,
        "visit_id": visit_id,
        "signal": signal,
        "label": label,
        "metadata": metadata,
    }


def covidred_prediction_fn(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task function for early COVID-19 prediction from wearable device data.
    
    This function processes a sample from the COVID-RED dataset for the early
    detection task - predicting COVID-19 onset before symptom appearance.
    
    Parameters
    ----------
    sample : Dict[str, Any]
        A sample dictionary from COVIDREDDataset containing:
        - participant_id: Participant identifier
        - window_start_date: Start date of the measurement window
        - window_end_date: End date of the measurement window
        - features: Feature tensor of shape (window_days, n_features)
        - label: Binary label (1=pre-symptomatic period, 0=normal)
    
    Returns
    -------
    Dict[str, Any]
        Processed sample dictionary in PyHealth format:
        - patient_id: Participant identifier (str)
        - visit_id: Unique identifier for this window (str)
        - signal: Time series tensor of shape (n_features, window_days)
        - label: Binary prediction label (int, 0 or 1)
        - metadata: Additional information (dict)
    
    Examples
    --------
    >>> from pyhealth.datasets import COVIDREDDataset
    >>> from pyhealth.tasks import covidred_prediction_fn
    >>> 
    >>> dataset = COVIDREDDataset(
    ...     root="/path/to/covidred",
    ...     split="train",
    ...     task="prediction"
    ... )
    >>> sample = dataset[0]
    >>> processed_sample = covidred_prediction_fn(sample)
    >>> print(f"Signal shape: {processed_sample['signal'].shape}")
    >>> print(f"Label: {processed_sample['label']}")
    
    Notes
    -----
    The prediction task focuses on identifying pre-symptomatic patterns in the
    1-14 days before symptom onset, which is critical for early intervention
    and reducing transmission.
    """
    # Use the same processing as detection task
    # The distinction is in how the dataset creates labels
    return covidred_detection_fn(sample)


def covidred_multiclass_fn(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task function for multiclass COVID-19 severity classification.
    
    This function extends the basic detection to classify COVID-19 cases
    into multiple severity categories: negative, mild, moderate, severe.
    
    Parameters
    ----------
    sample : Dict[str, Any]
        A sample dictionary from COVIDREDDataset with additional severity info.
    
    Returns
    -------
    Dict[str, Any]
        Processed sample dictionary with multiclass label:
        - 0: COVID-19 negative
        - 1: Mild (recovered at home, no assistance)
        - 2: Moderate (recovered at home with assistance)
        - 3: Severe (hospitalized)
    
    Examples
    --------
    >>> from pyhealth.datasets import COVIDREDDataset
    >>> from pyhealth.tasks import covidred_multiclass_fn
    >>> 
    >>> # Assuming dataset includes severity information
    >>> dataset = COVIDREDDataset(root="/path/to/covidred", split="train")
    >>> sample = dataset[0]
    >>> processed_sample = covidred_multiclass_fn(sample)
    >>> print(f"Severity class: {processed_sample['label']}")
    
    Notes
    -----
    This task requires the dataset to include severity information.
    Check dataset documentation for availability.
    """
    # Extract patient and visit identifiers
    patient_id = str(sample["participant_id"])
    visit_id = f"{patient_id}_{sample['window_start_date'].strftime('%Y%m%d')}"
    
    # Transpose features
    signal = sample["features"].transpose(0, 1)
    
    # Extract severity label if available
    # Default to binary if severity not provided
    label = sample.get("severity_label", sample["label"])
    
    # Create metadata
    metadata = {
        "window_start_date": sample["window_start_date"],
        "window_end_date": sample["window_end_date"],
        "window_days": signal.shape[1],
        "n_features": signal.shape[0],
        "task_type": "multiclass_severity",
    }
    
    return {
        "patient_id": patient_id,
        "visit_id": visit_id,
        "signal": signal,
        "label": int(label),
        "metadata": metadata,
    }
