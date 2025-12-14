"""Blood Glucose Level Prediction Task for OhioT1DM Dataset.

This module provides task functions for blood glucose prediction using the 
OhioT1DM dataset. The primary task is to predict future blood glucose levels
given historical CGM readings and other contextual data.

Reference:
    Marling, C., & Bunescu, R. (2020). The OhioT1DM Dataset for Blood Glucose 
    Level Prediction: Update 2020. CEUR Workshop Proceedings, 2675, 71-74.

Example:
    >>> from pyhealth.datasets import OhioT1DMDataset
    >>> from pyhealth.tasks import blood_glucose_prediction_ohiot1dm_fn
    >>> dataset = OhioT1DMDataset(root="/path/to/OhioT1DM/")
    >>> dataset = dataset.set_task(blood_glucose_prediction_ohiot1dm_fn)
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np


def _parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse timestamp string to datetime object."""
    formats = [
        "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%m-%d-%Y %H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def _get_glucose_at_intervals(
    glucose_data: List[Dict],
    start_idx: int,
    num_points: int,
    interval_minutes: int = 5
) -> Optional[np.ndarray]:
    """Extract glucose values at regular intervals.
    
    Args:
        glucose_data: List of glucose readings with 'ts' and 'value'.
        start_idx: Starting index in glucose_data.
        num_points: Number of points to extract.
        interval_minutes: Expected interval between readings.
        
    Returns:
        Numpy array of glucose values, or None if data is incomplete.
    """
    if start_idx + num_points > len(glucose_data):
        return None
    
    values = []
    for i in range(num_points):
        values.append(glucose_data[start_idx + i]["value"])
    
    return np.array(values, dtype=np.float32)


def blood_glucose_prediction_ohiot1dm_fn(
    patient: Dict,
    history_minutes: int = 60,
    prediction_horizon: int = 30,
    step_minutes: int = 5,
) -> List[Dict]:
    """Task function for blood glucose level prediction.

    Creates samples using a sliding window approach. Each sample contains:
    - Historical glucose values (input)
    - Future glucose value at prediction horizon (target)
    
    Args:
        patient: Patient data dict from OhioT1DMDataset.
        history_minutes: Minutes of historical data to use. Default 60.
        prediction_horizon: Minutes ahead to predict. Default 30.
        step_minutes: Step size for sliding window. Default 5.

    Returns:
        List of sample dicts with keys:
            - patient_id: Patient identifier
            - record_id: Unique sample identifier
            - glucose_history: Historical glucose values (numpy array)
            - glucose_target: Target glucose value to predict
            - prediction_horizon: Prediction horizon in minutes

    Example:
        >>> samples = blood_glucose_prediction_ohiot1dm_fn(patient)
        >>> print(f"Input shape: {samples[0]['glucose_history'].shape}")
        >>> print(f"Target: {samples[0]['glucose_target']}")
    """
    samples = []
    patient_id = patient["patient_id"]
    glucose_data = patient.get("glucose_level", [])
    
    if len(glucose_data) < 20:  # Need minimum data
        return samples
    
    # Calculate number of points needed
    # CGM readings are every 5 minutes
    cgm_interval = 5
    history_points = history_minutes // cgm_interval
    horizon_points = prediction_horizon // cgm_interval
    step_points = step_minutes // cgm_interval
    
    sample_idx = 0
    
    # Slide through the data
    for i in range(0, len(glucose_data) - history_points - horizon_points, step_points):
        # Get historical glucose values
        history = _get_glucose_at_intervals(glucose_data, i, history_points)
        
        if history is None:
            continue
        
        # Get target glucose value
        target_idx = i + history_points + horizon_points - 1
        if target_idx >= len(glucose_data):
            continue
        
        target_value = glucose_data[target_idx]["value"]
        
        # Skip samples with invalid values
        if np.any(np.isnan(history)) or np.isnan(target_value):
            continue
        
        samples.append({
            "patient_id": patient_id,
            "record_id": f"{patient_id}_{sample_idx}",
            "glucose_history": history,
            "glucose_target": float(target_value),
            "prediction_horizon": prediction_horizon,
        })
        sample_idx += 1
    
    return samples


def blood_glucose_prediction_30min_fn(patient: Dict) -> List[Dict]:
    """Blood glucose prediction with 30-minute horizon (standard benchmark)."""
    return blood_glucose_prediction_ohiot1dm_fn(
        patient, 
        history_minutes=60, 
        prediction_horizon=30
    )


def blood_glucose_prediction_60min_fn(patient: Dict) -> List[Dict]:
    """Blood glucose prediction with 60-minute horizon."""
    return blood_glucose_prediction_ohiot1dm_fn(
        patient, 
        history_minutes=60, 
        prediction_horizon=60
    )


def hypoglycemia_detection_ohiot1dm_fn(
    patient: Dict,
    history_minutes: int = 60,
    prediction_horizon: int = 30,
    hypo_threshold: float = 70.0,
    step_minutes: int = 5,
) -> List[Dict]:
    """Task function for hypoglycemia detection/prediction.

    Binary classification task to predict if blood glucose will fall
    below the hypoglycemia threshold within the prediction horizon.

    Args:
        patient: Patient data dict from OhioT1DMDataset.
        history_minutes: Minutes of historical data. Default 60.
        prediction_horizon: Minutes ahead to predict. Default 30.
        hypo_threshold: Blood glucose threshold for hypoglycemia (mg/dL). Default 70.
        step_minutes: Step size for sliding window. Default 5.

    Returns:
        List of sample dicts with keys:
            - patient_id: Patient identifier
            - record_id: Unique sample identifier
            - glucose_history: Historical glucose values
            - label: 1 if hypoglycemia occurs, 0 otherwise

    Example:
        >>> samples = hypoglycemia_detection_ohiot1dm_fn(patient)
        >>> print(f"Label: {samples[0]['label']}")
    """
    samples = []
    patient_id = patient["patient_id"]
    glucose_data = patient.get("glucose_level", [])
    
    if len(glucose_data) < 20:
        return samples
    
    cgm_interval = 5
    history_points = history_minutes // cgm_interval
    horizon_points = prediction_horizon // cgm_interval
    step_points = step_minutes // cgm_interval
    
    sample_idx = 0
    
    for i in range(0, len(glucose_data) - history_points - horizon_points, step_points):
        history = _get_glucose_at_intervals(glucose_data, i, history_points)
        
        if history is None:
            continue
        
        # Check if any glucose value in prediction horizon is below threshold
        future_start = i + history_points
        future_end = future_start + horizon_points
        
        if future_end > len(glucose_data):
            continue
        
        future_values = [glucose_data[j]["value"] for j in range(future_start, future_end)]
        
        if np.any(np.isnan(history)) or any(np.isnan(v) for v in future_values):
            continue
        
        # Label: 1 if hypoglycemia occurs in prediction window
        label = 1 if any(v < hypo_threshold for v in future_values) else 0
        
        samples.append({
            "patient_id": patient_id,
            "record_id": f"{patient_id}_{sample_idx}",
            "glucose_history": history,
            "label": label,
        })
        sample_idx += 1
    
    return samples


def hyperglycemia_detection_ohiot1dm_fn(
    patient: Dict,
    history_minutes: int = 60,
    prediction_horizon: int = 30,
    hyper_threshold: float = 180.0,
    step_minutes: int = 5,
) -> List[Dict]:
    """Task function for hyperglycemia detection/prediction.

    Binary classification task to predict if blood glucose will rise
    above the hyperglycemia threshold within the prediction horizon.

    Args:
        patient: Patient data dict from OhioT1DMDataset.
        history_minutes: Minutes of historical data. Default 60.
        prediction_horizon: Minutes ahead to predict. Default 30.
        hyper_threshold: Blood glucose threshold for hyperglycemia (mg/dL). Default 180.
        step_minutes: Step size for sliding window. Default 5.

    Returns:
        List of sample dicts with keys:
            - patient_id: Patient identifier
            - record_id: Unique sample identifier
            - glucose_history: Historical glucose values
            - label: 1 if hyperglycemia occurs, 0 otherwise
    """
    samples = []
    patient_id = patient["patient_id"]
    glucose_data = patient.get("glucose_level", [])
    
    if len(glucose_data) < 20:
        return samples
    
    cgm_interval = 5
    history_points = history_minutes // cgm_interval
    horizon_points = prediction_horizon // cgm_interval
    step_points = step_minutes // cgm_interval
    
    sample_idx = 0
    
    for i in range(0, len(glucose_data) - history_points - horizon_points, step_points):
        history = _get_glucose_at_intervals(glucose_data, i, history_points)
        
        if history is None:
            continue
        
        future_start = i + history_points
        future_end = future_start + horizon_points
        
        if future_end > len(glucose_data):
            continue
        
        future_values = [glucose_data[j]["value"] for j in range(future_start, future_end)]
        
        if np.any(np.isnan(history)) or any(np.isnan(v) for v in future_values):
            continue
        
        # Label: 1 if hyperglycemia occurs in prediction window
        label = 1 if any(v > hyper_threshold for v in future_values) else 0
        
        samples.append({
            "patient_id": patient_id,
            "record_id": f"{patient_id}_{sample_idx}",
            "glucose_history": history,
            "label": label,
        })
        sample_idx += 1
    
    return samples


def glucose_range_classification_ohiot1dm_fn(
    patient: Dict,
    history_minutes: int = 60,
    prediction_horizon: int = 30,
    step_minutes: int = 5,
) -> List[Dict]:
    """Task function for glucose range classification.

    Multi-class classification to predict glucose range:
    - Class 0: Hypoglycemia (< 70 mg/dL)
    - Class 1: Normal (70-180 mg/dL)  
    - Class 2: Hyperglycemia (> 180 mg/dL)

    Args:
        patient: Patient data dict from OhioT1DMDataset.
        history_minutes: Minutes of historical data. Default 60.
        prediction_horizon: Minutes ahead to predict. Default 30.
        step_minutes: Step size for sliding window. Default 5.

    Returns:
        List of sample dicts with keys:
            - patient_id: Patient identifier
            - record_id: Unique sample identifier
            - glucose_history: Historical glucose values
            - label: 0 (hypo), 1 (normal), or 2 (hyper)
    """
    samples = []
    patient_id = patient["patient_id"]
    glucose_data = patient.get("glucose_level", [])
    
    if len(glucose_data) < 20:
        return samples
    
    cgm_interval = 5
    history_points = history_minutes // cgm_interval
    horizon_points = prediction_horizon // cgm_interval
    step_points = step_minutes // cgm_interval
    
    sample_idx = 0
    
    for i in range(0, len(glucose_data) - history_points - horizon_points, step_points):
        history = _get_glucose_at_intervals(glucose_data, i, history_points)
        
        if history is None:
            continue
        
        target_idx = i + history_points + horizon_points - 1
        if target_idx >= len(glucose_data):
            continue
        
        target_value = glucose_data[target_idx]["value"]
        
        if np.any(np.isnan(history)) or np.isnan(target_value):
            continue
        
        # Classify into ranges
        if target_value < 70:
            label = 0  # Hypoglycemia
        elif target_value > 180:
            label = 2  # Hyperglycemia
        else:
            label = 1  # Normal
        
        samples.append({
            "patient_id": patient_id,
            "record_id": f"{patient_id}_{sample_idx}",
            "glucose_history": history,
            "glucose_target": float(target_value),
            "label": label,
        })
        sample_idx += 1
    
    return samples
