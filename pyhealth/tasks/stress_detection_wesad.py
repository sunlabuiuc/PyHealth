"""Stress Detection Task for WESAD Dataset.

This module provides task functions for stress detection using the WESAD dataset.

Reference:
    Schmidt, P., et al. (2018). Introducing WESAD, a Multimodal Dataset for 
    Wearable Stress and Affect Detection. ICMI 2018.

Example:
    >>> from pyhealth.datasets import WESADDataset
    >>> from pyhealth.tasks import stress_detection_wesad_fn
    >>> dataset = WESADDataset(root="/path/to/WESAD/")
    >>> dataset = dataset.set_task(stress_detection_wesad_fn)
"""

from typing import Dict, List
import numpy as np

# Label constants
LABEL_BASELINE = 1
LABEL_STRESS = 2
LABEL_AMUSEMENT = 3

# Sampling rates
SAMPLING_RATES = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4, "label": 700}


def _resample_labels(labels: np.ndarray, target_length: int) -> np.ndarray:
    """Resample labels to match target signal length."""
    indices = np.linspace(0, len(labels) - 1, target_length).astype(int)
    return labels[indices]


def stress_detection_wesad_fn(
    patient: Dict,
    window_seconds: float = 60.0,
    step_seconds: float = 1.0,
    binary: bool = True,
) -> List[Dict]:
    """Task function for stress detection on WESAD dataset.

    Creates samples using sliding windows over EDA signal.

    For binary classification:
        - Label 0: Non-stress (baseline or amusement)
        - Label 1: Stress

    For three-class:
        - Label 0: Baseline
        - Label 1: Stress
        - Label 2: Amusement

    Args:
        patient: Patient data dict from WESADDataset.
        window_seconds: Window size in seconds. Default 60.
        step_seconds: Step size in seconds. Default 1.
        binary: If True, binary classification. Default True.

    Returns:
        List of sample dicts with keys: patient_id, record_id, eda, label.

    Example:
        >>> samples = stress_detection_wesad_fn(patient, binary=True)
        >>> print(samples[0].keys())
    """
    samples = []
    patient_id = patient["patient_id"]

    eda = patient["signal"]["wrist"]["EDA"].flatten()
    labels_700hz = patient["label"].flatten()

    # Resample labels to EDA rate (4 Hz)
    labels = _resample_labels(labels_700hz, len(eda))

    valid_labels = [LABEL_BASELINE, LABEL_STRESS, LABEL_AMUSEMENT]
    eda_sr = SAMPLING_RATES["EDA"]
    window_samples = int(window_seconds * eda_sr)
    step_samples = int(step_seconds * eda_sr)

    sample_idx = 0
    for start in range(0, len(eda) - window_samples, step_samples):
        end = start + window_samples
        window_labels = labels[start:end]

        # Skip if <80% valid labels
        valid_mask = np.isin(window_labels, valid_labels)
        if valid_mask.sum() < window_samples * 0.8:
            continue

        # Get majority label
        valid_window_labels = window_labels[valid_mask]
        unique, counts = np.unique(valid_window_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]

        # Convert to task label
        if binary:
            task_label = 1 if majority_label == LABEL_STRESS else 0
        else:
            task_label = {LABEL_BASELINE: 0, LABEL_STRESS: 1, LABEL_AMUSEMENT: 2}[majority_label]

        samples.append({
            "patient_id": patient_id,
            "record_id": f"{patient_id}_{sample_idx}",
            "eda": eda[start:end].astype(np.float32),
            "label": task_label,
        })
        sample_idx += 1

    return samples


def stress_detection_wesad_binary_fn(patient: Dict) -> List[Dict]:
    """Binary stress detection (stress vs non-stress)."""
    return stress_detection_wesad_fn(patient, binary=True)


def stress_detection_wesad_multiclass_fn(patient: Dict) -> List[Dict]:
    """Three-class classification (baseline vs stress vs amusement)."""
    return stress_detection_wesad_fn(patient, binary=False)
