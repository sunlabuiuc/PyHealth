"""
PyHealth task for classification using the Daily and Sports Activity dataset.

Dataset link:
    https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

Dataset paper:
    Zhang, H.; Zhan, D.; Lin, Y.; He, J.; Zhu, Q.; Shen, Z.-J.; and
    Zheng, Z. 2024. Daily Physical Activity Monitoring: Adaptive Learning
    from Multi-source Motion Sensor Data. Proceedings of the fifth Conference
    on Health, Inference, and Learning, volume 248 of Proceedings of Machine
    Learning Research, 39–54. PMLR

Dataset paper link:
    https://raw.githubusercontent.com/mlresearch/v248/main/assets/zhang24a/zhang24a.pdf

Authors:
    Niam Pattni (npattni2@illinois.edu)
    Sezim Zamirbekova (szami2@illinois.edu)
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Apply per-feature z-score normalization.
    
    Args:
        signal (np.ndarray): The signal tensor to normalize.

    Returns:
        np.ndarray: The normalized signal.
    """
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (signal - mean) / std

def _validate_and_select_features(
    signal: np.ndarray,
    selected_features: Optional[List[int]],
) -> np.ndarray:
    """
    Validate optional feature indices and subset the signal.
    
    Args:
        signal (np.ndarray): The original signal before feature selection.
        selected_features (Optional[List[int]]): Features to select from the
        specified signal.

    Returns:
        np.ndarray: The signal after feature selection.

    Raises:
        ValueError: selected_features is empty.
        ValueError: There is an index in selected_features which is out of bounds
        for the signal.
    """
    if selected_features is None:
        return signal

    if len(selected_features) == 0:
        raise ValueError("selected_features cannot be an empty list.")

    n_features = signal.shape[1]
    for idx in selected_features:
        if idx < 0 or idx >= n_features:
            raise ValueError(
                f"""Feature index {idx} is out of bounds for signal with {n_features} 
                features."""
            )

    return signal[:, selected_features]

def _sliding_windows(
    signal: np.ndarray,
    window_size: int, 
    stride: int,
) -> List[np.ndarray]:
    """
    Split a sequence into fixed-size sliding windows.
    
    Args:
        signal (np.ndarray): The signal to split.
        window_size (int): The size of the sliding window.
        stride (int): How far to slide the window at each step.

    Returns:
        List[np.ndarray]: Windows generated from the signal.

    Raises:
        ValueError: Window size is negative.
        ValueError: Stride is negative.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    n_steps = signal.shape[0]
    windows: List[np.ndarray] = []

    if n_steps < window_size:
        return windows

    for start in range(0, n_steps - window_size + 1, stride):
        end = start + window_size
        windows.append(signal[start:end])

    return windows

class DailyAndSportActivitiesClassification(BaseTask):
    """
    A PyHealth task class for classification of activities in the Daily and
    Sport Activities dataset.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.

    Examples:
        >>> from pyhealth.datasets import DailyAndSportActivitiesDataset
        >>> from pyhealth.tasks import DailyAndSportActivitiesClassification
        >>> dataset = DailyAndSportActivitiesDataset(download=True)
        >>> task = DailyAndSportActivitiesyClassification()
        >>> samples = dataset.set_task(task)
    """
    task_name: str = "DailyAndSportActivitiesClassification"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        signal_loader: Callable,
        window_size: int = 50,
        stride: int = 25,
        normalize: bool = True,
        selected_features: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes the DailyAndSportActivitiesClassification task.

        Args:
            signal_loader (Callable): The function to use for parsing signal data.
            window_size (int): The size of the sliding window on the input signal.
            Defaults to 50.
            stride (int): The size of the sliding window move. Defauts to 25.
            normalize (bool): Should the signal data be normalized. Defaults to True.
            selected_features (Optional[List[int]]): Features to select from the signal.
            Defaults to None (all features).

        Raises:
            ValueError: Window size is negative.
            ValueError: Stride is negative.
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.signal_loader = signal_loader
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.selected_features = selected_features
        
    def __call__(self, patient: Patient) -> List[Dict]:
        """
        Generate activity-recognition samples for a single patient.
        
        Args:
            patient (Patient): The patient to generate samples for.

        Returns:
            List[Dict]: The list of samples for the specified patient.
        """
        events: List[Event] = patient.get_events(event_type="daily_sport_activities")

        samples: List[Dict] = []

        for event in events:
            signal = self.signal_loader(event["file_path"])
            signal = _validate_and_select_features(signal, self.selected_features)

            if self.normalize:
                signal = _normalize_signal(signal)

            windows = _sliding_windows(
                signal=signal,
                window_size=self.window_size,
                stride=self.stride,
            )

            activity_id = event["activity_id"]
            if isinstance(activity_id, str) and activity_id.startswith("a"):
                label = int(activity_id[1:]) - 1
            else:
                label = int(activity_id)

            record_id = event["record_id"]
            visit_id = event["visit_id"]

            for idx, window in enumerate(windows):
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "visit_id": visit_id,
                        "record_id": f"{record_id}_win_{idx}",
                        "signal": window.astype(np.float32),
                        "label": label,
                    }
                )

        return samples
    