from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from pyhealth.data import Event, Patient
from pyhealth.tasks import BaseTask


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Apply per-feature z-score normalization."""
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (signal - mean) / std

def _validate_and_select_features(
    signal: np.ndarray, selected_features: Optional[List[int]]
) -> np.ndarray:
    """Validate optional feature indices and subset the signal."""
    if selected_features is None:
        return signal

    if len(selected_features) == 0:
        raise ValueError("selected_features cannot be an empty list.")

    n_features = signal.shape[1]
    for idx in selected_features:
        if idx < 0 or idx >= n_features:
            raise ValueError(
                f"Feature index {idx} is out of bounds for signal with {n_features} features."
            )

    return signal[:, selected_features]

def _sliding_windows(signal: np.ndarray, window_size: int, stride: int) -> List[np.ndarray]:
    """Split a sequence into fixed-size sliding windows."""
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

class DailyAndSportActivitiesTask(BaseTask):
    """Create task samples for activity recognition from DailyAndSportActivitiesDataset."""
    task_name: str = "DailyAndSportActivitiesTask"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        window_size: int = 50,
        stride: int = 25,
        normalize: bool = True,
        selected_features: Optional[List[int]] = None,
        signal_loader=None,
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.selected_features = selected_features
        self.signal_loader = signal_loader
        
    def __call__(self, patient: Patient) -> List[Dict]:
        """Generate activity-recognition samples for one patient."""
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
    