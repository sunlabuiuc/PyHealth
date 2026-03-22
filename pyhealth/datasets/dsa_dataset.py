"""
DSA (Daily and Sports Activities) Dataset for PyHealth.

This module loads the UCI Daily and Sports Activities dataset,
which contains motion sensor time series from 5 body-part sensors
across 19 activity classes and 8 subjects.

Dataset source:
    https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

Reference:
    Zhang et al. "Daily Physical Activity Monitoring: Adaptive Learning
    from Multi-source Motion Sensor Data." CHIL 2024.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple

from pyhealth.datasets import BaseDataset


# ── Constants ──────────────────────────────────────────────────────────────────

ACTIVITY_LABELS: Dict[str, str] = {
    "a01": "sitting",
    "a02": "standing",
    "a03": "lying_on_back",
    "a04": "lying_on_right",
    "a05": "ascending_stairs",
    "a06": "descending_stairs",
    "a07": "standing_in_elevator",
    "a08": "moving_in_elevator",
    "a09": "walking_in_parking_lot",
    "a10": "walking_on_treadmill_flat",
    "a11": "walking_on_treadmill_inclined",
    "a12": "running_on_treadmill",
    "a13": "exercising_on_stepper",
    "a14": "exercising_on_cross_trainer",
    "a15": "cycling_horizontal",
    "a16": "cycling_vertical",
    "a17": "rowing",
    "a18": "jumping",
    "a19": "playing_basketball",
}

SENSOR_LABELS: Dict[str, str] = {
    "s1": "torso",
    "s2": "right_arm",
    "s3": "left_arm",
    "s4": "right_leg",
    "s5": "left_leg",
}

TIMESTEPS = 125
N_CHANNELS = 9
N_SEGMENTS = 60
N_SUBJECTS = 8
N_ACTIVITIES = 19
N_SENSORS = 5


# ── Dataset Class ──────────────────────────────────────────────────────────────

class DSADataset(BaseDataset):
    """PyHealth dataset for the UCI Daily and Sports Activities (DSA) dataset.

    The DSA dataset contains motion sensor data recorded from 5 sensors
    placed on different body parts (torso, right arm, left arm, right leg,
    left leg). Eight subjects performed 19 daily and sports activities.
    Each sensor captures a 9-dimensional time series (accelerometer,
    gyroscope, magnetometer) at 25 Hz, segmented into 125-timestep windows.

    The dataset is structured in PyHealth as:
        - Patient  = one subject (p1-p8)
        - Visit    = one activity session per subject (e.g., subject 1 doing a01)
        - Event    = one 125-timestep window from one sensor

    Args:
        root (str): Path to the root folder of the DSA dataset.
            Expected structure: root/a{01-19}/p{1-8}/s{1-5}.txt
        target_sensor (str): The sensor to use as the target domain.
            One of: s1, s2, s3, s4, s5. Defaults to "s2" (right arm).
        dev (bool): If True, load only the first 2 subjects for fast
            development/testing. Defaults to False.

    Examples:
        >>> dataset = DSADataset(
        ...     root="/content/drive/MyDrive/DSA/data",
        ...     target_sensor="s2",
        ... )
        >>> print(len(dataset.patients))
        8
    """

    def __init__(
        self,
        root: str,
        target_sensor: str = "s2",
        dev: bool = False,
    ) -> None:
        self.root = root
        self.target_sensor = target_sensor
        self.dev = dev
        self.patients: Dict[str, Dict[str, List[Dict]]] = {}

        if target_sensor not in SENSOR_LABELS:
            raise ValueError(
                f"target_sensor must be one of {list(SENSOR_LABELS.keys())}, "
                f"got '{target_sensor}'."
            )

        if not os.path.exists(root):
            raise FileNotFoundError(
                f"Dataset root not found: {root}\n"
                "Please download from: "
                "https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities"
            )

        self._load_all()

    def _load_all(self) -> None:
        """Walk the directory tree and load all subjects, activities, sensors."""
        subjects = [f"p{i}" for i in range(1, N_SUBJECTS + 1)]
        if self.dev:
            subjects = subjects[:2]
        for subject_id in subjects:
            self.patients[subject_id] = self._parse_subject(subject_id)

    def _parse_subject(self, subject_id: str) -> Dict[str, List[Dict]]:
        """Parse all activity sessions for one subject.

        Args:
            subject_id (str): Subject folder name, e.g. "p1".

        Returns:
            Dict mapping visit_id (activity code) to list of event dicts.
        """
        visits: Dict[str, List[Dict]] = {}
        for activity_code, activity_name in ACTIVITY_LABELS.items():
            events = self._parse_activity(subject_id, activity_code, activity_name)
            if events:
                visits[activity_code] = events
        return visits

    def _parse_activity(
        self,
        subject_id: str,
        activity_code: str,
        activity_name: str,
    ) -> List[Dict]:
        """Parse all sensor windows for one subject/activity combination.

        Args:
            subject_id (str): e.g. "p1"
            activity_code (str): e.g. "a01"
            activity_name (str): e.g. "sitting"

        Returns:
            List of event dicts, each containing:
                - sensor_id (str): e.g. "s2"
                - sensor_name (str): e.g. "right_arm"
                - is_target (bool): True if this is the target sensor
                - activity_code (str): e.g. "a01"
                - activity_name (str): e.g. "sitting"
                - label (int): integer class label 0-18
                - segment_idx (int): which of the 60 windows this is
                - data (np.ndarray): shape (9, 125), float32
        """
        events: List[Dict] = []
        label = int(activity_code[1:]) - 1

        for sensor_id, sensor_name in SENSOR_LABELS.items():
            file_path = os.path.join(
                self.root, activity_code, subject_id, f"{sensor_id}.txt"
            )
            if not os.path.exists(file_path):
                continue
            raw = self._read_txt(file_path)
            if raw is None:
                continue
            segments = self._segment(raw)
            for seg_idx, segment in enumerate(segments):
                events.append({
                    "sensor_id": sensor_id,
                    "sensor_name": sensor_name,
                    "is_target": sensor_id == self.target_sensor,
                    "activity_code": activity_code,
                    "activity_name": activity_name,
                    "label": label,
                    "segment_idx": seg_idx,
                    "data": segment,
                })
        return events

    def _read_txt(self, file_path: str) -> Optional[np.ndarray]:
        """Read a DSA .txt file into a numpy array.

        Args:
            file_path (str): Full path to the .txt file.

        Returns:
            np.ndarray of shape (7500, 9), float32, or None on error.
        """
        try:
            data = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
            if data.shape != (N_SEGMENTS * TIMESTEPS, N_CHANNELS):
                raise ValueError(
                    f"Unexpected shape {data.shape} in {file_path}."
                )
            return data
        except Exception as e:
            print(f"[DSADataset] Warning: could not read {file_path}: {e}")
            return None

    def _segment(self, raw: np.ndarray) -> np.ndarray:
        """Split a (7500, 9) array into (60, 9, 125) segments.

        Args:
            raw (np.ndarray): Shape (7500, 9).

        Returns:
            np.ndarray of shape (60, 9, 125).
        """
        segments = raw.reshape(N_SEGMENTS, TIMESTEPS, N_CHANNELS)
        segments = segments.transpose(0, 2, 1)
        return segments

    def get_all_samples(
        self,
        split: str = "all",
        train_subjects: Optional[List[str]] = None,
        test_subjects: Optional[List[str]] = None,
    ) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """Return flat lists of (data, label, sensor_id) for model consumption.

        Args:
            split (str): "train", "test", or "all".
            train_subjects (List[str]): e.g. ["p1","p2","p3","p4","p5","p6"]
            test_subjects (List[str]): e.g. ["p7","p8"]

        Returns:
            Tuple of (X, y, sensors) where X is a list of (9,125) arrays,
            y is a list of int labels, sensors is a list of sensor_id strings.
        """
        X, y, sensors = [], [], []
        if split == "train":
            subjects = train_subjects or list(self.patients.keys())[:6]
        elif split == "test":
            subjects = test_subjects or list(self.patients.keys())[6:]
        else:
            subjects = list(self.patients.keys())

        for subj in subjects:
            if subj not in self.patients:
                continue
            for visit_events in self.patients[subj].values():
                for event in visit_events:
                    X.append(event["data"])
                    y.append(event["label"])
                    sensors.append(event["sensor_id"])
        return X, y, sensors

    def get_sensor_data(
        self, sensor_id: str, split: str = "all"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get all samples from a specific sensor as stacked arrays.

        Args:
            sensor_id (str): One of s1-s5.
            split (str): "train", "test", or "all".

        Returns:
            Tuple of (X, y) where X has shape (N, 9, 125) and y shape (N,).
        """
        X_all, y_all, sensors_all = self.get_all_samples(split=split)
        filtered = [
            (x, lbl)
            for x, lbl, s in zip(X_all, y_all, sensors_all)
            if s == sensor_id
        ]
        if not filtered:
            raise ValueError(f"No samples found for sensor '{sensor_id}'.")
        X_arr = np.stack([f[0] for f in filtered], axis=0)
        y_arr = np.array([f[1] for f in filtered], dtype=np.int64)
        return X_arr, y_arr

    def __len__(self) -> int:
        """Return total number of subjects."""
        return len(self.patients)

    def __repr__(self) -> str:
        n_subjects = len(self.patients)
        n_events = sum(
            len(events)
            for subj in self.patients.values()
            for events in subj.values()
        )
        return (
            f"DSADataset(\n"
            f"  root={self.root},\n"
            f"  target_sensor={self.target_sensor} "
            f"({SENSOR_LABELS[self.target_sensor]}),\n"
            f"  subjects={n_subjects},\n"
            f"  total_events={n_events}\n"
            f")"
        )
