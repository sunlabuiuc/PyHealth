"""
DSA (Daily and Sports Activities) Dataset for PyHealth.

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

ACTIVITY_LABELS: Dict[str, str] = {
    "a01": "sitting", "a02": "standing", "a03": "lying_on_back",
    "a04": "lying_on_right", "a05": "ascending_stairs",
    "a06": "descending_stairs", "a07": "standing_in_elevator",
    "a08": "moving_in_elevator", "a09": "walking_in_parking_lot",
    "a10": "walking_on_treadmill_flat", "a11": "walking_on_treadmill_inclined",
    "a12": "running_on_treadmill", "a13": "exercising_on_stepper",
    "a14": "exercising_on_cross_trainer", "a15": "cycling_horizontal",
    "a16": "cycling_vertical", "a17": "rowing",
    "a18": "jumping", "a19": "playing_basketball",
}

SENSOR_LABELS: Dict[str, str] = {
    "s1": "torso", "s2": "right_arm", "s3": "left_arm",
    "s4": "right_leg", "s5": "left_leg",
}

TIMESTEPS = 125
N_CHANNELS = 9
N_SEGMENTS = 60
N_SUBJECTS = 8
N_ACTIVITIES = 19
N_SENSORS = 5

SENSOR_COLUMNS: Dict[str, range] = {
    "s1": range(0,  9),
    "s2": range(9,  18),
    "s3": range(18, 27),
    "s4": range(27, 36),
    "s5": range(36, 45),
}


class DSADataset(BaseDataset):
    """PyHealth dataset for the UCI Daily and Sports Activities (DSA) dataset.

    File structure: root/a{01-19}/p{1-8}/s{01-60}.txt
    Each file: 125 rows x 45 columns (9 channels x 5 sensors).

    Dataset is structured as:
        Patient = one subject (p1-p8)
        Visit   = one activity session per subject
        Event   = one 125-timestep window from one sensor

    Args:
        root (str): Path to the DSA data folder.
        target_sensor (str): Sensor used as target domain. Default "s2".
        dev (bool): If True, load only 2 subjects for fast testing.

    Examples:
        >>> dataset = DSADataset(root="/path/to/DSA/data", target_sensor="s2")
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
            raise FileNotFoundError(f"Dataset root not found: {root}")

        self._load_all()

    def _load_all(self) -> None:
        """Load all subjects from the dataset directory."""
        subjects = [f"p{i}" for i in range(1, N_SUBJECTS + 1)]
        if self.dev:
            subjects = subjects[:2]
        for sid in subjects:
            self.patients[sid] = self._parse_subject(sid)

    def _parse_subject(self, subject_id: str) -> Dict[str, List[Dict]]:
        """Parse all activity sessions for one subject.

        Args:
            subject_id (str): e.g. "p1"

        Returns:
            Dict mapping activity_code to list of event dicts.
        """
        visits: Dict[str, List[Dict]] = {}
        for act_code, act_name in ACTIVITY_LABELS.items():
            events = self._parse_activity(subject_id, act_code, act_name)
            if events:
                visits[act_code] = events
        return visits

    def _parse_activity(
        self,
        subject_id: str,
        activity_code: str,
        activity_name: str,
    ) -> List[Dict]:
        """Parse all sensor windows for one subject/activity pair.

        Each file s01-s60 is one segment with 125 timesteps x 45 columns.

        Args:
            subject_id (str): e.g. "p1"
            activity_code (str): e.g. "a01"
            activity_name (str): e.g. "sitting"

        Returns:
            List of dicts with keys: sensor_id, sensor_name, is_target,
            activity_code, activity_name, label, segment_idx, data (9,125).
        """
        events = []
        label = int(activity_code[1:]) - 1

        for seg_idx in range(1, N_SEGMENTS + 1):
            fname = "s{:02d}.txt".format(seg_idx)
            fpath = os.path.join(self.root, activity_code, subject_id, fname)
            if not os.path.exists(fpath):
                continue
            raw = self._read_txt(fpath)
            if raw is None:
                continue
            for sensor_id, sensor_name in SENSOR_LABELS.items():
                cols = list(SENSOR_COLUMNS[sensor_id])
                sensor_data = raw[:, cols].T.astype(np.float32)
                events.append({
                    "sensor_id": sensor_id,
                    "sensor_name": sensor_name,
                    "is_target": sensor_id == self.target_sensor,
                    "activity_code": activity_code,
                    "activity_name": activity_name,
                    "label": label,
                    "segment_idx": seg_idx - 1,
                    "data": sensor_data,
                })
        return events

    def _read_txt(self, file_path: str) -> Optional[np.ndarray]:
        """Read one segment file into a (125, 45) numpy array.

        Args:
            file_path (str): Path to .txt file.

        Returns:
            np.ndarray of shape (125, 45) or None on error.
        """
        try:
            data = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
            expected = (TIMESTEPS, N_CHANNELS * N_SENSORS)
            if data.shape != expected:
                raise ValueError(
                    "Unexpected shape {}, expected {}.".format(
                        data.shape, expected
                    )
                )
            return data
        except Exception as e:
            print("[DSADataset] Warning: could not read {}: {}".format(
                file_path, e
            ))
            return None

    def get_all_samples(
        self,
        split: str = "all",
        train_subjects: Optional[List[str]] = None,
        test_subjects: Optional[List[str]] = None,
    ) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """Return flat lists of (data, label, sensor_id).

        Args:
            split (str): "train", "test", or "all".
            train_subjects (List[str]): Subject IDs for train split.
            test_subjects (List[str]): Subject IDs for test split.

        Returns:
            Tuple of (X, y, sensors).
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
        self,
        sensor_id: str,
        split: str = "all",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get stacked arrays for a specific sensor.

        Args:
            sensor_id (str): One of s1-s5.
            split (str): "train", "test", or "all".

        Returns:
            Tuple (X, y) where X shape is (N, 9, 125).
        """
        X_all, y_all, sensors_all = self.get_all_samples(split=split)
        filtered = [
            (x, lbl)
            for x, lbl, s in zip(X_all, y_all, sensors_all)
            if s == sensor_id
        ]
        if not filtered:
            raise ValueError(
                "No samples found for sensor '{}'.".format(sensor_id)
            )
        X_arr = np.stack([f[0] for f in filtered], axis=0)
        y_arr = np.array([f[1] for f in filtered], dtype=np.int64)
        return X_arr, y_arr

    def __len__(self) -> int:
        """Return total number of subjects."""
        return len(self.patients)

    def __repr__(self) -> str:
        n_events = sum(
            len(evts)
            for subj in self.patients.values()
            for evts in subj.values()
        )
        return (
            "DSADataset(\n"
            "  root={},\n"
            "  target_sensor={} ({}),\n"
            "  subjects={},\n"
            "  total_events={}\n"
            ")"
        ).format(
            self.root,
            self.target_sensor,
            SENSOR_LABELS[self.target_sensor],
            len(self.patients),
            n_events,
        )
