"""WESAD (Wearable Stress and Affect Detection) Dataset.

This module provides the WESADDataset class for loading the WESAD dataset,
which contains multimodal physiological signals from Empatica E4 wrist-worn
devices for stress and affect detection research.

Dataset Information:
    - Name: WESAD (Wearable Stress and Affect Detection)
    - Subjects: 15 (S2-S17, excluding S1 and S12)
    - Device: Empatica E4 wristband
    - Signals: EDA (4Hz), BVP (64Hz), ACC (32Hz), TEMP (4Hz)
    - Labels: Baseline (1), Stress (2), Amusement (3)
    
Dataset Reference:
    Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018).
    Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.
    In Proceedings of the 20th ACM International Conference on Multimodal Interaction 
    (ICMI 2018), pp. 400-408.
    https://doi.org/10.1145/3242969.3242985

Dataset Download:
    - UCI: https://archive.ics.uci.edu/ml/datasets/WESAD+(Wearable+Stress+and+Affect+Detection)
    - Kaggle: https://www.kaggle.com/datasets/qiriro/stress

Example:
    >>> from pyhealth.datasets import WESADDataset
    >>> dataset = WESADDataset(root="/path/to/WESAD/")
    >>> dataset.stat()
"""

import os
import pickle
import logging
from typing import Optional, List, Dict, Callable, Any

import numpy as np

from pyhealth.datasets import BaseSignalDataset

logger = logging.getLogger(__name__)

# Sampling rates for Empatica E4 wrist signals
SAMPLING_RATES: Dict[str, int] = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4,
    "label": 700,
}

# Valid subject IDs (S1 missing, S12 excluded due to sensor issues)
SUBJECT_IDS: List[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# Label mapping
LABEL_MAP: Dict[int, str] = {
    0: "transient",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}


class WESADDataset(BaseSignalDataset):
    """Dataset class for WESAD (Wearable Stress and Affect Detection).

    The WESAD dataset contains physiological data from 15 subjects during a 
    lab study using Empatica E4 wrist-worn devices.

    Wrist signals (Empatica E4):
        - ACC: 3-axis accelerometer (32 Hz)
        - BVP: Blood volume pulse (64 Hz)
        - EDA: Electrodermal activity (4 Hz)
        - TEMP: Skin temperature (4 Hz)

    Labels:
        - 1: Baseline (neutral)
        - 2: Stress (Trier Social Stress Test)
        - 3: Amusement (funny videos)

    Args:
        root: Root directory containing subject folders (S2/, S3/, ..., S17/).
        dataset_name: Name of the dataset. Default is "wesad".
        dev: If True, only load 3 subjects. Default is False.
        refresh_cache: If True, reprocess data. Default is False.

    Example:
        >>> from pyhealth.datasets import WESADDataset
        >>> dataset = WESADDataset(root="/path/to/WESAD/", dev=True)
        >>> dataset.stat()
        >>> patient = dataset.get_patient("S2")
        >>> eda = patient["signal"]["wrist"]["EDA"]
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = "wesad",
        dev: bool = False,
        refresh_cache: bool = False,
    ) -> None:
        self.root = root
        self.dataset_name = dataset_name
        self.dev = dev
        self.refresh_cache = refresh_cache

        self.filepath = os.path.join(
            os.path.dirname(os.path.abspath(root.rstrip("/"))),
            f".cache_{dataset_name}",
        )
        os.makedirs(self.filepath, exist_ok=True)

        self.task: Optional[str] = None
        self.samples: Optional[List[Dict]] = None
        self.patient_to_index: Optional[Dict[str, List[int]]] = None
        self.visit_to_index: Optional[Dict[str, List[int]]] = None
        self.patients: Dict[str, Dict[str, Any]] = {}

        self._load_data()

    def _load_data(self) -> None:
        """Load data from cache or process raw files."""
        cache_file = os.path.join(self.filepath, "processed_data.pkl")

        if os.path.exists(cache_file) and not self.refresh_cache:
            logger.info(f"Loading cached data from {cache_file}")
            with open(cache_file, "rb") as f:
                self.patients = pickle.load(f)
        else:
            logger.info("Processing raw WESAD data...")
            self._process_raw_data()
            with open(cache_file, "wb") as f:
                pickle.dump(self.patients, f)

    def _process_raw_data(self) -> None:
        """Process raw .pkl files from each subject folder."""
        subject_ids = SUBJECT_IDS[:3] if self.dev else SUBJECT_IDS

        for sid in subject_ids:
            pkl_file = os.path.join(self.root, f"S{sid}", f"S{sid}.pkl")

            if not os.path.exists(pkl_file):
                logger.warning(f"File not found: {pkl_file}")
                continue

            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f, encoding="latin1")

                self.patients[f"S{sid}"] = {
                    "patient_id": f"S{sid}",
                    "subject_id": sid,
                    "signal": {
                        "wrist": {
                            "ACC": np.array(data["signal"]["wrist"]["ACC"]),
                            "BVP": np.array(data["signal"]["wrist"]["BVP"]),
                            "EDA": np.array(data["signal"]["wrist"]["EDA"]),
                            "TEMP": np.array(data["signal"]["wrist"]["TEMP"]),
                        }
                    },
                    "label": np.array(data["label"]),
                }
                logger.info(f"Loaded S{sid}")
            except Exception as e:
                logger.error(f"Error loading S{sid}: {e}")

    def stat(self) -> None:
        """Print dataset statistics."""
        print("=" * 60)
        print(f"Dataset: {self.dataset_name.upper()}")
        print("=" * 60)
        print(f"Subjects: {len(self.patients)}")
        print(f"IDs: {list(self.patients.keys())}")
        print("\nSignals: ACC (32Hz), BVP (64Hz), EDA (4Hz), TEMP (4Hz)")
        print("\nPer-subject info:")
        
        for pid, patient in self.patients.items():
            duration = len(patient["signal"]["wrist"]["EDA"]) / 4 / 60
            labels = patient["label"]
            print(f"  {pid}: {duration:.1f}min, "
                  f"baseline={np.sum(labels==1)}, "
                  f"stress={np.sum(labels==2)}, "
                  f"amusement={np.sum(labels==3)}")

    def info(self) -> None:
        """Print dataset info."""
        print(f"Dataset: {self.dataset_name}")
        print(f"Root: {self.root}")
        print(f"Subjects: {len(self.patients)}")

    def set_task(self, task_fn: Callable) -> "WESADDataset":
        """Apply task function to create samples."""
        self.samples = []
        self.patient_to_index = {}

        for patient_id, patient_data in self.patients.items():
            start_idx = len(self.samples)
            self.samples.extend(task_fn(patient_data))
            self.patient_to_index[patient_id] = list(range(start_idx, len(self.samples)))

        return self

    def get_patient(self, patient_id: str) -> Dict:
        """Get patient data by ID."""
        if patient_id not in self.patients:
            raise KeyError(f"Patient {patient_id} not found")
        return self.patients[patient_id]

    def __len__(self) -> int:
        return len(self.patients)

    def __iter__(self):
        for pid in self.patients:
            yield self.patients[pid]

    def __getitem__(self, patient_id: str) -> Dict:
        return self.get_patient(patient_id)
