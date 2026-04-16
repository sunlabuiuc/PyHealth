"""
Physical Activity Task for PyHealth.

Converts DSADataset samples into normalized tensors and integer labels
for activity classification using the adaptive transfer learning framework.

Reference:
    Zhang et al. "Daily Physical Activity Monitoring: Adaptive Learning
    from Multi-source Motion Sensor Data." CHIL 2024.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union, Type

from pyhealth.tasks.base_task import BaseTask
from pyhealth.datasets.dsa_dataset import (
    DSADataset,
    SENSOR_LABELS,
    N_CHANNELS,
    TIMESTEPS,
    N_ACTIVITIES,
)


class PhysicalActivityTask(BaseTask):
    """Task for daily physical activity classification using DSA sensor data.

    Defines the input/output schema for classifying physical activities
    from wearable motion sensor time series. Supports multi-domain
    transfer learning by providing separate datasets for source and
    target sensor domains.

    Attributes:
        task_name (str): Name of the task.
        input_schema (Dict): Input feature schema — 9-channel time series.
        output_schema (Dict): Output label schema — 19-class activity label.

    Args:
        dsa_dataset (DSADataset): A loaded DSADataset instance.
        train_subjects (List[str]): Subject IDs for training. Defaults to p1-p6.
        test_subjects (List[str]): Subject IDs for testing. Defaults to p7-p8.
        normalize (bool): Apply min-max normalization. Default True.

    Examples:
        >>> from pyhealth.datasets.dsa_dataset import DSADataset
        >>> dsa = DSADataset(root="/path/to/DSA/data", target_sensor="s2")
        >>> task = PhysicalActivityTask(dsa)
        >>> train_ds = task.get_target_dataset(split="train")
        >>> x, y = train_ds[0]
        >>> print(x.shape)
        torch.Size([9, 125])
    """

    # ── BaseTask required class attributes ────────────────────────────────
    task_name: str = "physical_activity_classification"

    input_schema: Dict[str, Union[str, Type]] = {
        "sensor_timeseries": "float",   # shape (9, 125) — 9 channels x 125 timesteps
    }

    output_schema: Dict[str, Union[str, Type]] = {
        "activity_label": "multiclass",  # integer label in [0, 18]
    }

    # ── Default train/test split from paper ───────────────────────────────
    DEFAULT_TRAIN = ["p1", "p2", "p3", "p4", "p5", "p6"]
    DEFAULT_TEST  = ["p7", "p8"]

    def __init__(
        self,
        dsa_dataset: DSADataset,
        train_subjects: Optional[List[str]] = None,
        test_subjects: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> None:
        self.dsa_dataset    = dsa_dataset
        self.train_subjects = train_subjects or self.DEFAULT_TRAIN
        self.test_subjects  = test_subjects  or self.DEFAULT_TEST
        self.normalize      = normalize

    def __call__(self, patient: Dict) -> List[Dict]:
        """Process one patient record into a list of task samples.

        Required by BaseTask. Each sample contains one sensor window
        with its activity label.

        Args:
            patient (Dict): Patient dict from DSADataset with keys:
                sensor_id, data (9,125), label, activity_name.

        Returns:
            List of dicts, each with keys:
                - sensor_timeseries (np.ndarray): shape (9, 125)
                - activity_label (int): class label 0-18
                - sensor_id (str): which sensor this came from
                - is_target (bool): whether this is the target sensor
        """
        samples = []
        for event in patient:
            samples.append({
                "sensor_timeseries": event["data"],
                "activity_label":    event["label"],
                "sensor_id":         event["sensor_id"],
                "is_target":         event["is_target"],
            })
        return samples

    def get_dataset(
        self,
        sensor_id: str = "s2",
        split: str = "train",
    ) -> "PhysicalActivityDataset":
        """Build a PhysicalActivityDataset for a given sensor and split.

        Args:
            sensor_id (str): One of s1-s5, or "all".
            split (str): "train" or "test".

        Returns:
            PhysicalActivityDataset ready for DataLoader.
        """
        return PhysicalActivityDataset(
            dsa_dataset=self.dsa_dataset,
            sensor_id=sensor_id,
            split=split,
            train_subjects=self.train_subjects,
            test_subjects=self.test_subjects,
            normalize=self.normalize,
        )

    def get_source_datasets(
        self, split: str = "train"
    ) -> Dict[str, "PhysicalActivityDataset"]:
        """Build one dataset per source sensor domain.

        Source domains = all sensors except the target sensor.
        Used during the pre-training phase of AdaptiveTransferModel.

        Args:
            split (str): "train" or "test".

        Returns:
            Dict of sensor_id -> PhysicalActivityDataset for sources only.
        """
        target = self.dsa_dataset.target_sensor
        return {
            sid: self.get_dataset(sensor_id=sid, split=split)
            for sid in SENSOR_LABELS
            if sid != target
        }

    def get_target_dataset(
        self, split: str = "train"
    ) -> "PhysicalActivityDataset":
        """Build the target sensor domain dataset.

        The target is the daily-use wearable sensor (e.g. right arm).

        Args:
            split (str): "train" or "test".

        Returns:
            PhysicalActivityDataset for the target sensor only.
        """
        return self.get_dataset(
            sensor_id=self.dsa_dataset.target_sensor,
            split=split,
        )

    def summary(self) -> None:
        """Print a summary of the task configuration."""
        target  = self.dsa_dataset.target_sensor
        sources = [s for s in SENSOR_LABELS if s != target]
        print("=" * 50)
        print("PhysicalActivityTask Summary")
        print("=" * 50)
        print("task_name     : {}".format(self.task_name))
        print("input_schema  : {}".format(self.input_schema))
        print("output_schema : {}".format(self.output_schema))
        print("Target sensor : {} ({})".format(target, SENSOR_LABELS[target]))
        print("Source sensors: {}".format(sources))
        print("Train subjects: {}".format(self.train_subjects))
        print("Test subjects : {}".format(self.test_subjects))
        print("Num classes   : {}".format(N_ACTIVITIES))
        print("Input shape   : ({}, {})".format(N_CHANNELS, TIMESTEPS))
        print("=" * 50)


class PhysicalActivityDataset(Dataset):
    """PyTorch Dataset for physical activity classification.

    Wraps a DSADataset, filters by sensor and split, applies per-channel
    min-max normalization to [-1, 1], and returns (tensor, label) pairs.

    Args:
        dsa_dataset (DSADataset): A loaded DSADataset instance.
        sensor_id (str): Sensor to use. One of s1-s5, or "all".
        split (str): "train", "test", or "all".
        train_subjects (List[str]): Subject IDs for train split.
        test_subjects (List[str]): Subject IDs for test split.
        normalize (bool): Apply min-max normalization. Default True.

    Examples:
        >>> dsa = DSADataset(root="/path/to/DSA/data", dev=True)
        >>> ds = PhysicalActivityDataset(dsa, sensor_id="s2", split="train")
        >>> x, y = ds[0]
        >>> print(x.shape)
        torch.Size([9, 125])
    """

    def __init__(
        self,
        dsa_dataset: DSADataset,
        sensor_id: str = "s2",
        split: str = "train",
        train_subjects: Optional[List[str]] = None,
        test_subjects: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> None:
        self.sensor_id = sensor_id
        self.normalize = normalize

        valid = list(SENSOR_LABELS.keys()) + ["all"]
        if sensor_id not in valid:
            raise ValueError(
                "sensor_id must be one of {}, got '{}'.".format(
                    valid, sensor_id
                )
            )

        X_raw, y_raw, sensors_raw = dsa_dataset.get_all_samples(
            split=split,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
        )

        if sensor_id != "all":
            pairs = [
                (x, lbl)
                for x, lbl, s in zip(X_raw, y_raw, sensors_raw)
                if s == sensor_id
            ]
        else:
            pairs = list(zip(X_raw, y_raw))

        if len(pairs) == 0:
            raise ValueError(
                "No samples found for sensor='{}', split='{}'.".format(
                    sensor_id, split
                )
            )

        X_arr = np.stack([p[0] for p in pairs], axis=0).astype(np.float32)
        y_arr = np.array([p[1] for p in pairs], dtype=np.int64)

        if normalize:
            X_arr = self._minmax_normalize(X_arr)

        self.X = torch.tensor(X_arr, dtype=torch.float32)
        self.y = torch.tensor(y_arr, dtype=torch.long)

    @staticmethod
    def _minmax_normalize(X: np.ndarray) -> np.ndarray:
        """Normalize each channel to [-1, 1] across all samples.

        Formula: x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1

        Args:
            X (np.ndarray): Shape (N, 9, 125).

        Returns:
            np.ndarray: Shape (N, 9, 125), values in [-1, 1].
        """
        x_min = X.min(axis=(0, 2), keepdims=True)
        x_max = X.max(axis=(0, 2), keepdims=True)
        denom = x_max - x_min
        denom[denom == 0] = 1.0
        return 2.0 * (X - x_min) / denom - 1.0

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one (input_tensor, label) pair.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple of (x, y) where x shape is (9, 125) and y is scalar int.
        """
        return self.X[idx], self.y[idx]
