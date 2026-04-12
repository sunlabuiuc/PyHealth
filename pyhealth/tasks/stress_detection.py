"""
Stress detection task for the WESAD dataset.

Maps windowed EDA samples from WESADDataset to model-ready
input/output pairs for binary or three-class stress classification.

Authors:
    Megan Saunders, Jennifer Miranda, Jesus Torres
    {meganas4, jm123, jesusst2}@illinois.edu
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class StressDetectionDataset(Dataset):
    """PyTorch Dataset wrapping WESAD windows for stress detection.

    Takes the list of sample dicts produced by WESADDataset and returns
    tensors suitable for PyHealth model training.

    Attributes:
        samples (List[Dict]): Windowed EDA samples from WESADDataset.
        subject_ids (List[str]): Unique subject identifiers present in samples.
        num_classes (int): Number of output classes inferred from label set.

    Example::
        >>> from pyhealth.datasets.wesad import WESADDataset
        >>> from pyhealth.tasks.stress_detection import StressDetectionDataset
        >>> raw = WESADDataset(root="./WESAD")
        >>> task = StressDetectionDataset(raw.samples)
        >>> x, y = task[0]
        >>> print(x.shape, y)
    """

    def __init__(self, samples: List[Dict],
                 subject_filter: Optional[List[str]] = None) -> None:
        """Initializes the stress detection task dataset.

        Args:
            samples (List[Dict]): List of dicts with keys 'eda' (np.ndarray),
                'label' (int), and 'subject_id' (str), as produced by
                WESADDataset._load_and_window.
            subject_filter (Optional[List[str]]): If provided, only include
                samples from these subject IDs. Useful for LNSO cross-validation
                splits.
        """
        if subject_filter is not None:
            samples = [s for s in samples if s["subject_id"] in subject_filter]

        self.samples = samples
        self.subject_ids = sorted({s["subject_id"] for s in samples})
        labels = {s["label"] for s in samples}
        self.num_classes = len(labels)

        logger.info(
            f"StressDetectionDataset: {len(self.samples)} windows, "
            f"{self.num_classes} classes, "
            f"{len(self.subject_ids)} subjects."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Returns a single EDA window and its label as tensors.

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - eda: Float tensor of shape (window_size,)
                - label: Long tensor scalar
        """
        sample = self.samples[idx]
        eda = torch.tensor(sample["eda"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return eda, label

    def get_subject_splits(self, test_subjects: List[str]):
        """Returns train and test subsets by subject for LNSO cross-validation.

        Args:
            test_subjects (List[str]): Subject IDs to hold out as test set.

        Returns:
            Tuple[StressDetectionDataset, StressDetectionDataset]:
                Train dataset (all subjects not in test_subjects) and
                test dataset (only test_subjects).

        Example::
            >>> train_ds, test_ds = task.get_subject_splits(["S2", "S3"])
        """
        train_subjects = [s for s in self.subject_ids if s not in test_subjects]
        train_ds = StressDetectionDataset(self.samples, subject_filter=train_subjects)
        test_ds = StressDetectionDataset(self.samples, subject_filter=test_subjects)
        return train_ds, test_ds