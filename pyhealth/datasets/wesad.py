"""
Contributors: Kevin Lin(Kevin25@illinois.edu), Yen-Chu Yu(yenchuy2@illinois.edu) 
Related Paper: Contrastive Learning of Electrodermal Activity Representations for Stress Detection
Paper Link: https://openreview.net/pdf?id=bSC_xo8VQ1b

Description:
This dataset class wraps the WESAD dataset for use in contrastive or supervised deep learning tasks, especially 
physiological signal modeling. It handles multi-modal sensor fusion, temporal segmentation, and optional label assignment.

---

The WESAD(Wearable Stress and Affect Detection) dataset is a multimodal dataset collected for the purpose of stress and affect detection using wearable sensors. 
It includes signals from both a **chest-worn RespiBAN** device and a **wrist-worn Empatica E4**, but this class focuses on 
the *chest* device for its richer, high-frequency signals.

**Subjects**: 15 individuals (2–17, excluding S12)
**Recording Phases**: Each subject experiences 3 conditions:
- Baseline (neutral)
- Stress (Trier Social Stress Test)
- Amusement (watching a comedy clip)

**Signals from Chest Device**:
- **ACC**: 3-axis accelerometer @ 32 Hz
- **EDA**: Electrodermal activity @ 4 Hz
- **Temp**: Skin temperature @ 4 Hz
- **ECG**: Electrocardiogram @ 700 Hz
- **Resp**: Respiration @ 700 Hz

---

Dataset Processing:
This class performs the following steps:
1. **Loads** WESAD `.pkl` files per subject from the specified `root` directory.
2. **Resamples** all modalities to a common sampling frequency (default: 32 Hz) using Fourier interpolation.
3. **Concatenates** all signals along the feature dimension.
4. **Segments** the resulting time-series into fixed-length windows (default: 240 time steps = 7.5 seconds at 32 Hz).
5. **Stores** each segment along with metadata such as subject ID and segment ID.

Each sample in the final dataset is a dictionary:
```python
{
    "subject": "S11",
    "segment_id": "S11_42",
    "data": np.ndarray of shape (240, 7),  # concatenated [ACC(3) + EDA(1) + Temp(1) + ECG(1) + Resp(1)]
    "label": np.ndarray of shape (240,), 
}
"""

import logging
import os
import pickle
import numpy as np
from typing import List, Optional
from scipy.signal import resample

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class WESADDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        segment_length: int = 240,
        target_freq: int = 32,
        dataset_name: Optional[str] = None,
        dev: bool = False,
        **kwargs,
    ):
        """
        Args:
            root: path to the root directory of the WESAD dataset.
            segment_length: number of time points per segment (e.g., 240).
            target_freq: target sampling frequency for all signals (default: 32Hz).
        """
        self.root = root
        self.tables = []
        self.dataset_name = dataset_name or 'WESAD'
        self.config = {}
        self.dev = dev 

        logger.info(
            f"Initializing {self.dataset_name} dataset from {self.root} (dev mode: {self.dev})"
        )

        self.segment_length = segment_length
        self.target_freq = target_freq

        self.global_event_df = self.load_data()

        # Cached attributes
        self._collected_global_event_df = None
        self._unique_patient_ids = None

    def load_data(self):
        dataset = []
        for subject_folder in sorted(os.listdir(self.root)):
            if not subject_folder.startswith("S"):
                continue
            pkl_path = os.path.join(self.root, subject_folder, f"{subject_folder}.pkl")
            if not os.path.exists(pkl_path):
                continue
            with open(pkl_path, "rb") as f:
                sample = pickle.load(f)

            segments = self.extract_modalities(sample)

            for i, (segment, label) in enumerate(segments):
                dataset.append({
                    "subject": sample["subject"],
                    "segment_id": f"{sample['subject']}_{i}",
                    "data": segment,  # shape: (segment_length, D)
                    "label": label
                })

        return dataset

    def resample_signal(self, signal: np.ndarray, original_freq: int, target_freq: int, target_length: int) -> np.ndarray:
        """
        Resamples a 1D or 2D signal to match the target frequency and length.
        """
        duration = len(signal) / original_freq
        num_target_points = int(duration * target_freq)
        if signal.ndim == 1:
            return resample(signal, num_target_points)[:target_length]
        else:
            return resample(signal, num_target_points, axis=0)[:target_length]


    def extract_modalities(self, sample):
        """
        Extract and resample WESAD modalities to a uniform frequency, then segment.
        """
        chest = sample["signal"]["chest"]

        # Original sampling rates
        freqs = {
            "ACC": 32,
            "EDA": 4,
            "Temp": 4,
            "ECG": 700,
            "Resp": 700,
            "label": 700
        }

        # Resample each modality to target frequency (e.g., 32 Hz)
        acc = chest["ACC"]  # already at target frequency
        length = len(acc)  # target segment length in time steps
        ecg = self.resample_signal(chest["ECG"], freqs["ECG"], self.target_freq, length)[:, None]
        eda = self.resample_signal(chest["EDA"], freqs["EDA"], self.target_freq, length)[:, None]
        resp = self.resample_signal(chest["Resp"], freqs["Resp"], self.target_freq, length)[:, None]
        temp = self.resample_signal(chest["Temp"], freqs["Temp"], self.target_freq, length)[:, None]
        label = self.resample_signal(sample["label"], freqs["label"], self.target_freq, length)

        # Concatenate all signals (time aligned)
        combined = np.concatenate([acc, eda, temp, ecg, resp], axis=1)  # shape: (length, D)
        # Segment into fixed-length windows
        segments = [
            (combined[i:i+self.segment_length], label[i:i+self.segment_length])
            for i in range(0, length - self.segment_length + 1, self.segment_length)
        ]

        return segments