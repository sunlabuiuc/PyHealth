"""
SimulatedStressDataset

This dataset simulates EDA signals using NeuroKit2 to create synthetic
baseline and stress periods. Each 10-second window is labeled with
a binary stress label (0 = baseline, 1 = stress).

No raw data or pre-trained models are required.

Usage:
    ds = SimulatedStressDataset()
    features, label = ds[0]
"""

import neurokit2 as nk
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Dict


class SimulatedStressDataset(Dataset):
    def __init__(self, num_subjects: int = 10, sampling_rate: int = 4):
        """
        Args:
            num_subjects (int): Number of synthetic subjects to simulate
            sampling_rate (int): Sampling rate of the EDA signal
        """
        self.samples = []
        window_sec = 10
        window_size = sampling_rate * window_sec

        for subject_id in range(num_subjects):
            # Simulate baseline and stress segments
            eda_baseline = nk.eda_simulate(duration=180, sampling_rate=sampling_rate, scr_number=np.random.randint(2, 6))
            eda_stress = nk.eda_simulate(duration=120, sampling_rate=sampling_rate, scr_number=np.random.randint(8, 18))

            signal = np.concatenate([eda_baseline, eda_stress])
            labels = np.array([0] * len(eda_baseline) + [1] * len(eda_stress))

            for start in range(0, len(signal) - window_size + 1, window_size):
                end = start + window_size
                window = signal[start:end]
                label_window = labels[start:end]
                label = int(np.round(np.mean(label_window)))

                features = {
                    "mean_eda": float(np.mean(window)),
                    "std_eda": float(np.std(window)),
                    "min_eda": float(np.min(window)),
                    "max_eda": float(np.max(window)),
                }

                self.samples.append((features, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, float], int]:
        return self.samples[idx]

if __name__ == "__main__":
    ds = SimulatedStressDataset()
    print(f"Total samples: {len(ds)}")
    print("First sample:", ds[0])
