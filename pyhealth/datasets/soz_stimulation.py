# pyhealth/datasets/soz_stimulation.py

from typing import Dict, Tuple, Optional
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SOZStimulationDataset(Dataset):
    r"""
    SOZStimulationDataset

    Dataset for seizure onset zone (SOZ) localization using single-pulse
    electrical stimulation (SPES) EEG responses, based on the paper:

    Norris, J., Chari, A., van Blooijs, D., Cooray, G., Friston, K.,
    Tisdall, M., & Rosch, R. (2024). Localising the Seizure Onset Zone
    from Single-Pulse Electrical Stimulation Responses with a CNN
    Transformer. arXiv preprint arXiv:2403.20324.
    https://doi.org/10.48550/arXiv.2403.20324


    This class is designed to work with preprocessed numpy arrays generated
    from the original BIDS EEG dataset. We do NOT ship the raw or preprocessed
    data with PyHealth. Instead, users are expected to:

    1. Obtain access to the original SPES/BIDS dataset.
    2. Run a preprocessing script (e.g., from the reproduction repo) that
       saves numpy files in a format like:

        root/
          train_X_stim.npy        # shape: [N_train, C_stim, T]
          train_y.npy             # shape: [N_train]

          val_X_stim.npy
          val_y.npy

          test_X_stim.npy
          test_y.npy

    Each sample consists of:
        - "X_stim": stimulation-locked EEG response segment, shape [C_stim, T]
        - label: int in {0, 1}, indicating SOZ vs non-SOZ.

    Args:
        root: Root directory containing preprocessed numpy arrays.
        split: One of {"train", "val", "test"}.
        device: Optional torch.device to move tensors to on __getitem__.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"

        self.root = root
        self.split = split
        self.device = device

        stim_path = os.path.join(root, f"{split}_X_stim.npy")
        y_path = os.path.join(root, f"{split}_y.npy")

        if not os.path.exists(stim_path):
            raise FileNotFoundError(f"Cannot find {stim_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"Cannot find {y_path}")

        self.X_stim = np.load(stim_path)  # [N, C_stim, T]
        self.y = np.load(y_path)         # [N]

        if len(self.X_stim) != len(self.y):
            raise ValueError("X_stim and y must have the same number of samples")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        x_stim = torch.as_tensor(self.X_stim[idx], dtype=torch.float32)
        label = int(self.y[idx])

        sample: Dict[str, torch.Tensor] = {"X_stim": x_stim}

        if self.device is not None:
            sample = {k: v.to(self.device) for k, v in sample.items()}

        return sample, label


if __name__ == "__main__":
    # Simple smoke test (requires real preprocessed data under this folder)
    ds = SOZStimulationDataset(root="data/soz_spes_processed", split="train")
    print(f"Num samples: {len(ds)}")
    x, y = ds[0]
    print("Keys:", x.keys())
    print("X_stim shape:", x["X_stim"].shape)
    print("Label:", y)
