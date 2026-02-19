# tests/core/test_soz_stimulation_dataset.py

import os
import numpy as np
import torch

from pyhealth.datasets import SOZStimulationDataset


def test_soz_stimulation_dataset_loading(tmp_path):
    # Create temp directory with fake data
    root = tmp_path / "soz_spes_processed"
    os.makedirs(root, exist_ok=True)

    # Fake data
    N, C, T = 10, 4, 100
    X_stim = np.random.randn(N, C, T).astype("float32")
    y = np.random.randint(0, 2, size=(N,)).astype("int64")

    np.save(root / "train_X_stim.npy", X_stim)
    np.save(root / "train_y.npy", y)

    # Load dataset
    ds = SOZStimulationDataset(root=str(root), split="train")
    assert len(ds) == N

    sample, label = ds[0]
    assert "X_stim" in sample
    assert isinstance(sample["X_stim"], torch.Tensor)
    assert isinstance(label, int)
