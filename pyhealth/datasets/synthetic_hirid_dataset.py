import numpy as np
from torch.utils.data import Dataset


class SyntheticHiRIDDataset(Dataset):
    """
    Loads synthetic HiRID-style ICU time series data.
    """

    def __init__(self, data_path: str):
        self.data = np.load(f"{data_path}/synthetic_data.npy")
        self.labels = np.load(f"{data_path}/synthetic_labels.npy")
        assert self.data.shape[0] == self.labels.shape[0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]  # shape: (channels, time)
        y = self.labels[idx]
        return x, y
