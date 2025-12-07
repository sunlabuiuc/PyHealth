import os
import pandas as pd
import numpy as np
import torch

from pyhealth.datasets import BaseDataset
from pyhealth.data import Sample


class MITBIHArrhythmiaDataset(BaseDataset):
    """MIT-BIH Arrhythmia Dataset (Kaggle CSV format)
    
    Expected Files:
        - mitbih_train.csv
        - mitbih_test.csv

    Each row:
        [187 signal values..., label]

    Produces PyHealth Sample objects:
        Sample(
            patient_id,
            visit_id,
            data={"signal": Tensor(1,187)},
            label=int
        )
    """

    def __init__(self, root: str, split: str = "train", **kwargs):
        super().__init__(root, **kwargs)

        assert split in ["train", "test"], "split must be 'train' or 'test'"
        self.split = split

        file_map = {
            "train": "mitbih_train.csv",
            "test": "mitbih_test.csv",
        }
        csv_path = os.path.join(root, file_map[split])

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")

        self.df = pd.read_csv(csv_path, header=None)

        # X: shape (N, 187), y: shape (N,)
        self.X = self.df.iloc[:, :-1].values.astype(np.float32)
        self.y = self.df.iloc[:, -1].values.astype(int)

        self.n_samples = len(self.df)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index: int) -> Sample:
        # in MIT-BIH there are no patient/visit IDs → generate synthetic IDs
        patient_id = f"p{index}"
        visit_id = f"v{index}"

        signal = torch.tensor(self.X[index], dtype=torch.float32).unsqueeze(0)  # (1,187)
        label = int(self.y[index])

        return Sample(
            patient_id=patient_id,
            visit_id=visit_id,
            data={"signal": signal},
            label=label,
        )
