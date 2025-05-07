from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

from pyhealth.datasets import BaseDataset

class MIMICCXRDataset(BaseDataset):
    """MIMIC‑CXR image dataset (No‑Finding, Pneumothorax, Fracture, …).

    Expected directory layout::

        root/
            images/              # jpg or png
                <studydir>/<img>.jpg
            metadata.csv         # columns: path,label,sex,ethnicity,age,split

    Parameters
    ----------
    root: str or Path
        Path containing *images/* and *metadata.csv*.
    task: str {"No Finding", "Pneumothorax", "Fracture"}
        Target column in metadata.
    transform: torchvision.transforms
        Optional image transforms.
    split: str {"train","val","test"}
        Loads only rows where metadata["split"] == split.
    """
    def __init__(
        self,
        root: str | Path,
        task: str = "No Finding",
        split: str = "train",
        transform: transforms.Compose | None = None,
    ) -> None:
        super().__init__(root)
        self.root = Path(root)
        self.task = task
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # load metadata
        meta_path = self.root / "metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        df = pd.read_csv(meta_path)
        self.df = df[df["split"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_path = self.root / "images" / row["path"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(row[self.task], dtype=torch.float32)
        sample = {
            "path": row["path"],
            "label": label,
            "sex": row["sex"],
            "ethnicity": row["ethnicity"],
            "age": row["age"],
        }
        return img, sample

    @property
    def num_classes(self) -> int:  # required by some PyHealth models
        return 1
