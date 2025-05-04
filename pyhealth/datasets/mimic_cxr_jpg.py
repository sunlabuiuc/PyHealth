from torch.utils.data import Dataset
import pandas as pd
from typing import List, Tuple, Dict, Union
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import torch

class CXRDataset(Dataset):
    def __init__(
        self,
        csv_path:str,
        img_root: str,
        group_col: Union[str, None] = None,
        group_encodings: Union[Dict[Union[str, int, float], int], None] = None,
        labels: List[str]=[
            "No Finding",
            "Fracture", 
            "Pneumothorax"
        ],
        img_size: int=224, 
        train: bool =True, 
        uncertainty_strategy: str="zero",
    ):
        """
        A dataset to serve the research in:
        Improving Fairness of X-Ray Classifiers
            Haoran Zhang, Natalie Dullerud, Karsten Roth, Lauren Oakden-Rayner, Stephen Robert
            Pfohl, and Marzyeh Ghassemi
        Requires access to the MIMIC-CXR-JPG dataset and the stanford CheXpert dataset

        csv_path: str
            The path to the CheXpert dataset
        img_root: str
            The root of the images from the MIMIC-CXR-JPG dataset
        group_col: str
            The name of the column of the demographic axis to compare fairness on. None if that is not in scope
        group_encodings: dict
            A key value mapping of encoding to apply to the demographic groups along the demographic axis of interest. None if not in scope
        labels: List[str]
            The labels present in the dataset.
            Defaults to: [
                    "No Finding",
                    "Fracture", 
                    "Pneumothorax"
                ]
        img_size: int
            The dimesions of the images, defaults to 224 as this is what was used in the paper
        train: bool
            If this is a training dataset
        uncertainty_strategy: str 
            One of: "zero" (convert 0.5 -> 0), "one" (convert 0.5 -> 1), "ignore" (mask label)
        """
        if uncertainty_strategy not in ["zero", "one", "ignore"]:
            raise ValueError(f"{uncertainty_strategy} not in ['zero', 'one', 'ignore']")

        self.data = pd.read_csv(csv_path)
        self.labels = labels
        self.img_root = img_root
        self.group_col = group_col
        self.group_encodings = group_encodings
        self.transforms = self._get_transforms(train, img_size)
        self.uncertainty_strategy = uncertainty_strategy

    @staticmethod
    def _get_transforms(train: bool = True, img_size: int = 224) -> transforms.Compose:
        # the normalization parameters set based on the ImageNet dataset
        IMAGENET_MEAN = [0.485, 0.456, 0.406],
        IMAGENET_STD= [0.229, 0.224, 0.225]
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_root, row["path"])
        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)

        label = torch.tensor(row[self.labels].values.astype(np.float32), dtype=torch.float32)

        if self.uncertainty_strategy == "zero":
            label[label == 0.5] = 0
        elif self.uncertainty_strategy == "one":
            label[label == 0.5] = 1
        elif self.uncertainty_strategy == "ignore":
            pass
        group = self.group_encoding.get(row[self.group_col]) if self.group_col is not None else None

        return image, label, group
