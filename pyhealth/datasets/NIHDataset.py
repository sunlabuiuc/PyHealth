import os
import logging
from typing import Dict, Optional

import pandas as pd
from torchvision import transforms

from pyhealth.datasets.base_dataset import BaseDataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Adjust constants as required
DATA_DIR = "/tmp/CXR8"
SPLIT = "training"
ZIP_FILENAME = "CXR8.zip"
CSV_FILENAME = "Data_Entry_2017_v2020.csv"
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class NIHChestXray8Dataset(BaseDataset):
    """
    Download NIH CXR8 Dataset (45GB):
    https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345

    Table on `Data_Entry_2017_v2020.csv`
    Image Index:                           Unique identifier and filename for each image.
    Finding Labels:                        Diagnostic findings present in the image. Multiple labels are separated by `|`.
    Follow-up:                             Indicator for follow-up examinations
    Patient ID:                            Unique identifier assigned to each patient.
    Patient Age:                           Age of the patient at the time of image capture (in years).
    Patient Sex:                           Biological sex of the patient; typically `M` (male) or `F` (female).
    View Position:                         The orientation or view of the X-ray (e.g., PA for posteroanterior).
    Original Image Width:                  The width of the image in pixels.
    Original Image Height:                 The height of the image in pixels.
    Original Image Pixel Spacing (x):      The physical spacing (resolution) of a pixel along the x-axis.
    Original Image Pixel Spacing (y):      The physical spacing (resolution) of a pixel along the y-axis.
    """
    def __init__(self):
        self.dataset_dir = DATA_DIR
        self.split = SPLIT
        self.transform = TRANSFORM
        self.zip_file = os.path.join(self.dataset_dir, ZIP_FILENAME)

        self.labels = self.labels()
        self.samples = self.dataset()

    def labels(self) -> Dict[str, str]:
        csv_path = self.csv()
        if not csv_path:
            raise FileNotFoundError(f"{CSV_FILENAME} not found.")
        df = pd.read_csv(csv_path, dtype=str)
        return dict(zip(df["Image Index"], df["Finding Labels"]))

    def dataset(self) -> Dict[int, Dict[str, object]]:
        split_file = "train_val_list.txt" if self.split == "training" else "test_list.txt"
        split_path = os.path.join(self.dataset_dir, split_file)
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"Split file missing: {split_file}")

        with open(split_path, "r") as f:
            filenames = [line.strip() for line in f if line.strip()]

        image_paths = self.image_idx()
        missing = [f for f in filenames if f not in image_paths]
        if missing:
            raise FileNotFoundError(f"Missing images: {missing[:5]}...")

        records = []
        for fname in filenames:
            path = image_paths[fname]
            label_text = self.labels.get(fname, "No Finding")
            label = 0 if label_text == "No Finding" else 1
            records.append({"image_path": path, "label": label})

        return {i: r for i, r in enumerate(records)}

    def csv(self) -> Optional[str]:
        for dirpath, _, files in os.walk(self.dataset_dir):
            if CSV_FILENAME in files:
                return os.path.join(dirpath, CSV_FILENAME)
        return None

    def image_idx(self) -> Dict[str, str]:
        idx = {}
        for item in os.listdir(self.dataset_dir):
            img_dir = os.path.join(self.dataset_dir, item, "images")
            if os.path.isdir(img_dir):
                for img_name in os.listdir(img_dir):
                    idx[img_name] = os.path.join(img_dir, img_name)
        return idx

    def __len__(self) -> int:
        return len(self.samples)

    def stat(self) -> str:
        output = f"Dataset: NIH ChestXray 8 | Split: {self.split} | Total Samples: {len(self)}"
        return output


if __name__ == "__main__":
    dataset = NIHChestXray8Dataset()
    print(dataset.stat())
