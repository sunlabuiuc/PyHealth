import os
import subprocess
import zipfile
import logging
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image

from pyhealth.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INFO_MSG = (
    "The NIH Chest X-ray Dataset.\n"
    "URL: https://www.kaggle.com/datasets/nih-chest-xrays/data\n\n"
    "The dataset contains over 100,000 frontal-view X-ray images with 14 common "
    "thoracic disease labels.\n\n"
    "Citation:\n"
    "Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M.\n"
    '"ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on '
    "Weakly-Supervised Classification and Localization of Common Thorax "
    'Diseases."\nIEEE CVPR 2017, ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf\n'
    "doi: 10.1109/CVPR.2017.369"
)

DATASET_ZIP_FILENAME = "nih_chestxray.zip"


class NIHChestXrayDataset(BaseDataset):
    """NIH Chest X‑ray Dataset with binary labels (No Finding vs Finding)."""

    def __init__(
        self,
        root: str,
        split: str = "training",
        transform: Optional[callable] = None,
        download: bool = True,
    ):
        """
        Initialize the NIH Chest X‑ray dataset.

        Downloads, extracts, and processes the dataset, building a map of
        sample indices to image paths and binary labels.

        Args:
            root (str): Directory to download/extract into & read from.
            split (str): "training" uses train_val_list.txt; "test" uses test_list.txt.
            transform (callable, optional): Transform applied to loaded PIL images.
            download (bool): If True, download & extract archive before processing.

        Raises:
            ValueError: If split is not "training" or "test".
            FileNotFoundError: If required CSV or split files are missing.

        Example:
            >>> ds = NIHChestXrayDataset(root="/tmp/nih", split="training")
            >>> ds.stat()
        """
        if split not in ("training", "test"):
            raise ValueError("split must be 'training' or 'test'")
        self.root = root
        self.split = split
        self.transform = transform
        self.dataset_name = "NIH Chest X-ray"
        self.dataset_url = (
            "https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data"
        )
        self.dataset_zip = os.path.join(root, DATASET_ZIP_FILENAME)

        if download:
            self._download()
            self._prepare()

        csv_path = self._find_labels_csv()
        df = pd.read_csv(csv_path, dtype=str)
        self._raw_label_map: Dict[str, str] = dict(
            zip(df["Image Index"], df["Finding Labels"])
        )

        self.patients = self.process()

    def _find_labels_csv(self) -> str:
        """
        Recursively locate Data_Entry_2017.csv under the root directory.

        Searches all subdirectories under `self.root` for a file named 
        'Data_Entry_2017.csv' and returns its full path.

        Returns:
            str: Full path to the labels CSV.

        Raises:
            FileNotFoundError: If the CSV is not found anywhere under root.

        Example:
            >>> csv_path = ds._find_labels_csv()
        """
        for dirpath, _, files in os.walk(self.root):
            if "Data_Entry_2017.csv" in files:
                found = os.path.join(dirpath, "Data_Entry_2017.csv")
                logger.info(f"Found labels CSV at {found}")
                return found
        raise FileNotFoundError(
            f"Could not find Data_Entry_2017.csv anywhere under {self.root}"
        )

    def process(self) -> Dict[int, Dict[str, object]]:
        """
        Read split list, index images, attach binary label, and return mapping.

        Reads the appropriate split file (train_val_list.txt or test_list.txt),
        finds image paths via _index_all_images(), and maps each filename to
        its binary label (0=no finding, 1=finding).

        Returns:
            Dict[int, Dict[str, object]]: sample index -> {'image_path': str, 'label': int}

        Raises:
            FileNotFoundError: If the split file or images are missing.

        Example:
            >>> samples = ds.process()
        """
        list_file = (
            "train_val_list.txt"
            if self.split == "training"
            else "test_list.txt"
        )
        split_path = os.path.join(self.root, list_file)
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"Missing split file: {split_path}")

        with open(split_path, "r") as f:
            filenames = [ln.strip() for ln in f if ln.strip()]

        filename_to_path = self._index_all_images()
        missing = [fn for fn in filenames if fn not in filename_to_path]
        if missing:
            raise FileNotFoundError(f"Missing images for: {missing[:5]}…")

        samples: List[Dict[str, object]] = []
        for fn in filenames:
            img_path = filename_to_path[fn]
            raw_label = self._raw_label_map.get(fn, "No Finding")
            label = 0 if raw_label == "No Finding" else 1
            samples.append({"image_path": img_path, "label": label})

        return {i: s for i, s in enumerate(samples)}

    def _index_all_images(self) -> Dict[str, str]:
        """
        Scan all images_*/images/ subfolders to map filename to full path.

        Returns:
            Dict[str, str]: filename -> absolute image path

        Example:
            >>> mapping = ds._index_all_images()
        """
        mapping: Dict[str, str] = {}
        for entry in os.listdir(self.root):
            folder = os.path.join(self.root, entry)
            images_dir = os.path.join(folder, "images")
            if os.path.isdir(images_dir):
                for img_name in os.listdir(images_dir):
                    mapping[img_name] = os.path.join(images_dir, img_name)
        return mapping

    def _download(self) -> None:
        """
        Download the dataset zip file if it is not already present.

        Uses wget to download to self.dataset_zip.

        Example:
            >>> ds._download()
        """
        os.makedirs(self.root, exist_ok=True)
        if os.path.exists(self.dataset_zip):
            logger.info("Zip exists; skipping download.")
            return

        logger.info("Downloading NIH Chest X-ray dataset…")
        cmd = f"wget -nc -O {self.dataset_zip} {self.dataset_url}"
        subprocess.run(cmd, check=True, shell=True)
        logger.info("Download successful.")

    def _prepare(self) -> None:
        """
        Extract the dataset zip into root if not already extracted.

        Checks for Data_Entry_2017.csv to determine if extraction is needed.

        Example:
            >>> ds._prepare()
        """
        # we check for one known file to skip re-extraction
        if os.path.exists(os.path.join(self.root, "Data_Entry_2017.csv")):
            logger.info("Already extracted; skipping.")
            return

        logger.info("Extracting NIH Chest X-ray zip…")
        with zipfile.ZipFile(self.dataset_zip, "r") as zf:
            zf.extractall(self.root)
        logger.info("Extraction complete.")

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset split.

        Returns:
            int: total number of samples

        Example:
            >>> len(ds)
        """
        return len(self.patients)

    def __getitem__(self, index: int) -> Image.Image:
        """
        Load and return a single sample as a PIL Image (optionally transformed).

        Args:
            index (int): Sample index in range [0, len(self)).

        Returns:
            PIL.Image.Image: Loaded (and transformed) image.

        Raises:
            IndexError: If the index is out of bounds.

        Example:
            >>> img = ds[0]
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        p = self.patients[index]
        img = Image.open(p["image_path"]).convert("RGB")
        return self.transform(img) if self.transform else img

    def stat(self) -> str:
        """
        Print and return simple statistics about the dataset.

        Returns:
            str: Formatted statistics string.

        Example:
            >>> stats = ds.stat()
        """
        out = (
            f"Dataset: {self.dataset_name}\n"
            f"Split:   {self.split}\n"
            f"Samples: {len(self)}"
        )
        print(out)
        return out

    @staticmethod
    def info() -> None:
        """
        Print dataset citation and expected folder structure.

        Example:
            >>> NIHChestXrayDataset.info()
        """
        print(INFO_MSG)


if __name__ == "__main__":
    import argparse
    from torchvision import transforms
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="NIH CXR smoke test")
    parser.add_argument("--root",    default="/tmp/nih_chestxray", type=str)
    parser.add_argument("--split",   default="training",
                        choices=["training", "test"])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    my_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    ds = NIHChestXrayDataset(
        root=args.root,
        split=args.split,
        transform=my_transform,
        download=args.download,
    )

    NIHChestXrayDataset.info()
    ds.stat()

    # Verify that __getitem__ now returns a Tensor
    img_tensor = ds[0]
    print("Transformed image shape:", img_tensor.shape)  # e.g. [3,224,224]

    # Now DataLoader can batch them
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    for batch in loader:
        # batch is now a Tensor of shape [4,3,224,224]
        print("Batch tensor shape:", batch.shape)
        break

