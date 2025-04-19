# pyhealth/datasets/nih_cxr.py

import os
import subprocess
import zipfile
import logging
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from pyhealth.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INFO_MSG = (
    "The NIH Chest X-ray Dataset.\n"
    "After extraction, root must contain:\n"
    "  - train_val_list.txt (one filename per line)\n"
    "  - test_list.txt      (one filename per line)\n"
    "  - images_001/ … images_012/, each with subfolder images/ containing .png files\n\n"
    "Citation:\n"
    "Wang, X. et al. ChestX-ray8: Hospital-scale Chest X-ray Database..."
)

DATASET_ZIP_FILENAME = "nih_chestxray.zip"


class NIHChestXrayDataset(BaseDataset):
    """NIH Chest X‑ray Dataset.

    Supports download, extraction, split‑based indexing, and image loading.
    Each sample dict has key "image_path" pointing to a .png file.

    Example:
        >>> ds = NIHChestXrayDataset(root="/tmp/nih", split="training")
        >>> ds.stat()
        >>> img = ds[0]
    """

    def __init__(
        self,
        root: str,
        split: str = "training",
        transform: Optional[callable] = None,
        download: bool = True,
    ):
        """Initialize the NIH Chest X‑ray Dataset.

        Args:
            root (str): Directory to download/extract into & read from.
            split (str): "training" (uses train_val_list.txt) or "test".
            transform (callable, optional): Applied to loaded PIL images.
            download (bool): If True, download & extract before indexing.

        Raises:
            ValueError: If split is not "training" or "test".
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

        # Build patients dict: sample_id → {"image_path": full_path}
        self.patients = self.process()

    def process(self) -> Dict[int, Dict[str, str]]:
        """Index split file & images, return sample mapping.

        Reads the split list (train_val_list.txt or test_list.txt),
        indexes all images under images_*/images/, and maps each listed
        filename to its full path.

        Returns:
            Dict[int, Dict]: sample_id → {"image_path": str}

        Raises:
            FileNotFoundError: If split file missing or images missing.
        """
        # choose the correct list file
        list_filename = "train_val_list.txt" if self.split == "training" else "test_list.txt"
        list_path = os.path.join(self.root, list_filename)
        if not os.path.isfile(list_path):
            raise FileNotFoundError(f"Missing split file: {list_path}")

        # read all filenames for this split
        with open(list_path, "r") as f:
            filenames = [ln.strip() for ln in f if ln.strip()]

        # build a mapping of filename → full path
        filename_to_path = self._index_all_images()
        missing = [fn for fn in filenames if fn not in filename_to_path]
        if missing:
            raise FileNotFoundError(
                f"Could not locate {len(missing)} files, e.g. {missing[:5]}"
            )

        # assemble file paths in split order
        file_paths: List[str] = [filename_to_path[fn] for fn in filenames]

        # return as patients dict
        return {idx: {"image_path": p} for idx, p in enumerate(file_paths)}

    def _index_all_images(self) -> Dict[str, str]:
        """Scan images_*/images/ subfolders to map filename → full path.

        Returns:
            Dict[str, str]: e.g. {"00000001_000.png": "/tmp/nih/images_001/images/00000001_000.png"}
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
        """Download the dataset zip if not already present."""
        os.makedirs(self.root, exist_ok=True)
        if os.path.exists(self.dataset_zip):
            logger.info("Zip already exists; skipping download.")
            return

        logger.info("Downloading NIH Chest X-ray dataset...")
        cmd = f"wget -nc -O {self.dataset_zip} {self.dataset_url}"
        try:
            subprocess.run(cmd, check=True, shell=True)
            logger.info("Download successful.")
        except subprocess.CalledProcessError as e:
            logger.error("Download failed.")
            raise e

    def _prepare(self) -> None:
        """Extract the zip into root if not already extracted."""
        # assume split lists and images_*/ folders are inside the zip
        # after extraction
        # we check for one known file to decide if extraction is needed
        sample_list = os.path.join(self.root, "train_val_list.txt")
        if os.path.exists(sample_list):
            logger.info("Already extracted; skipping.")
            return

        logger.info("Extracting NIH Chest X-ray zip...")
        with zipfile.ZipFile(self.dataset_zip, "r") as zf:
            zf.extractall(self.root)
        logger.info("Extraction complete.")

    def __len__(self) -> int:
        """Return number of samples in this split."""
        return len(self.patients)

    def __getitem__(self, index: int) -> Image.Image:
        """Load and return the image at sample index.

        Args:
            index (int): sample index

        Returns:
            PIL.Image.Image: loaded (and transformed) image

        Raises:
            IndexError: if index is out of range
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        path = self.patients[index]["image_path"]
        img = Image.open(path).convert("RGB")
        return self.transform(img) if self.transform else img

    def stat(self) -> str:
        """Print and return dataset statistics (name, split, count)."""
        lines = [
            f"Dataset: {self.dataset_name}",
            f"Split:   {self.split}",
            f"Samples: {len(self)}",
        ]
        out = "\n".join(lines)
        print(out)
        return out

    @staticmethod
    def info() -> None:
        """Print citation and expected folder layout."""
        print(INFO_MSG)


if __name__ == "__main__":
    # demo / smoke test
    my_transforms = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    ds = NIHChestXrayDataset(
        root="/tmp/nih_chestxray",
        split="training",
        transform=my_transforms,
        download=True,
    )
    NIHChestXrayDataset.info()
    ds.stat()
    img = ds[0]
    print("Loaded image size:", img.size)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    for batch in loader:
        print("Batch shape:", batch.shape)
        break
