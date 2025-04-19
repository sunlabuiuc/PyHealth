import os
import logging
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from pyhealth.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INFO_MSG = (
    "The NIH Chest X-ray Dataset.\n"
    "Root directory must contain:\n"
    "  - train_val_list.txt  (one image filename per line)\n"
    "  - test_list.txt       (one image filename per line)\n"
    "  - images_001/ … images_012/, each with subfolder images/ containing .png files\n\n"
    "Citation:\n"
    "Wang, X. et al. ChestX-ray8: Hospital-scale Chest X-ray Database..."
)


class NIHChestXrayDataset(BaseDataset):
    """NIH Chest X‑ray Dataset for image classification tasks.

    Example:
        >>> ds = NIHChestXrayDataset(root="/data/nih", split="training")
        >>> ds.stat()
        >>> img = ds[0]    # PIL.Image.Image
    """

    def __init__(
        self,
        root: str,
        split: str = "training",
        transform=None,
    ):
        """Initialize the dataset.

        Args:
            root (str): Path to dataset root (contains .txt splits & images_* folders).
            split (str): Either "training" or "test".  
            transform (callable, optional): transform to apply on loaded PIL image.

        Raises:
            ValueError: If split is not one of {"training", "test"}.
        """
        if split not in ("training", "test"):
            raise ValueError("split must be either 'training' or 'test'")
        self.root = root
        self.split = split
        self.transform = transform
        self.dataset_name = "nih_cxr"
        # Build patients map: id → {"image_path": …}
        self.patients = self.process()

    def process(self) -> Dict[int, Dict[str, str]]:
        """Build the sample dictionary by reading split file and indexing images.

        Returns:
            Dict[int, Dict]: sample_id → {"image_path": full_path}

        Example:
            Called automatically on init to populate self.patients.
        """
        # load split filenames
        list_file = "train_val_list.txt" if self.split == "training" else "test_list.txt"
        split_path = os.path.join(self.root, list_file)
        if not os.path.isfile(split_path):
            raise FileNotFoundError(f"Missing split file: {split_path}")

        with open(split_path, "r") as f:
            filenames = [line.strip() for line in f if line.strip()]

        # index all images once
        filename_to_path = self._index_all_images()

        # map filenames to actual paths
        file_paths: List[str] = []
        missing = []
        for fn in filenames:
            try:
                file_paths.append(filename_to_path[fn])
            except KeyError:
                missing.append(fn)
        if missing:
            raise FileNotFoundError(f"Could not locate {len(missing)} files: {missing[:5]}...")

        # build the patients dict
        return {idx: {"image_path": p} for idx, p in enumerate(file_paths)}

    def _index_all_images(self) -> Dict[str, str]:
        """Scan images_*/images/ and build a map from filename → full path.

        Returns:
            Dict[str, str]: mapping of "00000001_000.png" → "/full/path/.../images_005/images/00000001_000.png"
        """
        mapping: Dict[str, str] = {}
        for subdir in os.listdir(self.root):
            subdir_path = os.path.join(self.root, subdir)
            images_folder = os.path.join(subdir_path, "images")
            if os.path.isdir(images_folder):
                for img_name in os.listdir(images_folder):
                    mapping[img_name] = os.path.join(images_folder, img_name)
        return mapping

    def __len__(self) -> int:
        """Number of samples in this split."""
        return len(self.patients)

    def __getitem__(self, index: int) -> Image.Image:
        """Load and return a single image (after optional transform).

        Args:
            index (int): sample index

        Returns:
            PIL.Image.Image: the loaded (and transformed, if set) image

        Raises:
            IndexError: if index out of bounds
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")
        img_path = self.patients[index]["image_path"]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img) if self.transform else img

    def stat(self) -> str:
        """Print and return basic dataset statistics.

        Returns:
            str: formatted statistics
        """
        stats = [
            f"Dataset: {self.dataset_name}",
            f"Split:   {self.split}",
            f"Samples: {len(self)}",
        ]
        out = "\n".join(stats)
        print(out)
        return out

    @staticmethod
    def info() -> None:
        """Print dataset citation and structure info."""
        print(INFO_MSG)


if __name__ == "__main__":
    # simple smoke test
    ds = NIHChestXrayDataset(root="/tmp/nih_chestxray", split="training")
    NIHChestXrayDataset.info()
    ds.stat()
    img = ds[0]
    print("Loaded image size:", img.size)
