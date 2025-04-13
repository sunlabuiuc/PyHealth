import os
import subprocess
import zipfile
import logging
from typing import Callable, Dict, Optional

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

# Import BaseDataset from the PyHealth package.
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
    """NIH Chest X-ray Dataset.

    This dataset class supports downloading, extracting, and loading
    the NIH Chest X-ray images for either the "training" or "test" splits.
    Each sample is represented as a dictionary containing the image path.

    Example:
        >>> from pyhealth.datasets.nih_cxr import NIHChestXrayDataset
        >>> from torchvision import transforms
        >>> my_transforms = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor()
        ... ])
        >>> dataset = NIHChestXrayDataset(
        ...     root="/tmp/nih_chestxray",
        ...     split="training",
        ...     transform=my_transforms,
        ...     download=True
        ... )
        >>> dataset.stat()
        >>> sample_image = dataset[0]
    """

    def __init__(
        self,
        root: str,
        split: str = "training",
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        """Initialize the NIH Chest X-ray Dataset.

        Args:
            root (str): Base directory for storing the dataset files.
            split (str): The dataset split to load, must be either "training" or "test".
            transform (Callable, optional): A function/transform applied to images.
            download (bool, optional): Whether to download and extract the dataset
                if it is not present locally.

        Raises:
            ValueError: If split is not "training" or "test".
        """
        if split not in ("training", "test"):
            raise ValueError("split must be either 'training' or 'test'.")
        self.split = split
        self.transform = transform
        self.dataset_name = "NIH Chest X-ray"
        self.dataset_url = (
            "https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data"
        )
        self.dataset_zip = os.path.join(root, DATASET_ZIP_FILENAME)
        self.extracted_folder = os.path.join(root, "database")
        self.root = root

        if download:
            self._download()
            self._prepare()

        # Populate the patients dict by processing the dataset.
        # Each sample is stored as a dict with key "image_path".
        self.patients = self.process()

    def process(self) -> Dict[int, Dict]:
        """Process the dataset by scanning the file system and building a sample map.

        This method will:
          1. Check the appropriate subdirectory based on the split.
          2. List all JPEG image files.
          3. Build a dictionary mapping sample IDs (int) to a dict with the image
             path.

        Returns:
            Dict[int, Dict]: A dictionary where the key is the sample ID and the value
                is a dict containing the key "image_path" (str).

        Example Use:
            Called automatically during initialization via BaseDataset.
        """
        file_paths = self._load_file_paths()
        patients = {idx: {"image_path": path} for idx, path in enumerate(file_paths)}
        return patients

    def _download(self) -> None:
        """Download the dataset archive if it does not exist locally.

        Uses a wget command to download the dataset zip file.

        Returns:
            None

        Example Use:
            Called during initialization if 'download' is True.
        """
        os.makedirs(self.root, exist_ok=True)
        if os.path.exists(self.dataset_zip):
            logger.info("Dataset zip file already exists; skipping download.")
            return

        logger.info("Downloading dataset...")
        command = f"wget -nc -O {self.dataset_zip} {self.dataset_url}"
        try:
            subprocess.run(command, check=True, shell=True)
            logger.info("Download completed successfully.")
        except subprocess.CalledProcessError as err:
            logger.error("Download failed.")
            raise err

    def _prepare(self) -> None:
        """Extract the dataset archive if not already extracted.

        Returns:
            None

        Example Use:
            Called during initialization after downloading.
        """
        if os.path.exists(self.extracted_folder):
            logger.info("Dataset already extracted; skipping extraction.")
            return

        logger.info("Extracting dataset files...")
        with zipfile.ZipFile(self.dataset_zip, "r") as zip_ref:
            zip_ref.extractall(self.root)
        logger.info("Extraction completed.")

    def _load_file_paths(self) -> list:
        """Retrieve the list of image file paths for the selected split.

        Assumes that the extracted folder contains a "database" directory with
        "train" and "test" subdirectories.

        Returns:
            list: A sorted list of file paths (str) for JPEG images.

        Raises:
            FileNotFoundError: If the expected subdirectory is missing.
            RuntimeError: If no JPEG images are found in the directory.

        Example Use:
            Called by process() to generate the mapping of samples.
        """
        subfolder = "train" if self.split == "training" else "test"
        base_dir = os.path.join(self.extracted_folder, subfolder)
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Expected folder {base_dir} does not exist.")

        image_files = sorted(
            [
                os.path.join(base_dir, fname)
                for fname in os.listdir(base_dir)
                if fname.lower().endswith(".jpg")
            ]
        )
        if not image_files:
            raise RuntimeError(f"No image files found in {base_dir}.")
        return image_files

    def stat(self) -> str:
        """Print and return statistics about the dataset.

        Statistics include the dataset name, split, and number of samples.

        Returns:
            str: A formatted string containing dataset statistics.

        Example Use:
            >>> stats_str = dataset.stat()
            >>> print(stats_str)
        """
        stats_lines = [
            "",
            "Statistics of Dataset:",
            f"\t- Dataset Name: {self.dataset_name}",
            f"\t- Split: {self.split}",
            f"\t- Number of samples: {len(self.patients)}",
            "",
        ]
        stats_str = "\n".join(stats_lines)
        logger.info(stats_str)
        print(stats_str)
        return stats_str

    @staticmethod
    def info() -> None:
        """Print information about the NIH Chest X-ray Dataset.

        Returns:
            None

        Example Use:
            >>> NIHChestXrayDataset.info()
        """
        print(INFO_MSG)

    def __getitem__(self, index: int) -> Image.Image:
        """Retrieve a single sample from the dataset.

        Loads the image at the given index, applies the optional transform,
        and returns the processed image.

        Args:
            index (int): The sample index.

        Returns:
            Image.Image: The loaded (and transformed, if applicable) image.

        Raises:
            IndexError: If index is out of the range of available samples.
            Exception: If an error occurs while loading the image.

        Example Use:
            >>> image = dataset[0]
        """
        if index < 0 or index >= len(self.patients):
            raise IndexError("Index out of range.")

        image_path = self.patients[index]["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as err:
            logger.error(f"Error loading image {image_path}: {err}")
            raise err

        if self.transform:
            image = self.transform(image)

        return image


if __name__ == "__main__":
    # Example test case to verify dataset integration.
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    # Initialize the dataset with the training split and download enabled.
    dataset = NIHChestXrayDataset(
        root="/tmp/nih_chestxray",
        split="training",
        transform=my_transforms,
        download=True,
    )

    # Print dataset info and statistics.
    NIHChestXrayDataset.info()
    dataset.stat()

    # Retrieve a sample image.
    sample_image = dataset[0]
    logger.info(f"Loaded sample image with size: {sample_image.size}")

    # Create a DataLoader for iterating over the dataset.
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for batch in data_loader:
        logger.info(f"Processed a batch with shape: {batch.shape}")
        break
