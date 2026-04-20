"""
Chest X-Ray Lung Segmentation Dataset implementation.

Assumes that masks and images are storded in separate
folders, where pairs have the same filename.
"""

import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.datasets.utils import MODULE_CACHE_PATH
from pyhealth.tasks import CXRSegmentationTask

logger = logging.getLogger(__name__)


class CXRSegmentationDataset(BaseDataset):
    """
    Base image dataset for CXR Lung Segmentation.

    One such dataset is available at:
    https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels

    Args:
        root (str, optional): Root directory of the raw data containing the dataset files.
            If not provided, defaults to the 'cxr_segmentation' folder in the pyhealth cache.
            Expected structure:
                - <root>/CXR_png/
                - <root>/masks/
            The masks subdir is expected to contain a binary mask of the same
            spatial size for each example in the CXR_png subdir.
        dataset_name (str | None, optional): Name of the dataset.
            Defaults to "cxr_segmentation".
        config_path (str | Path | None, optional): Path to the
            configuration file. If not provided, uses the default config
            in the configs directory. Defaults to None.
        cache_dir (str | Path | None, optional): Directory for caching
            processed data. Defaults to None.
        num_workers (int | None, optional): Number of parallel workers for
            data processing. Defaults to the number of CPUs on the system.
        dev (bool, optional): If True, only loads a small subset of data for
            development/testing. Defaults to False.
        download (bool, optional): Whether to download the dataset if it is not
            found in the root directory. Defaults to False.

    Examples:
        >>> dataset = CXRSegmentationDataset(
        ...     root="/path/to/cxr_segmentation/dataset",
        ...     dev=False,
        ...     download=True,
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str | None = None,
        dataset_name: str | None = None,
        config_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        num_workers: int | None = None,
        dev: bool = False,
        download: bool = False,
    ) -> None:
        if root is None:
            root = os.path.join(MODULE_CACHE_PATH, "cxr_segmentation")

        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "cxr_segmentation.yaml"

        num_workers = num_workers or os.cpu_count() or 1

        if download:
            self._download(root)

        self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=["cxr_segmentation"],
            dataset_name=dataset_name or "cxr_segmentation",
            config_path=str(config_path),
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @property
    def default_task(self) -> CXRSegmentationTask:
        return CXRSegmentationTask()

    def _download(self, root: str) -> None:
        """
        Downloads and extracts the CXR Lung Segmentation dataset from:
        https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
        """
        if os.path.exists(os.path.join(root, "CXR_png")) and os.path.exists(os.path.join(root, "masks")):
            logger.info(f"Dataset already exists at {root}. Skipping download.")
            return

        os.makedirs(root, exist_ok=True)

        url = "https://www.kaggle.com/api/v1/datasets/download/nikhilpandey360/chest-xray-masks-and-labels"
        zip_path = os.path.join(root, "chest-xray-masks-and-labels.zip")

        logger.info(f"Downloading dataset from {url} to {zip_path}...")
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc="Downloading CXR Segmentation Dataset",
        ) as t:

            def reporthook(blocknum, blocksize, totalsize):
                if totalsize > 0:
                    t.total = totalsize
                t.update(blocknum * blocksize - t.n)

            urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)

        logger.info(f"Extracting {zip_path} to {root}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(root)

        # The Kaggle zip contains files nested in 'Lung Segmentation'
        # Move 'CXR_png' and 'masks' to be directly in root for cleanliness
        nested_dir = os.path.join(root, "Lung Segmentation")
        for folder in ["CXR_png", "masks"]:
            src = os.path.join(nested_dir, folder)
            dst = os.path.join(root, folder)
            logger.info(f"Moving {folder} from {nested_dir} to {root}...")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)

        # Clean up the nested dir
        logger.info(f"Removing nested directory {nested_dir}...")
        shutil.rmtree(nested_dir)
        # Remove the "data" dir, which is a duplicate
        duplicate_data = os.path.join(root, "data")
        logger.info(f"Removing duplicate data at  {duplicate_data}...")
        shutil.rmtree(duplicate_data)
        logger.info("Cleaning up zip file...")
        os.remove(zip_path)

        logger.info("Download and extraction complete.")

    def prepare_metadata(self, root: str) -> None:
        """
        Scan directories and create metadata CSV for CXR_png.

        This writes:
        - <root>/cxr_segmentation-pyhealth.csv
        """
        root = os.path.abspath(root)

        # Directory to scan for images
        image_dir = os.path.join(root, "CXR_png")
        mask_dir = os.path.join(root, "masks")
        metadata_file = os.path.join(root, "cxr_segmentation-pyhealth.csv")

        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Required directory 'masks' not found in {root}")

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Required directory 'CXR_png' not found in {root}")

        # If the file already exists, we skip scanning
        if os.path.exists(metadata_file):
            return

        logger.info(f"Scanning directory {image_dir} for dataset...")
        data = []
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        for f in image_files:
            patient_id = f.replace(".png", "")
            image_path = os.path.join(image_dir, f)

            # Check for mask with and without _mask suffix
            mask_options = [f"{patient_id}_mask.png", f"{patient_id}.png"]
            mask_abs_path = None
            for opt in mask_options:
                candidate = os.path.join(mask_dir, opt)
                if os.path.exists(candidate):
                    mask_abs_path = candidate
                    break

            if mask_abs_path:
                data.append(
                    {
                        "patient_id": patient_id,
                        "image_path": image_path,
                        "mask_path": mask_abs_path,
                    }
                )

        if len(data) > 0:
            df = pd.DataFrame(data)
            df.to_csv(metadata_file, index=False)
            logger.info(f"Generated metadata with {len(df)} samples: {metadata_file}")
        else:
            logger.warning(f"No valid samples found in {image_dir}.")
