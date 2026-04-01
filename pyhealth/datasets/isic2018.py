"""
PyHealth dataset for the ISIC 2018 Skin Lesion Classification dataset (Task 3).

Dataset link:
    https://challenge.isic-archive.com/data/#2018

License:
    CC-BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)

Dataset paper: (please cite if you use this dataset)
    [1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, et al. "Skin Lesion
    Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the
    International Skin Imaging Collaboration (ISIC)", 2018;
    https://arxiv.org/abs/1902.03368

    [2] Tschandl, P., Rosendahl, C. & Kittler, H. "The HAM10000 dataset, a large
    collection of multi-source dermatoscopic images of common pigmented skin
    lesions." Sci. Data 5, 180161 (2018).

Dataset paper link:
    https://doi.org/10.1038/sdata.2018.161

Author:
    Fan Zhang (fanz6@illinois.edu)
"""

import logging
import os
import zipfile
from functools import wraps
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

from pyhealth.datasets import BaseDataset
from pyhealth.processors import ImageProcessor
from pyhealth.tasks import ISIC2018Classification

logger = logging.getLogger(__name__)

# Official ISIC 2018 Task 3 download URLs
_IMAGES_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2018/"
    "ISIC2018_Task3_Training_Input.zip"
)
_LABELS_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2018/"
    "ISIC2018_Task3_Training_GroundTruth.zip"
)

_GROUNDTRUTH_CSV = "ISIC2018_Task3_Training_GroundTruth.csv"
_IMAGES_DIR = "ISIC2018_Task3_Training_Input"


class ISIC2018Dataset(BaseDataset):
    """Dataset class for the ISIC 2018 Skin Lesion Classification challenge (Task 3).

    The dataset contains 10,015 dermoscopy images across seven diagnostic
    categories of pigmented skin lesions.

    Attributes:
        root (str): Root directory of the raw data.
        dataset_name (str): Name of the dataset.
        config_path (str): Path to the configuration file.
        classes (List[str]): List of skin lesion class labels.

    The expected directory structure under ``root`` is::

        <root>/
            ISIC2018_Task3_Training_GroundTruth.csv
            ISIC2018_Task3_Training_Input/
                ISIC_0024306.jpg
                ISIC_0024307.jpg
                ...
    """

    classes: List[str] = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]

    def __init__(
        self,
        root: str = ".",
        config_path: Optional[str] = str(
            Path(__file__).parent / "configs" / "isic2018.yaml"
        ),
        download: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the ISIC 2018 dataset.

        Args:
            root (str): Root directory of the raw data. Defaults to the
                working directory.
            config_path (Optional[str]): Path to the configuration file.
                Defaults to the bundled ``configs/isic2018.yaml``.
            download (bool): Whether to download the dataset. Defaults to
                False.

        Raises:
            FileNotFoundError: If the dataset root path does not exist.
            FileNotFoundError: If the ground-truth CSV is not found under
                ``root``.
            FileNotFoundError: If the images directory is not found under
                ``root``.
            ValueError: If no JPEG images are found in the images directory.

        Example::
            >>> dataset = ISIC2018Dataset(root="./data/isic2018")
        """
        self._label_path: str = os.path.join(root, _GROUNDTRUTH_CSV)
        self._image_path: str = os.path.join(root, _IMAGES_DIR)

        if download:
            self._download(root)

        self._verify_data(root)
        self._index_data(root)

        super().__init__(
            root=root,
            tables=["isic2018"],
            dataset_name="ISIC2018",
            config_path=config_path,
            **kwargs,
        )

    @property
    def default_task(self) -> ISIC2018Classification:
        """Returns the default task for this dataset.

        Returns:
            ISIC2018Classification: The default multiclass classification task.

        Example::
            >>> dataset = ISIC2018Dataset()
            >>> task = dataset.default_task
        """
        return ISIC2018Classification()

    @wraps(BaseDataset.set_task)
    def set_task(self, *args, **kwargs):
        input_processors = kwargs.get("input_processors", None)

        if input_processors is None:
            input_processors = {}

        if "image" not in input_processors:
            input_processors["image"] = ImageProcessor(mode="RGB")

        kwargs["input_processors"] = input_processors

        return super().set_task(*args, **kwargs)

    set_task.__doc__ = (
        f"{set_task.__doc__}\n"
        "        Note:\n"
        "            If no image processor is provided, a default RGB "
        "`ImageProcessor(mode='RGB')` is injected for dermoscopy images."
    )

    def _download(self, root: str) -> None:
        """Downloads the ISIC 2018 Task 3 images and ground-truth labels.

        Downloads and extracts:
        1. The ground-truth CSV from the ISIC challenge S3 bucket.
        2. The training image archive from the ISIC challenge S3 bucket.

        Args:
            root (str): Root directory where files will be saved.

        Raises:
            requests.HTTPError: If a download request fails.
        """
        os.makedirs(root, exist_ok=True)

        labels_zip = os.path.join(root, "ISIC2018_Task3_Training_GroundTruth.zip")
        images_zip = os.path.join(root, "ISIC2018_Task3_Training_Input.zip")

        logger.info("Downloading ISIC 2018 ground-truth labels...")
        _download_file(_LABELS_URL, labels_zip)
        logger.info("Extracting ground-truth labels...")
        _extract_zip(labels_zip, root)
        os.remove(labels_zip)

        logger.info("Downloading ISIC 2018 training images (this may take a while)...")
        _download_file(_IMAGES_URL, images_zip)
        logger.info("Extracting training images...")
        _extract_zip(images_zip, root)
        os.remove(images_zip)

        logger.info("Download complete.")

    def _verify_data(self, root: str) -> None:
        """Verifies the presence and structure of the dataset directory.

        Args:
            root (str): Root directory of the raw data.

        Raises:
            FileNotFoundError: If the dataset root path does not exist.
            FileNotFoundError: If the ground-truth CSV is missing.
            FileNotFoundError: If the images directory is missing.
            ValueError: If no JPEG images are found in the images directory.
        """
        if not os.path.exists(root):
            msg = "Dataset path does not exist!"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not os.path.isfile(self._label_path):
            msg = f"Dataset path must contain '{_GROUNDTRUTH_CSV}'!"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not os.path.isdir(self._image_path):
            msg = f"Dataset path must contain a '{_IMAGES_DIR}' directory!"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not list(Path(self._image_path).glob("*.jpg")):
            msg = f"'{_IMAGES_DIR}' directory must contain JPEG images!"
            logger.error(msg)
            raise ValueError(msg)

    def _index_data(self, root: str) -> pd.DataFrame:
        """Parses and indexes metadata for all available images in the dataset.

        Reads the ground-truth CSV, filters to images present on disk,
        normalises column names to lowercase, and writes a consolidated
        metadata CSV for the PyHealth config to consume.

        Args:
            root (str): Root directory of the raw data.

        Returns:
            pd.DataFrame: Table of image paths and per-class binary labels.
        """
        df = pd.read_csv(self._label_path)

        # Keep only rows whose image file is present on disk
        image_names = {f.stem for f in Path(self._image_path).glob("*.jpg")}
        df = df[df["image"].isin(image_names)].copy()

        # Normalise class column names to lowercase
        class_rename = {c.upper(): c for c in self.classes}
        # The CSV uses the original casing: MEL, NV, BCC, AKIEC, BKL, DF, VASC
        df.rename(columns=class_rename, inplace=True)

        df.rename(columns={"image": "image_id"}, inplace=True)

        # Use image_id as the patient identifier (ISIC images are independent)
        df["patient_id"] = df["image_id"]

        df["path"] = df["image_id"].apply(
            lambda img_id: os.path.join(self._image_path, f"{img_id}.jpg")
        )

        df.to_csv(os.path.join(root, "isic2018-metadata-pyhealth.csv"), index=False)

        return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: str) -> None:
    """Streams a file from *url* to *dest* with a progress log."""
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    logger.info(f"  {pct:.1f}% ({downloaded}/{total} bytes)")


def _extract_zip(zip_path: str, dest_dir: str) -> None:
    """Safely extracts a zip archive to *dest_dir*."""
    abs_dest = os.path.abspath(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = os.path.abspath(os.path.join(abs_dest, member.filename))
            if not member_path.startswith(abs_dest + os.sep):
                raise ValueError(f"Unsafe path in zip archive: '{member.filename}'!")
        zf.extractall(dest_dir)
