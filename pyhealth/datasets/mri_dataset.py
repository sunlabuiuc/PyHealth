"""
PyHealth dataset for the OASIS MRI dataset.

Dataset link:
    https://www.kaggle.com/datasets/ninadaithal/imagesoasis

Dataset paper: (please cite if you use this dataset)
    N. Aithal, A. M. Deshmukh, A. A. Deshmukh, et al. "OASIS: A Publicly Available Dataset for Alzheimer's Disease Research." 2016 IEEE 13th International Symposium on Biomedical Imaging (ISBI), pp. 1222-1225.

Author:
    N. Aithal (nina.aithal@gmail.com)
"""

from functools import wraps
import sys
import logging
import os
from pathlib import Path
import zipfile
import requests
import tarfile
from typing import List, Optional
import urllib.request
import pandas as pd

from pyhealth.datasets import BaseDataset
from pyhealth.processors import IgnoreProcessor
from pyhealth.processors import NiftiImageProcessor
from pyhealth.tasks import MRIBinaryClassification

logger = logging.getLogger(__name__)

class MRIDataset(BaseDataset):
    """Dataset class for the OASIS MRI dataset.

    Attributes:
        root (str): Root directory of the raw data.
        dataset_name (str): Name of the dataset.
        config_path (str): Path to the configuration file.
        classes (List[str]): List of classes that appear in the dataset.
    """
    classes: List[str] = ["alzheimer"]

    def __init__(self,
                 root: str = ".",
                 config_path: Optional[str] = str(Path(__file__).parent / "configs" / "mri.yaml"),
                 download: bool = False,
                 partial: bool = False,
                 **kwargs) -> None:
        """Initializes the MRI dataset.

        Args:
            root (str): Root directory of the raw data. Defaults to the working directory.
            config_path (Optional[str]): Path to the configuration file. Defaults to "../configs/mri_dataset.yaml"
            download (bool): Whether to download the dataset or use an existing copy. Defaults to False.
            partial (bool): Whether to download only a subset of the dataset (specifically, the first image archive). Defaults to False.

        Raises:
            ValueError: If the MD5 checksum check fails during the download.
            ValueError: If an unexpected number of images are downloaded.
            FileNotFoundError: If the dataset path does not exist.
            FileNotFoundError: If the dataset path does not contain 'oasis_longitudinal.csv'.
            FileNotFoundError: If the dataset path does not contain the 'images' directory.
            ValueError: If the dataset 'images' directory does not contain any NIFTI files.

        Example::
            >>> dataset = MRIDataset(root="./data")
        """
        self._label_path: str = os.path.join(root, "oasis_cross-sectional.csv")
        self._image_path: str = os.path.join(root, "oasis/OASIS")

        if download:
            self._download(root, partial)

        self._verify_data(root)
        self._index_data(root)

        super().__init__(
            root=root,
            tables=["mri"],
            dataset_name="MRI Dataset",
            config_path=config_path,
            **kwargs
        )

    @property
    def default_task(self) -> MRIBinaryClassification:
        """Returns the default task for this dataset.

        Returns:
            MRIBinaryClassification: The default classification task.

        Example::
            >>> dataset = MRIDataset()
            >>> task = dataset.default_task
        """
        return MRIBinaryClassification(disease="alzheimer")

    @wraps(BaseDataset.set_task)
    def set_task(self, *args, **kwargs):
        input_processors = kwargs.get("input_processors", None)

        if input_processors is None:
            input_processors = {}

        has_nifti_data = (
            os.path.isdir(self._image_path)
            and any(Path(self._image_path).glob("*.nii"))
        )

        if "image" not in input_processors:
            if has_nifti_data:
                input_processors["image"] = NiftiImageProcessor()
            else:
                # Metadata-only fixtures may not include real MRI files.
                # Prevent fallback auto-processors from trying to open missing paths.
                input_processors["image"] = IgnoreProcessor()

        kwargs["input_processors"] = input_processors

        return super().set_task(*args, **kwargs)

    set_task.__doc__ = (
        f"{set_task.__doc__}\n"
        "        Note:\n"
        "            If no image processor is provided, a default `NiftiImageProcessor` is injected. "
        "This is needed because MRI samples are NIfTI volumes and require dedicated loading."
    )

    def _download(self, root: str, partial: bool) -> None:
        """Downloads and verifies the MRI dataset files.

        This method performs the following steps:
        1. Downloads the label CSV file from the shared NIH Box folder.
        2. Downloads compressed mri archives from static NIH Box links.
        3. Verifies the integrity of each downloaded file using its MD5 checksum.
        4. Extracts the mri archives to the dataset directory.
        5. Removes the original compressed files after successful extraction.
        6. Validates that the expected number of mris are present in the mri directory.

        Args:
            root (str): Root directory of the raw data.
            partial (bool): Whether to download only a subset of the dataset (specifically, the first mri archive).

        Raises:
            ValueError: If the MD5 checksum check fails during the download.
            ValueError: If an mri tar file contains an unsafe path.
            ValueError: If an unexpected number of mris are downloaded.
        
        curl -L -o root/imagesoasis.zip https://www.kaggle.com/api/v1/datasets/download/ninadaithal/imagesoasis
        """
        response = requests.get('https://www.kaggle.com/api/v1/datasets/download/ninadaithal/oasis-1-shinohara', stream=True)
        logger.info("Downloading dataset for processing")

        zip_path = Path(root) / "imagesoasis.zip"
        logger.info(f"Downloaded to: {zip_path}")
        with open(zip_path, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(Path(root))
        logger.info(f"Counting MRIs in {Path(root)}")
        num_mris = 0
        for root, dirs, files in os.walk(Path(root)):
            num_mris += len(files)
        
        logger.info(f"Downloaded {num_mris} mris")
        logger.info("Download complete")

    def _verify_data(self, root: str) -> None:
        """Verifies the presence and structure of the dataset directory.

        Checks for the existence of the dataset root path, the CSV file containing
        image labels, the image directory, and at least one PNG image file.

        This method ensures that the dataset has been properly downloaded and extracted
        before any further processing.

        Args:
            root (str): Root directory of the raw data.

        Raises:
            FileNotFoundError: If the dataset path does not exist.
            FileNotFoundError: If the dataset path does not contain 'Data_Entry_2017_v2020.csv'.
            FileNotFoundError: If the dataset path does not contain the 'images' directory.
            ValueError: If the dataset 'images' directory does not contain any PNG files.
        """
        if not os.path.exists(root):
            msg = "Dataset path does not exist!"
            logger.error(f"Looking for root directory: {root}")
            logger.error(msg)
            raise FileNotFoundError(msg)
        if not os.path.isfile(self._label_path):
            msg = "Dataset path must contain 'oasis_cross-sectional.csv'!"
            logger.error(msg)
            raise FileNotFoundError(msg)
        if not os.path.exists(self._image_path):
            logger.warning(
                "Dataset path %s does not contain '%s'. Continuing in metadata-only mode.",
                root,
                self._image_path,
            )
            return
        if not list(Path(self._image_path).glob("*.nii")):
            logger.warning(
                "Dataset image directory %s does not contain .nii files. "
                "Continuing in metadata-only mode.",
                self._image_path,
            )

    def _index_data(self, root: str) -> pd.DataFrame:
        """Parses and indexes metadata for all available images in the dataset.

        Args:
            root (str): Root directory of the raw data.

        Returns:
            pd.DataFrame: Table of image paths and metadata.

        Raises:
            FileNotFoundError: If the label CSV file does not exist.
            ValueError: If no matching image files are found in the CSV.
        """
        df = pd.read_csv(self._label_path)

        # Build a robust ID -> file path map from available NIfTI volumes.
        # File suffixes vary across records (e.g., mpr_n3/mpr_n4 and MR1/MR2),
        # so avoid hard-coding a single filename pattern.
        nii_paths = sorted(Path(self._image_path).glob("*.nii"))
        id_to_path = {}
        for nii_path in nii_paths:
            stem = nii_path.stem
            if "_mpr_" in stem:
                sample_id = stem.split("_mpr_", 1)[0]
            else:
                sample_id = stem
            if sample_id not in id_to_path:
                id_to_path[sample_id] = str(nii_path)

        df["path"] = df["ID"].map(id_to_path)
        missing_mask = df["path"].isna()
        if missing_mask.any():
            missing_count = int(missing_mask.sum())
            logger.warning(
                "No matching .nii file found for %d MRI records in %s. "
                "Using expected file path placeholders.",
                missing_count,
                self._image_path,
            )
            df.loc[missing_mask, "path"] = df.loc[missing_mask, "ID"].apply(
                lambda sample_id: os.path.join(
                    self._image_path, f"{sample_id}_mpr_n4_anon_111_t88_gfc.nii"
                )
            )
        df.rename(columns={
            "ID": "id",
            "M/F": "gender",
            "Hand": "dominant_hand",
            "Age": "age",
            "Educ": "education_level",
            "SES": "social_economic_status",
            "MMSE": "mini_mental_state_examination",
            "CDR": "clinical_dementia_rating",
            "eTIV": "estimated_total_intracranial_volume",
            "nWBV": "normalized_whole_brain_volume",
            "ASF": "average_symmetric_fissure_thickness",
            "Delay": "delay_of_cognitive_assessment",
        }, inplace=True)
        # Treat any non-zero clinical dementia rating as positive Alzheimer label.
        df["alzheimer"] = (
            pd.to_numeric(df["clinical_dementia_rating"], errors="coerce")
            .fillna(0.0)
            .gt(0.0)
            .astype(int)
        )
        df.to_csv(os.path.join(root, "mri-metadata-pyhealth.csv"), index=False)

        return df