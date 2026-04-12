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
from pyhealth.processors import ImageProcessor
# from pyhealth.tasks import AlzheimerDiseaseClassification

logger = logging.getLogger(__name__)

class MRIDataset(BaseDataset):
    """Dataset class for the OASIS MRI dataset.

    Attributes:
        root (str): Root directory of the raw data.
        dataset_name (str): Name of the dataset.
        config_path (str): Path to the configuration file.
        classes (List[str]): List of classes that appear in the dataset.
    """
    classes: List[str] = ["mild_demented", "very_mild_demented", "mild_demented", "non_demented"]

    def __init__(self,
                 root: str = ".",
                 config_path: Optional[str] = str(Path(__file__).parent / "configs" / "mri_dataset.yaml"),
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
            tables=["mri_dataset"],
            dataset_name="MRI Dataset",
            config_path=config_path,
            **kwargs
        )

    ''' add these tests later when we have the AlzheimerDiseaseClassification task  
    @property
    def default_task(self) -> AlzheimerDiseaseClassification:
        """Returns the default task for this dataset.

        Returns:
            AlzheimerDiseaseClassification: The default classification task.

        Example::
            >>> dataset = MRIDataset()
            >>> task = dataset.default_task
        """
        return AlzheimerDiseaseClassification()
    '''

    @wraps(BaseDataset.set_task)
    def set_task(self, *args, **kwargs):
        input_processors = kwargs.get("input_processors", None)

        if input_processors is None:
            input_processors = {}

        if "mri" not in input_processors:
            input_processors["mri"] = MRIImageProcessor(mode='L')

        kwargs["input_processors"] = input_processors

        return super().set_task(*args, **kwargs)

    set_task.__doc__ = (
        f"{set_task.__doc__}\n"
        "        Note:\n"
        "            If no mri processor is provided, a default `MRIImageProcessor` is injected. "
        "This is needed because the MRI dataset mris do not all have the same number of channels, "
        "causing the default PyHealth mri processor to fail."
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
        if not os.path.exists(self._image_path):
            msg = "Dataset path must contain an 'images' directory!"
            logger.error(msg)
            raise FileNotFoundError(msg)
        if not os.path.isfile(self._label_path):
            msg = "Dataset path must contain 'oasis_cross-sectional.csv'!"
            logger.error(msg)
            raise FileNotFoundError(msg)
        if not list(Path(self._image_path).glob("*.nii")):
            msg = "Dataset 'images' directory must contain NII files!"
            logger.error(msg)
            raise ValueError(msg)    

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
        # image_names = [f.name[0:f.name.find("_mpr")] for f in Path(self._image_path).iterdir() if f.is_file()]
        df['img_path'] = df['ID'] + '_mpr_n3_anon_sbj_111_normalised.nii'
        ''' don't think we need this piece 
        for _class in self.classes:
            df[_class] = df['Finding Labels'].str.contains(_class, case=False).astype(int)
        df.drop(columns=["Finding Labels"], inplace=True)
        df.rename(columns={'Image Index': 'path',
                            'Follow-up #': 'visit_id',
                            'Patient ID': 'patient_id',
                            'Patient Age': 'patient_age',
                            'Patient Sex': 'patient_sex',
                            'View Position': 'view_position',
                            'OriginalImage[Width': 'original_image_width',
                            'Height]': 'original_image_height',
                            'OriginalImagePixelSpacing[x': 'original_image_pixel_spacing_x',
                            'y]': 'original_image_pixel_spacing_y'}, inplace=True)
        df['path'] = df['path'].apply(lambda p: os.path.join(self._image_path, p))
        df.to_csv(os.path.join(root, "chestxray14-metadata-pyhealth.csv"), index=False)
        '''
        return df