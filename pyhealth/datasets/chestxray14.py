"""
PyHealth dataset for the ChestX-ray14 dataset.

Dataset link:
    https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345

Dataset paper: (please cite if you use this dataset)
    Xiaosong Wang, Yifan Peng, Le Lu, et al. "ChestX-ray8: Hospital-scale Chest
    X-ray Database and Benchmarks on Weakly-Supervised Classification and
    Localization of Common Thorax Diseases." 2017 IEEE Conference on Computer
    Vision and Pattern Recognition (CVPR), pp. 3462-3471.

Dataset paper link:
    https://arxiv.org/abs/1705.02315

Author:
    Eric Schrock (ejs9@illinois.edu)
"""
import argparse
import hashlib
import logging
import os
from pathlib import Path
import requests
import tarfile
from typing import List, Optional
import urllib.request

import pandas as pd

from .base_dataset import BaseDataset
from ..tasks.chestxray14_multilabel_classification import ChestXray14MultilabelClassification

logger = logging.getLogger(__name__)

class ChestXray14Dataset(BaseDataset):
    """Dataset class for the NIH ChestX-ray14 dataset.

    Attributes:
        root (str): Root directory of the raw data.
        dataset_name (str): Name of the dataset.
        config_path (str): Path to the configuration file.
        classes (List[str]): List of diseases that appear in the dataset.

    Methods:
        __len__(): Returns the number of entries in the dataset.
        __getitem__(index: int): Retrieves a specific image and its metadata.
        stats(): Prints statistics about the dataset's content.
    """
    classes: List[str] = ["atelectasis", "cardiomegaly", "consolidation",
               "edema", "effusion", "emphysema",
               "fibrosis", "hernia", "infiltration",
               "mass", "nodule", "pleural_thickening",
               "pneumonia", "pneumothorax"]

    def __init__(self,
                 root: str = ".",
                 config_path: Optional[str] = str(Path(__file__).parent / "configs" / "chestxray14.yaml"),
                 download: bool = True,
                 partial: bool = False) -> None:
        """Initializes the ChestX-ray14 dataset.

        Args:
            root (str): Root directory of the raw data. Defaults to the working directory.
            config_path (Optional[str]): Path to the configuration file. Defaults to "../configs/chestxray14.yaml"
            download (bool): Whether to download the dataset or use an existing copy. Defaults to True.
            partial (bool): Whether to download only a subset of the dataset. Defaults to False.

        Raises:
            ValueError: If the MD5 checksum check fails during the download.
            ValueError: If an unexpected number of images are downloaded.
            FileNotFoundError: If the dataset path does not exist.
            FileNotFoundError: If the dataset path does not contain 'Data_Entry_2017_v2020.csv'.
            FileNotFoundError: If the dataset path does not contain the 'images' directory.
            ValueError: If the dataset 'images' directory does not contain any PNG files.

        Example:
            >>> dataset = ChestXray14Dataset(root="./data")
        """
        self._partial = partial

        self._label_path: str = os.path.join(root, "Data_Entry_2017_v2020.csv")
        self._image_path: str = os.path.join(root, "images")

        if download:
            self._download(root)

        self._verify_data(root)
        self._data: pd.DataFrame = self._index_data(root)

        super().__init__(
            root=root,
            tables=["chestxray14"],
            dataset_name="ChestX-ray14",
            config_path=config_path,
        )

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.

        Example:
            >>> dataset = ChestXray14Dataset()
            >>> print(len(dataset))
        """
        return len(self._data)

    def __getitem__(self, index: int) -> pd.Series:
        """Retrieves a single sample from the dataset at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            pd.Series: Requested row of the data table.

        Raises:
            IndexError: If the index is out of bounds.

        Example:
            >>> dataset = ChestXray14Dataset()
            >>> print(dataset[0])
        """
        return self._data.iloc[index]

    def stats(self) -> None:
        """Prints information on the contents of the dataset

        Example:
            >>> dataset = ChestXray14Dataset()
            >>> dataset.stats()
        """
        lines = list()
        lines.append("")
        lines.append(f"Statistics (partial={self._partial}):")
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Number of images: {self.__len__()}")
        lines.append(f"\t- Average number of findings per image: {self._data[self.classes].sum().sum() / self.__len__():.2f}")
        lines.append(f"\t- Max number of findings in an image: {self._data[self.classes].sum(axis=1).max()}")
        num_no_finding = (self._data[self.classes].sum(axis=1) == 0).sum()
        lines.append(f"\t- Number with no finding: {num_no_finding} ({(num_no_finding / self.__len__()) * 100:.1f}%)")
        for _class in self.classes:
            num_finding = self._data[_class].sum()
            lines.append(f"\t- Number with {_class}: {num_finding} ({(num_finding / self.__len__()) * 100:.1f}%)")
        lines.append("")
        print("\n".join(lines))

    @property
    def default_task(self) -> ChestXray14MultilabelClassification:
        """Returns the default task for this dataset.

        Returns:
            ChestXray14MultilabelClassification: The default classification task.

        Example:
            >>> dataset = ChestXray14Dataset()
            >>> task = dataset.default_task
        """
        return ChestXray14MultilabelClassification()

    def _download(self, root: str) -> None:
        """Downloads and verifies the ChestX-ray14 dataset files.

        This method performs the following steps:
        1. Downloads the label CSV file from a Google Drive mirror.
        2. Downloads compressed image archives from NIH Box links.
        3. Verifies the integrity of each downloaded file using its MD5 checksum.
        4. Extracts the image archives to the dataset directory.
        5. Removes the original compressed files after successful extraction.
        6. Validates that the expected number of images are present in the image directory.

        If `self._partial` is True, only a subset of the dataset is downloaded and verified
        (specifically, the first two image archives).

        Args:
            root (str): Root directory of the raw data.

        Raises:
            ValueError: If the MD5 checksum check fails during the download.
            ValueError: If an image tar file contains an unsafe path.
            ValueError: If an unexpected number of images are downloaded.
        """
        # https://nihcc.app.box.com/v/ChestXray-NIHCC/file/219760887468 (mirrored to Google Drive)
        # I couldn't figure out a way to download this file directly from box.com
        response = requests.get('https://drive.google.com/uc?export=download&id=1mkOZNfYt-Px52b8CJZJANNbM3ULUVO3f')
        with open(self._label_path, "wb") as file:
            file.write(response.content)

        # https://nihcc.app.box.com/v/ChestXray-NIHCC/file/371647823217
        links = [
            'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
            'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
            'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
            'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
            'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
            'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
            'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
            'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
            'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
            'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
            'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
            'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
        ]

        # https://nihcc.app.box.com/v/ChestXray-NIHCC/file/249502714403
        md5_checksums = [
            'fe8ed0a6961412fddcbb3603c11b3698',
            'ab07a2d7cbe6f65ddd97b4ed7bde10bf',
            '2301d03bde4c246388bad3876965d574',
            '9f1b7f5aae01b13f4bc8e2c44a4b8ef6',
            '1861f3cd0ef7734df8104f2b0309023b',
            '456b53a8b351afd92a35bc41444c58c8',
            '1075121ea20a137b87f290d6a4a5965e',
            'b61f34cec3aa69f295fbb593cbd9d443',
            '442a3caa61ae9b64e61c561294d1e183',
            '09ec81c4c31e32858ad8cf965c494b74',
            '499aefc67207a5a97692424cf5dbeed5',
            'dc9fda1757c2de0032b63347a7d2895c'
        ]

        if self._partial:
            links = links[:2]
            md5_checksums = md5_checksums[:2]

        for idx, link in enumerate(links):
            fn = os.path.join(root, f"images_{idx+1:02d}.tar.gz")

            logger.info(f'Downloading {fn}...')
            urllib.request.urlretrieve(link, fn)

            logger.info(f"Checking MD5 checksum for {fn}...")
            with open(fn, 'rb') as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()

            if file_md5 != md5_checksums[idx]:
                msg = "Invalid MD5 checksum!"
                logger.error(msg)
                raise ValueError(msg)

            logger.info(f"Extracting {fn}...")
            with tarfile.open(fn, 'r:gz') as tar:
                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                    return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

                for member in tar.getmembers():
                    member_path = os.path.join(root, member.name)
                    if not is_within_directory(root, member_path):
                        msg = f"Unsafe path detected in tar file: '{member.name}'!"
                        logger.error(msg)
                        raise ValueError(msg)

                tar.extractall(path=root)

            logger.info(f"Deleting {fn}...")
            os.remove(fn)

        num_images = len([f for f in os.listdir(self._image_path) if os.path.isfile(os.path.join(self._image_path, f))])
        num_images_expected = 14999 if self._partial else 112120
        if num_images != num_images_expected:
            msg = f"Expected {num_images_expected} images but found {num_images}!"
            logger.error(msg)
            raise ValueError(msg)

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
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not os.path.isfile(self._label_path):
            msg = "Dataset path must contain 'Data_Entry_2017_v2020.csv'!"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not os.path.exists(self._image_path):
            msg = "Dataset path must contain an 'images' directory!"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not list(Path(self._image_path).glob("*.png")):
            msg = "Dataset 'images' directory must contain PNG files!"
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
        image_names = [f.name for f in Path(self._image_path).iterdir() if f.is_file()]
        df = df[df["Image Index"].isin(image_names)]

        for _class in self.classes:
            df[_class] = df['Finding Labels'].str.contains(_class, case=False).astype(int)

        df.drop(columns=["Finding Labels", "Follow-up #", "Patient ID", "View Position", "OriginalImage[Width", "Height]", "OriginalImagePixelSpacing[x", "y]"], inplace=True)
        df.rename(columns={'Image Index': 'path', 'Patient Age': 'patient_age', 'Patient Sex': 'patient_sex'}, inplace=True)
        df['path'] = df['path'].apply(lambda p: os.path.join(self._image_path, p))
        df.to_csv(os.path.join(root, "chestxray14-metadata-pyhealth.csv"), index=False)

        return df


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()

    dataset = ChestXray14Dataset(config_path=args.config, download=args.download, partial=args.partial)

    dataset.stats()
    print(dataset[0])
