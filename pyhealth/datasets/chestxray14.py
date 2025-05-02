import argparse
import hashlib
import logging
import os
from pathlib import Path
import requests
import tarfile
from typing import List, Literal, Tuple, TypedDict, Union
import urllib.request

import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import Compose

from pyhealth.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

INFO_MSG = """
dataset: index -> Tuple[Image.Image, <ChestXray>]

<ChestXray>
    - image_name: str
    - patient_age: str
    - patient_sex: Literal['M', 'F']
    - labels: <ChestXrayLabels>

    <ChestXrayLabels>
        - atelectasis: bool
        - cardiomegaly: bool
        - consolidation: bool
        - edema: bool
        - effusion: bool
        - emphysema: bool
        - fibrosis: bool
        - hernia: bool
        - infiltration: bool
        - mass: bool
        - nodule: bool
        - pleural_thickening: bool
        - pneumonia: bool
        - pneumothorax: bool
"""

class ChestXrayLabels(TypedDict):
    """Typed dictionary representing disease labels for a chest X-ray.

    Each key corresponds to a medical condition potentially observable in a
    chest radiograph. The value is a boolean indicating the presence (`True`)
    or absence (`False`) of that condition in the associated image.

    Attributes:
        atelectasis (bool): Collapse or closure of lung tissue.
        cardiomegaly (bool): Enlarged heart.
        consolidation (bool): Solidification of lung tissue due to accumulation of fluids.
        edema (bool): Fluid accumulation in lung tissue.
        effusion (bool): Fluid buildup between the layers of tissue lining the lungs.
        emphysema (bool): Air sacs in the lungs are damaged and enlarged.
        fibrosis (bool): Scarring or thickening of lung tissue.
        hernia (bool): Protrusion of an organ through the chest wall.
        infiltration (bool): Diffuse or patchy lung opacity.
        mass (bool): Larger abnormal growth in the lungs.
        nodule (bool): Small abnormal round growth in the lungs.
        pleural_thickening (bool): Thickening of the pleura, the membrane surrounding the lungs.
        pneumonia (bool): Infection causing inflammation in the air sacs of the lungs.
        pneumothorax (bool): Collapsed lung due to air in the pleural space.
    """
    atelectasis: bool
    cardiomegaly: bool
    consolidation: bool
    edema: bool
    effusion: bool
    emphysema: bool
    fibrosis: bool
    hernia: bool
    infiltration: bool
    mass: bool
    nodule: bool
    pleural_thickening: bool
    pneumonia: bool
    pneumothorax: bool

class ChestXray(TypedDict):
    """Typed dictionary representing a single chest X-ray metadata entry.

    This structure encapsulates all relevant information for a single image
    in the ChestX-ray14 dataset, including patient metadata and associated
    disease labels.

    Attributes:
        image_name (str): Filename of the chest X-ray image.
        patient_age (int): Age of the patient at the time of imaging.
        patient_sex (Literal['M', 'F']): Sex of the patient ('M' for male, 'F' for female).
        labels (ChestXrayLabels): Dictionary mapping each disease label to a boolean
            indicating presence (`True`) or absence (`False`) of that finding.
    """
    image_name: str
    patient_age: int
    patient_sex: Literal['M', 'F']
    labels: ChestXrayLabels

class ChestXray14Dataset(BaseDataset):
    """Dataset class for the NIH ChestX-ray14 dataset.

    This class handles downloading, verifying, indexing, and accessing the
    ChestX-ray14 dataset. It provides functionality to load the dataset,
    retrieve individual samples with optional transformations, and display
    dataset statistics and structure.

    Attributes:
        dataset_name (str): Name of the dataset.
        paper_url (str): URL of the original paper introducing the dataset.
        dataset_url (str): URL to download the dataset from the NIH repository.
        root (str): Filesystem path to the dataset root directory.
        download (bool): Whether to download the dataset if not already present.
        partial (bool): Whether to download a smaller subset of the dataset.
        transform (Compose): Transformations applied to each image sample.
        label_path (Path): Path to the CSV file containing image labels.
        image_path (Path): Path to the directory containing image files.
        data (List[ChestXray]): Parsed list of dataset entries, including image
            metadata and associated disease labels.

    Methods:
        __len__(): Returns the number of entries in the dataset.
        __getitem__(index): Retrieves a specific image and its metadata.
        info(): Prints information about the dataset's structure.
        stat(): Prints statistics about the dataset's content.
        _download(): Handles downloading, verifying, and extracting the dataset.
        _verify_data(): Ensures the dataset structure and contents are correct.
        _index_data(): Parses and indexes the dataset into `self.data`.

    Example:
        >>> from pathlib import Path
        >>> from torchvision.transforms import Compose, Resize, ToTensor
        >>> transform = Compose([Resize((224, 224)), ToTensor()])
        >>> dataset = ChestXray14Dataset(root="./data", download=True, transform=transform)
        >>> print(len(dataset))
        >>> image, metadata = dataset[0]
    """
    def __init__(self,
                 dataset_name: str = "ChestX-ray14",
                 paper_url: str = "https://arxiv.org/abs/1705.02315",
                 dataset_url: str = "https://nihcc.app.box.com/v/ChestXray-NIHCC",
                 root: str = "",
                 download: bool = True,
                 partial: bool = False,
                 transform: Compose = None) -> None:
        """Initializes the ChestX-ray14 dataset.

        Args:
            dataset_name (str): Name of the dataset. Defaults to "ChestX-ray14".
            paper_url (str): URL to the dataset's reference paper. Defaults to the original ChestX-ray14 paper.
            dataset_url (str): URL to download the dataset. Defaults to the NIHCC Box link.
            root (str): Local path to store or load the dataset. Defaults to the current directory.
            download (bool): Whether to download the dataset or use an existing copy. Defaults to True.
            partial (bool): Whether to download only a subset of the dataset. Defaults to False.
            transform (Compose): Optional torchvision transform pipeline to apply to the images. Defaults to None.

        Raises:
            ValueError: If the MD5 checksum check fails during the download.
            ValueError: If an unexpected number of images are downloaded.
            FileNotFoundError: If the dataset path does not exist.
            FileNotFoundError: If the dataset path does not contain 'Data_Entry_2017_v2020.csv'.
            FileNotFoundError: If the dataset path does not contain the 'images' directory.
            ValueError: If the dataset 'images' directory does not contain any PNG files.

        Example:
            >>> from pathlib import Path
            >>> from torchvision.transforms import Compose, Resize, ToTensor
            >>> transform = Compose([Resize((224, 224)), ToTensor()])
            >>> dataset = ChestXray14Dataset(root="./data", download=True, transform=transform)
        """
        super().__init__(
            root=root,
            tables=[dataset_name],
            dataset_name=dataset_name,
        )

        self.paper_url = paper_url
        self.dataset_url = dataset_url
        self.download = download
        self.partial = partial
        self.transform = transform

        self.label_path: Path = os.path.join(self.root, "Data_Entry_2017_v2020.csv")
        self.image_path: Path = os.path.join(self.root, "images")

        if self.download:
            self._download()

        self._verify_data()

        self.data: List[ChestXray] = []
        self._index_data()

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.

        Example:
            >>> dataset = ChestXray14Dataset()
            >>> print(len(dataset))
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Union[Image.Image, torch.Tensor], ChestXray]:
        """Retrieves a single sample from the dataset at the specified index.

        Loads the image from disk, applies the transform if `self.transform` is set,
        and returns the image along with its corresponding metadata.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[Union[Image.Image, torch.Tensor], ChestXray]: A tuple containing
            the image (either as a PIL Image or a torch.Tensor, depending on the
            transform) and the associated `ChestXray` metadata entry.

        Raises:
            IndexError: If the index is out of bounds.

        Example:
            >>> dataset = ChestXray14Dataset()
            >>> print(dataset[0])
        """
        image_name = self.data[index]["image_name"]
        image_path = os.path.join(self.image_path, image_name)

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.data[index]

    def info(self) -> None:
        """Prints information on the structure of the dataset

        Example:
            >>> dataset = ChestXray14Dataset()
            >>> dataset.info()
        """
        print(INFO_MSG)

    def stat(self) -> None:
        """Prints information on the contents of the dataset

        Example:
            >>> dataset = ChestXray14Dataset()
            >>> dataset.stat()
        """
        lines = list()
        lines.append("")
        lines.append(f"Statistics (partial={self.partial}):")
        lines.append(f"\t- Dataset: {self.dataset_name}")
        lines.append(f"\t- Paper: {self.paper_url}")
        lines.append(f"\t- Source: {self.dataset_url}")
        lines.append(f"\t- Number of images: {self.__len__()}")
        lines.append(f"\t- Average number of findings per image: {sum([sum(xray['labels'].values()) for xray in self.data]) / self.__len__():.2}")
        lines.append(f"\t- Number with no finding: {sum([not any(xray['labels'].values()) for xray in self.data])}")

        for label in ChestXrayLabels.__annotations__:
            lines.append(f"\t- Number with {label}: {sum([xray['labels'][label] for xray in self.data])}")

        lines.append("")
        print("\n".join(lines))

    def _download(self) -> None:
        """Downloads and verifies the ChestX-ray14 dataset files.

        This method performs the following steps:
        1. Downloads the label CSV file from a Google Drive mirror.
        2. Downloads compressed image archives from NIH Box links.
        3. Verifies the integrity of each downloaded file using its MD5 checksum.
        4. Extracts the image archives to the dataset directory.
        5. Removes the original compressed files after successful extraction.
        6. Validates that the expected number of images are present in the image directory.

        If `self.partial` is True, only a subset of the dataset is downloaded and verified
        (specifically, the first two image archives).

        Raises:
            ValueError: If the MD5 checksum check fails during the download.
            ValueError: If an unexpected number of images are downloaded.
        """
        # https://nihcc.app.box.com/v/ChestXray-NIHCC/file/219760887468 (mirrored to Google Drive)
        # I couldn't figure out a way to download this file directly from box.com
        response = requests.get('https://drive.google.com/uc?export=download&id=1mkOZNfYt-Px52b8CJZJANNbM3ULUVO3f')
        with open(self.label_path, "wb") as file:
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

        if self.partial:
            links = links[:2]
            md5_checksums = md5_checksums[:2]

        for idx, link in enumerate(links):
            fn = self.path.joinpath(f"images_{idx+1:02d}.tar.gz")

            logger.info(f'Downloading {fn}...')
            urllib.request.urlretrieve(link, fn)

            logger.info(f"Checking MD5 checksum for {fn}...")
            with open(fn, 'rb') as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()

            if file_md5 != md5_checksums[idx]:
                msg = "Invalid MD5 checksum"
                logger.error(msg)
                raise ValueError(msg)

            logger.info(f"Extracting {fn}...")
            with tarfile.open(fn, 'r:gz') as tar:
                tar.extractall(path=self.root)

            logger.info(f"Deleting {fn}...")
            os.remove(fn)

        num_images = len([f for f in os.listdir(self.image_path) if os.path.isfile(os.path.join(self.image_path, f))])
        num_images_expected = 14999 if self.partial else 112120
        if num_images != num_images_expected:
            msg = f"Expected {num_images_expected} images but found {num_images}!"
            logger.error(msg)
            raise ValueError(msg)

        logger.info("Download complete")

    def _verify_data(self) -> None:
        """Verifies the presence and structure of the dataset directory.

        Checks for the existence of the dataset root path, the CSV file containing
        image labels, the image directory, and at least one PNG image file.

        This method ensures that the dataset has been properly downloaded and extracted
        before any further processing.

        Raises:
            FileNotFoundError: If the dataset path does not exist.
            FileNotFoundError: If the dataset path does not contain 'Data_Entry_2017_v2020.csv'.
            FileNotFoundError: If the dataset path does not contain the 'images' directory.
            ValueError: If the dataset 'images' directory does not contain any PNG files.
        """
        if not os.path.exists(self.root):
            msg = "Dataset path does not exist!"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not os.path.isfile(self.label_path):
            msg = "Dataset path must contain 'Data_Entry_2017_v2020.csv'!"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not os.path.exists(self.image_path):
            msg = "Dataset path must contain an 'images' directory!"
            logger.error(msg)
            raise FileNotFoundError(msg)

        if not list(self.image_path.glob("*.png")):
            msg = "Dataset 'images' directory must contain PNG files!"
            logger.error(msg)
            raise ValueError(msg)

    def _index_data(self) -> None:
        """Parses and indexes metadata for all available images in the dataset.

        Reads the label CSV file and filters it to include only entries that have
        corresponding image files in the dataset directory. For each valid entry,
        extracts patient metadata and creates a multi-label disease dictionary
        indicating the presence of specific conditions.

        The resulting structured data is stored in `self.data` as a list of dictionaries,
        each representing a single chest X-ray image and its associated metadata.

        Each entry in `self.data` contains:
            - image_name (str): Filename of the X-ray image.
            - patient_age (int): Age of the patient.
            - patient_sex (str): Sex of the patient.
            - labels (dict[str, bool]): Presence of each disease label as a boolean.

        Raises:
            FileNotFoundError: If the label CSV file does not exist.
            ValueError: If no matching image files are found in the CSV.
        """
        df = pd.read_csv(self.label_path)
        image_names = [f.name for f in self.image_path.iterdir() if f.is_file()]
        filtered_df = df[df["Image Index"].isin(image_names)]

        for _, row in filtered_df.iterrows():
            self.data.append({
                "image_name": row["Image Index"],
                "patient_age": row["Patient Age"],
                "patient_sex": row["Patient Sex"],
                "labels": {
                    "atelectasis": "Atelectasis" in row["Finding Labels"],
                    "cardiomegaly": "Cardiomegaly" in row["Finding Labels"],
                    "consolidation": "Consolidation" in row["Finding Labels"],
                    "edema": "Edema" in row["Finding Labels"],
                    "effusion": "Effusion" in row["Finding Labels"],
                    "emphysema": "Emphysema" in row["Finding Labels"],
                    "fibrosis": "Fibrosis" in row["Finding Labels"],
                    "hernia": "Hernia" in row["Finding Labels"],
                    "infiltration": "Infiltration" in row["Finding Labels"],
                    "mass": "Mass" in row["Finding Labels"],
                    "nodule": "Nodule" in row["Finding Labels"],
                    "pleural_thickening": "Pleural_Thickening" in row["Finding Labels"],
                    "pneumonia": "Pneumonia" in row["Finding Labels"],
                    "pneumothorax": "Pneumothorax" in row["Finding Labels"],
                },
            })

if __name__ == "__main__":
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    dataset = ChestXray14Dataset(download=(not args.no_download), partial=True)

    dataset.stat()
    dataset.info()
    print(dataset[0])
