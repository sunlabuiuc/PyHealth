"""Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset.

Author: Bryan Lau (bryan16@illinois.edu)
Description:
    A Pyhealth dataset for Alzheimer's Disease Neuroimaging Initiative (ADNI) MRI 
    brain scan images in Neuroimaging Informatics Technology Initiative (NIftI) 
    format.
"""
import os
import pandas as pd
import random

from lxml import etree
from pyhealth.datasets import BaseDataset
from pathlib import Path
from typing import Optional


class ADNIDataset(BaseDataset):
    """Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset.

    A Pyhealth dataset for Alzheimer's Disease Neuroimaging Initiative (ADNI) 
    MRI brain scan images.

    Since the ADNI dataset itself is not public, users wishing to use the 
    actual files must apply for access at:

    https://adni.loni.usc.edu/data-samples/adni-data/

    The required pre-processing parameters are:

    - Multiplanar reconstruction (MPR)
    - Gradient warping correction (GradWarp)
    - B1 non-uniformity correction
    - N3 intensity normalization

    These parameters produce the directory structure that this dataset 
    looks for when it attempts to parse the downloaded files.

    When extracted, the downloaded zip files produce a directory 
    structure containing:

    - MRI image files in NifTI format (*.nii)
    - Corresponding XML metadata files (for each image)

    The directory structure has the following layout:

    - root
        - subject id
            - pre-processing transform
                - date acquired
                    - image uid
                        MRI image file
                        metadata xml file
        - subject id
        - subject id
        metadata xml file
        metadata xml file
        metadata xml file

    A separate catalog CSV file can also be downloaded, but is not used by 
    this dataset.

    The dataset provides:

    - Structural brain MRI scans
    - Patient diagnostic labels, indicating:
        - Cognitively Normal (CN)
        - Mild Cognitive Impairment (MCI)
        - Alzheimer's Disease (AD)
    - Demographic information
        - Gender
        - Age
        - Weight (Kg)

    Args:
        root (str ): Root directory of extracted ADNI dataset files.
        config_path (str): Path to the configuration file.
        dataset_name (str): Name identifying this dataset, default "ADNI".
        dev (bool): If True, number of loaded images is restricted to 100.

    Raises:
        FileNotFoundError:  If either the root directory, subject directories 
                            or MRI images are missing.

    Example:
        >>> from pyhealth.datasets import ADNIDataset
        >>> from pyhealth.tasks import AlzheimersDiseaseClassification
        >>> adni_dataset = ADNIDataset(root="/path/to/adni_data", dev=True)
        >>> ad_task = AlzheimersDiseaseClassification()
        >>> samples = adni_dataset.set_task(ad_task)
    """

    def __init__(
        self,
        root: str,
        config_path: Optional[str] = str(
            Path(__file__).parent / "configs" / "adni.yaml"),
        dataset_name: str = "ADNI",
        dev: bool = False,
        **kwargs
    ):
        """Initialize the ADNI dataset.

        Args:
            root (str): Root directory of extracted ADNI dataset files.
            config_path (Optional[str]): Path to the configuration file, 
                defaults to "../configs/adni-metadata-pyhealth.csv"
            dataset_name (str): Name identifying this dataset, default "ADNI".
            dev: If True, number of loaded images is restricted to 100.
        """
        self.root = Path(root)
        self.dev = dev

        self.DEV_LIMIT = 1000

        # Validate data path
        self._validate_data_path(self.root)

        # Write patient catalog
        self._write_patient_catalog(self.root)

        super().__init__(
            root=root,
            tables=["adni"],
            config_path=config_path,
            dataset_name=dataset_name,
            dev=dev,
            **kwargs
        )

    def _write_patient_catalog(self, data_path):
        """Scan the data path and build a patient catalog from metadata xml files.

        Builds catalog and writes to adni-metadata-pyhealth.csv.

        Args:
            data_path: Root directory path of the extracted ADNI files.
        """

        patient_catalog = []
        patient_set = set()

        # Find all metadata xml files located in the root directory
        # e.g. ADNI_002_S_0295_MPR__GradWarp__B1_Correction__N3_S13408_I45107.xml
        metadata_files = Path(data_path).glob("ADNI*.xml")

        # If a dev limit is requested, then sample the list randomly
        if self.dev:
            metadata_files = random.sample(
                sorted(metadata_files), k=self.DEV_LIMIT)

        # Read MRI subject and image metadata from each xml file
        for idx, file in enumerate(metadata_files):

            xml_root = etree.parse(file)

            # Subject node
            xml_subject = xml_root.find(".//project/subject")
            subject = xml_subject.find("./subjectIdentifier")
            gender = xml_subject.find("./subjectSex")
            group = xml_subject.find("./researchGroup")

            # Study node
            xml_study = xml_root.find(".//project/subject/study")
            age = xml_study.find("./subjectAge")
            weight = xml_study.find("./weightKg")

            # Series node
            xml_series = xml_root.find(".//project/subject/study/series")
            date_acquired = xml_series.find("./dateAcquired")
            image_uid = xml_series.find(
                "./seriesLevelMeta/derivedProduct/imageUID")

            # Dev limit
            patient_set.add(subject.text)
            if len(patient_set) > self.DEV_LIMIT:
                break

            # Locate image
            image_glob = Path(data_path).glob(
                f"{subject.text}/**/{date_acquired.text}*/I{image_uid.text}/ADNI_{subject.text}_*.nii")
            image_path = next(image_glob)

            # Patient record
            patient_record = {
                "patient_id": subject.text,
                "gender": gender.text,
                "age": float(age.text),
                "weight": float(weight.text),
                "group": group.text,
                "timestamp": date_acquired.text,
                "image_uid": image_uid.text,
                "image_path": image_path
            }
            patient_catalog.append(patient_record)

        patient_catalog_df = pd.DataFrame(patient_catalog)
        patient_catalog_df.to_csv(os.path.join(
            self.root, "adni-metadata-pyhealth.csv"), index=False)

    def _validate_data_path(self, data_path):
        """Validate the provided root data path.

        Checks that the specified root data path:
        - Exists
        - Contains at least one subject directory
        - Contains at least one metadata xml file
        - Contains at least one subject MRI image in NifTI format

        Raises:
            FileNotFoundError:  If the root directory is missing.
            FileNotFoundError:  If patient subject directories are missing.
            FileNotFoundError:  If metadata xml files are missing.
            FileNotFoundError:  If MRI images are missing.
        """

        # Exists
        if not os.path.exists(data_path):
            msg = "Root dataset path does not exist!"
            raise FileNotFoundError(msg)

        # Contains subject directories
        subject_dirs = list(Path(data_path).glob("*_S_*/"))
        if not subject_dirs:
            msg = "Dataset path must contain subject directories!"
            raise FileNotFoundError(msg)

        # Contains metadata xml files
        if not list(Path(data_path).glob("*.xml")):
            msg = "Dataset path must contain metadata xml files!"
            raise FileNotFoundError(msg)

        # First subject has at least one NifTI MRI image
        if not list(Path(subject_dirs[0]).glob("**/*.nii")):
            msg = "Dataset path must contain MRI images!"
            raise FileNotFoundError(msg)
