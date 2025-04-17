import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from ..tasks import COVID19CXRClassification
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class COVID19CXRDataset(BaseDataset):
    """Base image dataset for COVID-19 Radiography Database.

    Dataset is available at:
    https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

    Data Sources:
    ------------
    COVID-19 data:
        - 2473 CXR images from padchest dataset[1]
        - 183 CXR images from a Germany medical school[2]
        - 559 CXR images from SIRM, Github, Kaggle & Tweeter[3,4,5,6]
        - 400 CXR images from another Github source[7]

    Normal images:
        - 8851 from RSNA [8]
        - 1341 from Kaggle [9]

    Lung opacity images:
        - 6012 from Radiological Society of North America (RSNA) CXR dataset[8]

    Viral Pneumonia images:
        - 1345 from the Chest X-Ray Images (pneumonia) database[9]

    Citations:
    ---------
    If you use this dataset, please cite:
    1. M.E.H. Chowdhury, T. Rahman, A. Khandakar, et al. "Can AI help in
       screening Viral and COVID-19 pneumonia?" IEEE Access, Vol. 8, 2020,
       pp. 132665-132676.
    2. Rahman, T., Khandakar, A., Qiblawey, Y., et al. "Exploring the Effect
       of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray
       Images." arXiv preprint arXiv:2012.02238.

    References:
    ----------
    [1] https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/
    [2] https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png
    [3] https://sirm.org/category/senza-categoria/covid-19/
    [4] https://eurorad.org
    [5] https://github.com/ieee8023/covid-chestxray-dataset
    [6] https://figshare.com/articles/COVID-19_Chest_X-Ray_Image_Repository/12580328
    [7] https://github.com/armiro/COVID-CXNet
    [8] https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
    [9] https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

    Args:
        root: Root directory of the raw data containing the dataset files.
        dataset_name: Optional name of the dataset. Defaults to "covid19_cxr".
        config_path: Optional path to the configuration file. If not provided,
            uses the default config in the configs directory.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import COVID19CXRDataset
        >>> dataset = COVID19CXRDataset(
        ...     root="/path/to/covid19_cxr"
        ... )
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> print(samples[0])
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = (
                Path(__file__).parent / "configs" / "covid19_cxr.yaml"
            )
        if not os.path.exists(os.path.join(root, "covid19_cxr-metadata-pyhealth.csv")):
            self.prepare_metadata(root)
        default_tables = ["covid19_cxr"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "covid19_cxr",
            config_path=config_path,
        )
        return

    def prepare_metadata(self, root: str) -> None:
        """Prepare metadata for the COVID-19 CXR dataset.

        Args:
            root: Root directory containing the dataset files.

        This method:
        1. Reads metadata from Excel files for each class
        2. Processes file paths and labels
        3. Combines all data into a single DataFrame
        4. Saves the processed metadata to a CSV file
        """
        # process and merge raw xlsx files from the dataset
        covid = pd.DataFrame(
            pd.read_excel(f"{root}/COVID.metadata.xlsx")
        )
        covid["FILE NAME"] = covid["FILE NAME"].apply(
            lambda x: f"{root}/COVID/images/{x}.png"
        )
        covid["label"] = "COVID"
        lung_opacity = pd.DataFrame(
            pd.read_excel(f"{root}/Lung_Opacity.metadata.xlsx")
        )
        lung_opacity["FILE NAME"] = lung_opacity["FILE NAME"].apply(
            lambda x: f"{root}/Lung_Opacity/images/{x}.png"
        )
        lung_opacity["label"] = "Lung Opacity"
        normal = pd.DataFrame(
            pd.read_excel(f"{root}/Normal.metadata.xlsx")
        )
        normal["FILE NAME"] = normal["FILE NAME"].apply(
            lambda x: x.capitalize()
        )
        normal["FILE NAME"] = normal["FILE NAME"].apply(
            lambda x: f"{root}/Normal/images/{x}.png"
        )
        normal["label"] = "Normal"
        viral_pneumonia = pd.DataFrame(
            pd.read_excel(f"{root}/Viral Pneumonia.metadata.xlsx")
        )
        viral_pneumonia["FILE NAME"] = viral_pneumonia["FILE NAME"].apply(
            lambda x: f"{root}/Viral Pneumonia/images/{x}.png"
        )
        viral_pneumonia["label"] = "Viral Pneumonia"
        df = pd.concat(
            [covid, lung_opacity, normal, viral_pneumonia],
            axis=0,
            ignore_index=True
        )
        df = df.drop(columns=["FORMAT", "SIZE"])
        df.columns = ["path", "url", "label"]
        for path in df.path:
            assert os.path.isfile(path), f"File {path} does not exist"
        df.to_csv(
            os.path.join(root, "covid19_cxr-metadata-pyhealth.csv"),
            index=False
        )
        return

    @property
    def default_task(self) -> COVID19CXRClassification:
        """Returns the default task for this dataset.

        Returns:
            COVID19CXRClassification: The default classification task.
        """
        return COVID19CXRClassification()
