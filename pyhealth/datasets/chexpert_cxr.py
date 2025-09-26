import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from ..tasks import CheXpertCXRClassification
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CheXpertCXRDataset(BaseDataset):
    """Base image dataset for Chest Radiography Database.

    Dataset is available at: https://www.kaggle.com/datasets/ashery/chexpert

    Citations:
    ---------
    If you use this dataset, please cite:
    1. Joseph Janizek, Gabriel Erion, Alex DeGrave, et al. "An Adversarial Approach for the Robust Classification of
       Pneumonia from Chest Radiographs"

    References:
    ----------
    [1] https://arxiv.org/abs/2001.04051

    Args:
        root: Root directory of the raw data containing the dataset files.
        dataset_name: Optional name of the dataset. Defaults to "chexpert_cxr".
        config_path: Optional path to the configuration file. If not provided,
            uses the default config in the configs directory.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import CheXpertCXRClassification
        >>> dataset = CheXpertCXRClassification(
        ...     root="/path/to/chexpert_cxr"
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
        dev: Optional[str] = None,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = (
                Path(__file__).parent / "configs" / "chexpert_cxr.yaml"
            )
        logger.info("No config path provided, using default config")
        print("No config path provided, using default config")
        if not os.path.exists(os.path.join(root, "chexpert_cxr-pyhealth.csv")):
            self.prepare_metadata(root)
        default_tables = ["chexpert_cxr"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "chexpert_cxr",
            config_path=config_path,
            dev=dev,
        )
        self.dataset_df = None
        return


    def prepare_metadata(self, root: str) -> None:
        """Prepare metadata for the CheXpert CXR dataset.

        Args:
            root: Root directory containing the dataset files.

        This method:
        1. Reads the train and valid dataset from CheXpert mini dataset
        2. Processes file paths
        3. Combines all data into a single DataFrame
        4. Saves the processed metadata to a CSV file
        """
        cxr_train = pd.DataFrame(
            pd.read_csv(f"{root}/train.csv")
        )

        cxr_valid = pd.DataFrame(
            pd.read_csv(f"{root}/valid.csv")
        )

        df = pd.concat(
            [cxr_train, cxr_valid],
            axis=0,
            ignore_index=True
        )

        df["Path"] = df["Path"].apply(
            lambda x: os.path.join(os.path.dirname(root), x)
        )

        for path in df.Path:
            path = os.path.join(os.path.dirname(root), path)
            assert os.path.isfile(path), f"File {path} does not exist"
            
        df.to_csv(
            os.path.join(root, "chexpert_cxr-pyhealth.csv"),
            index=False
        )
        self.dataset_df = df.copy()
        return

    @property
    def default_task(self) -> CheXpertCXRClassification:
        """Returns the default task for this dataset.

        Returns:
            CheXpertCXRClassification: The default classification task.
        """
        return CheXpertCXRClassification()