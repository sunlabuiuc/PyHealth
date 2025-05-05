# Name (s): Chris Yu, Jimmy Lee
# NetId (s) (If applicable for UIUC students): hmyu2, jl279
# The paper title : Revisit Deep Cox Mixtures For Survival Regression
# The paper link: https://github.com/chrisyu-uiuc/revisit-deepcoxmixtures-cs598-uiuc/blob/main/Revisit_DeepCoxMixturesForSurvivalRegression.pdf
# Implementation of the GBSGDataset class for loading and managing the German Breast Cancer Study Group dataset within the pyhealth framework.

import logging
from pathlib import Path
from typing import List, Optional

import polars as pl

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class GBSGDataset(BaseDataset):
    """
    A dataset class for handling GBSG (German Breast Cancer Study Group) data.

    This class is responsible for loading and managing the GBSG dataset,
    which includes breast cancer patient data with various clinical attributes
    relevant for survival analysis.

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.
    """

    def __init__(
        self,
        root: str,
        tables: List[str] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
            Initializes the GBSGDataset with the specified parameters.

            This function sets up the dataset by specifying the root directory,
            tables to load, dataset name, and configuration path. It calls the
            constructor of the base class (`BaseDataset`) to handle the core loading
            logic based on the provided parameters.

            Args:
                root: The root directory (str) where the dataset files (e.g., gbsg.csv) are stored.
                tables: A list of strings specifying additional tables to include. Defaults to None,
                    which will result in loading the default 'gbsg' table.
                dataset_name: An optional string to name the dataset instance. Defaults to "gbsg".
                config_path: An optional string specifying the path to a configuration file.
                    If not provided, a default configuration file specific to GBSG will be used.
                **kwargs: Additional keyword arguments passed to the BaseDataset constructor.

            Returns:
                None

            Example Usage:
                # Assuming gbsg.csv and gbsg.yaml are in /path/to/data
                dataset = GBSGDataset(
                    root="/path/to/data/",
                    config_path="/path/to/pyhealth/datasets/configs/gbsg.yaml",
                    dataset_name="my_gbsg_study"
                )
                # This will be called when instantiating the GBSGDataset class.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "gbsg.yaml"
        
        tables = tables or ["gbsg"]
        
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "gbsg",
            config_path=config_path,
            **kwargs
        )
        return
