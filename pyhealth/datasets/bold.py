import logging
import warnings
from pathlib import Path
from typing import List, Optional

import polars as pl

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class BoldDataset(BaseDataset):
    """
    A dataset class for handling BOLD data.

    This class is responsible for loading and managing the BOLD dataset,
    which includes tables such as "patients","demographics","hospital","abgdata","vitalsdata","labdata",
                                   "coagulationlabs","bmpdata","hfpdata","otherlabdata","sofascores"

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initializes the BoldDataset with the specified parameters.

        Args:
            root (str): The root directory where the dataset is stored.
            tables (List[str]): A list of additional tables to include.
            dataset_name (Optional[str]): The name of the dataset. Defaults to "bold".
            config_path (Optional[str]): The path to the configuration file. If not provided, a default config is used.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "bold.yaml"

        default_tables = ["patients","demographics","hospital","abgdata","vitalsdata","labdata","coagulationlabs","bmpdata","hfpdata","otherlabdata","sofascores"]

        tables = default_tables + tables

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "bold",
            config_path=config_path,
            **kwargs
        )
        return

