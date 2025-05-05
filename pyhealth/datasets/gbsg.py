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
    which includes breast cancer patient data with various clinical attributes.

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

        Args:
            root (str): The root directory where the dataset is stored.
            tables (List[str]): A list of additional tables to include.
            dataset_name (Optional[str]): The name of the dataset. Defaults to "gbsg".
            config_path (Optional[str]): The path to the configuration file. If not provided, a default config is used.
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