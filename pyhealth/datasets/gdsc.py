import logging
from pathlib import Path
from typing import List, Optional
from .base_dataset import BaseDataset

import polars as pl

logger = logging.getLogger(__name__)


class GDSCDataset(BaseDataset):

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initializes the GDSC Dataset with the specified parameters.

        Args:
            root (str): The root directory where the dataset is stored.
            tables (List[str]): A list of additional tables to include.
            dataset_name (Optional[str]): The name of the dataset. Defaults to "gdsc".
            config_path (Optional[str]): The path to the configuration file. If not provided, a default config is used.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "gdsc.yaml"
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "gdsc",
            config_path=config_path,
            **kwargs
        )
        return
