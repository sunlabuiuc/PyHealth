import logging
import warnings
from pathlib import Path
from typing import List, Optional

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MIMIC3Dataset(BaseDataset):
    """
    A dataset class for handling MIMIC-III data.

    This class is responsible for loading and managing the MIMIC-III dataset,
    which includes tables such as patients, admissions, and icustays.

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
        Initializes the MIMIC4Dataset with the specified parameters.

        Args:
            root (str): The root directory where the dataset is stored.
            tables (List[str]): A list of additional tables to include.
            dataset_name (Optional[str]): The name of the dataset. Defaults to "mimic3".
            config_path (Optional[str]): The path to the configuration file. If not provided, a default config is used.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "mimic3.yaml"
        default_tables = ["patients", "admissions", "icustays"]
        tables = default_tables + tables
        if "prescriptions" in tables:
            warnings.warn(
                "Events from prescriptions table only have date timestamp (no specific time). "
                "This may affect temporal ordering of events.",
                UserWarning,
            )
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "mimic3",
            config_path=config_path,
            **kwargs
        )
        return
