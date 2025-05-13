import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class DSADataset(BaseDataset):
    """
    A dataset class for handling the Daily and Sport Activities.

    This class is responsible for loading and managing the DSA dataset.

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.
    """

    def __init__(
        self,
        root: str,
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
            config_path = Path(__file__).parent / "configs" / "DSA.yaml"
        tables = [] # The DSA dataset doesn't have the concept of tables
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "DSA",
            config_path=config_path,
            **kwargs
        )
        return

    def load_data(self):
        segments, labels, metadata = [], [], []
        for activity in sorted(os.listdir(self.root)):
            for person in sorted(os.listdir(os.path.join(data_root, activity))):
                for session_file in sorted(os.listdir(os.path.join(data_root, activity, person))):
                    if session_file.endswith('.txt'):
                        path = os.path.join(data_root, activity, person, session_file)
                        raw = np.loadtxt(path, delimiter=",")
                        segments.append(raw)
                        labels.append(activity)
                        metadata.append((activity, person, session_file))
        return segments, labels, metadata