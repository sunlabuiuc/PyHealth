import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from ..tasks.dailysportsactivities_classification import DSAClassification
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class DSADataset(BaseDataset):
    """Base dataset for Daily Sports Activities Data.

    Dataset is available at:
    https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

    Args:
        root: Root directory of the raw data containing the dataset files.
        dataset_name: Optional name of the dataset. Defaults to "dsa".
        config_path: Optional path to the configuration file. If not provided,
            uses the default config in the configs directory.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import DSADataset
        >>> dataset = DSADataset(
        ...     root="/path/to/dsa_data"
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
                Path(__file__).parent / "configs" / "dsa.yaml"
            )
        if not os.path.exists(os.path.join(root, "daily-sports-activities.csv")):
            self.prepare_metadata(root)
        default_tables = ["dsa"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "dsa",
            config_path=config_path,
        )
        return

    def prepare_metadata(self, root: str) -> None:
        """Prepare metadata for the DSA dataset.

        Args:
            root: Root directory containing the dataset files.

        This method:
        1. Processes each .txt file with data for a particular 5 sec segment
        2. Combines the data into a single DataFrame
        3. Saves the processed metadata to a CSV file
        """
        data_list = []

        for activity_idx in range(1, 20): 
            activity_folder = f"a{activity_idx:02d}"
            for person_idx in range(1, 9): 
                person_folder = f"p{person_idx}"
                for segment_idx in range(1, 61):
                    segment_file = f"s{segment_idx:02d}.txt"
                    segment_path = os.path.join(root, activity_folder, person_folder, segment_file)

                    data = np.loadtxt(segment_path, delimiter=",")
                    data_flattened = data.flatten()

                    data_dict = {'person': person_idx, 'segment_num': segment_idx, 'sensor_data': data_flattened, 'activity': activity_idx}

                    data_list.append(data_dict)

        df = pd.DataFrame(data_list)
        df['sensor_data'] = df['sensor_data'].apply(lambda x : '|'.join(map(str, x)))
        df.to_csv(os.path.join(root, "daily-sports-activities.csv"),index=False)
        return

    @property
    def default_task(self) -> DSAClassification:
        """Returns the default task for this dataset.

        Returns:
            DSAClassification: The default classification task.
        """
        return DSAClassification()
