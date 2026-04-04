"""Daily and Sports Activities dataset for PyHealth.

This module provides the DSADataset class for loading and processing
DSA (Daily and Sports Activities) data for machine learning tasks.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class DSADataset(BaseDataset):
    """Daily and Sports Activities dataset for motion sensor data analysis.

    Daily and Sports Activities (DSA) dataset contains motion sensor data of 19
    daily and sports activities each performed by 8 subjects for 5 minutes in 
    their own style. The 5-min period is divided into 5-sec segments so that 60 
    segments are obtained for each activity and each subject. 
    
    Five sensor units are placed on the torso, arms, and legs to capture 
    time-series data. Sensors are configured to capture data at 25 Hz frequency. 

    In each text file, there are 45 columns (5 units * 9 sensors) and 125 rows 
    (5 seconds * 25 Hz). Each row represents sensor dimensions captured at a p
    articular sampling instant from all sensors for one activity, one patient.

    The raw dataset structure looks like this: 
    - <root>
        - data
            - a01: activity 01
                - p1: patient 1
                    - s01.txt: segment 01
                    - s02.txt: segment 02
                    - ... ...
                    - s60.txt: segment 60
                - p2: patient 2
                - ... ...
                - p8
            - a02: activity 02
            - ... ...
            - a19

    This dataset class loads raw data files and reformats into a single CSV file
    with columns activity, patient, segment, sample_id, and 45 sensor columns.

    Dataset is available at:
    https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

    Note:
        This dataset is licensed under a Creative Commons Attribution 4.0 
        International (CC BY 4.0) license.

    Args:
        root: Root directory of the raw data containing the DSA dataset files.
        dataset_name: Optional name of the dataset. Defaults to "dsa".
        config_path: Optional path to the configuration file. If not provided,
            uses the default config in the configs directory.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import DSADataset
        >>> dataset = DSADataset(root="/path/to/dsa")
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> print(samples[0])
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        download : bool = False,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "dsa.yaml"

        if download: 
            self.download_dataset(root=root)

        # Prepare standardized CSV if not exists
        pyhealth_csv = os.path.join(root, "dsa-pyhealth.csv")
        if not os.path.exists(pyhealth_csv):
            if len(os.listdir(root)) != 19: 
                logger.info(
                    f"""
                    The contexts in {root} are unexpected. You root directory is likely wrong. 
                    You can download the dataset manually from https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities
                    or directly set download=True.
                    """
                )
                raise 
                
            logger.info("Preparing DSA data...")
            self.prepare_data(root)

        super().__init__(
            root=root,
            tables=["activities"],
            dataset_name=dataset_name or "dsa",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def download_dataset(root : str = None) -> None: 
        """Download DSA dataset and extract files.
        
        Args: 
            root: directory to extract downloaded files, default to ./daily-and-sports-activities
        """
        import urllib.request
        import zipfile
        import os

        url = "https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip"
        zip_path = "./daily+and+sports+activities.zip"
        if root is None: 
            root = "./daily-and-sports-activities"

        logger.info(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)

        logger.info("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)

        os.remove(zip_path)

        logger.info(f"Done! Files extracted to: {root}")
        return

    @staticmethod
    def prepare_data(root: str) -> None:
        """Prepare data for DSA dataset.

        Converts raw DSA text files to standardized CSV format.

        Args:
            root: Root directory containing the DSA files.
        """

        data = []

        columns = [
            f"{x}_{z}{y}" 
            for z in ["x", "y", "z"]
            for y in ["acc", "gyro", "mag"]
            for x in ["T", "RA", "LA", "RL", "LL"]
        ]

        logger.info(f"Loading raw sensors data ...")

        for a in range(1, 20): 
            for p in range(1, 9): 
                for s in range(1, 61): 
                    df = pd.read_csv(
                        root / "data" / f"a{a:02d}" / f"p{p}" / f"s{s:02d}.txt", 
                        header=None,
                    )
                    df.columns = columns
                    df = df.assign(
                        activity=a,
                        patient=p,
                        segment=s,
                    )
                    df["sample_id"] = range(1, 126)
                    data.append(df)
                    
        df_out = pd.concat(data)

        # Save to standardized CSV
        output_path = os.path.join(root, "dsa-pyhealth.csv")
        df_out.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_out)} sensors to {output_path}")

    @property
    def default_task(self):
        """Returns the default task for this dataset.

        Returns:
            ActivityClassification: The default classification task.
        """
        from pyhealth.tasks import ActivityClassification

        return ActivityClassification()
