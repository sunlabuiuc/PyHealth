"""
MIMIC-SBDH Dataset Loader

Authors: Youye Xie, Lanou Qu
NetID: youyex2, lanouqu2
Paper Title: MIMIC-SBDH: A Dataset for Social and Behavioral Determinants of Health
Paper Link: https://proceedings.mlr.press/v149/ahsan21a/ahsan21a.pdf
"""

import logging
from pathlib import Path
from typing import Optional
import polars as pl
import pandas as pd
import re
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class SBDHDataset(BaseDataset):
    """
    A dataset class for handling MIMIC-SBDH data.

    This module implements the `SBDHDataset` class, a custom dataset class for loading 
    and managing the MIMIC-SBDH dataset using the PyHealth framework. It combines 
    social history free-text data from MIMIC-III NOTEEVENTS with structured SBDH 
    labels provided by the MIMIC-SBDH project.

    SBDH label (MIMIC-SBDH.csv) is available at https://github.com/hibaahsan/MIMIC-SBDH/blob/main/MIMIC-SBDH.csv
    The MIMIC-III NOTEEVENTS (NOTEEVENTS.csv) is available at https://physionet.org/content/mimiciii/1.4/

    Key Features:
    - Loads both `MIMIC-SBDH.csv` (labels) and `NOTEEVENTS.csv` (clinical notes) located under a specified root directory.
    - Extracts the "Social History" sections from clinical notes using regular expressions.
    - Preprocesses text fields and prepares them as global events in the dataset.
    - Supports standard PyHealth dataset methods such as `stats()`, `get_patient()`, and `get_events()`.

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.
    """

    def __init__(
        self,
        root: str,
        tables: Optional[str] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initializes the MIMIC-SBDH with the specified parameters.

        Args:
            root (str): The root directory where the dataset is stored.
            tables (Optional[str]): A list of additional tables to include.
            dataset_name (Optional[str]): The name of the dataset. Defaults to "sbdh".
            config_path (Optional[str]): The path to the configuration file. If not provided, a default config is used.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "sbdh.yaml"
        if tables is None:
            tables = ["sbdh"]
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "sbdh",
            config_path=config_path,
            **kwargs
        )
 
        self.global_event_df = self.global_event_df.with_columns(pl.col('sbdh/text').map_elements(self._extract_social_history,return_dtype=pl.Utf8))
        return
    
    def _extract_social_history(self, text: str) -> str:
        """
        Extract the social history section from the MIMIC-III NOTEEVENTS

        Args:
            text (str): The clinical note in the 'TEXT' column of NOTEEVENTS table.
       
        Returns:
            match_pattern (str): The social history extracted from the clinical note.

        """
        if pd.isna(text):
                return ""
        text = text.lower()
        # Look for the content between 'social history:' and the next section head or end of the file.
        pattern = r"social history:\s*(.*?)(?=\n[a-z\s]+:|\Z)"
        match_pattern = re.search(pattern, text, re.DOTALL)
        if match_pattern:
            return match_pattern.group(1).strip()
        return ""
    
if __name__ == "__main__":
    """
    Usage:
    1. Place `MIMIC-SBDH.csv` and `NOTEEVENTS.csv` in the same folder.
    2. Initialize the dataset:
        dataset = SBDHDataset(root="/path/to/dataset/folder")
    3. Basic exploration:
        dataset.stats()
        patient_id = dataset.unique_patient_ids[0]
        dataset.get_patient(patient_id).get_events()
    """
    dataset = SBDHDataset(
        root="/path/to/dataset/folder"
    )
    dataset.stats()
    patient_id = dataset.unique_patient_ids[0]
    dataset.get_patient(patient_id).get_events()

