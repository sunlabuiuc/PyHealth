"""
MIMIC-SBDH Dataset
Authors: Duy Nguyen, Khang Nguyen
NetID: duyn2, khangn2
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
    It integrates free-text social history data from MIMIC-III NOTEEVENTS
    with structured SBDH labels from the MIMIC-SBDH project,
    creating a unified dataset for comprehensive analysis.

    Data Sources:
    MIMIC-SBDH.csv is available at https://github.com/hibaahsan/MIMIC-SBDH/blob/main/MIMIC-SBDH.csv
    This file contains the labels for the seven SBDHs:
    Community-Present and Community-Absent (0: False, 1: True)
    Education (0: False, 1: True)
    Economics (0: None, 1: True, 2: False)
    Environment (0: None, 1: True, 2: False)
    Alcohol Use (0: None, 1: Present, 2: Past, 3: Never, 4: Unsure)
    Tobacco Use (0: None, 1: Present, 2: Past, 3: Never, 4: Unsure)
    Drug Use (0: None, 1: Present, 2: Past, 3: Never, 4: Unsure)

    The MIMIC-III NOTEEVENTS (NOTEEVENTS.csv) is available at https://physionet.org/content/mimiciii/1.4/

    Key Features:
    - Loads both `MIMIC-SBDH.csv` (labels) and `NOTEEVENTS.csv` (clinical notes) located under a specified root directory.
    - Extracts the "Social History" sections from clinical notes
    - Preprocesses text fields and prepares them as global events in the dataset.
    - Supports standard PyHealth dataset methods such as `stats()`, `get_patient()`, and `get_events()`.

    Attributes:
        root (str): root directory of the raw data (MIMIC-SBDH.csv and NOTEEVENTS.csv).
        tables (List[str]): list of tables to be loaded (e.g., ["sbdh"]).
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
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "mimic3_sbdh.yaml"
        if tables is None:
            tables = ["sbdh"]
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "mimic-sbdh",
            config_path=config_path,
            **kwargs
        )

        self.global_event_df = self.global_event_df.with_columns(
            pl.col('sbdh/text').map_elements(self._extract_social_history, return_dtype=pl.Utf8))
        return

    def _extract_social_history(self, text: str) -> str:
        """
        Extract the social history section from the MIMIC-III NOTEEVENTS
        Args:
            text (str): The clinical note in the 'TEXT' column of NOTEEVENTS table.

        Returns:
            match_pattern (str): The social history extracted from the clinical note.
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""

        text = text.lower().strip()

        # Regex pattern to extract content following 'social history:' until the next section header or end of text.
        pattern = r"social history:\s*(.*?)(?=\n[a-z\s]+:|\Z)"
        match = re.search(pattern, text, re.DOTALL)

        return match.group(1).strip() if match else ""


