"""
MIMIC-SBDH Dataset With Social History
Authors: Jalen Jiang, Rodrigo Mata
NetID: jalenj4, mata6
Paper Title: MIMIC-SBDH: A Dataset for Social and Behavioral Determinants of Health
Paper Link: https://proceedings.mlr.press/v149/ahsan21a/ahsan21a.pdf
"""

import logging
import re
from pathlib import Path
from typing import Optional, List

import polars as pl

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class SBDHDataset(BaseDataset):
    """
    A dataset class for handling MIMIC-SBDH data and extracting social history from discharge summaries.

    It joins EHR social history data from MIMIC-III NOTEEVENTS
    with structured SBDH labels from the MIMIC-SBDH project.

    Data Sources:
    MIMIC-SBDH.csv -- https://github.com/hibaahsan/MIMIC-SBDH/blob/main/MIMIC-SBDH.csv

    This file labels each discharge summary according to eight SBDHs, the numerical labels embed the following statuses:
    Community-Present (0: False, 1: True)
    Community-Absent (0: False, 1: True)
    Education (0: False, 1: True)
    Economics (0: None, 1: True, 2: False)
    Environment (0: None, 1: True, 2: False)
    Alcohol Use (0: None, 1: Present, 2: Past, 3: Never, 4: Unsure)
    Tobacco Use (0: None, 1: Present, 2: Past, 3: Never, 4: Unsure)
    Drug Use (0: None, 1: Present, 2: Past, 3: Never, 4: Unsure)

    Args:
        root: Root directory of the raw data.
        tables: List of tables to be loaded (e.g., ["sbdh"]).
        dataset_name: Name of the dataset. Defaults to "mimic-sbdh".
        config_path: Path to the configuration file. If None, uses default config.

    """

    def __init__(
        self,
        root: str,
        tables: List[str] = ["sbdh"],
        dataset_name: str = "mimic-sbdh",
        config_path: Optional[str] = None,
        **kwargs
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "mimic_sbdh.yml"
        
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs
        )
        
        # extract social history from noteevents, only the social history section
        self.extract_social_history()
        return

    def extract_social_history(self) -> pl.DataFrame:
        """
        Extract the social history section from the unstructured text notes in the data. Adds a new column 'social_history' to the global_event_df
        containing the extracted social history sections.
        
        Returns:
            The updated global_event_df with the social_history column
        """
        logger.info("Extracting social history from notes")
        
        self.global_event_df = self.global_event_df.with_columns(
            pl.col('sbdh/TEXT').map_elements(self._extract_social_history, return_dtype=pl.Utf8).alias('social_history')
        )
        
        logger.info(f"Extracted social history for {self.global_event_df.height} notes")
        return self.global_event_df

    @staticmethod
    def _extract_social_history(text: str) -> str:
        """
        Extract the social history section from a sinngle value for TEXT
        
        Args:
            text (str): The clinical note text from NOTEEVENTS table
            
        Returns:
            str: The extracted social history, or empty string if not found
        """
        if not isinstance(text, str) or pl.is_null(text):
            return ""
        
        # social history can be all caps or Title Cased
        pattern = r"(?:SOCIAL HISTORY:|Social History:)(.*?)(?=\n\s*(?:[A-Z][A-Z\s]+:|[A-Z][a-z]+\s+[A-Z][a-z]+:)|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # try the section-splitting approach if the more strict matching wasn't a hit
        sections = re.split(r"\n\s*\n", text)
        for section in sections:
            if re.match(r"(?:SOCIAL HISTORY:|Social History:)", section.strip(), re.IGNORECASE):
                return re.sub(r"(?:SOCIAL HISTORY:|Social History:)", "", section.strip(), 
                          flags=re.IGNORECASE, count=1).strip()
        
        return ""
    
    def export_social_history(self, output_path: str) -> None:
        """
        Export the extracted social history data to a CSV file
        
        Args:
            output_path (str): Path to save the CSV file
        """
        logger.info(f"Exporting social history to {output_path}")
        
        # select cols
        social_history_df = self.global_event_df.select([
            'sbdh/ROW_ID', 'sbdh/SUBJECT_ID', 'sbdh/CHARTTIME', 'social_history', 
            'sbdh/sdoh_community_present', 'sbdh/sdoh_community_absent', 
            'sbdh/sdoh_education', 'sbdh/sdoh_economics', 'sbdh/sdoh_environment',
            'sbdh/behavior_alcohol', 'sbdh/behavior_tobacco', 'sbdh/behavior_drug'
        ])
        
        rename_map = {col: col.replace('sbdh/', '') for col in social_history_df.columns if col.startswith('sbdh/')}
        social_history_df = social_history_df.rename(rename_map)

        # export to csv
        social_history_df.write_csv(output_path)
        logger.info(f"Exported {social_history_df.height} records to {output_path}")
