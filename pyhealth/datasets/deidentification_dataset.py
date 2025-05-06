"""
DeidentificationDataset Class
Author: Varshini R R
NetID: vrr4

This module implements the `DeidentificationDataset` class for loading, processing, 
and preprocessing synthetic hospital discharge summaries used in de-identification 
tasks. The class reads data from a specified JSON file, validates the contents, 
and provides basic preprocessing for the text data.

Usage:
    This class is used in de-identification workflows to load discharge summary 
    data, perform preprocessing tasks (such as text normalization), and support 
    downstream machine learning models or evaluations.

Methods:
    - `load_data`: Loads and validates the data from a JSON file, converting it into a pandas DataFrame.
    - `get_patient_data`: Retrieves a subset of the dataset for a specific patient based on their ID.
    - `preprocess_data`: Normalizes and cleans the text data, storing it in a new `processed_text` column.
    - `log_memory_usage`: Logs the current memory usage of the process.
    
Attributes:
    - `config`: Dictionary containing configuration details such as file path, patient ID, and attributes.
    - `data`: A pandas DataFrame containing the processed dataset.
    - `patient_id`: Column name for the unique identifier of the patient/document.
    - `timestamp`: Column name for the timestamp (optional).
    - `attributes`: List of attributes to extract from each record.
    - `dev`: Flag indicating if the dataset should be loaded in development mode with a subset of records.

Dependencies:
    - pandas
    - json
    - logging
    - psutil (optional for memory logging)
    - pyhealth (for BaseDataset)

"""

import os
import json
import logging
import pandas as pd
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

from .base_dataset import BaseDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def log_memory_usage(tag: str = "") -> None:
    """Logs current memory usage of the process.

    Args:
        tag: Optional label to indicate where memory is being logged.
    """
    if psutil is not None:
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            logger.info(f"Memory usage {tag}: {mem_info.rss / (1024 * 1024):.1f} MB")
        except Exception as e:
            logger.warning(f"Memory logging failed at {tag}: {e}")
    else:
        logger.warning(f"psutil not available. Unable to log memory at {tag}")


class DeidentificationDataset(BaseDataset):
    """Dataset loader for discharge summaries used in de-identification tasks."""

    def __init__(self, config: Dict, dev: bool = False) -> None:
        """Initializes the DeidentificationDataset.

        Args:
            config: Dictionary with required keys:
                - 'file_path': Path to JSON file.
                - 'patient_id': Column name for unique patient/document ID.
                - 'timestamp': Column name for timestamp (optional).
                - 'attributes': List of attributes to extract from each record.
            dev: Whether to run in development mode (loads only first 100 records).

        Raises:
            ValueError: If required configuration keys are missing or invalid.
        """
        self.config = config
        self.file_path = config.get("file_path")
        self.patient_id = config.get("patient_id")
        self.timestamp = config.get("timestamp")
        self.attributes = config.get("attributes", [])
        self.dev = dev

        if not self.file_path:
            raise ValueError("Missing 'file_path' in configuration.")
        if not self.patient_id:
            raise ValueError("Missing 'patient_id' in configuration.")
        if not self.attributes:
            raise ValueError("Missing 'attributes' list in configuration.")

        log_memory_usage("Before loading data")
        self.data = self.load_data()
        log_memory_usage("After loading data")

        if self.dev:
            self.data = self.data.head(100)
            logger.info("Development mode: limited to first 100 records.")

    def load_data(self) -> pd.DataFrame:
        """Loads and validates JSON data from the configured file path.

        Returns:
            A pandas DataFrame containing the dataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If JSON is malformed or required columns are missing.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found at: {self.file_path}")

        try:
            with open(self.file_path, "r") as file:
                raw_data = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error reading file: {e}")

        if not isinstance(raw_data, list):
            raise ValueError("Expected a list of records in JSON file.")

        try:
            df = pd.json_normalize(raw_data)
        except Exception as e:
            raise ValueError(f"Failed to convert JSON to DataFrame: {e}")

        required_columns = [self.patient_id, "text"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        return df

    def get_patient_data(self, patient_id: str) -> pd.DataFrame:
        """Retrieves data for a specific patient/document ID.

        Args:
            patient_id: Unique identifier to filter records by.

        Returns:
            Subset of the dataset matching the given patient ID.

        Logs:
            A warning if no records are found.
        """
        subset = self.data[self.data[self.patient_id] == patient_id]
        if subset.empty:
            logger.warning(f"No records found for patient ID: {patient_id}")
        return subset

    def preprocess_data(self) -> None:
        """Cleans and normalizes the 'text' column.

        Adds a new column `processed_text` with:
            - Lowercase text
            - Newlines removed
            - Leading/trailing whitespace stripped

        Raises:
            ValueError: If the 'text' column is missing or processing fails.
        """
        if "text" not in self.data.columns:
            raise ValueError("Missing 'text' column in dataset.")

        try:
            self.data["processed_text"] = (
                self.data["text"]
                .astype(str)
                .str.lower()
                .str.replace("\n", " ", regex=False)
                .str.strip()
            )
        except Exception as e:
            raise ValueError(f"Failed during text preprocessing: {e}")


# Example Usage
if __name__ == "__main__":
    dataset_config = {
        "table_name": "discharge_summaries",
        "file_path": "data/deid_raw/discharge/discharge_summaries.json",
        "patient_id": "document_id",
        "timestamp": "discharge_date",
        "attributes": [
            "document_id", "text", "patient_name", "dob", "age", "sex", "service",
            "chief_complaint", "diagnosis", "treatment", "follow_up_plan",
            "discharge_date", "attending_physician"
        ]
    }

    try:
        dataset = DeidentificationDataset(dataset_config, dev=True)
        dataset.preprocess_data()
        print(f"First record:\n{dataset.data.iloc[0]}")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
