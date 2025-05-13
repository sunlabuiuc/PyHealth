import os
import pickle
import logging
from pathlib import Path
from typing import List, Optional

import polars as pl
import numpy as np

from ..data import Patient
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class WESADDataset(BaseDataset):
    """
    Custom dataset class for the WESAD dataset in the PyHealth framework.
    This class handles the loading and parsing of the physiological and 
    contextual data from the WESAD dataset.

    Attributes:
        dataset_name (str): Name of the dataset.
        root_dir (str): Path to the root directory of the dataset.
        subjects (List[str]): List of subject IDs to include.
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = "WESAD",
        config_path: Optional[str] = None,
        dev: bool = False,
    ):
        """
        Initializes the WESADDataset object.

        Args:
            dataset_name (str): Name to assign to the dataset instance.
            root_dir (str): Path to the WESAD dataset directory.
            subjects (Optional[List[str]]): List of subject IDs to include. If None, include all.

        Returns:
            None

        Description:
            Loads and prepares the WESAD dataset for downstream processing. Filters subjects
            if provided.

        Example:
            dataset = WESADDataset("wesad", "/path/to/data", subjects=["S2", "S3"])
        """
        self.subject_ids = [f"S{sid}" for sid in range(2, 17) if sid not in (1, 12)]
        self.pkl_files = [
            os.path.join(root, f"{subj}", f"{subj}.pkl") for subj in self.subject_ids
        ]
        self.pkl_files = [f for f in self.pkl_files if os.path.exists(f)]

        logger.info(f"Found {len(self.pkl_files)} WESAD pkl files.")

        super().__init__(root, tables or ["wesad"], dataset_name, config_path, dev)

    def load_data(self) -> pl.LazyFrame:
        """
        Loads and parses WESAD .pkl data into a LazyFrame for further processing.

        Returns:
            pl.LazyFrame: Lazy-loaded DataFrame containing sensor readings and labels.

        Description:
            For each .pkl file corresponding to a subject:
            - Load data with pickle
            - Extract labels and sensor signals (from chest and wrist)
            - Convert each time step into a data row with sensor values and timestamp
        """
        data_rows = []
        for pkl_path in self.pkl_files:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            
            # Accessing subject, label, and signal data
            subject = data.get("subject", "Unknown")
            label_array = data.get("label", [])
            signal = data.get("signal", {})

            # Access chest and wrist signal data safely
            chest_data = signal.get("chest", {})
            wrist_data = signal.get("wrist", {})

            sample_count = len(label_array)
            for i in range(sample_count):
                row = {
                    "patient_id": str(subject),
                    "timestamp": i,  # simple index-based timestamp
                    "event_type": "wesad",
                    "label": int(label_array[i]),
                }

                # Add features from chest data
                for key, arr in chest_data.items():
                    row[f"chest_{key}"] = arr[i] if len(arr) > i else np.nan

                # Add features from wrist data
                for key, arr in wrist_data.items():
                    row[f"wrist_{key}"] = arr[i] if len(arr) > i else np.nan

                data_rows.append(row)

        # Convert data into Polars LazyFrame
        df = pl.DataFrame(data_rows)
        return df.lazy()

    def preprocess_wesad(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Optional preprocessing if needed."""
        return df  # placeholder — extend with filtering/label mapping if needed