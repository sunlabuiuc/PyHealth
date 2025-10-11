"""
Authors: Asmita Chihnara (asmitac2) and Tithi Sreemany (tithis2)

Dataset: ICU Mortality Prediction Dataset (PhysioNet Challenge 2012)
Description:
    This dataset contains ICU patient records from the PhysioNet Challenge 2012,
    designed for predicting in-hospital mortality. The data includes:
    - Demographic information (age, gender, height, weight, ICU type)
    - Time-series measurements (vital signs, lab values)
    - Outcome information (in-hospital mortality)

    The dataset is organized into three sets (set-a, set-b, set-c) with corresponding
    outcome files. Each set contains patient records in individual text files, with
    measurements recorded at various time points during the ICU stay.
"""

import os
import logging
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import polars as pl
from tqdm import tqdm

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseDataset
from pyhealth.datasets.utils import strptime, padyear

logger = logging.getLogger(__name__)


class ICUMortalityDataset(BaseDataset):
    """Dataset for ICU Mortality Prediction using PhysioNet Challenge 2012 data.

    This dataset contains ICU patient records used for predicting in-hospital mortality.
    The data is sourced from the PhysioNet Challenge 2012 dataset, available at
    https://physionet.org/content/challenge-2012/1.0.0/.

    The dataset consists of multiple text files containing time-series data for
    each patient. Each file contains:
        - Demographic information (Age, Gender, Height, Weight, ICUType)
        - Time-series measurements (vital signs, lab values)
        - Outcome information (in-hospital mortality)

    The data is organized into three sets (set-a, set-b, set-c) with corresponding
    outcome files (Outcomes-a.txt, Outcomes-b.txt, Outcomes-c.txt). Each set can be
    loaded using a specific configuration file.

    Args:
        root: Root directory of the raw data (should contain set-a, set-b, set-c directories).
        tables: List of tables to be loaded. Default is ["patients"].
        dataset_name: Name of the dataset. Default is "icumortality".
        config_path: Path to the configuration file. If None, uses default config.
        dev: Whether to enable dev mode (only use a small subset of the data).
            Default is False.

    Examples:
        >>> from pyhealth.datasets import ICUMortalityDataset
        >>> # Load set-a with default config
        >>> dataset = ICUMortalityDataset(root="temp_data/challenge-2012")
        >>> # Load set-b with custom config
        >>> dataset = ICUMortalityDataset(
        ...     root="temp_data/challenge-2012",
        ...     config_path="pyhealth/datasets/configs/icumortality.yaml"
        ... )
        >>> # Load in dev mode (first 100 patients)
        >>> dataset = ICUMortalityDataset(root="temp_data/challenge-2012", dev=True)
        >>> # Get dataset statistics
        >>> dataset.stat()
        >>> # Print dataset information
        >>> dataset.info()
    """

    def __init__(
        self,
        root: str,
        tables: List[str] = ["patients"],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
        **kwargs,
    ):
        """Initialize the ICU Mortality dataset.

        Args:
            root: Root directory of the raw data.
            tables: List of tables to be loaded.
            dataset_name: Name of the dataset.
            config_path: Path to the configuration file.
            dev: Whether to enable dev mode.
            **kwargs: Additional arguments passed to the base class.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "icumortality.yaml"
            )
            logger.info(f"Using default config: {config_path}")

        # Initialize patient data storage
        self._patients: Dict[str, Dict] = {}

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "icumortality",
            config_path=config_path,
            dev=dev,
            **kwargs,
        )

        # Load patient data
        self.load_patient_data()

    def load_patient_data(self):
        """Load patient data from individual text files.

        This method:
        1. Reads the outcomes file specified in the config
        2. Processes each patient file in the specified set directory
        3. Extracts demographic information and time-series measurements
        4. Stores the data in the self._patients dictionary

        The data is loaded from the set directory specified in the config file
        (e.g., set-a, set-b, set-c). In dev mode, only the first 100 patients
        are processed.
        """
        # Get set name from config
        set_name = self.config.tables["patients"].file_path
        outcomes_file = self.config.tables["patients"].join[0].file_path

        # Read outcomes file
        outcomes_df = pd.read_csv(
            os.path.join(self.root, outcomes_file), sep=",", dtype={"RecordID": str}
        )
        outcomes_dict = dict(
            zip(outcomes_df["RecordID"], outcomes_df["In-hospital_death"])
        )

        # Process each patient file
        set_dir = os.path.join(self.root, set_name)
        patient_files = os.listdir(set_dir)
        if self.dev:
            patient_files = patient_files[:100]  # Use only first 100 files in dev mode

        for patient_file in tqdm(
            patient_files, desc=f"Parsing patients from {set_name}"
        ):
            if not patient_file.endswith(".txt"):
                continue

            patient_id = patient_file.split(".")[0]
            file_path = os.path.join(set_dir, patient_file)

            # Read patient data
            df = pd.read_csv(file_path, sep=",")

            # Get demographic information from first row
            first_row = df.iloc[0]
            patient_data = {
                "patient_id": patient_id,
                "gender": first_row.get("Gender", None),
                "age": first_row.get("Age", None),
                "height": first_row.get("Height", None),
                "weight": first_row.get("Weight", None),
                "icu_type": first_row.get("ICUType", None),
                "mortality": outcomes_dict.get(patient_id, 0),
                "measurements": [],
            }

            # Process time-series measurements
            for _, row in df.iterrows():
                if row["Parameter"] not in [
                    "RecordID",
                    "Age",
                    "Gender",
                    "Height",
                    "Weight",
                    "ICUType",
                ]:
                    measurement = {
                        "code": row["Parameter"],
                        "time": row["Time"],
                        "value": row["Value"],
                    }
                    patient_data["measurements"].append(measurement)

            # Add patient data
            self._patients[patient_id] = patient_data

    def load_data(self) -> pl.LazyFrame:
        """Convert patient data to a polars LazyFrame format.

        This method:
        1. Converts the patient data into a format suitable for polars
        2. Creates a DataFrame with columns: patient_id, event_type, timestamp, code, value
        3. Returns a LazyFrame for efficient processing

        Returns:
            pl.LazyFrame: A lazy DataFrame containing all patient measurements.
        """
        # Convert patient data to a format suitable for polars
        records = []
        for patient_id, patient_data in self._patients.items():
            for measurement in patient_data["measurements"]:
                # Convert time string to datetime
                time_parts = measurement["time"].split(":")
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                timestamp = datetime(2012, 1, 1) + timedelta(
                    hours=hours, minutes=minutes
                )

                record = {
                    "patient_id": patient_id,
                    "event_type": "measurement",
                    "timestamp": timestamp,
                    "code": measurement["code"],
                    "value": measurement["value"],
                }
                records.append(record)

        # Create a polars DataFrame
        if records:
            # Create a DataFrame and convert it to lazy
            df = pl.DataFrame(records)
            logger.info(f"Created DataFrame with {len(df)} records")
            return df.lazy()
        else:
            # Create an empty DataFrame with the correct schema
            logger.warning("No records found, creating empty DataFrame")
            return pl.DataFrame(
                {
                    "patient_id": pl.Series([], dtype=pl.Utf8),
                    "event_type": pl.Series([], dtype=pl.Utf8),
                    "timestamp": pl.Series([], dtype=pl.Datetime),
                    "code": pl.Series([], dtype=pl.Utf8),
                    "value": pl.Series([], dtype=pl.Utf8),
                }
            ).lazy()

    @property
    def patients(self) -> Dict[str, Dict]:
        """Return the patients dictionary.

        Returns:
            Dict[str, Dict]: A dictionary mapping patient IDs to their data.
        """
        return self._patients

    def stat(self) -> Dict[str, int]:
        """Returns basic statistics of the dataset.

        The statistics include:
            - Number of patients
            - Number of measurements
            - Number of deaths
            - Mortality rate

        Returns:
            Dict[str, int]: A dictionary containing the dataset statistics.
        """
        num_patients = len(self.patients)
        if num_patients == 0:
            return {
                "# patients": 0,
                "# measurements": 0,
                "# deaths": 0,
                "mortality rate": "0.00%",
            }

        num_measurements = sum(
            len(patient["measurements"]) for patient in self.patients.values()
        )
        num_deaths = sum(patient["mortality"] for patient in self.patients.values())
        return {
            "# patients": num_patients,
            "# measurements": num_measurements,
            "# deaths": num_deaths,
            "mortality rate": f"{num_deaths/num_patients*100:.2f}%",
        }

    def info(self) -> None:
        """Prints basic information about the dataset.

        This method prints:
            - Dataset name
            - Root directory
            - Number of patients
            - Number of measurements
            - Number of deaths
            - Mortality rate
        """
        stats = self.stat()
        print(f"Dataset: ICUMortality")
        print(f"Root: {self.root}")
        print(f"Number of patients: {stats['# patients']}")
        print(f"Number of measurements: {stats['# measurements']}")
        print(f"Number of deaths: {stats['# deaths']}")
        print(f"Mortality rate: {stats['mortality rate']}")


if __name__ == "__main__":
    dataset = ICUMortalityDataset(
        root="temp_data/challenge-2012",
        dev=True,
    )
    dataset.stat()
    dataset.info()
