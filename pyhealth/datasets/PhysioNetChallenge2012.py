import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import polars as pl
from tqdm import tqdm
import datetime  # For time conversion

# --- Step 1: Adjust these imports based on your project structure ---
# Assuming BaseDataset and load_yaml_config are in the same directory for this example
# In a real PyHealth scenario, it might be:
# from pyhealth.datasets.base_dataset import BaseDataset
# from pyhealth.datasets.configs import load_yaml_config
from .base_dataset import BaseDataset  # If base_dataset.py is in the same package
from .configs import load_yaml_config  # If configs.py (containing load_yaml_config) is in the same package

# --- End of import adjustment section ---

logger = logging.getLogger(__name__)


class PhysioNet2012Dataset(BaseDataset):
    """
    Dataset class for "Predicting Mortality of ICU Patients:
    The PhysioNet/Computing in Cardiology Challenge 2012".

    Dataset website: https://physionet.org/content/challenge-2012/1.0.0/

    Args:
        root (str): Root directory of the dataset.
        subset (str): "train" or "test". Defaults to "train".
        dataset_name (Optional[str]): Defaults to "physionet_2012_mortality".
        config_path (Optional[str]): Path to YAML config. If None, attempts to load
                                     a default config named 'PhysioNetChallenge2012.yaml'
                                     from a 'configs' subdirectory relative to this file's location.
        dev (bool): Development mode.
    """

    def __init__(
            self,
            root: str,
            subset: str = "train",
            dataset_name: Optional[str] = None,
            config_path: Optional[str] = None,
            dev: bool = False,
    ):
        self.root_path = Path(root)
        self.subset = subset.lower()

        if dataset_name is None:
            dataset_name = "physionet_2012_mortality"

        # --- Step 2: Load and use the YAML configuration ---
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "PhysioNetChallenge2012.yaml"

        # self.config will be loaded by BaseDataset's __init__ using the provided or default config_path
        # super().__init__ will call load_yaml_config(config_path)
        # and store it in self.config.
        # Our overridden load_data() will then be called.
        # We can access self.config in our methods AFTER super().__init__()


        super().__init__(
            root=root,
            tables=[],  # We override load_data, so BaseDataset's table processing is bypassed for main data
            dataset_name=dataset_name,
            config_path=config_path,
            dev=dev,
        )

    def _get_patient_files_and_ids(self) -> List[Tuple[Path, str]]:
        """
        Gets list of patient .txt file paths and their corresponding RecordIDs for the subset.
        Uses values from the loaded YAML configuration if available.
        """
        data_dir_name = ""
        if self.subset == "train":
            # Use config value, fallback to hardcoded if not in config
            data_dir_name = "set-a"
        elif self.subset == "test":
            data_dir_name = "set-b"
        else:
            raise ValueError(f"Invalid subset '{self.subset}'. Choose 'train' or 'test'.")

        data_dir = self.root_path / data_dir_name
        if not data_dir.exists() or not data_dir.is_dir():
            # Fallback to old hardcoded path if config-derived path doesn't exist and config was minimal
            if data_dir_name in ["set-a", "set-b"] and not (
                    self.root_path / {"set-a": "set-a", "set-b": "set-b"}[data_dir_name]).exists():
                logger.error(f"Data directory '{data_dir}' (from config or default) not found.")
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
            elif data_dir_name not in ["set-a", "set-b"]:  # If config specified something else that wasn't found
                logger.error(f"Data directory '{data_dir}' (from config '{data_dir_name}') not found.")
                raise FileNotFoundError(f"Data directory not found: {data_dir}")
            # If we are here, it means the default set-a/set-b was used but not found.

        patient_files_with_ids = []
        for file_path in sorted(list(data_dir.glob("*.txt"))):
            record_id = file_path.stem
            patient_files_with_ids.append((file_path, record_id))

        if not patient_files_with_ids:
            logger.warning(f"No .txt files found in {data_dir}")
            return []

        if self.dev:
            logger.info(f"Dev mode: Limiting to first 10 patient files from {self.subset} set.")
            return patient_files_with_ids[:10]
        return patient_files_with_ids

    def _parse_time_to_datetime(self, time_str: str, base_date: datetime.datetime) -> Optional[datetime.datetime]:
        """Converts HH:MM string to a full datetime object by adding to a base_date."""
        if not time_str:
            return None
        try:
            # timestamp_format = self.patient_file_format_config.get("timestamp_format", "%H:%M")
            # For this dataset, it's fixed as HH:MM, so direct parsing is fine.
            # If format could vary, using strptime with format from config would be better.
            hours, minutes = map(int, time_str.split(':'))
            return base_date + datetime.timedelta(hours=hours, minutes=minutes)
        except ValueError:
            return None

    def _parse_patient_file(self, file_path: Path, patient_id: str) -> List[Dict]:
        """
        Parses a single patient's .txt file.
        Uses patient_file_format from config if available for header/column names.
        """
        events = []
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []

        # Get header info from config, fallback to default
        expected_header_list = ["Time", "Parameter", "Value"]
        expected_header_str_lower = ",".join(expected_header_list).lower()

        if not lines or lines[0].strip().lower() != expected_header_str_lower:
            logger.warning(
                f"File {file_path} does not have the expected header '{expected_header_str_lower}' or is empty. Found: '{lines[0].strip().lower() if lines else ''}'")
            return []

        arbitrary_base_date = datetime.datetime(1970, 1, 1, 0, 0, 0)
        # Column names from config or default
        time_col_name = expected_header_list[0]  # "Time"
        param_col_name = expected_header_list[1]  # "Parameter"
        value_col_name = expected_header_list[2]  # "Value"
        # patient_id_param_in_txt = self.patient_file_format_config.get("patient_id_parameter", "RecordID")

        for line_num, line_content in enumerate(lines[1:], start=2):
            try:
                # This assumes a simple 3-column CSV. A more robust CSV parser might be needed
                # if values themselves can contain commas (though not expected for this dataset).
                parts = line_content.strip().split(',', 2)
                if len(parts) < 3:
                    # logger.debug(f"Skipping malformed line {line_num} (not enough columns) in {file_path}: {line_content.strip()}")
                    continue
                time_str, parameter_str, value_str = parts

                timestamp = self._parse_time_to_datetime(time_str, arbitrary_base_date)
                if timestamp is None:
                    continue

                event_type_str = parameter_str.strip()
                # patient_id_param_in_txt is the value of "Parameter" that identifies the RecordID
                # e.g. if patient_file_format_config.patient_id_parameter is "RecordID"
                if event_type_str == "RecordID":
                    if patient_id != value_str.strip():
                        logger.warning(
                            f"RecordID mismatch in {file_path} (line {line_num}): file-derived ID {patient_id}, file content ID {value_str.strip()}")
                    continue

                try:
                    parsed_value = float(value_str)
                except ValueError:
                    parsed_value = value_str.strip()

                event_data = {
                    "patient_id": patient_id,
                    "event_type": event_type_str,
                    "timestamp": timestamp,
                    "value": parsed_value,
                }
                events.append(event_data)

            except ValueError:
                # logger.debug(f"Skipping malformed line {line_num} in {file_path}: {line_content.strip()}")
                continue
        return events

    def load_data(self) -> pl.LazyFrame:
        logger.info(f"Starting PhysioNet 2012 data loading for subset: '{self.subset}' from root '{self.root_path}'")
        patient_files_with_ids = self._get_patient_files_and_ids()

        schema = {
            "patient_id": pl.Utf8,
            "event_type": pl.Utf8,
            "timestamp": pl.Datetime,
            "value": pl.Float64,  # Try to cast to float, non-castable become null
        }

        if not patient_files_with_ids:
            logger.warning(f"No patient files found for subset '{self.subset}'. Returning empty LazyFrame with schema.")
            return pl.LazyFrame(schema=schema)

        all_events_list = []
        for file_path, patient_id in tqdm(patient_files_with_ids, desc=f"Parsing {self.subset} patient files"):
            patient_events = self._parse_patient_file(file_path, patient_id)
            all_events_list.extend(patient_events)

        if not all_events_list:
            logger.warning("No events parsed. Returning empty LazyFrame with schema.")
            return pl.LazyFrame(schema=schema)

        global_event_df = pl.DataFrame(all_events_list)

        try:
            global_event_df = global_event_df.with_columns([
                pl.col("patient_id").cast(pl.Utf8),
                pl.col("event_type").cast(pl.Utf8),
                pl.col("timestamp").cast(pl.Datetime),
                pl.col("value").cast(pl.Float64, strict=False),
            ])
        except pl.SchemaError as e:
            logger.error(
                f"Schema error during final casting: {e}. DF columns: {global_event_df.columns}, Head:\n{global_event_df.head(3)}")
            return pl.LazyFrame(schema=schema)

        if global_event_df.height == 0:
            logger.warning("DataFrame empty after processing. Returning empty LazyFrame with schema.")
            return pl.LazyFrame(schema=schema)

        logger.info(
            f"Successfully loaded {global_event_df.height} events for {global_event_df.get_column('patient_id').n_unique()} patients in subset '{self.subset}'.")
        return global_event_df.lazy()

    def _load_outcomes_df(self) -> Optional[pl.DataFrame]:
        outcome_file_name = ""
        # Use config, fallback to hardcoded if not in config
        if self.subset == "train":
            outcome_file_name = self.data_layout_config.get("train_outcome_file", "Outcomes-a.txt")
        elif self.subset == "test":
            # outcome_file_name = self.data_layout_config.get("test_outcome_file", "") # e.g. "Outcomes-b.txt"
            # if not outcome_file_name:
            logger.info(f"Outcome file for subset '{self.subset}' is not loaded by default for training purposes.")
            return None
        else:
            logger.warning(f"No outcome file defined for subset '{self.subset}'.")
            return None

        outcome_file_path = self.root_path / outcome_file_name
        if not outcome_file_path.is_file():
            logger.warning(f"Outcome file '{outcome_file_path}' (from config or default) not found.")
            return None

        try:
            df_outcomes = pl.read_csv(outcome_file_path)
            # Use config for column names, fallback to default
            patient_id_col = self.outcome_file_format_config.get("patient_id_column", "RecordID")
            label_col = self.outcome_file_format_config.get("label_column", "In-hospital_death")

            df_outcomes = df_outcomes.select([
                pl.col(patient_id_col).cast(pl.Utf8).alias("patient_id"),
                pl.col(label_col).cast(pl.Int32).alias("label")
            ])
            logger.info(f"Loaded outcomes from {outcome_file_path} for {df_outcomes.height} records.")
            return df_outcomes
        except Exception as e:
            logger.error(f"Error reading or processing outcome file {outcome_file_path}: {e}")
            return None

    @property
    def default_task(self):
        # from ..tasks import YourMortalityPredictionTask # Example
        # return YourMortalityPredictionTask()
        logger.warning("Default task not yet implemented for PhysioNet2012Dataset.")
        return None