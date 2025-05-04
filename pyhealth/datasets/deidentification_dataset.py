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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_memory_usage(tag=""):
    """Log current memory usage."""
    if psutil is not None:
        try:
            process = psutil.Process(os.getpid())  # Get the current process
            mem_info = process.memory_info()  # Retrieve memory info
            # Log the memory usage in MB
            logger.info(f"Memory usage {tag}: {mem_info.rss / (1024 * 1024):.1f} MB")
        except Exception as e:
            logger.warning(f"Memory logging failed at {tag}: {e}")
    else:
        logger.warning(f"psutil not available. Unable to log memory usage at {tag}")

class DeidentificationDataset(BaseDataset):
    def __init__(self, config: dict, dev: bool = False):
        self.config = config
        self.file_path = config.get('file_path')
        self.patient_id = config.get('patient_id')
        self.timestamp = config.get('timestamp')
        self.attributes = config.get('attributes', [])
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

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found at: {self.file_path}")

        try:
            with open(self.file_path, 'r') as file:
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

        # Validate essential columns
        required_columns = [self.patient_id, 'text']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        return df

    def get_patient_data(self, patient_id: str) -> pd.DataFrame:
        """Retrieve data for a specific patient ID."""
        subset = self.data[self.data[self.patient_id] == patient_id]
        if subset.empty:
            logger.warning(f"No records found for patient ID: {patient_id}")
        return subset

    def preprocess_data(self):
        """Clean and normalize the 'text' column."""
        if 'text' not in self.data.columns:
            raise ValueError("Missing 'text' column in dataset.")

        try:
            self.data['processed_text'] = (
                self.data['text']
                .astype(str)
                .str.lower()
                .str.replace('\n', ' ', regex=False)
                .str.strip()
            )
        except Exception as e:
            raise ValueError(f"Failed during text preprocessing: {e}")

# Usage Example
if __name__ == "__main__":
    dataset_config = {
        'table_name': 'discharge_summaries',
        'file_path': 'data/deid_raw/discharge/discharge_summaries.json',
        'patient_id': 'document_id',
        'timestamp': 'discharge_date',
        'attributes': [
            'document_id', 'text', 'patient_name', 'dob', 'age', 'sex', 'service',
            'chief_complaint', 'diagnosis', 'treatment', 'follow_up_plan', 'discharge_date', 'attending_physician'
        ]
    }

    try:
        dataset = DeidentificationDataset(dataset_config, dev=True)
        dataset.preprocess_data()
        print(f"First record:\n{dataset.data.iloc[0]}")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
