"""
Name: Ankit Pasi
NetId: pasi2
Paper Title: Interpretation of Intracardiac Electrograms Through Textual Representations
Paper Link: https://arxiv.org/abs/2402.01115

Description:
This script defines the EGMDataset class for loading and processing the
PhysioNet Intracardiac Atrial Fibrillation Database within the PyHealth framework.
It reads WFDB EGM signals, performs Z-score normalization, segments the signals
into fixed-length (potentially overlapping) windows, and assigns an AFib label
based on record comments (defaulting to 1 if indeterminate). Each signal segment
is treated as an event and organized into a Polars DataFrame following the
pyhealth.datasets.BaseDataset structure, storing the raw signal segment as an
object within the DataFrame. The script includes an example usage block/test case (`if __name__ == "__main__":`)
that can run with either user-provided data (via --data-root) or auto-generated
dummy data.
"""
import os
import wfdb
import numpy as np
import polars as pl
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
from pyhealth.datasets import BaseDataset
from pyhealth.datasets.utils import MODULE_CACHE_PATH
import logging
import argparse # Import argparse

logger = logging.getLogger(__name__)

# Define constants for signal processing
DEFAULT_SEGMENT_LENGTH = 1000
DEFAULT_STEP_SIZE = None # No overlap by default

class EGMDataset(BaseDataset):
    """
    PyHealth dataset class for the Intracardiac Atrial Fibrillation Database.

    This dataset processes EGM signals from the PhysioNet database.
    Each catheter placement is treated as a patient.
    Each segment of the signal is treated as a single event ('EGM_Segment').
    The raw signal data for each segment is stored within the event attributes.

    Args:
        root: Root directory of the dataset. Data should be pre-downloaded and extracted here.
        dataset_name: Name of the dataset.
        segment_length: Length of each signal segment.
        step_size: Step size for creating overlapping segments. If None, segments are non-overlapping.
    """

    # Updated __init__ - Avoids super().__init__ due to constraints
    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = "egm_intra",
        segment_length: int = DEFAULT_SEGMENT_LENGTH,
        step_size: Optional[int] = DEFAULT_STEP_SIZE,
    ):
        # Manually replicate necessary BaseDataset initialization steps
        self.root = root
        self.tables = [] # EGMDataset doesn't load standard tables
        self.dataset_name = dataset_name or self.__class__.__name__
        self.config = None # EGMDataset doesn't use a config file

        # Set specific attributes for EGMDataset
        self.segment_length = segment_length
        self.step_size = step_size

        # Manually log initialization message (similar to BaseDataset)
        logger.info(
            f"Initializing {self.dataset_name} dataset from {self.root}"
        )

        # Manually call load_data (as BaseDataset.__init__ would)
        self.global_event_df = self.load_data()

        # Manually initialize cached attributes (from BaseDataset)
        self._collected_global_event_df = None
        self._unique_patient_ids = None

        # Note: We have skipped calling super().__init__ to avoid the
        # error caused by load_yaml_config(None) when config_path is None,
        # adhering to the constraint of not modifying base_dataset.py.

    def _read_wfdb_record(self, record_path: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Reads a single WFDB record."""
        try:
            record = wfdb.rdrecord(record_path)
            # Assuming EGM signals start from the 4th channel (index 3)
            # as per the original preprocess_intra.py script
            if record.p_signal.shape[1] > 3:
                 egm_signals = record.p_signal[:, 3:]
                 return egm_signals, record.comments
            else:
                logger.warning(f"Record {record_path} has fewer than 4 channels. Skipping EGM signals.")
                return None, None
        except Exception as e:
            logger.error(f"Error reading WFDB record {record_path}: {e}")
            return None, None

    def _z_score_normalization(self, data: np.ndarray) -> np.ndarray:
        """Applies z-score normalization to the signal data."""
        mean_val = np.mean(data, axis=0, keepdims=True) # Normalize along time axis
        std_val = np.std(data, axis=0, keepdims=True)
        std_val = np.maximum(std_val, 1e-10)  # Avoid division by zero
        normalized_data = (data - mean_val) / std_val
        normalized_data = np.nan_to_num(normalized_data, nan=0.0) # Handle potential NaNs
        return normalized_data

    def _segment_signal(self, data: np.ndarray, segment_length: int, step_size: Optional[int]) -> Optional[np.ndarray]:
        """Segments the signal data."""
        n_time_points = data.shape[0]

        if n_time_points < segment_length:
            logger.warning(f"Signal length ({n_time_points}) is shorter than segment length ({segment_length}). Skipping segmentation.")
            return None # Cannot segment if signal is too short

        if step_size is not None:
            if step_size <= 0:
                 logger.error("Step size must be positive.")
                 return None
            n_segments = 1 + (n_time_points - segment_length) // step_size
        else: # No overlap
            step_size = segment_length
            n_segments = n_time_points // segment_length

        if n_segments <= 0:
            logger.warning(f"No full segments possible with length {segment_length} and step {step_size}. Signal length {n_time_points}.")
            return None

        # Pre-allocate array: shape (n_segments, segment_length, n_channels)
        segmented_data = np.zeros((n_segments, segment_length, data.shape[1]))

        for i in range(n_segments):
            start_idx = i * step_size
            end_idx = start_idx + segment_length
            # Ensure we don't exceed bounds (although logic above should prevent this)
            if end_idx > n_time_points:
                 logger.warning(f"Calculated end index {end_idx} exceeds data length {n_time_points}. Adjusting last segment.")
                 end_idx = n_time_points
                 start_idx = end_idx - segment_length # Ensure last segment has correct length

            segmented_data[i] = data[start_idx:end_idx, :]

        return segmented_data

    def load_data(self) -> pl.LazyFrame:
        """
        Loads and processes the WFDB records into a Polars LazyFrame.

        Overrides the BaseDataset.load_data method.

        Returns:
            A Polars LazyFrame where each row represents an EGM segment event.
            Columns include 'patient_id', 'event_type', 'timestamp' (segment index),
            'label', 'signal_raw', and 'signal_channels'.
        """
        # Locate the data directory
        data_dir = self.root
        extracted_folder_name = "intracardiac-atrial-fibrillation-database-1.0.0"
        potential_data_path = os.path.join(self.root, extracted_folder_name)

        if os.path.isdir(potential_data_path) and any(f.endswith(".hea") for f in os.listdir(potential_data_path)):
             data_dir = potential_data_path
             logger.info(f"Found data directory: {data_dir}")
        elif not any(f.endswith(".hea") for f in os.listdir(self.root)):
             # In BaseDataset, data loading errors should raise exceptions.
             # Automatic download logic is not implemented here.
             raise FileNotFoundError(
                 f"Could not find .hea files in {self.root} or {potential_data_path}. "
                 f"Please ensure the data is downloaded and extracted correctly into {potential_data_path}."
             )
        else:
             logger.info(f"Using root directory for data: {self.root}")

        header_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".hea")])
        # Note: dev mode filtering is not handled by BaseDataset __init__.
        # Implement here if needed, or handle outside the class.
        # if self.dev: # 'dev' is not an attribute anymore
        #    header_files = header_files[:5]

        all_segment_events = [] # List to hold segment dictionaries

        logger.info(f"Processing {len(header_files)} records...")
        for header_file in tqdm(header_files, desc="Processing EGM Records"):
            record_name = header_file.split('.')[0]
            record_path = os.path.join(data_dir, record_name)

            egm_signals, comments = self._read_wfdb_record(record_path)
            if egm_signals is None:
                continue

            normalized_signals = self._z_score_normalization(egm_signals)

            segmented_signals = self._segment_signal(
                normalized_signals,
                self.segment_length,
                self.step_size
            )

            if segmented_signals is None:
                continue

            # Determine AFib label (remains the same logic)
            afib_label = 1
            if comments:
                 if any("AF" in c.upper() or "FIBRILLATION" in c.upper() for c in comments):
                     afib_label = 1
                 elif any("SR" in c.upper() or "SINUS" in c.upper() for c in comments):
                     afib_label = 0
                 else:
                     logger.warning(f"Could not determine AFib label for {record_name} from comments: {comments}. Defaulting to {afib_label}.")

            patient_id = record_name

            # Each segment becomes an event
            for segment_idx in range(segmented_signals.shape[0]):
                segment_data = segmented_signals[segment_idx] # Shape: (segment_length, n_channels)
                channels = [f"EGM_{j}" for j in range(segment_data.shape[1])]

                # Create the event dictionary matching BaseDataset's expected structure (loosely)
                # We add specific fields like 'signal_raw' and 'label'
                event_dict = {
                    "patient_id": patient_id,
                    # Using segment index as a pseudo-timestamp. BaseDataset expects datetime,
                    # but we'll store as int here. Might need adjustment depending on task usage.
                    "timestamp": segment_idx,
                    "event_type": "EGM_Segment", # Define a type for these events
                    "label": afib_label,
                    # Store the raw signal data and channel names.
                    # Storing NumPy arrays directly might be inefficient or problematic in Polars.
                    # Using dtype=object might be necessary.
                    "signal_raw": segment_data,
                    "signal_channels": channels,
                }
                all_segment_events.append(event_dict)

        if not all_segment_events:
             logger.warning("No segments were generated. Returning empty DataFrame.")
             # Define schema for empty frame to avoid errors downstream
             schema = {
                 "patient_id": pl.Utf8,
                 "timestamp": pl.Int64, # Changed from datetime
                 "event_type": pl.Utf8,
                 "label": pl.Int64,
                 "signal_raw": pl.Object, # Use Object for NumPy arrays
                 "signal_channels": pl.List(pl.Utf8)
             }
             return pl.DataFrame(schema=schema).lazy()


        # Convert list of dictionaries to Polars DataFrame
        # Explicitly define schema, especially for the NumPy array column
        try:
             df = pl.DataFrame(all_segment_events, schema={
                 "patient_id": pl.Utf8,
                 "timestamp": pl.Int64, # Using Int64 for segment index
                 "event_type": pl.Utf8,
                 "label": pl.Int64, # Assuming label is integer
                 "signal_raw": pl.Object, # Critical: Use Object type for NumPy arrays
                 "signal_channels": pl.List(pl.Utf8)
             })
        except Exception as e:
            logger.error(f"Failed to create Polars DataFrame: {e}")
            # Consider how to handle this - maybe return an empty frame or re-raise
            raise e

        # Return as LazyFrame as expected by BaseDataset structure
        return df.lazy()


# Example Usage (Optional: for testing the dataset class directly)
if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Argument Parsing --- #
    parser = argparse.ArgumentParser(description="Load and test the EGMDataset.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional path to the root directory containing the extracted dataset "
             "(e.g., 'path/to/intracardiac-atrial-fibrillation-database-1.0.0'). "
             "If not provided, a dummy test dataset will be created."
    )
    args = parser.parse_args()

    # --- Determine Dataset Root and Prepare Data --- #
    if args.data_root:
        # Use the provided data root
        dataset_root = args.data_root
        # Check if the provided path looks like the expected extracted folder
        if not os.path.basename(dataset_root) == "intracardiac-atrial-fibrillation-database-1.0.0":
             potential_data_path = os.path.join(dataset_root, "intracardiac-atrial-fibrillation-database-1.0.0")
             if os.path.isdir(potential_data_path):
                 logger.warning(f"--data-root provided is '{dataset_root}'. "
                                f"Using subdirectory: '{potential_data_path}'")
                 # dataset_root is already correctly set by load_data logic inside EGMDataset if parent dir is passed
             else:
                 logger.warning(f"Provided --data-root '{dataset_root}' does not look like the expected extracted folder name. "
                                f"Ensure .hea/.dat files are directly within this directory or a subdirectory named "
                                f"'intracardiac-atrial-fibrillation-database-1.0.0'.")
        print(f"Using provided data root: {dataset_root}")
    else:
        # Default to creating and using a dummy test directory
        dataset_root = os.path.join(MODULE_CACHE_PATH, "data", "egm_intra_test")
        dummy_extract_path = os.path.join(dataset_root, "intracardiac-atrial-fibrillation-database-1.0.0")
        os.makedirs(dummy_extract_path, exist_ok=True)
        print(f"--data-root not provided. Using dummy test directory: {dataset_root}")
        print(f"Expected dummy data path: {dummy_extract_path}")

        # Create dummy WFDB files ONLY if using the default test directory
        try:
            print("Creating dummy WFDB files for testing...")
            # Create a small dummy signal
            dummy_signal = np.random.randn(5000, 8) * 10 # 5000 samples, 8 channels
            fs = 1000 # Dummy sampling frequency

            # Write dummy record 1 (simulate AFib)
            wfdb.wrsamp(
                record_name='iaf01_test',
                fs=fs,
                units=['mV'] * 8,
                sig_name=[f'CH{i+1}' for i in range(8)],
                p_signal=dummy_signal,
                fmt=['16'] * 8,
                comments=['Diagnosis: AF'],
                write_dir=dummy_extract_path
            )
            print(f"Created dummy file: {os.path.join(dummy_extract_path, 'iaf01_test.hea')}")

            # Write dummy record 2 (simulate Sinus Rhythm)
            dummy_signal_sr = np.sin(np.linspace(0, 100, 5000))[:, np.newaxis] * 5
            dummy_signal_sr = np.hstack([dummy_signal_sr + np.random.randn(5000, 1) * 0.5] * 8)
            wfdb.wrsamp(
                record_name='iaf02_test',
                fs=fs,
                units=['mV'] * 8,
                sig_name=[f'CH{i+1}' for i in range(8)],
                p_signal=dummy_signal_sr,
                fmt=['16'] * 8,
                comments=['Diagnosis: Sinus Rhythm'],
                write_dir=dummy_extract_path
            )
            print(f"Created dummy file: {os.path.join(dummy_extract_path, 'iaf02_test.hea')}")
            print("Dummy files created.")

        except Exception as e:
            print(f"Could not create dummy WFDB files: {e}. "
                  "Make sure 'wfdb' is installed and you have write permissions.")
            print("Skipping dataset instantiation test.")
            exit() # Exit if we can't create test files


    # --- Instantiate and Test the Dataset --- #
    # Use the determined dataset_root
    try:
        egm_dataset = EGMDataset(
            root=dataset_root,
            segment_length=1000,
            step_size=500
        )

        # Access some data using BaseDataset properties/methods
        print(f"\nDataset loaded successfully!")
        print(f"Dataset Name: {egm_dataset.dataset_name}")
        print(f"Number of unique patients (records): {len(egm_dataset.unique_patient_ids)}")

        # Access the global event dataframe (which now contains segments)
        global_df = egm_dataset.collected_global_event_df
        print(f"Total number of events (segments): {global_df.height}")
        if global_df.height > 0:
             print(f"Schema of global event DataFrame: {global_df.schema}")
             print("\nFirst 5 events (segments):")
             print(global_df.head())

             # Example: Get data for the first patient
             first_patient_id = egm_dataset.unique_patient_ids[0]
             print(f"\nGetting patient object for patient_id '{first_patient_id}'...")
             patient_obj = egm_dataset.get_patient(first_patient_id)
             print(f"Patient object type: {type(patient_obj)}")
             # Use DataFrame height to get number of events
             print(f"Number of events (segments) for patient '{first_patient_id}': {patient_obj.data_source.height}")

             if patient_obj.data_source.height > 0:
                 # Access the patient's data (which is a filtered DataFrame)
                 patient_df = patient_obj.data_source # Access the DataFrame directly
                 print(f"\nFirst event for patient '{first_patient_id}':")
                 first_event_data = patient_df.row(0, named=True) # Get first row as dict
                 print(f"  Event Type: {first_event_data['event_type']}")
                 print(f"  Timestamp (Segment Index): {first_event_data['timestamp']}")
                 print(f"  Label: {first_event_data['label']}")
                 # Accessing the signal data stored as object
                 signal_data = first_event_data['signal_raw']
                 if isinstance(signal_data, np.ndarray):
                     print(f"  Signal shape: {signal_data.shape}")
                     print(f"  Signal channels: {first_event_data['signal_channels']}")
                     print(f"  Signal sample (first 5 values, first channel): {signal_data[:5, 0]}")
                 else:
                     print(f"  Signal data type: {type(signal_data)}")

        # Example: Get a sample using __getitem__ (inherited from BaseDataset -> provides patient by index)
        # print(f"\nGetting item at index 0 (patient '{egm_dataset.unique_patient_ids[0]}'):")
        # sample_data = egm_dataset[0] # Gets Patient object for the first patient_id
        # print(f"Type of sample_data: {type(sample_data)}")
        # print(f"Number of events in sample: {sample_data.num_events}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the dataset is correctly placed in the specified root directory.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during dataset processing: {e}")

    # Clean up dummy files/directory (optional)
    # Only attempt cleanup if we used the default dummy path
    if not args.data_root:
        try:
            import shutil
            shutil.rmtree(dataset_root) # Use dataset_root which is the test_root here
            print(f"\nCleaned up dummy test directory: {dataset_root}")
        except Exception as e:
            print(f"\nWarning: Could not clean up dummy test directory {dataset_root}: {e}") 