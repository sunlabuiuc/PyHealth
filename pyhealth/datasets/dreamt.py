import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

#: DREAMT sleep-stage strings mapped to canonical five-class integers.
# Original stage map: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/dataset_preparation/dreamt/dreamt_extract_ibi_se.py#L42
DREAMT_STAGE_MAPPING = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
    "Missing": -1,
}


def map_dreamt_stages(values) -> np.ndarray:
    """Maps DREAMT sleep stages to canonical integers.

    Args:
        values: Iterable of stage strings or already-numeric stages.

    Returns:
        ``int64`` array. ``"Missing"``, unrecognized tokens such as ``"P"``,
        and out-of-range numbers become ``-1``.

    Examples:
        >>> map_dreamt_stages(["W", "N3", "Missing", "P"]).tolist()
        [0, 3, -1, -1]
    """
    values = list(values)
    if values and isinstance(values[0], str):
        # Original unknown-stage fallback: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/dataset_preparation/dreamt/dreamt_extract_ibi_se.py#L43
        return np.asarray(
            [DREAMT_STAGE_MAPPING.get(str(v).strip(), -1) for v in values],
            dtype=np.int64,
        )
    return np.asarray(values, dtype=np.int64)


def load_dreamt_frame(
    file_path: str,
    # Original wearable channel: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/dataset_preparation/dreamt/dreamt_extract_ibi_se.py#L183
    signal_cols: tuple[str, ...] = ("BVP",),
    epoch_duration: float = 30.0,
) -> pd.DataFrame:
    """Reads the signal and stage columns of one DREAMT recording.

    The frame is returned at its native sampling rate; epoch construction and
    IBI extraction belong to the task layer. Non-finite signal samples are
    preserved so the task can interpolate them consistently.

    Args:
        file_path: Path to ``*_whole_df.csv`` or ``*_PSG_df.csv``.
        signal_cols: Waveform columns to read. Defaults to ``BVP``, the
            reference DREAMT wearable channel.
        epoch_duration: Epoch length in seconds, used only for validation.

    Returns:
        DataFrame containing the requested signal columns and a canonical
        integer ``Sleep_Stage`` column.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If no signal column or no stage column is present, or if
            ``epoch_duration`` is not positive.

    Examples:
        >>> frame = load_dreamt_frame("S001_PSG_df.csv", ("BVP",))
    """
    if epoch_duration <= 0:
        raise ValueError("epoch_duration must be positive")
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    header = pd.read_csv(path, nrows=0)
    wanted = [col for col in signal_cols if col in header.columns]
    if not wanted:
        raise ValueError(f"none of {signal_cols} found in {path}")
    if "Sleep_Stage" not in header.columns:
        raise ValueError(f"Sleep_Stage not found in {path}")
    frame = pd.read_csv(path, usecols=[*wanted, "Sleep_Stage"])
    for col in wanted:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float64)
    # Original Sleep_Stage column: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/dataset_preparation/dreamt/dreamt_extract_ibi_se.py#L35
    frame["Sleep_Stage"] = map_dreamt_stages(frame["Sleep_Stage"].values)
    return frame


class DREAMTDataset(BaseDataset):
    """
    Base Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology (DREAMT)

    Dataset accepts current versions of DREAMT (1.0.0, 1.0.1, 2.0.0, 2.1.0), available at:
    https://physionet.org/content/dreamt/

    DREAMT includes wrist-based wearable and polysomnography (PSG) sleep data from 100 participants
    recruited from the Duke University Health System (DUHS) Sleep Disorder Lab. This includes 
    wearable signals, PSG signals, sleep labels, and clinical data related to sleep health and disorders. 

    Citations:
    ---------
    When using this dataset, please cite:

    Wang, K., Yang, J., Shetty, A., & Dunn, J. (2025). DREAMT: Dataset for Real-time sleep stage EstimAtion 
    using Multisensor wearable Technology (version 2.1.0). PhysioNet. RRID:SCR_007345. 
    https://doi.org/10.13026/7r9r-7r24

    Will Ke Wang, Jiamu Yang, Leeor Hershkovich, Hayoung Jeong, Bill Chen, Karnika Singh, Ali R Roghanizad, 
    Md Mobashir Hasan Shandhi, Andrew R Spector, Jessilyn Dunn. (2024). Proceedings of the fifth 
    Conference on Health, Inference, and Learning, PMLR 248:380-396.

    Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). 
    PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex 
    physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

    Note: 
    ---------
    Dataset follows file and folder structure of dataset version, looks for participant_info.csv and data folders, 
    so root path should be version downloaded, example: root = ".../dreamt/2.1.0/" (or an older version folder such as ".../dreamt/1.0.0/")

    Getting the data:
        - Source: https://physionet.org/content/dreamt/
        - Access: restricted. Create a PhysioNet account and sign the data use
          agreement on the dataset page (no credentialing or CITI course needed).
        - Download: ``wget -r -N -c -np --user <username> --ask-password
          https://physionet.org/files/dreamt/2.1.0/``
        - ``root``: the version folder of that download (``.../dreamt/2.1.0/``;
          the older versions listed above also work).

    Args:
        root: root directory containing the dataset files
        dataset_name: optional name of dataset, defaults to "dreamt_sleep"
        config_path: optional configuration file, defaults to "dreamt.yaml"
    
    Attributes:
        root: root directory containing the dataset files
        dataset_name: name of dataset
        config_path: path to configuration file

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> dataset = DREAMTDataset(root = "/path/to/dreamt/data/version")
        >>> dataset.stats()
        >>>
        >>> # Get all patient ids
        >>> unique_patients = dataset.unique_patient_ids
        >>> print(f"There are {len(unique_patients)} patients")
        >>>
        >>> # Get single patient data
        >>> patient = dataset.get_patient("S002")
        >>> print(f"Patient has {len(patient.data_source)} event")
        >>>
        >>> # Get event
        >>> event = patient.get_events(event_type="dreamt_sleep")
        >>>
        >>> # Get Apnea-Hypopnea Index (AHI)
        >>> ahi = event[0].ahi
        >>> print(f"AHI is {ahi}")
        >>> 
        >>> # Get 64Hz sleep file path
        >>> file_path = event[0].file_64hz
        >>> print(f"64Hz sleep file path: {file_path}") 
    """

    def __init__(
            self,
            root: str,
            dataset_name: str | None = None,
            config_path: str | None = None,
    ) -> None:
        if config_path is None:
            logger.info("No config provided, using default config")
            config_path = Path(__file__).parent / "configs" / "dreamt.yaml"
        
        metadata_file = Path(root) / "dreamt-metadata.csv"

        if not os.path.exists(metadata_file):
            logger.info(f"{metadata_file} does not exist")
            self.prepare_metadata(root)
        
        default_tables = ["dreamt_sleep"]

        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "dreamt_sleep",
            config_path=config_path
        )
    
    def get_patient_file(
        self, patient_id: str, root: str, file_path: str
    ) -> str | None:
        """Returns the 64 Hz or 100 Hz CSV for a patient, or ``None``.

        Args:
            patient_id: Patient identifier.
            root: Root directory containing the dataset files.
            file_path: Folder name for the requested sampling rate.

        Returns:
            Absolute path string to the CSV, or ``None`` when the folder or
            file is missing.

        Examples:
            >>> dataset.get_patient_file("S001", root, "data_100Hz")
        """
        file: Path | None = None
        if file_path in ("data_64Hz", "data"):
            # Original legacy data directory: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/dataset_preparation/dreamt/dreamt_extract_ibi_se.py#L172
            file = Path(root) / file_path / f"{patient_id}_whole_df.csv"
        elif file_path == "data_100Hz":
            # Original 100 Hz directory: https://github.com/WillKeWang/WatchSleepNet_public/blob/main/dataset_preparation/dreamt/dreamt_extract_ibi_se.py#L202
            file = Path(root) / file_path / f"{patient_id}_PSG_df.csv"

        if file is None or not file.exists():
            logger.info("%s not found", file)
            return None
        return str(file)

    @property
    def default_task(self):
        """Returns a fresh per-recording IBI sleep-staging task.

        Examples:
            >>> isinstance(dataset.default_task, SleepStagingIBIDREAMT)
            True
        """
        from pyhealth.tasks.sleep_staging_ibi import SleepStagingIBIDREAMT

        return SleepStagingIBIDREAMT()


    def prepare_metadata(self, root: str) -> None:
        """
        Prepares metadata csv file for the DREAMT dataset by performing the following:
        1. Obtain clinical data from participant_info.csv file
        2. Process file paths based on patients found in clinical data
        3. Organize all data into a single DataFrame
        4. Save the processed DataFrame to a CSV file

        Args:
            root: root directory containing the dataset files
        """

        output_path = Path(root) / "dreamt-metadata.csv"

        # Obtain patient clinical data
        participant_info_path = Path(root) / "participant_info.csv"
        participant_info = pd.read_csv(participant_info_path)


        # Determine folder structure, assign associated file paths based on folder structure
        all_folders = [item.name for item in Path(root).iterdir() if item.is_dir()]
        file_path_64hz = "data_64Hz" if "data_64Hz" in all_folders else "data"
        file_path_100hz = "data_100Hz"

        # Determine paths for 64Hz and 100Hz files for each patient
        participant_info['file_64hz'] = participant_info['SID'].apply(
            lambda sid: self.get_patient_file(sid, root, file_path_64hz)
        )
        participant_info['file_100hz'] = participant_info['SID'].apply(
            lambda sid: self.get_patient_file(sid, root, file_path_100hz)
        )

        # Remove "%" from mean SaO2 recording
        participant_info['Mean_SaO2'] = participant_info['Mean_SaO2'].str[:-1]

        # Format columns to align with BaseDataset
        participant_info = participant_info.rename(columns = {
            'SID': 'patient_id',
            'AGE': 'age',
            'GENDER': 'gender',
            'BMI': 'bmi',
            'OAHI': 'oahi',
            'AHI': 'ahi',
            'Mean_SaO2': 'mean_sao2',
            'Arousal Index': 'arousal_index',
            "MEDICAL_HISTORY": 'medical_history',
            "Sleep_Disorders": 'sleep_disorders'
        })
        
        # Create csv
        participant_info.to_csv(output_path, index=False)
