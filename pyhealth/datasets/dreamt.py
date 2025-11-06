import logging
import os
from pathlib import Path
from typing import  Optional, Union

import pandas as pd
from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

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
    so root path should be version downloaded, example: root = ".../dreamt/1.0.0/" or ".../dreamt/2.0.0/"

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
            dataset_name: Optional[str] = None,
            config_path: Optional[str] = None,
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
    
    def get_patient_file(self, patient_id: str, root: str, file_path: str) -> Union[str | None]:
        """
        Returns file path of 64Hz and 100Hz data for a patient, or None if no file found

        Args:
            patient_id: patient identifier
            root: root directory containing the dataset files
            file_path: path to location of 64Hz or 100Hz file
            
        Returns:
            file: path to file location or None if no file found
        """
        
        if file_path == "data_64Hz" or file_path == "data":
            file = Path(root) / f"{file_path}" / f"{patient_id}_whole_df.csv"
    
        if file_path == "data_100Hz":
            file = Path(root) / f"{file_path}" / f"{patient_id}_PSG_df.csv"

        if not os.path.exists(str(file)):
            logger.info(f"{file} not found")
            file = None
        
        return file


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
