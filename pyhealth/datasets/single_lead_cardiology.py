import os
import numpy as np
from typing import Optional, List
from itertools import islice
from pyhealth.datasets import BaseSignalDataset

class SingleLeadCardiologyDataset(BaseSignalDataset):
    """Single short ECG lead recording dataset for cardiology
    
    Dataset is available at https://physionet.org/content/challenge-2017/1.0.0/

    Args:
        root: root directory of the raw data.
        dataset_name: name of the dataset.
        dev: enable dev mode or not. 
        split: either 'training' or 'validation' folder.
        refresh_cache: whether to refresh the cache.
    
    Attributes:
        task: Optional[str], name of the task (e.g., "arrhythmia detection").
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, record_id, and other task-specific attributes as key.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices

    Examples:
        >>> from pyhealth.datasets import SingleLeadCardiologyDataset
        >>> dataset = SingleLeadCardiologyDataset(
        ...         root="/srv/local/data/physionet.org/files/challenge-2017/1.0.0/",
        ...         split="training",
        ...         dev=True
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def __init__(self, root: str, 
                 dataset_name: Optional[str] = None, 
                 dev: bool = False, 
                 split: str = 'training',
                 refresh_cache: bool = False):
        
        super().__init__(
            dataset_name=dataset_name, 
            root=root, 
            dev=dev,
            refresh_cache=refresh_cache
        ) 
        self.root = root
        self.dev = dev
        self.split = split
        self.refresh_cache = refresh_cache

    def process_single_lead_ECG_data(self):
        """Processes the single-lead ECG data from PhysioNet Challenge 2017.
        
        Returns:
            dict: A dictionary mapping patient_ids, 
                in which we define as ex.A0001, to their ECG records.
        """
        # Validate split parameter
        if self.split not in ['training', 'validation']:
            raise ValueError("split must be either 'training' or 'validation'")
            
        # Path to the data folder based on split
        data_folder = os.path.join(self.root, self.split)
        
        # Get all .mat files (ECG signals) and corresponding .hea files (headers)
        record_files = []
        for file in os.listdir(data_folder):
            if file.endswith('.mat'):
                base_name = file[:-4]  # Remove .mat extension
                hea_file = base_name + '.hea'
                if os.path.exists(os.path.join(data_folder, hea_file)):
                    record_files.append(base_name)
        
        # In dev mode
        if self.dev:
            record_files = record_files[:50]
        
        # Create patient dictionary
        # Using record names as patient IDs since this dataset doesn't have explicit patient IDs
        patients = {
            record: [{
                "load_from_path": data_folder,
                "patient_id": record,
                "signal_file": record + ".mat",
                "label_file": record + ".hea",
                "save_to_path": self.filepath,
            }]
            for record in record_files
        }
        
        return patients


if __name__ == "__main__":
    # Example usage
    dataset = SingleLeadCardiologyDataset(
        root="/content/local/data/physionet.org/files/challenge-2017/1.0.0/",
        split="training",
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
    
    keys = list(dataset.patients.keys())
    print(f"First patient ID: {keys[0]}")
