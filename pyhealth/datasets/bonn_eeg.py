"""
Name: Vignesh Ravi
Net ID: ravi14

Bonn EEG Dataset for Seizure Detection (Andrzejak et al., 2001).
Dataset Link: https://repositori.upf.edu/handle/10230/42894

The dataset consists of 5 sets (A, B, C, D, E) containing 100 single-channel 
EEG segments each. Each segment is 23.6 seconds duration at 173.61 Hz sampling rate.
- Set E (S) is Ictal (Seizure).
- Sets A, B, C, D (Z, O, N, F) are Non-Seizure.

Paper: Indications of nonlinear deterministic and finite-dimensional structures... (Andrzejak et al., 2001)
"""
import os
import glob
import pandas as pd
from typing import Optional, List
from pyhealth.datasets import BaseDataset

class BonnEEGDataset(BaseDataset):
    """
    Bonn EEG Dataset for Seizure Detection (Andrzejak et al., 2001).

    Dataset Structure:
    ------------------
    root/
      Z/                    # Set A: Healthy, Eyes Open
        Z001.txt            # .txt file with 4097 float values (ASCII)
        ...
      O/                    # Set B: Healthy, Eyes Closed
      N/                    # Set C: Interictal, Hippocampal formation
      F/                    # Set D: Interictal, Epileptogenic zone
      S/                    # Set E: Ictal, Seizure activity

    Generated Metadata (index.csv):
    -------------------------------
    The class automatically scans the folders above and generates an index.csv:
      - filepath: Absolute path to the .txt file
      - label_class: Folder name (Z, O, N, F, S)
      - label_desc: Description of the physiological state

    Args:
        root (str): Root directory of the dataset.
        dataset_name (str, optional): Name of the dataset. Defaults to 'BonnEEG'.
        dev (bool, optional): Whether to enable dev mode (load small subset). Defaults to False.
    
    Example:
    >>> from pyhealth.datasets import BonnEEGDataset
    >>> from pyhealth.tasks import BonnEEGSeizureDetection
    >>> dataset = BonnEEGDataset(root="/path/to/BonnEEG/")
    >>> dataset.stat()
    >>> # Set task to Seizure Detection (Binary: Set S vs Others)
    >>> samples = dataset.set_task(BonnEEGSeizureDetection())
    >>> print(samples[0])
    """
    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = "BonnEEG",
        dev: bool = False,
    ):
        # We create a index CSV so the BaseDataset can load it via streaming when generating samples.
        self.prepare_metadata(root, dev)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "configs", "bonn_eeg.yaml")

        super().__init__(
            root=root,
            tables=["index"], 
            dataset_name=dataset_name,
            config_path=config_path,
            dev=dev,
        )

    @staticmethod
    def prepare_metadata(root: str, dev: bool):
        """
        Scans the directory and builds an index.csv containing file paths.
        """
        index_path = os.path.join(root, "index.csv")
        
        # Skip if index already exists (caching mechanism)
        if os.path.exists(index_path) and not dev:
            return

        folders = {
            "Z": "Set_A_Healthy_EyesOpen",
            "O": "Set_B_Healthy_EyesClosed",
            "N": "Set_C_Interictal_Opposite",
            "F": "Set_D_Interictal_Zone",
            "S": "Set_E_Ictal_Seizure",
        }

        records = []
        for folder_code, desc in folders.items():
            folder_path = os.path.join(root, folder_code)
            if not os.path.exists(folder_path): continue
            
            files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
            if dev: files = files[:5]

            for filepath in files:
                # Store absolute path so we can find it later
                records.append({
                    "filepath": os.path.abspath(filepath),
                    "label_class": folder_code,
                    "label_desc": desc
                })
        
        # Save Metadata Catalog
        pd.DataFrame(records).to_csv(index_path, index=False)