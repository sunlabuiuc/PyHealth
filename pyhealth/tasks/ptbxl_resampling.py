"""
A PyHealth task that performs dynamic resampling of ECG signals.

Dataset link:
    https://physionet.org/content/ptb-xl/1.0.1/

Dataset paper: (please cite if you use this dataset)
    Wagner, P., Strodthoff, N., Bousseljot, R. D., Samek, W., & Schaeffter, T. 
    "PTB-XL, a large publicly available electrocardiography dataset." 
    Scientific Data, 7(1), 1-15. (2020).

Dataset paper link:
    https://www.nature.com/articles/s41597-020-0495-6

Author:
    Jovian Wang (jovianw2@illinois.edu)
    Matthew Pham (mdpham2@illinois.edu)
    Yiyun Wang (yiyunw3@illinois.edu)
"""
import logging
import os
import wfdb
import numpy as np
from scipy import signal
from typing import Dict, List

from pyhealth.data import Patient, Event
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

class PTBXLResampling(BaseTask):
    """
    This task reads the 100Hz (low-res) raw signal and uses scipy to 
    mathematically generate the 500Hz (high-res) version as the target.
    """
    task_name: str = "PTBXLResampling"
    
    # Input: The original 100Hz signal
    # Output: The target signal generated via resampling logic
    input_schema: Dict[str, str] = {"low_res": "signal"}
    output_schema: Dict[str, str] = {"high_res": "signal"}

    def __init__(self, root: str) -> None:
        """
        Args:
            root (str): The path to the PTB-XL dataset (containing records100).
        """
        if not os.path.exists(root):
            raise FileNotFoundError(f"Path not found: {root}")
        self.root = root

    def __call__(self, patient: Patient) -> List[Dict]:
        events: List[Event] = patient.get_events(event_type="ptbxl")
        samples = []

        for event in events:
            ecg_id = int(event["ecg_id"])
            subfolder = f"{str((ecg_id // 1000) * 1000).zfill(5)}"
            
            lr_path = os.path.join(self.root, "records100", subfolder, f"{str(ecg_id).zfill(5)}_lr")

            try:
                record = wfdb.rdrecord(lr_path)
                lr_data = record.p_signal.T  # Shape: (12, 1000)
                num_samples_target = 2500
                hr_data = signal.resample(lr_data, num_samples_target, axis=1)

                samples.append({
                    "low_res": lr_data.astype(np.float32),
                    "high_res": hr_data.astype(np.float32),
                    "record_id": ecg_id
                })
                
            except Exception as e:
                logger.debug(f"Skipping record {ecg_id}: {e}")
                continue

        return samples