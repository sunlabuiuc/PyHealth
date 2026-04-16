"""
A PyHealth task for ECG signal resampling/super-resolution on the PTB-XL dataset.

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
import scipy.io as sio
from typing import Dict, List

from pyhealth.data import Patient, Event
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

class PTBXLResamplingTask(BaseTask):
    task_name: str = "PTBXLResampling"
    
    # Input: 100Hz Signal | Output: 500Hz Signal (the ground truth)
    input_schema: Dict[str, str] = {"low_res": "signal"}
    output_schema: Dict[str, str] = {"high_res": "signal"}

    def __init__(self, root: str) -> None:
        """
        Args:
            root (str): Root directory containing BOTH 'records100' and 'records500'.
        """
        if not os.path.exists(root):
            raise FileNotFoundError(f"Root path does not exist: {root}")
        self.root = root

    def __call__(self, patient: Patient) -> List[Dict]:
        events: List[Event] = patient.get_events(event_type="ptbxl")
        samples = []

        for event in events:
            ecg_id = int(event["ecg_id"])
            subfolder = f"{str((ecg_id // 1000) * 1000).zfill(5)}"
            
            lr_path = os.path.join(self.root, "records100", subfolder, f"{str(ecg_id).zfill(5)}_lr")
            hr_path = os.path.join(self.root, "records500", subfolder, f"{str(ecg_id).zfill(5)}_hr")

            try:
                lr_record = wfdb.rdrecord(lr_path)
                lr_signal = lr_record.p_signal.T # Transpose to (12, 1000)
                
                hr_record = wfdb.rdrecord(hr_path)
                hr_signal = hr_record.p_signal.T # Transpose to (12, 2500)

                samples.append({
                    "low_res": lr_signal.astype(np.float32),
                    "high_res": hr_signal.astype(np.float32),
                    "record_id": ecg_id
                })
                
            except Exception as e:
                continue

        return samples