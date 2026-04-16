"""
A PyHealth task that performs dynamic resampling of ECG signals.

Dataset link:
    https://physionet.org/content/ptb-xl/1.0.3/

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

from pyhealth.data import Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

class PTBXLResampling(BaseTask):
    """
    Task: Downsample high-resolution (500Hz) signals to 250Hz.
    This provides a balance between detail and computational efficiency.
    """
    task_name: str = "PTBXLResampling"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"signal": "tensor"}

    def __init__(self, root: str) -> None:
        self.root = root

    def __call__(self, patient: Patient) -> List[Dict]:
        events = patient.get_events(event_type="ptb-xl")
        samples = []

        ecg_id = int(patient.patient_id)
        subfolder = f"{str((ecg_id // 1000) * 1000).zfill(5)}"
        hr_path = os.path.join(self.root, "records500", subfolder, f"{str(ecg_id).zfill(5)}_hr")

        try:
            record = wfdb.rdrecord(hr_path)
            data_500hz = record.p_signal.T # Shape: (12, 5000)

            # Downsample to 250Hz (2500 samples for a 10s record)
            num_samples_target = 2500
            data_250hz = signal.resample(data_500hz, num_samples_target, axis=1)

            samples.append({
                "signal": data_250hz.astype(np.float32),
                "record_id": ecg_id
            })
        except Exception:
            pass

        return samples
