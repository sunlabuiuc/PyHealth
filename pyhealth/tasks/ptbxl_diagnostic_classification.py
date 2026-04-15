"""
PyHealth task for ECG diagnostic classification using the PTB-XL dataset.

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

class PTBXLDiagnosticClassification(BaseTask):
    """
    A PyHealth task class for multi-label diagnostic classification of ECGs
    in the PTB-XL dataset, utilizing resampled 500Hz signal features.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input (signal).
        output_schema (Dict[str, str]): The schema for the task output (multi-label).
    """
    task_name: str = "PTBXLDiagnosticClassification"
    # x is the resampled signal, metadata provides clinical context
    input_schema: Dict[str, str] = {"signal": "signal", "metadata": "tabular"}
    output_schema: Dict[str, str] = {"label": "multilabel"}

    def __init__(self, root: str) -> None:
        """
        Initializes the PTBXLDiagnosticClassification task.

        Args:
            root (str): The root directory where the resampled .mat files 
                       (12 leads, 2500 samples) are stored.
        """
        if not os.path.exists(root):
            msg = f"Signal root path does not exist: {root}"
            logger.error(msg)
            raise FileNotFoundError(msg)
            
        self.root = root

    def __call__(self, patient: Patient) -> List[Dict]:
        events: List[Event] = patient.get_events(event_type="ptbxl")
        samples = []

        for event in events:
            ecg_id = int(event["ecg_id"])       
            subfolder = f"{str((ecg_id // 1000) * 1000).zfill(5)}"
            
            file_name = f"{str(ecg_id).zfill(5)}_hr.mat"
            signal_path = os.path.join(self.root, "records100", subfolder, file_name)

            try:
                mat_data = sio.loadmat(signal_path)
                signal_data = mat_data['feats']
                
                samples.append({
                    "signal": signal_data,
                    "metadata": [patient.age, patient.sex],
                    "label": event["label"],
                    "record_id": ecg_id
                })
            except Exception as e:
                logger.warning(f"Could not load signal for record {ecg_id}: {e}")
                continue

        return samples