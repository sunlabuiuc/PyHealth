"""
PyHealth task for extracting features with STFT and Frequency Bands using the Temple University Hospital (TUH) EEG Seizure Corpus (TUSZ) dataset V2.0.5.

Dataset link: (Application needed)
    https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml

Dataset paper:
    Vinit Shah, Eva von Weltin, Silvia Lopez, et al., “The Temple University Hospital Seizure Detection Corpus,” arXiv preprint arXiv:1801.08085, 2018. Available: https://arxiv.org/abs/1801.08085

Dataset paper link:
    https://arxiv.org/abs/1801.08085

Author:
    To be named (tbn01@illinois.edu)
"""

# process: downsample -> STFT -> frequency band

# downsample: 
#     https://github.com/otoolej/downsample_open_eeg
# kaggle TUSZ: 
#     https://www.kaggle.com/code/seanbearden/processing-the-tuh-eeg-seizure-corpus#Load-Data
# official paper link: 
#     https://github.com/AITRICS/EEG_real_time_seizure_detection/tree/master


# self. normalization
# self.compute_stft

from typing import Dict, List, Tuple
import numpy as np
import mne
from pyhealth.tasks import BaseTask

class EEGSignalPreProcess(BaseTask):

    task_name: str = "temple_uni_eeg_preprocessing_task"
    input_schema: Dict[str, str] = {"signal": "tensor", "stft": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self,
                 file_name: str,
                 resample_rate: float = 200,
                 bandpass_filter: Tuple[float, float] = (0.1, 75.0),
                 notch_filter: float = 50.0,
                 ) -> None:
        super().__init__()
        self.file_name = file_name
        self.resample_rate = resample_rate
        self.bandpass_filter = bandpass_filter
        self.notch_filter = notch_filter
        
    def __call__(self) -> Tuple[np.ndarray, List[str]]:
        Rawdata = mne.io.read_raw_edf(self.file_name, preload=True, verbose="error")

        if self.bandpass_filter != (-1, -1):
            Rawdata.filter(l_freq=self.bandpass_filter[0], h_freq=self.bandpass_filter[1], verbose="error")
        if self.notch_filter != -1:
            Rawdata.notch_filter(self.notch_filter, verbose="error")
        if self.resample_rate != -1:
            Rawdata.resample(self.resample_rate, n_jobs=5, verbose="error")

        raw_data = Rawdata.get_data(units="uV")
        ch_name = Rawdata.ch_names
        return raw_data, ch_name