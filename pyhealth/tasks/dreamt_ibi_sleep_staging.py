"""
DREAMT IBI Sleep Staging Task
---------------------------------
Contributors:
- Andrew Salazar (aas15)
- Sharon Lin (xinyiyl2)
- Soumya Mazumder (soumyam4)

Associated Paper:
- WatchSleepNet: A Novel Model and Pretraining Approach for Advancing Sleep Staging with Smartwatches
  https://proceedings.mlr.press/v287/wang25a.html

Description:
This file implements a new PyHealth Task that enables sleep stage classification from
inter-beat interval (IBI) sequences extracted from the DREAMT wearable dataset.

The task:
- Uses `DREAMTDataset` and its `dreamt_sleep` event structure.
- Loads the wearable 64Hz CSV files associated with each patient.
- Extracts IBI sequences and the corresponding sleep-stage labels.
- Cleans and normalizes IBI values.
- Produces fixed-length sliding windows of IBI data suitable for training deep
  learning models such as WatchSleepNet.
- Output model-ready samples with:
    input:  {"ibi_seq": np.ndarray(window_size,)}
    output: {"stage": int sleep-stage label}

This Task is fully tested, self-contained, and compatible with the PyHealth task registry
and model pipelines.
"""

from typing import List, Dict, Any
from pyhealth.tasks import BaseTask
import numpy as np
import pandas as pd

STAGE_MAP = {
    "W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4
}

class DreamtIBISleepStagingTask(BaseTask):
    """
    PyHealth Task: Sleep Stage Classification from IBI (DREAMT Dataset)

    This task extracts inter-beat interval (IBI) sequences from DREAMT wearable
    data and converts them into fixed-length windows paried with sleep stage labels.

    Input:
        - ibi_seq : np.ndarray (window_size,)
    Output:
        - stage : integer sleep stage (0-4)

    Notes:
        - This task uses DREAMTDataset event_type="dreamt_sleep".
        - Produces multiple samples per patient (one per IBI window).
    """

    task_name = "DreamtSleepStaging"
    input_schema = {"ibi_seq": "series"}
    output_schema = {"stage": "multiclass"}

    def __init__(self, window_size: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def __call__(self, patient) -> List[Dict[str, Any]]:
        """Extracts IBI sequences + stage labels from each event."""

        events = patient.get_events(event_type="dreamt_sleep")
        results = []

        for event in events:
            file_path = event.file_64hz
            if file_path is None:
                continue

            df = pd.read_csv(file_path)

            if "IBI" not in df or "Sleep_Stage" not in df:
                continue

            ibi = df["IBI"].astype(float).values
            labels = df["Sleep_Stage"].values

            # Drop implausible values
            mask = (ibi >= 300) & (ibi <= 2000)
            ibi = ibi[mask]
            labels = labels[mask]

            if len(ibi) < self.window_size:
                continue

            # Normalize
            mean, std = np.mean(ibi), np.std(ibi) + 1e-6
            ibi = (ibi - mean) / std

            # Sliding windows
            W = self.window_size
            for i in range(0, len(ibi) - W + 1, W):
                window = ibi[i:i+W]
                stage_raw = labels[i+W-1]

                if isinstance(stage_raw, str):
                    stage = STAGE_MAP.get(stage_raw, None)
                else:
                    stage = int(stage_raw)

                if stage is None:
                    continue

                results.append({"ibi_seq": window, "stage": stage})

        return results