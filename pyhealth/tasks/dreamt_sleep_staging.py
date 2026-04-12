from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.signal import resample

from pyhealth.tasks import BaseTask
from pyhealth.data import Patient

class SleepStagingDREAMT(BaseTask):
    """Sleep staging task for WatchSleepNet on the DREAMT dataset.

    Each sample is a sequence of `seq_len` consecutive 30-second epochs.
    Each epoch has `target_hz * epoch_seconds` = 750 features (resampled IBI).

    The 3-class mapping groups sleep stages as:
        - Wake (W) -> 0
        - Light sleep (N1, N2, R) -> 1
        - Deep sleep (N3) -> 2

    Attributes:
        task_name: name of the task.
        input_schema: specifies that "signal" is a tensor.
        output_schema: specifies that "label" is a multiclass target.

    Note:
        Uses the 100Hz PSG file (file_100hz) by default because IBI values
        are populated there. In the 64Hz wearable file, IBI is empty and
        must be estimated from BVP.

    Args:
        epoch_seconds: duration of each epoch in seconds (default 30).
        seq_len: number of consecutive epochs per sequence (default 20).
        target_hz: target sampling rate after resampling (default 25).
        signal_col: which column to use from the CSV (default "IBI").
        use_100hz: if True, read from file_100hz (default True).
            If False, read from file_64hz.

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> from pyhealth.tasks import SleepStagingDREAMT
        >>> dataset = DREAMTDataset(root="/path/to/dreamt")
        >>> task = SleepStagingDREAMT()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "SleepStagingDREAMT"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    STAGE_MAP = {
        "W": 0,
        "N1": 1,
        "N2": 1,
        "R": 1,
        "N3": 2
    }

    def __init__(
        self,
        epoch_seconds: int = 30,
        seq_len: int = 20,
        target_hz: int = 25,
        signal_col: str = "IBI",
        use_100hz: bool = True,
    ):
        self.epoch_seconds = epoch_seconds
        self.seq_len = seq_len
        self.target_hz = target_hz
        self.signal_col = signal_col
        self.use_100hz = use_100hz
        super().__init__()

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Processes a single patient for sleep staging with WatchSleepNet.

        Args:
            patient: a Patient object from the DREAMTDataset.

        Returns:
            samples: list of dicts, each with patient_id, record_id,
                signal (seq_len, 750), and label (int).
        """
        pid = patient.patient_id
        event = patient.get_events(event_type="dreamt_sleep")[0]
        file_path = event.file_100hz if self.use_100hz else event.file_64hz

        if file_path is None:
            return []

        df = pd.read_csv(file_path)

        source_hz = 100 if self.use_100hz else 64
        samples_per_epoch_source = source_hz * self.epoch_seconds
        samples_per_epoch_target = self.target_hz * self.epoch_seconds

        signal = df[self.signal_col].values
        stages = df["Sleep_Stage"].values

        num_epochs = len(signal) // samples_per_epoch_source
        epochs = []
        epoch_labels = []

        for i in range(num_epochs):
            start = i * samples_per_epoch_source
            end = start + samples_per_epoch_source

            epoch_signal = resample(signal[start:end], samples_per_epoch_target)
            epochs.append(epoch_signal)

            epoch_stage = pd.Series(stages[start:end]).mode()[0]
            epoch_labels.append(self.STAGE_MAP.get(epoch_stage, 0))

        if len(epochs) < self.seq_len:
            return []

        epochs = np.array(epochs, dtype=np.float32)

        samples = []
        for i in range(len(epochs) - self.seq_len + 1):
            samples.append({
                "patient_id": pid,
                "record_id": f"{pid}-{i}",
                "signal": epochs[i : i + self.seq_len],
                "label": epoch_labels[i + self.seq_len - 1],
            })

        return samples
