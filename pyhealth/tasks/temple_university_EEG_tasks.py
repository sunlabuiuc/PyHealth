from __future__ import annotations

import os
import pickle
import mne
import numpy as np

import os
from typing import Any, Dict, List, Tuple

import mne
import numpy as np

from pyhealth.tasks import BaseTask


class EEGEventsTUEV(BaseTask):
    """Multi-class classification task for EEG event detection on TUEV.

    For each EDF recording, this task:
      1) reads the EDF
      2) applies bandpass (0.1-75 Hz), notch (50 Hz), resamples to 256 Hz
      3) loads the paired .rec file (same path, .edf -> .rec)
      4) constructs 5-second event-centered windows (16 bipolar channels)
      5) returns one sample per event

    Each returned sample contains:
      - "signal": np.ndarray, shape (16, 256*5)
      - "offending_channel": int
      - "label": int

    Examples:
        >>> from pyhealth.datasets import TUEVDataset
        >>> from pyhealth.tasks import EEGEventsTUEV
        >>> dataset = TUEVDataset(
        ...         root="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf/",
        ...     )
        >>> sample_dataset = dataset.set_task(EEGEventsTUEV())
        >>> sample = sample_dataset[0]
        >>> print(sample['label'])

        For a complete example, see `examples/conformal_eeg/tuev_eeg_quickstart.ipynb`.
    """

    task_name: str = "EEG_events"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(self) -> None:
        super().__init__()


    @staticmethod
    def BuildEvents(
        signals: np.ndarray, times: np.ndarray, EventData: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Ensure 2D in case a .rec has only one row
        EventData = np.atleast_2d(EventData)

        numEvents, _ = EventData.shape
        fs = 256.0
        numChan, _ = signals.shape

        features = np.zeros([numEvents, numChan, int(fs) * 5])
        offending_channel = np.zeros([numEvents, 1])
        labels = np.zeros([numEvents, 1])

        offset = signals.shape[1]
        signals = np.concatenate([signals, signals, signals], axis=1)

        for i in range(numEvents):
            chan = int(EventData[i, 0])
            start = np.where((times) >= EventData[i, 1])[0][0]
            end = np.where((times) >= EventData[i, 2])[0][0]

            features[i, :] = signals[
                :, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)
            ]
            offending_channel[i, :] = int(chan)
            labels[i, :] = int(EventData[i, 3])

        return features, offending_channel, labels

    @staticmethod
    def convert_signals(signals: np.ndarray, Rawdata: mne.io.BaseRaw) -> np.ndarray:
        signal_names = {
            k: v
            for (k, v) in zip(
                Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"])))
            )
        }

        new_signals = np.vstack(
            (
                signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F7-REF"]],
                signals[signal_names["EEG F7-REF"]] - signals[signal_names["EEG T3-REF"]],
                signals[signal_names["EEG T3-REF"]] - signals[signal_names["EEG T5-REF"]],
                signals[signal_names["EEG T5-REF"]] - signals[signal_names["EEG O1-REF"]],
                signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F8-REF"]],
                signals[signal_names["EEG F8-REF"]] - signals[signal_names["EEG T4-REF"]],
                signals[signal_names["EEG T4-REF"]] - signals[signal_names["EEG T6-REF"]],
                signals[signal_names["EEG T6-REF"]] - signals[signal_names["EEG O2-REF"]],
                signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F3-REF"]],
                signals[signal_names["EEG F3-REF"]] - signals[signal_names["EEG C3-REF"]],
                signals[signal_names["EEG C3-REF"]] - signals[signal_names["EEG P3-REF"]],
                signals[signal_names["EEG P3-REF"]] - signals[signal_names["EEG O1-REF"]],
                signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F4-REF"]],
                signals[signal_names["EEG F4-REF"]] - signals[signal_names["EEG C4-REF"]],
                signals[signal_names["EEG C4-REF"]] - signals[signal_names["EEG P4-REF"]],
                signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]],
            )
        )
        return new_signals

    @staticmethod
    def readEDF(fileName: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, mne.io.BaseRaw]:
        Rawdata = mne.io.read_raw_edf(fileName, preload=True, verbose="error")

        Rawdata.filter(l_freq=0.1, h_freq=75.0, verbose="error")
        Rawdata.notch_filter(50.0, verbose="error")
        Rawdata.resample(256, n_jobs=5, verbose="error")

        _, times = Rawdata[:]
        signals = Rawdata.get_data(units="uV")
        RecFile = fileName[0:-3] + "rec"

        eventData = np.genfromtxt(RecFile, delimiter=",")

        Rawdata.close()
        return signals, times, eventData, Rawdata


    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes one patient. Creates one sample per event in the .rec file.

        Expected patient events to include a `signal_file` attribute that points to an .edf file.
        """
        pid = patient.patient_id
        events = patient.get_events()

        samples: List[Dict[str, Any]] = []

        for event in events:
            edf_path = event.signal_file

            signals, times, rec, raw = self.readEDF(edf_path)
            signals = self.convert_signals(signals, raw)
            feats, offending_channels, labels = self.BuildEvents(signals, times, rec)

            for idx, (signal, offending_channel, label) in enumerate(
                zip(feats, offending_channels, labels)
            ):
                samples.append(
                    {
                        "patient_id": pid,
                        "signal_file": edf_path,
                        "signal": signal,
                        "offending_channel": int(offending_channel.squeeze()),
                        "label": int(label.squeeze())-1,
                    }
                )

        return samples
    
    
    
    
