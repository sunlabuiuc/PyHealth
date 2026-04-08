import numpy as np
import torch
import mne
from typing import Any, Dict, List, Tuple
from pyhealth.tasks import BaseTask

class TUSZTask(BaseTask):
    """Multi-class classification task for seizure detection on TUSZ.

    For each EDF recording, this task:
      1) reads the EDF
      2) applies bandpass, notch, and resamples
      3) loads paired event annotations from the .rec file
      4) constructs 5-second windows per event
      5) returns one sample per event

    Each returned sample contains:
      - "signal": torch.FloatTensor, shape (16, n_samples)
      - "label": int (0 = non-seizure, 1 = seizure)
    """

    task_name: str = "EEG_seizure"
    input_schema: Dict[str, str] = {"signal": "tensor", "stft": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        resample_rate: float = 200,
        bandpass_filter: Tuple[float, float] = (0.5, 70.0),
        notch_filter: float = 50.0,
        normalization: str = None,  # '95th_percentile', 'div_by_100'
        compute_stft: bool = True,
    ) -> None:
        super().__init__()
        self.resample_rate = resample_rate
        self.bandpass_filter = bandpass_filter
        self.notch_filter = notch_filter
        self.normalization = normalization
        self.compute_stft = compute_stft
        if not compute_stft:
            self.input_schema = {"signal": "tensor"}

    @staticmethod
    def readEDF(fileName: str,
                resample_rate: float = 200,
                bandpass_filter: Tuple[float, float] = (0.5, 70.0),
                notch_filter: float = 50.0
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, mne.io.BaseRaw]:
        """Reads EDF and corresponding .rec file, applies filters and resampling."""
        raw = mne.io.read_raw_edf(fileName, preload=True, verbose="error")
        raw.filter(l_freq=bandpass_filter[0], h_freq=bandpass_filter[1], verbose="error")
        raw.notch_filter(notch_filter, verbose="error")
        raw.resample(resample_rate, n_jobs=1, verbose="error")
        signals = raw.get_data(units="uV")
        _, times = raw[:]
        rec_file = fileName[:-3] + "rec"
        events = np.genfromtxt(rec_file, delimiter=",")
        raw.close()
        return signals, times, events, raw

    @staticmethod
    def convert_signals(signals: np.ndarray, raw: mne.io.BaseRaw) -> np.ndarray:
        """Convert raw signals to standard 16-channel bipolar montage."""
        ch_names = {k: v for k, v in zip(raw.info["ch_names"], range(len(raw.info["ch_names"])))}
        bipolar = np.vstack((
            signals[ch_names["EEG FP1-REF"]] - signals[ch_names["EEG F7-REF"]],
            signals[ch_names["EEG F7-REF"]] - signals[ch_names["EEG T3-REF"]],
            signals[ch_names["EEG T3-REF"]] - signals[ch_names["EEG T5-REF"]],
            signals[ch_names["EEG T5-REF"]] - signals[ch_names["EEG O1-REF"]],
            signals[ch_names["EEG FP2-REF"]] - signals[ch_names["EEG F8-REF"]],
            signals[ch_names["EEG F8-REF"]] - signals[ch_names["EEG T4-REF"]],
            signals[ch_names["EEG T4-REF"]] - signals[ch_names["EEG T6-REF"]],
            signals[ch_names["EEG T6-REF"]] - signals[ch_names["EEG O2-REF"]],
            signals[ch_names["EEG FP1-REF"]] - signals[ch_names["EEG F3-REF"]],
            signals[ch_names["EEG F3-REF"]] - signals[ch_names["EEG C3-REF"]],
            signals[ch_names["EEG C3-REF"]] - signals[ch_names["EEG P3-REF"]],
            signals[ch_names["EEG P3-REF"]] - signals[ch_names["EEG O1-REF"]],
            signals[ch_names["EEG FP2-REF"]] - signals[ch_names["EEG F4-REF"]],
            signals[ch_names["EEG F4-REF"]] - signals[ch_names["EEG C4-REF"]],
            signals[ch_names["EEG C4-REF"]] - signals[ch_names["EEG P4-REF"]],
            signals[ch_names["EEG P4-REF"]] - signals[ch_names["EEG O2-REF"]],
        ))
        return bipolar

    @staticmethod
    def BuildEvents(signals: np.ndarray, times: np.ndarray, EventData: np.ndarray,
                    resample_rate: float = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        EventData = np.atleast_2d(EventData)
        numEvents, _ = EventData.shape
        numChan = signals.shape[0]
        fs = resample_rate
        offset = signals.shape[1]
        signals = np.concatenate([signals, signals, signals], axis=1)
        features = np.zeros([numEvents, numChan, int(fs) * 5])
        offending_channel = np.zeros([numEvents, 1])
        labels = np.zeros([numEvents, 1])
        for i in range(numEvents):
            chan = int(EventData[i, 0])
            start = np.where(times >= EventData[i, 1])[0][0]
            end = np.where(times >= EventData[i, 2])[0][0]
            features[i, :] = signals[:, offset + start - 2 * int(fs): offset + end + 2 * int(fs)]
            offending_channel[i, :] = chan
            labels[i, :] = int(EventData[i, 3])
        return features, offending_channel, labels

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Generates samples for one patient."""
        pid = patient.patient_id
        samples: List[Dict[str, Any]] = []

        for split in ("train", "eval"):
            events = patient.get_events(split)
            for event in events:
                edf_path = event.signal_file
                try:
                    signals, times, rec, raw = self.readEDF(
                        edf_path, self.resample_rate, self.bandpass_filter, self.notch_filter
                    )
                    signals = self.convert_signals(signals, raw)
                except (ValueError, KeyError):
                    continue

                feats, offending_channels, labels = self.BuildEvents(signals, times, rec, self.resample_rate)

                for feat, chan, label in zip(feats, offending_channels, labels):
                    if self.normalization == "95th_percentile":
                        feat = feat / (np.quantile(np.abs(feat), q=0.95, axis=-1, keepdims=True) + 1e-8)
                    elif self.normalization == "div_by_100":
                        feat = feat / 100
                    feat_tensor = torch.FloatTensor(feat)
                    sample = {
                        "patient_id": pid,
                        "signal_file": edf_path,
                        "split": split,
                        "signal": feat_tensor,
                        "offending_channel": int(chan.squeeze()),
                        "label": int(label.squeeze()) - 1,
                    }
                    if self.compute_stft:
                        from pyhealth.models.tfm_tokenizer import get_stft_torch
                        sample["stft"] = get_stft_torch(feat_tensor.unsqueeze(0)).squeeze(0)
                    samples.append(sample)

        return samples