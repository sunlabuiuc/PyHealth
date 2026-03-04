import numpy as np
import torch
import mne
from typing import Any, Dict, List, Tuple
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

    def __init__(self,
                 resample_rate: float = 200,
                 bandpass_filter: Tuple[float, float] = (0.1, 75.0),
                 notch_filter: float = 50.0,
                 normalization: str = None # '95th_percentile', 'div_by_100'
                 ) -> None:
        super().__init__()

        self.resample_rate = resample_rate
        self.bandpass_filter = bandpass_filter
        self.notch_filter = notch_filter
        self.normalization = normalization

    @staticmethod
    def BuildEvents(
        signals: np.ndarray, times: np.ndarray, EventData: np.ndarray,
        resample_rate: float = 200,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Ensure 2D in case a .rec has only one row
        EventData = np.atleast_2d(EventData)

        numEvents, _ = EventData.shape
        fs = resample_rate
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
    def readEDF(fileName: str,
                 resample_rate: float = 200,
                 bandpass_filter: Tuple[float, float] = (0.1, 75.0),
                 notch_filter: float = 50.0,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, mne.io.BaseRaw]:
        Rawdata = mne.io.read_raw_edf(fileName, preload=True, verbose="error")

        Rawdata.filter(l_freq=bandpass_filter[0], h_freq=bandpass_filter[1], verbose="error")
        Rawdata.notch_filter(notch_filter, verbose="error")
        Rawdata.resample(resample_rate, n_jobs=5, verbose="error")

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

            signals, times, rec, raw = self.readEDF(edf_path, self.resample_rate, self.bandpass_filter, self.notch_filter)
            signals = self.convert_signals(signals, raw)
            feats, offending_channels, labels = self.BuildEvents(signals, times, rec, self.resample_rate)

            for idx, (signal, offending_channel, label) in enumerate(
                zip(feats, offending_channels, labels)
            ):
                
                if self.normalization == '95th_percentile':
                    signal = signal/(np.quantile(np.abs(signal), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-8)
                elif self.normalization == 'div_by_100':
                    signal = signal/100
                    
                signal = torch.FloatTensor(signal)
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
    
    

class EEGAbnormalTUAB(BaseTask):
    """Binary classification task for abnormal EEG detection on TUAB.

    For each EDF recording, this task:
      1) reads the EDF
      2) Applies bandpass (0.1-75 Hz), notch (50 Hz) and resamples to 200 Hz
      3) Constructs 16 bipolar channels from the TCP montage of non-overlapping 10 second windows # following BIOT
      4) assigns a binary label (1 = abnormal, 0 = normal) derived from the
         metadata ``label`` attribute

    Each returned sample contains:
      - "signal": np.ndarray, shape (16, 2000)
      - "label": int (0 = normal, 1 = abnormal)

    Examples:
        >>> from pyhealth.datasets import TUABDataset
        >>> from pyhealth.tasks import EEGAbnormalTUAB
        >>> dataset = TUABDataset(
        ...         root="/srv/local/data/TUH/tuh_eeg_abnormal/v3.0.0/edf/",
        ...     )
        >>> sample_dataset = dataset.set_task(EEGAbnormalTUAB())
        >>> sample = sample_dataset[0]
        >>> print(sample['signal'].shape)  # (16, 2000)
        >>> print(sample['label'])         # 0 or 1
    """

    task_name: str = "EEG_abnormal"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self,
                 resample_rate: float = 200,
                 bandpass_filter: Tuple[float, float] = (0.1, 75.0),
                 notch_filter: float = 50.0,
                 normalization: str = None # '95th_percentile', 'div_by_100'
                 ) -> None:
        super().__init__()
        self.resample_rate = resample_rate
        self.bandpass_filter = bandpass_filter
        self.notch_filter = notch_filter
        self.normalization = normalization
    @staticmethod
    def read_and_process_edf(fileName: str,
                             resample_rate: float = 200,
                             bandpass_filter: Tuple[float, float] = (0.1, 75.0),
                             notch_filter: float = 50.0,
                             ) -> Tuple[np.ndarray, List[str]]:
        Rawdata = mne.io.read_raw_edf(fileName, preload=True, verbose="error")

        Rawdata.filter(l_freq=bandpass_filter[0], h_freq=bandpass_filter[1], verbose="error")
        Rawdata.notch_filter(notch_filter, verbose="error")
        Rawdata.resample(resample_rate, n_jobs=5, verbose="error")

        raw_data = Rawdata.get_data(units="uV")
        ch_name = Rawdata.ch_names
        return raw_data, ch_name

    @staticmethod
    def convert_to_bipolar(raw_data: np.ndarray, ch_name: List[str]) -> np.ndarray:
        """Convert raw EEG channels to 16 bipolar montage channels following BIOT.
        
        Returns:
            np.ndarray of shape (16, n_samples)
        """
        channeled_data = np.zeros((16, raw_data.shape[1]))
        
        channeled_data[0] = (
            raw_data[ch_name.index("EEG FP1-REF")]
            - raw_data[ch_name.index("EEG F7-REF")]
        )
        channeled_data[1] = (
            raw_data[ch_name.index("EEG F7-REF")]
            - raw_data[ch_name.index("EEG T3-REF")]
        )
        channeled_data[2] = (
            raw_data[ch_name.index("EEG T3-REF")]
            - raw_data[ch_name.index("EEG T5-REF")]
        )
        channeled_data[3] = (
            raw_data[ch_name.index("EEG T5-REF")]
            - raw_data[ch_name.index("EEG O1-REF")]
        )
        channeled_data[4] = (
            raw_data[ch_name.index("EEG FP2-REF")]
            - raw_data[ch_name.index("EEG F8-REF")]
        )
        channeled_data[5] = (
            raw_data[ch_name.index("EEG F8-REF")]
            - raw_data[ch_name.index("EEG T4-REF")]
        )
        channeled_data[6] = (
            raw_data[ch_name.index("EEG T4-REF")]
            - raw_data[ch_name.index("EEG T6-REF")]
        )
        channeled_data[7] = (
            raw_data[ch_name.index("EEG T6-REF")]
            - raw_data[ch_name.index("EEG O2-REF")]
        )
        channeled_data[8] = (
            raw_data[ch_name.index("EEG FP1-REF")]
            - raw_data[ch_name.index("EEG F3-REF")]
        )
        channeled_data[9] = (
            raw_data[ch_name.index("EEG F3-REF")]
            - raw_data[ch_name.index("EEG C3-REF")]
        )
        channeled_data[10] = (
            raw_data[ch_name.index("EEG C3-REF")]
            - raw_data[ch_name.index("EEG P3-REF")]
        )
        channeled_data[11] = (
            raw_data[ch_name.index("EEG P3-REF")]
            - raw_data[ch_name.index("EEG O1-REF")]
        )
        channeled_data[12] = (
            raw_data[ch_name.index("EEG FP2-REF")]
            - raw_data[ch_name.index("EEG F4-REF")]
        )
        channeled_data[13] = (
            raw_data[ch_name.index("EEG F4-REF")]
            - raw_data[ch_name.index("EEG C4-REF")]
        )
        channeled_data[14] = (
            raw_data[ch_name.index("EEG C4-REF")]
            - raw_data[ch_name.index("EEG P4-REF")]
        )
        channeled_data[15] = (
            raw_data[ch_name.index("EEG P4-REF")]
            - raw_data[ch_name.index("EEG O2-REF")]
        )

        return channeled_data

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes one patient. Creates one sample per class
        """
        pid = patient.patient_id
        events = patient.get_events()
        
        samples: List[Dict[str, Any]] = []
        fs = self.resample_rate
        
        for event in events:
            edf_path = event.signal_file
            label = event.label
            if label == 'normal':
                label = 0
            elif label == 'abnormal':
                label = 1
            raw_data, ch_name = self.read_and_process_edf(edf_path, self.resample_rate, self.bandpass_filter, self.notch_filter)
            bipolar_data = self.convert_to_bipolar(raw_data, ch_name)
            
            num_samples = int(bipolar_data.shape[1] // (fs * 10))
            for i in range(num_samples):
                start = i * fs * 10
                end = start + fs * 10
                signal = bipolar_data[:, start:end]
                if self.normalization == '95th_percentile':
                    signal = signal/(np.quantile(np.abs(signal), q=0.95, axis=-1, method = 'linear',keepdims=True)+1e-8)
                elif self.normalization == 'div_by_100':
                    signal = signal/100
                    
                signal = torch.FloatTensor(signal)
                samples.append(
                    {
                        "patient_id": pid,
                        "signal_file": edf_path,
                        "signal": signal,
                        "label": label,
                        'segment_id': f'{i}',
                        'start_time': start,
                        'end_time': end,
                    }
                )
        return samples
            
    
    
    
