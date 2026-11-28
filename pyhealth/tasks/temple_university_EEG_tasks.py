import numpy as np
import torch
import mne
from typing import Any, Dict, List, Tuple
import logging
import numpy as np
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

# The 16 electrodes that EEGAbnormalTUAB.convert_to_bipolar combines.
_TUAB_ELECTRODES = tuple(
    f"EEG {name}-REF"
    for name in (
        "FP1", "F7", "T3", "T5", "O1", "FP2", "F8", "T4",
        "T6", "O2", "F3", "C3", "P3", "F4", "C4", "P4",
    )
)  # fmt: skip


# get_stft_torch and TFM-Tokenizer assume 200 Hz: a 200-sample window with a
# 100-sample hop and the first 100 frequency bins (1 s windows, 1 Hz bins).
_STFT_RATE = 200


def _check_stft_rate(resample_rate: float, compute_stft: bool) -> None:
    """Reject a ``stft`` request at a rate other than 200 Hz.

    The ``stft`` field is computed with ``get_stft_torch``, whose window,
    hop and number of bins are fixed for 200 Hz signals and match the
    raw-signal branch of TFM-Tokenizer. At any other rate the window length
    and bin width would change without notice.

    Args:
        resample_rate: Rate of the signals in Hz.
        compute_stft: Whether the task computes the ``stft`` field.

    Raises:
        ValueError: If ``compute_stft`` is True and ``resample_rate`` is not
            200 Hz.

    Examples:
        >>> _check_stft_rate(200, compute_stft=True)
        >>> _check_stft_rate(256, compute_stft=False)
    """
    if compute_stft and resample_rate != _STFT_RATE:
        raise ValueError(
            f"compute_stft=True requires resample_rate={_STFT_RATE} Hz, got "
            f"{resample_rate}. The stft field uses a fixed 200-sample window "
            f"and 100 frequency bins, built for 200 Hz signals. Pass "
            f"compute_stft=False to use another rate."
        )


def read_edf_uv(
    file_name: str,
    bandpass_filter: tuple[float, float] | None = None,
    notch_filter: float | None = None,
    resample_rate: float | None = None,
) -> tuple[np.ndarray, list[str], float]:
    """Read an EDF in microvolts, with optional filtering and resampling.

    Args:
        file_name: Path to an EDF recording.
        bandpass_filter: ``(low, high)`` in Hz, or None to skip it.
        notch_filter: Notch frequency in Hz, or None to skip it.
        resample_rate: Target rate in Hz, or None to keep the native rate.

    Returns:
        ``(data, ch_names, sfreq)``: data shaped ``(n_channels, n_samples)``
        in uV, the channel names, and the rate of ``data`` in Hz.

    Examples:
        >>> from pyhealth.tasks.temple_university_EEG_tasks import read_edf_uv
        >>> data, ch_names, sfreq = read_edf_uv(
        ...     "chb01_03.edf", bandpass_filter=(0.1, 75.0), notch_filter=60.0
        ... )
    """
    raw = mne.io.read_raw_edf(file_name, preload=True, verbose="error")
    if bandpass_filter is not None:
        raw.filter(
            l_freq=bandpass_filter[0], h_freq=bandpass_filter[1], verbose="error"
        )
    if notch_filter is not None:
        raw.notch_filter(notch_filter, verbose="error")
    if resample_rate is not None:
        raw.resample(resample_rate, n_jobs=1, verbose="error")
    data = raw.get_data(units="uV")
    ch_names = list(raw.ch_names)
    sfreq = float(raw.info["sfreq"])
    raw.close()
    return data, ch_names, sfreq


class EEGEventsTUEV(BaseTask):
    """Multi-class classification task for EEG event detection on TUEV.

    Works with :class:`~pyhealth.datasets.TUEVDataset`.

    For each EDF recording, this task:
      1) reads the EDF
      2) applies bandpass (0.1-75 Hz), notch (50 Hz), resamples to
         ``resample_rate`` (200 Hz by default)
      3) loads the paired .rec file (same path, .edf -> .rec)
      4) constructs 5-second event-centered windows (16 bipolar channels)
      5) returns one sample per event

    Recordings without the ``EEG XX-REF`` channels of the averaged-reference
    montage (TUEV also has linked-ear recordings) are skipped with a
    warning, as in the BIOT preprocessing this task follows.

    Each returned sample contains:
      - "signal": torch.FloatTensor, shape (16, 5 * resample_rate) for an
        integer rate. Otherwise the window holds round(5 * resample_rate)
        samples, starting round(2 * resample_rate) samples before the event.
        The window length is fixed and the end time in the .rec file is not
        used: for a 1 s event the window covers the 2 s before it, the event
        and the 2 s after it.
      - "offending_channel": int, the channel number from the .rec file.
        It indexes the 22-channel TCP montage (0-21), not a row of
        ``signal``.
      - "label": int
      - "stft": the STFT of "signal", when ``compute_stft`` is True. It is
        built for 200 Hz signals (1 s windows, 1 Hz bins, first 100 bins),
        as TFM-Tokenizer expects, so ``compute_stft=True`` requires
        ``resample_rate=200``; any other rate raises a ``ValueError``.

    Examples:
        >>> from pyhealth.datasets import TUEVDataset
        >>> from pyhealth.tasks import EEGEventsTUEV
        >>> dataset = TUEVDataset(
        ...         root="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf/",
        ...     )
        >>> sample_dataset = dataset.set_task(EEGEventsTUEV())
        >>> sample = sample_dataset[0]
        >>> print(sample['label'])

        For a complete example, see `examples/conformal_eeg/contrawr_tuev_eeg.ipynb`.
    """

    task_name: str = "EEG_events"
    input_schema: Dict[str, str] = {"signal": "tensor", "stft": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(self,
                 resample_rate: float = 200,
                 bandpass_filter: Tuple[float, float] = (0.1, 75.0),
                 notch_filter: float = 50.0,
                 normalization: str = None, # '95th_percentile', 'div_by_100'
                 compute_stft: bool = True,
                 ) -> None:
        super().__init__()
        _check_stft_rate(resample_rate, compute_stft)

        self.resample_rate = resample_rate
        self.bandpass_filter = bandpass_filter
        self.notch_filter = notch_filter
        self.normalization = normalization
        self.compute_stft = compute_stft
        # input_schema must reflect whether stft is produced
        if not compute_stft:
            self.input_schema = {"signal": "tensor"}

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

        # A fixed number of samples per window: 2 s before the event onset,
        # 5 s in all (TUEV events last 1 s). Rounding keeps the slice and the
        # buffer the same length when resample_rate is not a whole number.
        pre = round(2 * fs)
        window = round(5 * fs)
        features = np.zeros([numEvents, numChan, window])
        offending_channel = np.zeros([numEvents, 1])
        labels = np.zeros([numEvents, 1])

        offset = signals.shape[1]
        signals = np.concatenate([signals, signals, signals], axis=1)

        for i in range(numEvents):
            chan = int(EventData[i, 0])
            start = np.where((times) >= EventData[i, 1])[0][0]
            first = offset + start - pre

            features[i, :] = signals[:, first : first + window]
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
        Rawdata.resample(resample_rate, n_jobs=1, verbose="error")

        _, times = Rawdata[:]
        signals = Rawdata.get_data(units="uV")
        RecFile = fileName[0:-3] + "rec"

        eventData = np.genfromtxt(RecFile, delimiter=",")

        Rawdata.close()
        return signals, times, eventData, Rawdata


    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes one patient. Creates one sample per event in the .rec file.

        Iterates over both 'train' and 'eval' splits.
        Each sample includes a 'split' field and a precomputed 'stft' tensor.

        Expected patient events to include a `signal_file` attribute pointing to an .edf file.
        """
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
                except KeyError as exc:
                    logger.warning(
                        "Skipping %s: channel %s not found. Only recordings "
                        "with the EEG XX-REF channels are used.",
                        edf_path,
                        exc,
                    )
                    continue
                except ValueError as exc:
                    logger.warning("Skipping %s: %s", edf_path, exc)
                    continue

                feats, offending_channels, labels = self.BuildEvents(
                    signals, times, rec, self.resample_rate
                )

                for idx, (signal, offending_channel, label) in enumerate(
                    zip(feats, offending_channels, labels)
                ):
                    if self.normalization == '95th_percentile':
                        signal = signal / (
                            np.quantile(np.abs(signal), q=0.95, axis=-1, method='linear', keepdims=True) + 1e-8
                        )
                    elif self.normalization == 'div_by_100':
                        signal = signal / 100

                    signal = torch.FloatTensor(signal)

                    sample = {
                        "patient_id": pid,
                        "signal_file": edf_path,
                        "split": split,
                        "signal": signal,
                        "offending_channel": int(offending_channel.squeeze()),
                        "label": int(label.squeeze()) - 1,
                    }
                    if self.compute_stft:
                        # get_stft_torch expects (B, C, T); unsqueeze/squeeze the batch dim
                        from pyhealth.models.tfm_tokenizer import get_stft_torch
                        sample["stft"] = get_stft_torch(signal.unsqueeze(0)).squeeze(0)

                    samples.append(sample)

        return samples
    
    

class EEGAbnormalTUAB(BaseTask):
    """Binary classification task for abnormal EEG detection on TUAB.

    Works with :class:`~pyhealth.datasets.TUABDataset`.

    For each EDF recording, this task:
      1) reads the EDF
      2) Applies bandpass (0.1-75 Hz), notch (50 Hz) and resamples to
         ``resample_rate`` (200 Hz by default)
      3) Constructs 16 bipolar channels from the TCP montage of non-overlapping 10 second windows # following BIOT
      4) assigns a binary label (1 = abnormal, 0 = normal) derived from the
         metadata ``label`` attribute

    Recordings that cannot be read, or that lack one of the 16 ``EEG XX-REF``
    electrodes of the bipolar montage, are skipped with a warning.

    Each returned sample contains:
      - "signal": torch.FloatTensor, shape (16, 10 * resample_rate) for an
        integer rate. Otherwise the window holds round(10 * resample_rate)
        samples, close to but not exactly 10 s.
      - "label": int (0 = normal, 1 = abnormal)
      - "stft": the STFT of "signal", when ``compute_stft`` is True. It is
        built for 200 Hz signals (1 s windows, 1 Hz bins, first 100 bins),
        as TFM-Tokenizer expects, so ``compute_stft=True`` requires
        ``resample_rate=200``; any other rate raises a ``ValueError``.

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
    input_schema: Dict[str, str] = {"signal": "tensor", "stft": "tensor"}
    # NOTE: TUAB is a binary classification task (normal=0 vs abnormal=1), but the
    # output schema is intentionally set to "multiclass" rather than "binary".
    # Reason: PyHealth's conformal prediction methods (LABEL, ClusterLabel,
    # NeighborhoodLabel, CovariateLabel) require multiclass mode — they calibrate
    # prediction sets by thresholding a full (n, K) probability matrix, which is
    # only produced by a softmax output (multiclass). Binary mode uses sigmoid and
    # outputs (n, 1), which is incompatible with the CP calibration math.
    # For a 2-class problem, 2-class softmax is mathematically equivalent to
    # sigmoid, so there is no loss of correctness, just a different representation.
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(self,
                 resample_rate: float = 200,
                 bandpass_filter: Tuple[float, float] = (0.1, 75.0),
                 notch_filter: float = 50.0,
                 normalization: str = None, # '95th_percentile', 'div_by_100'
                 compute_stft: bool = True,
                 ) -> None:
        super().__init__()
        _check_stft_rate(resample_rate, compute_stft)
        self.resample_rate = resample_rate
        self.bandpass_filter = bandpass_filter
        self.notch_filter = notch_filter
        self.normalization = normalization
        self.compute_stft = compute_stft
        if not compute_stft:
            self.input_schema = {"signal": "tensor"}
    @staticmethod
    def read_and_process_edf(fileName: str,
                             resample_rate: float = 200,
                             bandpass_filter: Tuple[float, float] = (0.1, 75.0),
                             notch_filter: float = 50.0,
                             ) -> Tuple[np.ndarray, List[str]]:
        raw_data, ch_name, _ = read_edf_uv(
            fileName, bandpass_filter, notch_filter, resample_rate
        )
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
        """Processes one patient. Creates one 10-second window sample per segment.

        Iterates over both 'train' and 'eval' splits. 
        Each sample includes a 'split' field and, when compute_stft=True, a precomputed 'stft' tensor.
        """
        pid = patient.patient_id
        samples: List[Dict[str, Any]] = []
        fs = self.resample_rate

        for split in ("train", "eval"):
            events = patient.get_events(split)

            for event in events:
                edf_path = event.signal_file
                label = event.label
                if label == 'normal':
                    label = 0
                elif label == 'abnormal':
                    label = 1

                try:
                    raw_data, ch_name = self.read_and_process_edf(
                        edf_path, self.resample_rate, self.bandpass_filter, self.notch_filter
                    )
                except (ValueError, KeyError) as exc:
                    logger.warning("Skipping %s: %s", edf_path, exc)
                    continue

                missing = [c for c in _TUAB_ELECTRODES if c not in ch_name]
                if missing:
                    logger.warning(
                        "Skipping %s: it has no %s %s, which the 16-channel "
                        "bipolar montage needs.",
                        edf_path,
                        ", ".join(missing),
                        "channel" if len(missing) == 1 else "channels",
                    )
                    continue
                bipolar_data = self.convert_to_bipolar(raw_data, ch_name)

                # Slice indices must be ints, and resample_rate may be a float.
                window = round(fs * 10)
                num_samples = bipolar_data.shape[1] // window
                for i in range(num_samples):
                    start = i * window
                    end = start + window
                    signal = bipolar_data[:, start:end]

                    if self.normalization == '95th_percentile':
                        signal = signal / (
                            np.quantile(np.abs(signal), q=0.95, axis=-1, method='linear', keepdims=True) + 1e-8
                        )
                    elif self.normalization == 'div_by_100':
                        signal = signal / 100

                    signal = torch.FloatTensor(signal)

                    sample = {
                        "patient_id": pid,
                        "signal_file": edf_path,
                        "split": split,
                        "signal": signal,
                        "label": label,
                        "segment_id": f'{i}',
                        "start_time": start,
                        "end_time": end,
                    }
                    if self.compute_stft:
                        # get_stft_torch expects (B, C, T); unsqueeze/squeeze the batch dim
                        from pyhealth.models.tfm_tokenizer import get_stft_torch
                        sample["stft"] = get_stft_torch(signal.unsqueeze(0)).squeeze(0)

                    samples.append(
                        sample
                    )

        return samples
            
    
    
    
