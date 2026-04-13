import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict
import numpy as np
from pyhealth.data import Patient
from pyhealth.tasks import BaseTask
import neurokit2 as nk

logger = logging.getLogger(__name__)

# Profusion stages → 3-class: 0=Wake, 1=NREM, 2=REM
_STAGE_MAP = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2}


class SleepStagingSHHS(BaseTask):
    """Sleep staging task for WatchSleepNet on the SHHS dataset.

    Implements the ECG to IBI preprocessing pipeline described in the
    WatchSleepNet paper (Wang et al., CHIL 2025). Each sample is a
    sequence of seq_len consecutive 30 second IBI epochs.

    Preprocessing pipeline:
        1. Read EDF, extract ECG channel
        2. Parse Profusion XML for 30-second sleep-stage annotations
        3. Detect R-peaks (biosppy) and compute inter-beat intervals (IBI)
        4. Remove physiologically implausible IBIs (>= 2 s)
        5. Downsample the IBI signal to target_hz (default 25 Hz)
        6. Segment into 30-second epochs (750 samples at 25 Hz)
        7. Map stages to 3 classes: Wake (0), NREM (1), REM (2)
        8. Generate sliding-window sequences of seq_len epochs

    Each returned sample contains:
        - signal: ndarray of shape (seq_len, target_hz * epoch_seconds)
        - label: int in {0, 1, 2} — the stage of the last epoch

    Args:
        epoch_seconds: Duration of each epoch in seconds (default 30)
        seq_len: Number of consecutive epochs per sample (default 20)
        target_hz: Target sampling rate after downsampling (default 25)
        max_epochs: Maximum epochs to keep per recording (default 1100)

    Examples:
        >>> from pyhealth.datasets import SHHSDataset
        >>> from pyhealth.tasks import SleepStagingSHHS
        >>> dataset = SHHSDataset(root="/path/to/shhs")
        >>> task = SleepStagingSHHS()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "SleepStagingSHHS"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        epoch_seconds: int = 30,
        seq_len: int = 20,
        target_hz: int = 25,
        max_epochs: int = 1100,
    ):
        self.epoch_seconds = epoch_seconds
        self.seq_len = seq_len
        self.target_hz = target_hz
        self.max_epochs = max_epochs
        super().__init__()

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        pid = patient.patient_id
        events = patient.get_events(event_type="shhs_sleep")
        all_samples: list[dict[str, Any]] = []

        for event in events:
            try:
                samples = self._process_event(pid, event)
                all_samples.extend(samples)
            except (ValueError, RuntimeError, OSError):
                logger.warning(
                    "Skipping %s visit %s due to processing error",
                    pid,
                    getattr(event, "visitnumber", "?"),
                    exc_info=True,
                )

        return all_samples

    def _process_event(
        self, pid: str, event: Any
    ) -> list[dict[str, Any]]:
        import mne
        import time

        visit = getattr(event, "visitnumber", "?")
        samples_per_epoch = self.target_hz * self.epoch_seconds

        logger.info("Patient %s visit %s: reading EDF...", pid, visit)
        t0 = time.time()
        raw = mne.io.read_raw_edf(
            event.signal_file, preload=True, verbose="error"
        )
        ecg_idx = _pick_ecg_channel(raw.ch_names)
        ecg_signal = raw.get_data(picks=[ecg_idx]).squeeze()
        source_hz = int(float(event.ecg_sample_rate))
        logger.info(
            "Patient %s visit %s: EDF loaded (%.1fs, %d samples at %d Hz)",
            pid, visit, time.time() - t0, len(ecg_signal), source_hz,
        )

        stages = _parse_profusion_stages(event.annotation_file)

        logger.info(
            "Patient %s visit %s: detecting R-peaks...", pid, visit
        )
        t0 = time.time()
        ibi = _ecg_to_ibi(ecg_signal, source_hz)
        logger.info(
            "Patient %s visit %s: R-peak detection done (%.1fs)",
            pid, visit, time.time() - t0,
        )

        # Downsample IBI to target_hz
        ibi_ds = _downsample(ibi, source_hz, self.target_hz)

        # Segment into epochs
        num_signal_epochs = len(ibi_ds) // samples_per_epoch
        num_epochs = min(num_signal_epochs, len(stages), self.max_epochs)

        if num_epochs < self.seq_len:
            return []

        epoch_signals = ibi_ds[: num_epochs * samples_per_epoch].reshape(
            num_epochs, samples_per_epoch
        )

        # Map stages to 3 classes
        epoch_labels = [_STAGE_MAP.get(stages[i], -1) for i in range(num_epochs)]
        valid_mask = [lbl != -1 for lbl in epoch_labels]

        # Build samples: slide a window of seq_len epochs, label = last epoch
        samples: list[dict[str, Any]] = []
        for i in range(num_epochs - self.seq_len + 1):
            window_end = i + self.seq_len
            if not all(valid_mask[i:window_end]):
                continue

            samples.append(
                {
                    "patient_id": pid,
                    "record_id": f"{pid}-v{visit}-{i}",
                    "signal": epoch_signals[i:window_end].astype(np.float32),
                    "label": epoch_labels[window_end - 1],
                }
            )

        logger.info(
            "Patient %s visit %s: %d epochs, %d samples generated",
            pid, visit, num_epochs, len(samples),
        )
        return samples


def _pick_ecg_channel(ch_names: list[str]) -> int:
    """Return the index of the ECG channel."""
    for i, name in enumerate(ch_names):
        if "ecg" in name.lower():
            return i
    raise ValueError(f"No ECG channel found in {ch_names}")


def _parse_profusion_stages(xml_path: str) -> list[int]:
    """Parse per-epoch sleep stages from a Profusion XML file.

    Returns a list of integer stage codes (0-5), one per 30-second epoch.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    stage_elements = root.find("SleepStages")
    if stage_elements is None:
        raise ValueError(f"No <SleepStages> element in {xml_path}")
    return [
        int(s.text)
        for s in stage_elements.findall("SleepStage")
        if s.text is not None
    ]


def _ecg_to_ibi(ecg_signal: np.ndarray, fs: int) -> np.ndarray:
    """Compute continuous IBI array from an ECG signal.

    Uses neurokit2 for R-peak detection, then forward-fills the IBI value
    between consecutive peaks. IBIs >= 2.0 s are zeroed out.
    """

    cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    rpeaks = info["ECG_R_Peaks"]

    ibi = np.zeros(len(ecg_signal), dtype=np.float32)
    if len(rpeaks) < 2:
        return ibi

    ibi_values = np.diff(rpeaks) / float(fs)
    for i in range(len(rpeaks) - 1):
        ibi[rpeaks[i]:rpeaks[i + 1]] = ibi_values[i]
    ibi[rpeaks[-1]:] = ibi_values[-1]

    ibi[ibi >= 2.0] = 0.0
    return ibi


def _downsample(signal: np.ndarray, source_hz: int, target_hz: int) -> np.ndarray:
    """Downsample a signal from source_hz to target_hz."""
    if source_hz == target_hz:
        return signal
    if source_hz % target_hz == 0:
        return signal[:: source_hz // target_hz]

    from scipy.signal import resample_poly

    return np.asarray(resample_poly(signal, target_hz, source_hz), dtype=np.float32)
