import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, cheby2, filtfilt

from .base_task import BaseTask

logger = logging.getLogger(__name__)

# Sampling frequency of the E4 after upsampling (paper sec 2.3)
SAMPLE_RATE_HZ = 64
EPOCH_SEC = 30
EPOCH_SAMPLES = SAMPLE_RATE_HZ * EPOCH_SEC  # 1920 samples per 30-sec epoch

# E4 signal columns used as features (paper sec 2.3)
SIGNAL_COLS = ["BVP", "ACC_X", "ACC_Y", "ACC_Z", "EDA", "TEMP", "HR"]

# PSG sleep stage label constants
WAKE_LABEL = "W"
SLEEP_LABELS = {"R", "N1", "N2", "N3"}
MISSING_LABEL = "Missing"

# Fine-grained label map for multi-class staging
FINE_LABEL_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}

# Coarse label map for binary wake/sleep detection
BINARY_LABEL_MAP = {"W": 1, "N1": 0, "N2": 0, "N3": 0, "R": 0}


def _butter_bandpass(
    signal: np.ndarray,
    low: float,
    high: float,
    fs: int,
    order: int = 5,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to a 1D signal.

    Args:
        signal: 1D numpy array of signal values
        low: lower cutoff frequency in Hz
        high: upper cutoff frequency in Hz
        fs: sampling frequency in Hz
        order: filter order

    Returns:
        Filtered signal as numpy array
    """
    try:
        nyq = fs / 2.0
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, signal).astype(np.float32)
    except Exception as e:
        logger.debug(f"_butter_bandpass failed: {e}")
        return signal.astype(np.float32)
   
    
def _cheby_bandpass(
    signal: np.ndarray,
    low: float,
    high: float,
    fs: int,
    order: int = 4,
    rs: float = 20.0,
) -> np.ndarray:
    """Apply a Chebyshev type II bandpass filter to a 1D signal.

    Args:
        signal: 1D numpy array of signal values
        low: lower cutoff frequency in Hz
        high: upper cutoff frequency in Hz
        fs: sampling frequency in Hz
        order: filter order
        rs: Minimum stopband attenuation in dB.

    Returns:
        Filtered signal as numpy array
    """
    try:
        nyq = fs / 2.0
        b, a = cheby2(order, rs, [low / nyq, high / nyq], btype="bandpass")
        return filtfilt(b, a, signal).astype(np.float32)
    except Exception as e:
        logger.debug(f"_cheby_bandpass failed: {e}")
        return signal.astype(np.float32)


def _butter_lowpass(
    signal: np.ndarray,
    cutoff: float,
    fs: int,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth lowpass filter to a 1D signal.

    Args:
        signal: 1D numpy array of signal values
        cutoff: cutoff frequency in Hz
        fs: sampling frequency in Hz
        order: filter order

    Returns:
        Filtered signal as numpy array
    """
    try:
        nyq = fs / 2.0
        b, a = butter(order, cutoff / nyq, btype="low")
        return filtfilt(b, a, signal).astype(np.float32)
    except Exception as e:
        logger.debug(f"_butter_lowpass failed: {e}")
        return signal.astype(np.float32)


def _segment_detrend(
    signal: np.ndarray,
    segment_seconds: int = 5,
    fs: int = 64,
) -> np.ndarray:
    """Detrend signal by subtracting least-squares line from each segment.

    Follows EDA preprocessing from paper section 2.5.

    Args:
        signal: 1D numpy array of EDA values
        segment_seconds: length of each detrending segment in seconds
        fs: sampling frequency in Hz

    Returns:
        Detrended signal as numpy array
    """
    seg_len = segment_seconds * fs
    out = signal.copy().astype(np.float32)
    for start in range(0, len(signal), seg_len):
        seg = signal[start: start + seg_len]
        if len(seg) < 2:
            continue
        x = np.arange(len(seg), dtype=np.float32)
        coeffs = np.polyfit(x, seg, 1)
        out[start: start + len(seg)] = seg - np.polyval(coeffs, x)
    return out


def extract_epoch_features(
    epoch_df: pd.DataFrame,
    fs: int = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Extract statistical and signal-processing features from one 30-sec epoch.

    Implements the feature engineering described in paper section 2.5:
    - ACC: bandpass filtered (3-11 Hz), trimmed mean, max, IQR, MAD per axis
    - TEMP: winsorized to 31-40C, mean, min, max, std
    - BVP: bandpass filtered (0.5-20 Hz), basic HRV-proxy stats
    - EDA: detrended, lowpass filtered, mean and std of phasic component
    - HR: mean and std

    Args:
        epoch_df: DataFrame slice for one 30-second epoch containing
            BVP, ACC_X, ACC_Y, ACC_Z, EDA, TEMP, HR columns
        fs: sampling frequency in Hz (default 64)

    Returns:
        np.ndarray of shape (n_features,) — float32 feature vector
    """
    feats: List[float] = []

    def safe_col(col: str) -> np.ndarray:
        if col in epoch_df.columns:
            return epoch_df[col].to_numpy(dtype=np.float32)
        return np.zeros(len(epoch_df), dtype=np.float32)

    def summary_stats(x: np.ndarray) -> List[float]:
        """Mean, std, min, max of array."""
        return [
            float(np.mean(x)),
            float(np.std(x)),
            float(np.min(x)),
            float(np.max(x)),
        ]

    def trimmed_stats(x: np.ndarray) -> List[float]:
        """Trimmed mean (10%), max, IQR of absolute values."""
        if len(x) == 0:
            return [0.0, 0.0, 0.0]
        n = max(1, int(0.1 * len(x)))
        sorted_x = np.sort(np.abs(x))
        trimmed = sorted_x[n:-n] if len(sorted_x) > 2 * n else sorted_x
        return [
            float(np.mean(trimmed)),
            float(np.max(np.abs(x))),
            float(np.percentile(np.abs(x), 75) - np.percentile(np.abs(x), 25)),
        ]

    # ── ACC features (paper sec 2.5) ─────────────────────────────────────────
    for col in ["ACC_X", "ACC_Y", "ACC_Z"]:
        raw = safe_col(col)
        # Bandpass filter 3-11 Hz (Oura method cited in paper)
        filtered = _butter_bandpass(raw, low=3.0, high=11.0, fs=fs, order=5)
        feats.extend(trimmed_stats(filtered))
        # MAD from vector magnitude
        mag = np.sqrt(np.mean(raw ** 2))
        feats.append(float(np.mean(np.abs(raw - mag))))

    # ── TEMP features (paper sec 2.5) ────────────────────────────────────────
    temp = np.clip(safe_col("TEMP"), 31.0, 40.0)  # winsorize to 31-40C
    feats.extend(summary_stats(temp))

    # ── BVP / HRV features (paper sec 2.5) ───────────────────────────────────
    bvp = safe_col("BVP")
    bvp_filt = _cheby_bandpass(
        bvp, 
        low=0.5, 
        high=20.0, 
        fs=fs, 
        order=4, 
        rs=20.0)
    feats.extend(summary_stats(bvp_filt))

    # ── EDA features (paper sec 2.5) ─────────────────────────────────────────
    eda = safe_col("EDA")
    eda_detrended = _segment_detrend(eda, segment_seconds=5, fs=fs)
    eda_filtered = _butter_lowpass(eda_detrended, cutoff=1.0, fs=fs, order=4)
    feats.extend([float(np.mean(eda_filtered)), float(np.std(eda_filtered))])

    # ── HR features ──────────────────────────────────────────────────────────
    hr = safe_col("HR")
    feats.extend([float(np.mean(hr)), float(np.std(hr))])

    return np.array(feats, dtype=np.float32)


class SleepWakeDetectionDREAMT(BaseTask):
    """Binary sleep/wake detection task for the DREAMT dataset.

    Based on: Wang et al. (2024). Addressing wearable sleep tracking inequity:
    a new dataset and novel methods for a population with sleep disorders.
    CHIL 2024, PMLR 248:380-396.

    This task processes overnight Empatica E4 wearable recordings from patients
    with sleep disorders. Each night is sliced into non-overlapping 30-second
    epochs. Each epoch is labeled as Wake (1) or Sleep (0) based on concurrent
    PSG annotations. Clinical metadata (AHI, BMI) is included per epoch to
    support mixed-effects modeling as described in paper section 2.6.

    Feature engineering follows paper section 2.5:
    - ACC: Butterworth bandpass (3-11 Hz), trimmed mean, max, IQR, MAD
    - TEMP: winsorized (31-40C), mean, std, min, max
    - BVP: Chebyshev bandpass (0.5-20 Hz), summary stats
    - EDA: segment detrended, lowpass filtered, mean and std
    - HR: mean and std

    Attributes:
        task_name (str): "SleepWakeDetectionDREAMT"
        input_schema (Dict[str, str]): Input features per epoch:
            - "signal": float feature vector from extract_epoch_features()
            - "ahi": float Apnea-Hypopnea Index (random effect)
            - "bmi": float Body Mass Index (random effect)
        output_schema (Dict[str, str]): Binary label:
            - "label": 1 = Wake, 0 = Sleep

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> from pyhealth.tasks import SleepWakeDetectionDREAMT
        >>> dataset = DREAMTDataset(root="/path/to/dreamt/2.1.0")
        >>> task = SleepWakeDetectionDREAMT()
        >>> samples = dataset.set_task(task)
        >>> samples[0]
        {
            'patient_id': 'S002',
            'epoch_index': 0,
            'signal': array of shape (n_features,),
            'ahi': 22.1,
            'bmi': 33.7,
            'label': 0
        }
    """

    task_name: str = "SleepWakeDetectionDREAMT"

    input_schema: Dict[str, str] = {
        # Engineered feature vector per 30-sec epoch
        "signal": "float",
        # Clinical metadata for mixed-effects modeling (paper sec 2.6)
        "ahi": "float",
        "bmi": "float",
    }

    output_schema: Dict[str, str] = {
        # Binary classification: 1 = Wake, 0 = Sleep
        "label": "binary",
    }

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process one DREAMT patient into a list of 30-second epoch samples.

        Args:
            patient: A patient object from DREAMTDataset. Must have a
                dreamt_sleep event containing file_64hz, ahi, and bmi.

        Returns:
            List of dicts, one per valid epoch. Each dict contains:
                - patient_id (str): participant identifier
                - epoch_index (int): index of this epoch in the night
                - signal (np.ndarray): float32 feature vector
                - ahi (float): Apnea-Hypopnea Index
                - bmi (float): Body Mass Index
                - label (int): 1 = Wake, 0 = Sleep
        """
        return _process_patient(
            patient=patient,
            label_map=BINARY_LABEL_MAP,
        )


class SleepStagingDREAMT(BaseTask):
    """Multi-class sleep staging task for the DREAMT dataset.

    Based on: Wang et al. (2024). Addressing wearable sleep tracking inequity:
    a new dataset and novel methods for a population with sleep disorders.
    CHIL 2024, PMLR 248:380-396.

    This task extends SleepWakeDetectionDREAMT to predict all five PSG-derived
    sleep stages: Wake (0), N1 (1), N2 (2), N3 (3), REM (4). Feature
    engineering follows the same pipeline as SleepWakeDetectionDREAMT.

    Attributes:
        task_name (str): "SleepStagingDREAMT"
        input_schema (Dict[str, str]): Input features per epoch:
            - "signal": float feature vector from extract_epoch_features()
            - "ahi": float Apnea-Hypopnea Index
            - "bmi": float Body Mass Index
        output_schema (Dict[str, str]): Multi-class label:
            - "label": 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> from pyhealth.tasks import SleepStagingDREAMT
        >>> dataset = DREAMTDataset(root="/path/to/dreamt/2.1.0")
        >>> task = SleepStagingDREAMT()
        >>> samples = dataset.set_task(task)
        >>> samples[0]
        {
            'patient_id': 'S002',
            'epoch_index': 0,
            'signal': array of shape (n_features,),
            'ahi': 22.1,
            'bmi': 33.7,
            'label': 2
        }
    """

    task_name: str = "SleepStagingDREAMT"

    input_schema: Dict[str, str] = {
        "signal": "float",
        "ahi": "float",
        "bmi": "float",
    }

    output_schema: Dict[str, str] = {
        # Multi-class: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
        "label": "multiclass",
    }

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process one DREAMT patient into multi-class sleep stage samples.

        Args:
            patient: A patient object from DREAMTDataset.

        Returns:
            List of dicts, one per valid epoch, with label in {0,1,2,3,4}.
        """
        return _process_patient(
            patient=patient,
            label_map=FINE_LABEL_MAP,
        )


def _process_patient(
    patient: Any,
    label_map: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Shared processing logic for both task classes.

    Reads the patient's overnight E4 CSV, slices into 30-second epochs,
    performs signal quality filtering, extracts features, and maps PSG 
    labels using the provided label_map.

    Args:
        patient: A patient object from DREAMTDataset
        label_map: dict mapping PSG stage strings to integer labels.
            Use BINARY_LABEL_MAP for wake/sleep or FINE_LABEL_MAP for
            5-class staging.

    Returns:
        List of sample dicts, one per valid labeled epoch.
    """
    samples: List[Dict[str, Any]] = []

    events = patient.get_events(event_type="dreamt_sleep")
    if not events:
        return samples
    event = events[0]

    file_path = event.file_64hz
    if file_path is None:
        return samples

    try:
        ahi = float(event.ahi) if event.ahi is not None else 0.0
        bmi = float(event.bmi) if event.bmi is not None else 0.0
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Invalid AHI/BMI for patient {patient.patient_id}: {e}"
        )
        ahi, bmi = 0.0, 0.0

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.warning(f"Failed to read file {file_path}: {e}")
        return samples

    # Verify required columns
    required = SIGNAL_COLS + ["Sleep_Stage"]
    if any(c not in df.columns for c in required):
        logger.warning(
            f"Missing required columns in file {file_path}"
        )
        return samples

    n_epochs = len(df) // EPOCH_SAMPLES

    for epoch_idx in range(n_epochs):
        start = epoch_idx * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        epoch_df = df.iloc[start:end]

        # Get PSG label for this epoch
        stage = epoch_df["Sleep_Stage"].iloc[-1]
        if stage == MISSING_LABEL or pd.isna(stage):
            continue

        label = label_map.get(str(stage).strip())
        if label is None:
            continue

        # Extract engineered features
        signal = extract_epoch_features(epoch_df)

        # Skip epochs with invalid features
        if np.isnan(signal).any() or np.isinf(signal).any():
            continue

        samples.append(
            {
                "patient_id": patient.patient_id,
                "epoch_index": epoch_idx,
                "signal": signal,
                "ahi": ahi,
                "bmi": bmi,
                "label": label,
            }
        )

    return samples