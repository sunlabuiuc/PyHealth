"""Sleep-wake detection task for the DREAMT dataset.

This task processes overnight Empatica E4 wearable recordings from the
DREAMT dataset into per-epoch feature vectors with binary sleep/wake labels,
suitable for classical and deep learning models.

The feature extraction follows the methodology described in:
    Wang et al. "Addressing wearable sleep tracking inequity: a new
    dataset and novel methods for a population with sleep disorders."
    CHIL 2024, PMLR 248:380-396.

Per-epoch feature extraction (30-second windows, 1920 samples at 64Hz):
    - **ACC (ACC_X, ACC_Y, ACC_Z)**: Trimmed mean, max, IQR, and MAD
      extracted from each axis after Butterworth bandpass filtering (3-11 Hz).
    - **TEMP**: Mean, min, max, and standard deviation after winsorization
      to [31, 40] degrees C.
    - **BVP**: Mean and standard deviation per epoch after Chebyshev Type II
      bandpass filtering (0.5-20 Hz).
    - **EDA**: Mean tonic and phasic components (mean and std per epoch).
    - **HR**: Mean and standard deviation per epoch.
    - **ACC_INDEX**: Activity index derived from accelerometry magnitude.
    - **HRV_HFD**: Higuchi Fractal Dimension derived from BVP — one of the
      two strongest predictors identified in the paper.

Labels are binary: wake (PSG Sleep_Stage == "W") mapped to 1, all sleep
stages (N1, N2, N3, R) mapped to 0. Preparation ("P") and "Missing"
epochs are excluded per the DREAMT 2.1.0 schema.

Dataset link:
    https://physionet.org/content/dreamt/2.1.0/
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, cheby2

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

EPOCH_SAMPLES = 1920        # 64 Hz × 30 seconds
WAKE_LABEL = "W"
EXCLUDE_LABELS = {"P", "Missing"}

# Feature names produced by extract_epoch_features()
FEATURE_COLUMNS = [
    "ACC_X_mean", "ACC_X_max", "ACC_X_iqr", "ACC_X_mad",
    "ACC_Y_mean", "ACC_Y_max", "ACC_Y_iqr", "ACC_Y_mad",
    "ACC_Z_mean", "ACC_Z_max", "ACC_Z_iqr", "ACC_Z_mad",
    "TEMP_mean", "TEMP_min", "TEMP_max", "TEMP_std",
    "BVP_mean", "BVP_std",
    "EDA_mean", "EDA_std",
    "HR_mean", "HR_std",
    "ACC_INDEX",
    "HRV_HFD",
]


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def _butterworth_bandpass(signal: np.ndarray, fs: float = 64.0) -> np.ndarray:
    """5th-order Butterworth bandpass filter, 3-11 Hz (ACC preprocessing)."""
    sos = butter(5, [3.0, 11.0], btype="band", fs=fs, output="sos")
    return sosfilt(sos, signal)


def _chebyshev_bandpass(signal: np.ndarray, fs: float = 64.0) -> np.ndarray:
    """Chebyshev Type II bandpass filter, 0.5-20 Hz (BVP preprocessing)."""
    sos = cheby2(4, 40, [0.5, 20.0], btype="band", fs=fs, output="sos")
    return sosfilt(sos, signal)


def _winsorize(signal: np.ndarray, low: float = 31.0, high: float = 40.0) -> np.ndarray:
    """Clip signal to [low, high] (TEMP preprocessing)."""
    return np.clip(signal, low, high)


def _trimmed_mean(signal: np.ndarray, pct: float = 0.10) -> float:
    """Mean after removing pct from each tail."""
    n = len(signal)
    k = int(n * pct)
    return float(np.mean(np.sort(signal)[k: n - k]))


def _acc_index(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Activity index: std of vector magnitude over the epoch."""
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    return float(np.std(magnitude))


def _higuchi_fd(signal: np.ndarray, kmax: int = 10) -> float:
    """Higuchi Fractal Dimension of a 1-D signal (HRV_HFD)."""
    n = len(signal)
    lk = []
    for k in range(1, kmax + 1):
        lm = []
        for m in range(1, k + 1):
            indices = np.arange(m - 1, n, k)
            if len(indices) < 2:
                continue
            norm = (n - 1) / (len(indices) * k)
            lm.append(norm * np.sum(np.abs(np.diff(signal[indices]))))
        if lm:
            lk.append(np.mean(lm))
    if len(lk) < 2:
        return 0.0
    x = np.log(np.arange(1, len(lk) + 1, dtype=float))
    y = np.log(np.array(lk, dtype=float))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def extract_epoch_features(epoch: pd.DataFrame) -> List[float]:
    """Extract the full feature vector from one 30-second epoch DataFrame.

    Args:
        epoch: DataFrame slice of 1920 rows with columns BVP, ACC_X, ACC_Y,
            ACC_Z, TEMP, EDA, HR.

    Returns:
        List of floats matching FEATURE_COLUMNS order.
    """
    # ACC — filter then extract
    features = []
    for axis in ["ACC_X", "ACC_Y", "ACC_Z"]:
        raw = epoch[axis].to_numpy(dtype=float)
        filtered = _butterworth_bandpass(raw)
        abs_filtered = np.abs(filtered)
        features += [
            _trimmed_mean(abs_filtered),
            float(np.max(abs_filtered)),
            float(np.percentile(abs_filtered, 75) - np.percentile(abs_filtered, 25)),
            float(np.mean(np.abs(raw - np.mean(raw)))),  # MAD from raw
        ]

    # TEMP — winsorize then stats
    temp = _winsorize(epoch["TEMP"].to_numpy(dtype=float))
    features += [
        float(np.mean(temp)),
        float(np.min(temp)),
        float(np.max(temp)),
        float(np.std(temp)),
    ]

    # BVP — filter then stats
    bvp = _chebyshev_bandpass(epoch["BVP"].to_numpy(dtype=float))
    features += [float(np.mean(bvp)), float(np.std(bvp))]

    # EDA — stats on raw
    eda = epoch["EDA"].to_numpy(dtype=float)
    features += [float(np.mean(eda)), float(np.std(eda))]

    # HR — stats on raw
    hr = epoch["HR"].to_numpy(dtype=float)
    features += [float(np.mean(hr)), float(np.std(hr))]

    # ACC_INDEX
    features.append(_acc_index(
        epoch["ACC_X"].to_numpy(dtype=float),
        epoch["ACC_Y"].to_numpy(dtype=float),
        epoch["ACC_Z"].to_numpy(dtype=float),
    ))

    # HRV_HFD — on filtered BVP
    features.append(_higuchi_fd(bvp))

    return features


# Task class

class SleepWakeDetectionDREAMT(BaseTask):
    """Binary sleep-wake classification task for the DREAMT dataset.

    Each sample corresponds to one 30-second epoch from a single participant.
    Raw E4 wristband signals are read from the patient's 64Hz CSV file,
    segmented into 1920-sample windows, and statistical features are extracted
    per epoch following Wang et al. (2024).

    Attributes:
        task_name (str): ``"SleepWakeDetectionDREAMT"``
        input_schema (Dict[str, str]): ``{"features": "tensor"}``
        output_schema (Dict[str, str]): ``{"label": "binary"}``
        feature_columns (List[str]): Feature names in extraction order.

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> from pyhealth.tasks import SleepWakeDetectionDREAMT
        >>> dataset = DREAMTDataset(root="/path/to/dreamt/2.1.0/")
        >>> task = SleepWakeDetectionDREAMT()
        >>> samples = dataset.set_task(task)
        >>> samples[0]
        {
            'patient_id': 'S002',
            'epoch_index': 0,
            'ahi': 18.0,
            'bmi': 37.0,
            'features': [...],
            'label': 0
        }
    """

    task_name: str = "SleepWakeDetectionDREAMT"
    input_schema: Dict[str, str] = {"features": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, feature_columns: Optional[List[str]] = None) -> None:
        """Initializes the task.

        Args:
            feature_columns: Subset of FEATURE_COLUMNS to include. If None,
                all features are used.
        """
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        super().__init__()

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes one patient into a list of per-epoch samples.

        Args:
            patient: A patient object from DREAMTDataset with a
                ``dreamt_sleep`` event containing ``file_64hz``.

        Returns:
            List of dicts, one per valid epoch, each with keys:
            ``patient_id``, ``epoch_index``, ``ahi``, ``bmi``,
            ``features``, ``label``.
        """
        events = patient.get_events(event_type="dreamt_sleep")
        if not events:
            logger.warning("Patient %s has no dreamt_sleep events.", patient.patient_id)
            return []

        event = events[0]
        feature_file = event.file_64hz

        if feature_file is None:
            logger.warning("Patient %s has no file_64hz path.", patient.patient_id)
            return []

        try:
            df = pd.read_csv(feature_file)
        except Exception as exc:
            logger.warning("Could not read file for patient %s: %s", patient.patient_id, exc)
            return []

        required = {"BVP", "ACC_X", "ACC_Y", "ACC_Z", "TEMP", "EDA", "HR", "Sleep_Stage"}
        if not required.issubset(df.columns):
            logger.warning("Patient %s missing required columns.", patient.patient_id)
            return []

        samples = []
        n_epochs = len(df) // EPOCH_SAMPLES

        for i in range(n_epochs):
            start = i * EPOCH_SAMPLES
            end = start + EPOCH_SAMPLES
            epoch = df.iloc[start:end]

            # Get the majority sleep stage label for this epoch
            stage = epoch["Sleep_Stage"].mode()[0]

            # Skip preparation and missing epochs
            if stage in EXCLUDE_LABELS:
                continue

            try:
                all_features = extract_epoch_features(epoch)
            except Exception as exc:
                logger.warning("Feature extraction failed for patient %s epoch %d: %s",
                               patient.patient_id, i, exc)
                continue

            # Subset to requested feature columns
            if self.feature_columns != FEATURE_COLUMNS:
                col_idx = [FEATURE_COLUMNS.index(c) for c in self.feature_columns]
                all_features = [all_features[j] for j in col_idx]

            label = int(stage == WAKE_LABEL)

            samples.append({
                "patient_id": patient.patient_id,
                "epoch_index": i,
                "ahi": float(getattr(event, "ahi", float("nan"))),
                "bmi": float(getattr(event, "bmi", float("nan"))),
                "features": all_features,
                "label": label,
            })

        return samples