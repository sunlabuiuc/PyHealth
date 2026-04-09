"""Sleep staging task for the DREAMT dataset.

This task processes overnight Empatica E4 wearable recordings from the
DREAMT dataset into per-epoch raw signal windows with sleep stage labels,
suitable for deep learning models.

The preprocessing follows the methodology described in:

    Wang et al. "Addressing wearable sleep tracking inequity: a new
    dataset and novel methods for a population with sleep disorders."
    CHIL 2024, PMLR 248:380-396.

Signal-specific preprocessing:
    - **ACC (ACC_X, ACC_Y, ACC_Z)**: 5th-order Butterworth bandpass
      filter, 3-11 Hz (Altini & Kinnunen 2021).
    - **BVP**: Chebyshev Type II bandpass filter, 0.5-20 Hz.
    - **TEMP**: Winsorized (clipped) to [31, 40] degrees C.
    - **EDA, HR, IBI**: No additional filtering.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, cheby2, filtfilt

from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

# Label mappings per classification granularity
LABEL_MAP_5CLASS: Dict[str, int] = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
}

LABEL_MAP_3CLASS: Dict[str, int] = {
    "W": 0,
    "N1": 1,
    "N2": 1,
    "N3": 1,
    "R": 2,
}

LABEL_MAP_2CLASS: Dict[str, int] = {
    "W": 0,
    "N1": 1,
    "N2": 1,
    "N3": 1,
    "R": 1,
}

LABEL_MAPS: Dict[int, Dict[str, int]] = {
    5: LABEL_MAP_5CLASS,
    3: LABEL_MAP_3CLASS,
    2: LABEL_MAP_2CLASS,
}

ALL_SIGNAL_COLUMNS: List[str] = [
    "BVP",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "EDA",
    "TEMP",
    "HR",
    "IBI",
]

# Excluded sleep stages
EXCLUDED_STAGES = {"P", "Missing"}


def _apply_butterworth_bandpass(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: int,
    order: int = 5,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter.

    Args:
        signal: 1-D array of signal values.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered signal array.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    try:
        return filtfilt(b, a, signal)
    except ValueError:
        logger.warning(
            "Butterworth filtfilt failed (likely too few samples); "
            "returning raw signal."
        )
        return signal


def _apply_chebyshev_bandpass(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: int,
    order: int = 4,
    rs: float = 40.0,
) -> np.ndarray:
    """Apply a Chebyshev Type II bandpass filter.

    Args:
        signal: 1-D array of signal values.
        lowcut: Lower cutoff frequency in Hz.
        highcut: Upper cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order.
        rs: Minimum stopband attenuation in dB.

    Returns:
        Filtered signal array.
    """
    nyq = 0.5 * fs
    b, a = cheby2(order, rs, [lowcut / nyq, highcut / nyq], btype="band")
    try:
        return filtfilt(b, a, signal)
    except ValueError:
        logger.warning(
            "Chebyshev filtfilt failed (likely too few samples); "
            "returning raw signal."
        )
        return signal


def _apply_filters(
    epoch: np.ndarray,
    columns: List[str],
    fs: int,
) -> np.ndarray:
    """Apply signal-specific filters to one epoch.

    Args:
        epoch: Array of shape ``(n_channels, epoch_len)``.
        columns: Column names corresponding to each channel.
        fs: Sampling rate in Hz.

    Returns:
        Filtered epoch array of the same shape.
    """
    filtered = epoch.copy()
    for i, col in enumerate(columns):
        if col in ("ACC_X", "ACC_Y", "ACC_Z"):
            filtered[i] = _apply_butterworth_bandpass(
                filtered[i], lowcut=3.0, highcut=11.0, fs=fs
            )
        elif col == "BVP":
            filtered[i] = _apply_chebyshev_bandpass(
                filtered[i], lowcut=0.5, highcut=20.0, fs=fs
            )
        elif col == "TEMP":
            filtered[i] = np.clip(filtered[i], 31.0, 40.0)
        # EDA, HR, IBI: no additional filtering
    return filtered


class SleepStagingDREAMT(BaseTask):
    """Sleep staging task for the DREAMT dataset.

    Transforms each participant's overnight Empatica E4 recording into
    non-overlapping 30-second epochs of raw multi-channel signal data
    with integer sleep stage labels. Supports 5-class, 3-class, and
    2-class classification granularities.

    Signal-specific preprocessing (from Wang et al. CHIL 2024):

    - **ACC**: 5th-order Butterworth bandpass, 3-11 Hz
    - **BVP**: Chebyshev Type II bandpass, 0.5-20 Hz
    - **TEMP**: Winsorized to [31, 40] C

    Epochs labeled ``"P"`` (preparation) or ``"Missing"`` are excluded.

    Attributes:
        task_name: ``"SleepStagingDREAMT"``
        input_schema: ``{"signal": "tensor"}``
        output_schema: ``{"label": "multiclass"}``

    Args:
        n_classes: Number of classification classes. Must be one of
            ``{2, 3, 5}``. Default ``5``.

            - **5-class**: W=0, N1=1, N2=2, N3=3, R=4
            - **3-class**: W=0, NREM(N1/N2/N3)=1, REM=2
            - **2-class**: W=0, Sleep(N1/N2/N3/R)=1

        signal_columns: List of signal column names to include.
            Default includes all 8 channels: ``["BVP", "ACC_X",
            "ACC_Y", "ACC_Z", "EDA", "TEMP", "HR", "IBI"]``.
        epoch_seconds: Duration of each epoch in seconds.
            Default ``30.0``.
        sampling_rate: Sampling rate in Hz. Default ``64``.
        apply_filters: Whether to apply signal-specific filters.
            Default ``True``.

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> ds = DREAMTDataset(root="/path/to/dreamt/2.1.0")
        >>> task = SleepStagingDREAMT(n_classes=5)
        >>> sample_ds = ds.set_task(task)
        >>> sample = sample_ds.samples[0]
        >>> sample.keys()
        dict_keys(['patient_id', 'signal', 'label', 'epoch_index'])
        >>> sample["signal"].shape  # (8, 1920) for 8 channels, 30s * 64Hz
        (8, 1920)

        >>> # 2-class (wake vs sleep) with ACC channels only
        >>> task_binary = SleepStagingDREAMT(
        ...     n_classes=2,
        ...     signal_columns=["ACC_X", "ACC_Y", "ACC_Z"],
        ... )
        >>> sample_ds = ds.set_task(task_binary)
        >>> sample_ds.samples[0]["signal"].shape
        (3, 1920)
    """

    task_name: str = "SleepStagingDREAMT"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "multiclass"}

    def __init__(
        self,
        n_classes: int = 5,
        signal_columns: Optional[List[str]] = None,
        epoch_seconds: float = 30.0,
        sampling_rate: int = 64,
        apply_filters: bool = True,
    ) -> None:
        if n_classes not in {2, 3, 5}:
            raise ValueError(
                f"n_classes must be one of {{2, 3, 5}}, got {n_classes}"
            )
        self.n_classes = n_classes
        self.signal_columns = (
            list(signal_columns) if signal_columns is not None
            else list(ALL_SIGNAL_COLUMNS)
        )
        self.epoch_seconds = epoch_seconds
        self.sampling_rate = sampling_rate
        self.apply_filters = apply_filters
        self.epoch_len = int(epoch_seconds * sampling_rate)
        self.label_map = LABEL_MAPS[n_classes]
        super().__init__()

    def __call__(
        self,
        patient: Any,
    ) -> List[Dict[str, Any]]:
        """Process one DREAMT patient into epoch samples.

        Args:
            patient: A ``Patient`` object from ``DREAMTDataset``.

        Returns:
            List of sample dicts, each containing:

            - ``patient_id`` (str): Patient identifier.
            - ``signal`` (np.ndarray): Shape ``(n_channels, epoch_len)``,
              dtype ``float32``.
            - ``label`` (int): Integer sleep stage label.
            - ``epoch_index`` (int): Sequential epoch position within
              the recording.
        """
        pid: str = patient.patient_id

        try:
            events = patient.get_events(event_type="dreamt_sleep")
        except (TypeError, KeyError):
            events = patient.get_events()

        if not events:
            return []

        event = events[0]
        file_path = getattr(event, "file_64hz", None)
        if file_path is None or (
            isinstance(file_path, str) and file_path.lower() == "none"
        ):
            return []

        try:
            df = pd.read_csv(str(file_path))
        except (FileNotFoundError, pd.errors.EmptyDataError, OSError) as exc:
            logger.warning("Could not read %s: %s", file_path, exc)
            return []

        # Build case-insensitive column mapping
        col_map: Dict[str, str] = {
            col.lower(): col for col in df.columns
        }

        # Check required columns exist
        required = set(c.lower() for c in self.signal_columns) | {
            "sleep_stage"
        }
        missing = [c for c in required if c not in col_map]
        if missing:
            logger.warning(
                "Patient %s missing columns: %s", pid, missing
            )
            return []

        # Resolve actual column names
        stage_col = col_map["sleep_stage"]
        signal_col_names = [
            col_map[c.lower()] for c in self.signal_columns
        ]

        # Drop excluded stages
        mask = ~df[stage_col].isin(EXCLUDED_STAGES)
        df = df.loc[mask].reset_index(drop=True)

        if len(df) < self.epoch_len:
            return []

        n_epochs = len(df) // self.epoch_len
        samples: List[Dict[str, Any]] = []
        epoch_counter = 0

        for i in range(n_epochs):
            start = i * self.epoch_len
            end = start + self.epoch_len
            epoch_df = df.iloc[start:end]

            # Label from the middle of the epoch
            mid = start + self.epoch_len // 2
            stage = str(df[stage_col].iloc[mid]).strip()

            if stage not in self.label_map:
                continue

            label = self.label_map[stage]

            # Extract signal channels as (n_channels, epoch_len)
            signal = np.stack(
                [
                    epoch_df[col].values.astype(np.float64)
                    for col in signal_col_names
                ],
                axis=0,
            )

            # Replace NaN with 0
            np.nan_to_num(signal, nan=0.0, copy=False)

            # Apply signal-specific filters
            if self.apply_filters:
                signal = _apply_filters(
                    signal, self.signal_columns, self.sampling_rate
                )

            signal = signal.astype(np.float32)

            samples.append(
                {
                    "patient_id": pid,
                    "signal": signal,
                    "label": label,
                    "epoch_index": epoch_counter,
                }
            )
            epoch_counter += 1

        return samples
