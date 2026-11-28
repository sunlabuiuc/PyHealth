"""Sleep staging task for the DREAMT dataset using inter-beat-interval (IBI).

Builds the input expected by :class:`~pyhealth.models.WatchSleepNet`: windows
of ``T`` consecutive 30 s epochs, each epoch holding 750 IBI values (25 Hz
grid), with one Wake / NREM / REM label per epoch.

Reference:
    Wang et al. "WatchSleepNet: A Novel Model and Pretraining Approach for
    Advancing Sleep Staging with Smartwatches." CHIL 2025 (PMLR 287).
"""

from pathlib import Path
from typing import Any, ClassVar
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

from pyhealth.tasks import BaseTask

IGNORE_INDEX = -100
STAGE_SCHEMES: dict[str, dict[str, int]] = {
    "3class": {"W": 0, "N1": 1, "N2": 1, "N3": 1, "R": 2},  # Wake/NREM/REM
    "4class": {"W": 0, "N1": 1, "N2": 1, "N3": 2, "R": 3},  # Wake/Light/Deep/REM
    "5class": {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4},  # Wake/N1/N2/N3/REM
}
NUM_CLASSES: dict[str, int] = {"3class": 3, "4class": 4, "5class": 5}


def map_stages(
    stages: pd.Series, ignore_prep: bool = False, scheme: str = "3class"
) -> np.ndarray:
    """Maps DREAMT stage strings to class indices (or -100 to ignore).

    Args:
        stages: Per-row ``Sleep_Stage`` values.
        ignore_prep: If True the preparation stage ``P`` is ignored; if False
            it is treated as Wake.
        scheme: ``"3class"``, ``"4class"`` or ``"5class"``; see
            ``STAGE_SCHEMES``.

    Returns:
        An int64 array with the same length as ``stages``. Unknown values
        (for example ``"Missing"``) become ``-100``.
    """
    mapping = dict(STAGE_SCHEMES[scheme])
    mapping["P"] = IGNORE_INDEX if ignore_prep else 0
    return stages.map(mapping).fillna(IGNORE_INDEX).to_numpy(dtype=np.int64)


def epoch_labels(
    row_labels: np.ndarray, rows_per_epoch: int, n_epochs: int, num_classes: int = 3
) -> np.ndarray:
    """Takes the majority class of each epoch.

    Args:
        row_labels: Per-row class labels (``-100`` for ignored rows).
        rows_per_epoch: Number of rows in one epoch.
        n_epochs: Number of full epochs to label.
        num_classes: Number of classes in the label scheme.

    Returns:
        An int64 array of shape ``(n_epochs,)``. An epoch is ``-100`` when
        fewer than half of its rows carry a valid label.
    """
    rows = row_labels[: n_epochs * rows_per_epoch].reshape(n_epochs, -1)
    counts = np.stack([(rows == c).sum(axis=1) for c in range(num_classes)], axis=1)
    labels = counts.argmax(axis=1).astype(np.int64)
    labels[counts.sum(axis=1) < rows_per_epoch / 2] = IGNORE_INDEX
    return labels


def beats_from_bvp(bvp: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Detects pulse peaks in a BVP signal and returns the beat intervals.

    Uses a band-pass filter and per-epoch standardisation before peak picking
    (a lightweight stand-in for the NeuroKit pipeline used in the paper).

    Args:
        bvp: Raw BVP samples.
        fs: Sampling rate in Hz.

    Returns:
        ``(times, intervals)`` in seconds, where ``intervals[k]`` is the gap
        between peak ``k`` and peak ``k + 1`` and ``times[k]`` the time of
        peak ``k + 1``.
    """
    bvp = np.nan_to_num(bvp.astype(float))
    sos = butter(2, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
    filt = sosfiltfilt(sos, bvp)
    win = int(30 * fs)
    n_win = len(filt) // win
    if n_win == 0:
        return np.array([]), np.array([])
    scale = np.ones_like(filt)
    for i in range(n_win + 1):
        seg = filt[i * win : (i + 1) * win]
        if len(seg):
            std = seg.std()
            scale[i * win : (i + 1) * win] = std if std > 1e-8 else np.inf
    peaks, _ = find_peaks(filt / scale, prominence=0.5, distance=int(0.33 * fs))
    return peaks[1:] / fs, np.diff(peaks) / fs


def beats_from_ibi_column(ibi: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Reads beats from the device's sparse ``IBI`` column.

    Args:
        ibi: Per-row IBI values in seconds, NaN where no beat was reported.
        fs: Sampling rate in Hz, used to convert row index to time.

    Returns:
        ``(times, intervals)`` in seconds.
    """
    idx = np.flatnonzero(~np.isnan(ibi))
    return idx / fs, ibi[idx]


def ibi_to_grid(
    times: np.ndarray,
    intervals: np.ndarray,
    n_epochs: int,
    epoch_seconds: int = 30,
    grid_hz: int = 25,
    max_ibi: float = 2.0,
) -> np.ndarray:
    """Places beat intervals on a fixed-rate grid (zero-order hold).

    Args:
        times: Beat times in seconds.
        intervals: IBI value for each beat, in seconds.
        n_epochs: Number of epochs to cover.
        epoch_seconds: Epoch length in seconds.
        grid_hz: Grid rate. 30 s at 25 Hz gives the paper's 750 points.
        max_ibi: Intervals above this (or non-positive) are treated as missing,
            as are grid points whose last beat is older than this.

    Returns:
        Float array of shape ``(n_epochs * epoch_seconds * grid_hz,)`` with NaN
        where the IBI is missing.
    """
    n_points = n_epochs * epoch_seconds * grid_hz
    series = np.full(n_points, np.nan)
    if len(times) == 0:
        return series
    grid_t = np.arange(n_points) / grid_hz
    idx = np.searchsorted(times, grid_t, side="right") - 1
    has_beat = idx >= 0
    safe = np.clip(idx, 0, None)
    value = intervals[safe]
    age = grid_t - times[safe]
    ok = has_beat & (value > 0) & (value <= max_ibi) & (age <= max_ibi)
    series[ok] = value[ok]
    return series


class SleepStagingDREAMT(BaseTask):
    """Per-epoch Wake / NREM / REM staging from DREAMT wrist IBI.

    Works with :class:`~pyhealth.datasets.DREAMTDataset`. Each sample is one
    window of ``window_epochs`` consecutive 30 s epochs from one night.

    Attributes:
        task_name: ``"SleepStagingDREAMT"``.
        input_schema: ``{"ibi": "tensor"}``. Shape ``(T, 750)``, or
            ``(T, 2, 750)`` when ``add_mask`` is True (IBI plus validity mask).
        output_schema: ``{"label": "tensor"}``. Shape ``(T,)`` with class
            indices from ``label_scheme`` and ``-100`` for ignored epochs
            (3 class: 0 Wake, 1 NREM, 2 REM).

    Args:
        ibi_source: ``"bvp"`` detects beats in the BVP waveform (the paper's
            approach); ``"e4"`` uses the device's own ``IBI`` column.
        fill: How missing IBI is filled: ``"zero"`` (paper), ``"interpolate"``
            (linear) or ``"median"`` (night median).
        normalize: ``"none"``, ``"night"`` or ``"epoch"`` z-scoring of valid
            IBI values.
        add_mask: Add a second channel marking valid (1) vs missing (0) IBI.
        label_scheme: ``"3class"`` (Wake/NREM/REM), ``"4class"``
            (Wake/Light/Deep/REM) or ``"5class"`` (Wake/N1/N2/N3/REM). The
            number of classes is available as ``num_classes``.
        window_epochs: Epochs per sample. Partial trailing windows are dropped.
        max_epochs: Truncate each night to this many epochs (paper: 1100).
        ignore_prep: Ignore the DREAMT ``P`` stage instead of mapping to Wake.
        sampling_rate: Row rate of the 64 Hz file.
        epoch_seconds: Epoch length in seconds.
        grid_hz: IBI grid rate (25 Hz gives 750 points per epoch).
        max_ibi: Maximum plausible IBI in seconds.
        patient_ids: If given, only these participants produce samples.

    Examples:
        >>> from pyhealth.datasets import DREAMTDataset
        >>> dataset = DREAMTDataset(root="/path/to/dreamt/2.1.0")
        >>> task = SleepStagingDREAMT(window_epochs=100, fill="interpolate")
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "SleepStagingDREAMT"
    input_schema: ClassVar[dict[str, str]] = {"ibi": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "tensor"}

    def __init__(
        self,
        ibi_source: str = "bvp",
        fill: str = "zero",
        normalize: str = "none",
        add_mask: bool = False,
        label_scheme: str = "3class",
        window_epochs: int = 100,
        max_epochs: int = 1100,
        ignore_prep: bool = False,
        sampling_rate: float = 64.0,
        epoch_seconds: int = 30,
        grid_hz: int = 25,
        max_ibi: float = 2.0,
        patient_ids: Sequence[str] | None = None,
    ) -> None:
        if ibi_source not in ("bvp", "e4"):
            raise ValueError("ibi_source must be 'bvp' or 'e4'.")
        if fill not in ("zero", "interpolate", "median"):
            raise ValueError("fill must be 'zero', 'interpolate' or 'median'.")
        if normalize not in ("none", "night", "epoch"):
            raise ValueError("normalize must be 'none', 'night' or 'epoch'.")
        if label_scheme not in STAGE_SCHEMES:
            raise ValueError("label_scheme must be '3class', '4class' or '5class'.")
        self.label_scheme = label_scheme
        self.num_classes = NUM_CLASSES[label_scheme]
        self.ibi_source = ibi_source
        self.fill = fill
        self.normalize = normalize
        self.add_mask = add_mask
        self.window_epochs = window_epochs
        self.max_epochs = max_epochs
        self.ignore_prep = ignore_prep
        self.sampling_rate = sampling_rate
        self.epoch_seconds = epoch_seconds
        self.grid_hz = grid_hz
        self.max_ibi = max_ibi
        self.patient_ids = list(patient_ids) if patient_ids is not None else None
        super().__init__()

    def build_night(
        self, bvp: np.ndarray, ibi: np.ndarray, stages: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Converts one night of raw columns into epoch arrays.

        Args:
            bvp: Per-row BVP samples.
            ibi: Per-row device IBI values (NaN where absent).
            stages: Per-row ``Sleep_Stage`` strings.

        Returns:
            ``(x, labels)`` where ``x`` has shape ``(N, 750)`` (or
            ``(N, 2, 750)`` with ``add_mask``) and ``labels`` has shape
            ``(N,)``, with ``N <= max_epochs``.
        """
        rows_per_epoch = int(self.epoch_seconds * self.sampling_rate)
        n_epochs = min(len(stages) // rows_per_epoch, self.max_epochs)
        if n_epochs == 0:
            return np.zeros((0, 750), dtype=np.float32), np.zeros(0, dtype=np.int64)
        points = self.epoch_seconds * self.grid_hz

        if self.ibi_source == "bvp":
            times, intervals = beats_from_bvp(bvp, self.sampling_rate)
        else:
            times, intervals = beats_from_ibi_column(ibi, self.sampling_rate)
        series = ibi_to_grid(
            times, intervals, n_epochs, self.epoch_seconds, self.grid_hz, self.max_ibi
        )
        valid = ~np.isnan(series)
        epochs = series.reshape(n_epochs, points)

        if self.normalize == "night" and valid.any():
            epochs = (epochs - np.nanmean(series)) / (np.nanstd(series) + 1e-8)
        elif self.normalize == "epoch":
            with np.errstate(all="ignore"):
                mean = np.nanmean(epochs, axis=1, keepdims=True)
                std = np.nanstd(epochs, axis=1, keepdims=True)
            epochs = (epochs - mean) / (std + 1e-8)

        flat = epochs.reshape(-1)
        missing = np.isnan(flat)
        if self.fill == "interpolate" and valid.any():
            flat[missing] = np.interp(
                np.flatnonzero(missing), np.flatnonzero(~missing), flat[~missing]
            )
        elif self.fill == "median" and valid.any():
            flat[missing] = np.nanmedian(flat)
        flat[np.isnan(flat)] = 0.0  # zero fill, and any leftover NaNs

        x = flat.reshape(n_epochs, points).astype(np.float32)
        if self.add_mask:
            mask = valid.reshape(n_epochs, points).astype(np.float32)
            x = np.stack([x, mask], axis=1)
        labels = epoch_labels(
            map_stages(stages, self.ignore_prep, self.label_scheme),
            rows_per_epoch,
            n_epochs,
            self.num_classes,
        )
        return x, labels

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds windowed samples for one DREAMT participant.

        Args:
            patient: A patient from :class:`~pyhealth.datasets.DREAMTDataset`.

        Returns:
            A list of samples with keys ``patient_id``, ``window_idx``,
            ``ibi`` and ``label``. Empty if the 64 Hz file is missing.
        """
        if self.patient_ids is not None and patient.patient_id not in self.patient_ids:
            return []
        events = patient.get_events(event_type="dreamt_sleep")
        if not events:
            return []
        path = getattr(events[0], "file_64hz", None)
        if path is None or pd.isna(path) or not Path(str(path)).exists():
            return []
        df = pd.read_csv(path, usecols=["BVP", "IBI", "Sleep_Stage"])
        x, labels = self.build_night(
            df["BVP"].to_numpy(), df["IBI"].to_numpy(dtype=float), df["Sleep_Stage"]
        )

        samples: list[dict[str, Any]] = []
        step = self.window_epochs
        for i, start in enumerate(range(0, len(labels) - step + 1, step)):
            window_labels = labels[start : start + step]
            if (window_labels == IGNORE_INDEX).all():
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "window_idx": i,
                    "ibi": x[start : start + step],
                    "label": window_labels,
                }
            )
        return samples
