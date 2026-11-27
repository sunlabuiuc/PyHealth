"""Per-recording inter-beat-interval (IBI) sleep-staging tasks.

The tasks in this module turn 30 s waveform epochs into a fixed-size IBI
sequence with shape ``(T, points_per_epoch)``. Each row is derived by detecting
beats in a waveform epoch, expanding the beat intervals with sample-and-hold,
and striding (or resampling) to ``points_per_epoch`` samples. Labels are
retrieved as three classes with ``-1`` reserved for unknown epochs:

.. code-block:: text

    W -> 0, N1/N2/N3/N4 -> 1, R/REM -> 2, unknown -> -1

No NPZ cache, custom task cache, feature processor, or class-remapping
machinery is involved; the tasks emit the final tensors directly.

Examples:
    >>> task = SleepStagingIBIDREAMT()
    >>> task.task_name
    'SleepStagingIBIDREAMT'
"""

import logging
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.signal import detrend, find_peaks, resample, resample_poly

from pyhealth.tasks.base_task import BaseTask
from pyhealth.tasks.sleep_staging_nsrr import (
    CHANNEL_ALIASES,
    LABEL_SCHEMES,
    _read_edf_channels,
    _read_stage_codes,
)

logger = logging.getLogger(__name__)

#: NSRR profusion codes -> canonical five-class stages.
PROFUSION_STAGE_MAP: dict[int, int] = LABEL_SCHEMES["five_class"]

#: Canonical five-class stages -> retrieved three-class labels.
CANONICAL_TO_THREE: dict[int, int] = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, -1: -1}

_AASM_STRING_MAP: dict[str, int] = {
    "W": 0,
    "WAKE": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "N4": 3,
    "R": 4,
    "REM": 4,
    "MISSING": -1,
    "UNKNOWN": -1,
}

_RK_STRING_MAP: dict[str, int] = {
    "W": 0,
    "WAKE": 0,
    "S1": 1,
    "N1": 1,
    "S2": 2,
    "N2": 2,
    "S3": 3,
    "N3": 3,
    "S4": 3,
    "N4": 3,
    "REM": 4,
    "R": 4,
    "MISSING": -1,
    "UNKNOWN": -1,
}

_PEAK_BACKENDS = ("auto", "scipy", "neurokit", "biosppy")


def _clean_signal(waveform: Sequence[float]) -> np.ndarray:
    """Interpolates non-finite samples in a 1-D waveform.

    Args:
        waveform: One-dimensional signal samples.

    Returns:
        Float64 array with no NaN or infinite values. An all-non-finite input
        becomes an all-zero array.

    Examples:
        >>> _clean_signal([0.0, np.nan, 2.0]).tolist()
        [0.0, 1.0, 2.0]
    """
    signal = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        return signal
    good = np.isfinite(signal)
    if good.all():
        return signal
    if not good.any():
        return np.zeros_like(signal)
    x = np.arange(signal.size)
    return np.interp(x, x[good], signal[good]).astype(np.float64)


def _peaks_to_ibi(
    peaks: Sequence[int],
    fs: float,
    max_ibi: float,
    length: int | None = None,
) -> np.ndarray:
    """Expands beat positions into a sample-and-hold IBI signal.

    Args:
        peaks: Beat sample indices.
        fs: Sampling rate in Hz.
        max_ibi: Intervals at or above this many seconds become zero.
        length: Desired output length. Defaults to the last peak plus one.

    Returns:
        Float64 IBI array of shape ``(length,)``. Fewer than two in-range
        peaks yields an all-zero array.

    Examples:
        >>> _peaks_to_ibi([0, 100, 250], 100, 2.0, 300)[:2].tolist()
        [1.0, 1.0]
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    if max_ibi <= 0:
        raise ValueError("max_ibi must be positive")
    peak_array = np.asarray(peaks, dtype=np.int64).reshape(-1)
    out_len = int(length) if length is not None else (
        int(peak_array[-1]) + 1 if peak_array.size else 0
    )
    ibi = np.zeros(max(out_len, 0), dtype=np.float64)
    in_range = peak_array[(peak_array >= 0) & (peak_array < ibi.size)]
    if in_range.size < 2:
        return ibi
    in_range = np.sort(in_range)
    for start, end in pairwise(in_range):
        interval = float(end - start) / float(fs)
        if end > start and interval < max_ibi:
            ibi[int(start) : int(end)] = interval
    return ibi


def _beat_annotations_to_ibi(
    samples: Sequence[int],
    fs: float,
    max_ibi: float,
    length: int | None = None,
) -> np.ndarray:
    """Builds an IBI signal from externally supplied beat annotations.

    Args:
        samples: Annotated beat sample indices.
        fs: Sampling rate in Hz.
        max_ibi: Intervals at or above this many seconds become zero.
        length: Desired output length.

    Returns:
        Float64 sample-and-hold IBI array.

    Examples:
        >>> _beat_annotations_to_ibi([0, 50], 50, 2.0, 60)[0]
        1.0
    """
    return _peaks_to_ibi(samples, fs, max_ibi, length)


def _heart_rate_to_ibi(
    times: Sequence[float],
    bpm: Sequence[float],
    max_ibi: float = 2.0,
    length: int | None = None,
) -> np.ndarray:
    """Interpolates instantaneous heart rate into an IBI signal.

    Args:
        times: Heart-rate sample times in seconds.
        bpm: Heart rate in beats per minute at each time.
        max_ibi: Intervals at or above this many seconds become zero.
        length: Optional output length; defaults to the time span at the
            median heart-rate sample rate.

    Returns:
        Float64 IBI array with one value per output sample.

    Examples:
        >>> _heart_rate_to_ibi([0.0, 1.0], [60.0, 60.0], 2.0).tolist()
        [1.0, 1.0]
    """
    time_array = np.asarray(times, dtype=np.float64).reshape(-1)
    rate_array = np.asarray(bpm, dtype=np.float64).reshape(-1)
    if time_array.size != rate_array.size:
        raise ValueError("times and bpm must have the same length")
    if time_array.size < 2:
        out_len = int(length) if length is not None else 0
        return np.zeros(max(out_len, 0), dtype=np.float64)

    order = np.argsort(time_array)
    time_array, rate_array = time_array[order], rate_array[order]
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.where(rate_array > 0, 60.0 / rate_array, 0.0)
    values[~np.isfinite(values)] = 0.0
    values[values >= max_ibi] = 0.0

    deltas = np.diff(time_array)
    deltas = deltas[deltas > 0]
    fs = 1.0 / float(np.median(deltas)) if deltas.size else 1.0
    span = float(time_array[-1] - time_array[0])
    out_len = int(length) if length is not None else max(int(span * fs) + 1, 2)
    out_times = np.arange(out_len, dtype=np.float64) / fs
    ibi = np.interp(out_times, time_array - time_array[0], values)
    ibi[ibi >= max_ibi] = 0.0
    return ibi.astype(np.float64)


def _resample_rows(signal: np.ndarray, points: int) -> np.ndarray:
    """Resamples each row of a 2-D array to ``points`` columns."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.shape[1] == points:
        return signal
    ratio = Fraction(points, signal.shape[1])
    return resample_poly(
        signal, ratio.numerator, ratio.denominator, axis=1
    ).astype(np.float64)


def _ibi_to_epoch_grid(
    ibi: Sequence[float],
    epoch_seconds: float,
    points_per_epoch: int,
    n_epochs: int,
    downsample_factor: int | None = None,
) -> np.ndarray:
    """Reshapes an IBI signal into one fixed-size row per epoch.

    An integer stride is used when ``downsample_factor`` divides the source
    epoch length exactly into ``points_per_epoch``; otherwise each epoch is
    resampled to ``points_per_epoch`` samples.

    Args:
        ibi: Continuous IBI samples at the source sampling rate.
        epoch_seconds: Seconds per epoch.
        points_per_epoch: Target number of samples per epoch.
        n_epochs: Number of epochs to emit.
        downsample_factor: Optional integer stride.

    Returns:
        Float32 array of shape ``(n_epochs, points_per_epoch)``.

    Examples:
        >>> ibi = np.ones(4 * 100)
        >>> _ibi_to_epoch_grid(ibi, 30.0, 100, 4, 1).shape
        (4, 100)
    """
    samples = np.asarray(ibi, dtype=np.float64).reshape(-1)
    if n_epochs <= 0:
        return np.zeros((0, points_per_epoch), dtype=np.float32)
    if samples.size < n_epochs:
        raise ValueError("IBI signal is shorter than n_epochs")
    native = samples.size // n_epochs
    if native <= 0:
        raise ValueError("IBI signal is too short for the requested epochs")
    epochs = samples[: n_epochs * native].reshape(n_epochs, native)
    if (
        downsample_factor is not None
        and downsample_factor > 1
        and native % downsample_factor == 0
        and native // downsample_factor == points_per_epoch
    ):
        grid = epochs[:, ::downsample_factor]
    else:
        grid = _resample_rows(epochs, points_per_epoch)
    return grid.astype(np.float32)


def _stage_to_canonical(stage: Any, scheme: str = "aasm") -> int:
    """Maps a stage label to the canonical five-class space.

    Args:
        stage: String or integer stage label.
        scheme: ``"aasm"`` or ``"rk"`` string vocabulary.

    Returns:
        Canonical stage in ``{-1, 0, 1, 2, 3, 4}``; unknown values map to
        ``-1``.

    Examples:
        >>> _stage_to_canonical("N4"), _stage_to_canonical("S4", "rk")
        (3, 3)
    """
    if scheme not in ("aasm", "rk"):
        raise ValueError("scheme must be 'aasm' or 'rk'")
    if stage is None:
        return -1
    mapping = _AASM_STRING_MAP if scheme == "aasm" else _RK_STRING_MAP
    if isinstance(stage, str):
        return mapping.get(stage.strip().upper(), -1)
    try:
        value = int(stage)
    except (TypeError, ValueError):
        return -1
    return value if value == -1 or 0 <= value <= 4 else -1


def _to_three_class(stage: Any, scheme: str = "aasm") -> Any:
    """Maps canonical stages to the retrieved three-class space.

    Args:
        stage: Scalar or array of stage labels.
        scheme: String vocabulary passed to :func:`_stage_to_canonical`.

    Returns:
        An integer for scalar input, otherwise an ``int64`` array with the
        same shape. ``-1`` is preserved for unknown stages.

    Examples:
        >>> _to_three_class([0, 1, 2, 3, 4, -1]).tolist()
        [0, 1, 1, 1, 2, -1]
    """
    if isinstance(stage, (list, tuple, np.ndarray)):
        array = np.asarray(stage)
        values = [_to_three_class(value, scheme) for value in array.ravel()]
        return np.asarray(values, dtype=np.int64).reshape(array.shape)
    canonical = _stage_to_canonical(stage, scheme)
    return CANONICAL_TO_THREE.get(canonical, -1)


def _truncate_epochs(
    signal: np.ndarray, label: np.ndarray, max_epochs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Truncates aligned signal rows and labels to ``max_epochs``.

    Args:
        signal: Float array of shape ``(T, points)``.
        label: Integer array of shape ``(T,)``.
        max_epochs: Maximum number of rows to keep.

    Returns:
        The truncated ``(signal, label)`` pair.

    Examples:
        >>> x, y = _truncate_epochs(np.ones((5, 2)), np.arange(5), 3)
        >>> x.shape, y.tolist()
        ((3, 2), [0, 1, 2])
    """
    rows = np.asarray(signal)
    labels = np.asarray(label)
    if rows.ndim != 2 or labels.ndim != 1 or rows.shape[0] != labels.shape[0]:
        raise ValueError("signal and label must be aligned (T, points) and (T,)")
    n = min(rows.shape[0], labels.shape[0], int(max_epochs))
    return rows[:n], labels[:n]


def _detect_peaks(
    waveform: Sequence[float],
    fs: float,
    signal_type: str,
    backend: str = "auto",
) -> np.ndarray:
    """Detects beats in a waveform epoch.

    ``auto`` prefers biosppy for ECG and neurokit for PPG when installed, falls
    back to :func:`scipy.signal.find_peaks` only when the package is missing,
    and treats degenerate epochs (including NeuroKit's "too few peaks" warning)
    as having no beats. An explicit backend is used strictly so validation runs
    can pin the detector.

    Args:
        waveform: One epoch of a PPG or ECG waveform.
        fs: Sampling rate in Hz.
        signal_type: ``"ecg"`` or ``"ppg"``.
        backend: One of ``"auto"``, ``"scipy"``, ``"neurokit"``, ``"biosppy"``.

    Returns:
        Integer array of peak sample indices.

    Examples:
        >>> x = np.zeros(100); x[20] = x[60] = 1.0
        >>> _detect_peaks(x, 100, "ecg", backend="scipy").tolist()
        [20, 60]
    """
    if signal_type not in ("ecg", "ppg"):
        raise ValueError("signal_type must be 'ecg' or 'ppg'")
    if backend not in _PEAK_BACKENDS:
        raise ValueError(f"unknown peak backend {backend!r}")
    signal = _clean_signal(waveform)
    if signal.size < 3:
        return np.empty(0, dtype=np.int64)

    def empty() -> np.ndarray:
        return np.empty(0, dtype=np.int64)

    def scipy_peaks() -> np.ndarray:
        cleaned = detrend(signal, type="linear") if signal.size > 3 else signal
        distance = max(1, round(0.35 * float(fs)))
        peaks, _ = find_peaks(cleaned, distance=distance)
        return peaks.astype(np.int64)

    detector_errors = (
        AttributeError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    )

    if backend == "scipy":
        return scipy_peaks()

    if backend == "biosppy":
        if signal_type != "ecg":
            raise ValueError("biosppy backend supports ECG only")
        try:
            from biosppy.signals import ecg as biosppy_ecg

            result = biosppy_ecg.ecg(
                signal=signal, sampling_rate=float(fs), show=False
            )
        except detector_errors as err:
            logger.debug("biosppy peak detection failed: %s", err)
            return empty()
        return np.asarray(result["rpeaks"], dtype=np.int64)

    if backend == "neurokit":
        try:
            import neurokit2 as nk

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if signal_type == "ecg":
                    _, info = nk.ecg_process(signal, sampling_rate=float(fs))
                    key = "ECG_R_Peaks"
                else:
                    _, info = nk.ppg_process(signal, sampling_rate=float(fs))
                    key = "PPG_Peaks"
            if any(
                "Too few peaks detected to compute the rate" in str(w.message)
                for w in caught
            ):
                return empty()
        except detector_errors as err:
            logger.debug("neurokit peak detection failed: %s", err)
            return empty()
        return np.asarray(info.get(key, []), dtype=np.int64)

    candidates = ["biosppy", "neurokit"] if signal_type == "ecg" else ["neurokit"]
    for candidate in candidates:
        try:
            return _detect_peaks(signal, fs, signal_type, backend=candidate)
        except ImportError as err:
            logger.debug("Peak backend %s is unavailable: %s", candidate, err)
    return scipy_peaks()


def _resample_to_rate(signal: np.ndarray, fs: float, target_fs: float) -> tuple[
    np.ndarray, float
]:
    """Fourier-resamples a waveform to ``target_fs`` when needed.

    The target length is truncated with ``int`` to match the NSRR/MESA
    reference readers, which call :func:`scipy.signal.resample` with
    ``int(len(signal) * target_fs / fs)`` samples.

    Args:
        signal: One-dimensional waveform.
        fs: Current sampling rate in Hz.
        target_fs: Desired sampling rate in Hz.

    Returns:
        The resampled waveform and its effective sampling rate.

    Examples:
        >>> out, rate = _resample_to_rate(np.ones(256), 256.0, 100.0)
        >>> out.shape, rate
        ((100,), 100.0)
    """
    array = np.asarray(signal, dtype=np.float64).reshape(-1)
    if fs <= 0 or target_fs <= 0:
        raise ValueError("sampling rates must be positive")
    if abs(fs - target_fs) < 1e-9:
        return array, float(fs)
    target_length = int(array.size * float(target_fs) / float(fs))
    if target_length <= 0:
        return np.empty(0, dtype=np.float64), float(target_fs)
    return resample(array, target_length), float(target_fs)


class _SleepStagingIBIBase(BaseTask):
    """Shared argument validation and sample emission for IBI tasks."""

    task_name = "SleepStagingIBIBase"
    input_schema = {"signal": "tensor", "lengths": "raw"}  # noqa: RUF012
    output_schema = {"label": ("tensor", {"dtype": torch.long})}  # noqa: RUF012

    signal_type: str = "ppg"
    source_fs: float = 100.0
    downsample_factor: int = 4

    def __init__(
        self,
        epoch_seconds: float = 30.0,
        points_per_epoch: int = 750,
        max_epochs: int = 1100,
        min_epochs: int = 1,
        max_ibi: float = 2.0,
        peak_backend: str = "auto",
    ) -> None:
        if not np.isfinite(epoch_seconds) or epoch_seconds <= 0:
            raise ValueError("epoch_seconds must be positive and finite")
        if points_per_epoch <= 0:
            raise ValueError("points_per_epoch must be positive")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if min_epochs <= 0:
            raise ValueError("min_epochs must be positive")
        if not np.isfinite(max_ibi) or max_ibi <= 0:
            raise ValueError("max_ibi must be positive and finite")
        if peak_backend not in _PEAK_BACKENDS:
            raise ValueError(f"unknown peak backend {peak_backend!r}")
        self.epoch_seconds = float(epoch_seconds)
        self.points_per_epoch = int(points_per_epoch)
        self.max_epochs = int(max_epochs)
        self.min_epochs = int(min_epochs)
        self.max_ibi = float(max_ibi)
        self.peak_backend = peak_backend
        super().__init__()

    def _waveforms_to_rows(self, waveforms: np.ndarray) -> np.ndarray:
        """Converts ``(T, native)`` waveform epochs into IBI rows."""
        waveforms = np.asarray(waveforms, dtype=np.float64)
        rows = np.empty((waveforms.shape[0], self.points_per_epoch), np.float32)
        for i, waveform in enumerate(waveforms):
            peaks = _detect_peaks(
                waveform, self.source_fs, self.signal_type, self.peak_backend
            )
            ibi = _peaks_to_ibi(
                peaks, self.source_fs, self.max_ibi, length=len(waveform)
            )
            rows[i] = _ibi_to_epoch_grid(
                ibi,
                self.epoch_seconds,
                self.points_per_epoch,
                1,
                self.downsample_factor,
            )[0]
        return rows

    def _finalize(
        self,
        signal: np.ndarray,
        label: np.ndarray,
        patient_id: str,
        record_id: str,
    ) -> list[dict[str, Any]]:
        """Truncates and wraps rows in the shared sample schema."""
        rows = np.asarray(signal, dtype=np.float32)
        labels = _to_three_class(label)
        labels = np.asarray(labels, dtype=np.int64)
        rows, labels = _truncate_epochs(rows, labels, self.max_epochs)
        n_epochs = int(labels.shape[0])
        if n_epochs < self.min_epochs:
            return []
        return [
            {
                "patient_id": str(patient_id),
                "record_id": str(record_id),
                "signal": rows,
                "label": labels,
                "lengths": n_epochs,
            }
        ]

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        raise NotImplementedError


def _nsrr_samples(task: _SleepStagingIBIBase, patient: Any) -> list[dict[str, Any]]:
    """Builds one IBI sample per NSRR EDF/XML recording of a patient."""
    samples: list[dict[str, Any]] = []
    for event in patient.get_events():
        try:
            channels = _read_edf_channels(event.signal_file, task.channel_aliases)
            channel = next(
                (a for a in task.channel_aliases if a in channels), None
            )
            if channel is None:
                raise ValueError(f"no IBI channel in {event.signal_file}")
            signal, fs = channels[channel]
            stages = _read_stage_codes(event.label_file)
            signal, fs = _resample_to_rate(signal, fs, task.source_fs)
            native = round(task.source_fs * task.epoch_seconds)
            n_epochs = min(len(stages), int(len(signal) // native))
            if n_epochs <= 0:
                continue
            canonical = np.asarray(
                [
                    PROFUSION_STAGE_MAP.get(int(stage), -1)
                    for stage in stages[:n_epochs]
                ],
                dtype=np.int64,
            )
            labels = canonical
            waveforms = signal[: n_epochs * native].reshape(n_epochs, native)
            rows = task._waveforms_to_rows(waveforms)
            record_id = getattr(event, "record_name", None) or Path(
                str(event.signal_file)
            ).stem
            samples.extend(
                task._finalize(rows, labels, patient.patient_id, str(record_id))
            )
        except (ValueError, OSError, ET.ParseError) as err:
            logger.warning(
                "Skipping recording of patient %s: %s", patient.patient_id, err
            )
    return samples


class SleepStagingIBIDREAMT(_SleepStagingIBIBase):
    """Per-recording IBI sleep staging for :class:`~pyhealth.datasets.DREAMTDataset`.

    The default ``psg_100hz`` source mirrors the reference DREAMT pipeline:
    the ``data_100Hz`` BVP column is sampled at 100 Hz, the 30 s epoch grid
    is recovered from stage changes, and IBI is strided by four to produce
    750 samples per epoch at 25 Hz. The full anchored night is kept, including
    leading and trailing wake epochs. The optional ``wearable_64hz`` source
    reads ``data_64Hz`` (or legacy ``data``) files instead.

    Attributes:
        task_name: ``"SleepStagingIBIDREAMT"``.
        input_schema: ``{"signal": "tensor", "lengths": "raw"}``.
        output_schema: ``{"label": ("tensor", {"dtype": torch.long})}``.

    Args:
        signal_source: ``"psg_100hz"`` or ``"wearable_64hz"``.
        signal_channel: Waveform column, normally ``"BVP"``.
        epoch_seconds: Seconds per epoch.
        points_per_epoch: Number of IBI samples per epoch.
        max_epochs: Maximum epochs per recording.
        min_epochs: Drop recordings with fewer epochs than this.
        max_ibi: Intervals at or above this many seconds become zero.
        peak_backend: Peak detector, or ``"auto"`` for the preferred
            installed backend with a scipy fallback.

    Examples:
        >>> from pyhealth.tasks import SleepStagingIBIDREAMT
        >>> task = SleepStagingIBIDREAMT()
        >>> task.downsample_factor
        4
    """

    task_name = "SleepStagingIBIDREAMT"
    signal_type = "ppg"

    def __init__(
        self,
        signal_source: str = "psg_100hz",
        signal_channel: str = "BVP",
        epoch_seconds: float = 30.0,
        points_per_epoch: int = 750,
        max_epochs: int = 1100,
        min_epochs: int = 1,
        max_ibi: float = 2.0,
        peak_backend: str = "auto",
    ) -> None:
        if signal_source not in ("psg_100hz", "wearable_64hz"):
            raise ValueError(f"unknown DREAMT signal source {signal_source!r}")
        if not signal_channel:
            raise ValueError("signal_channel must not be empty")
        self.signal_source = signal_source
        self.signal_channel = str(signal_channel)
        self.source_fs = 100.0 if signal_source == "psg_100hz" else 64.0
        super().__init__(
            epoch_seconds=epoch_seconds,
            points_per_epoch=points_per_epoch,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_ibi=max_ibi,
            peak_backend=peak_backend,
        )
        target_rate = self.points_per_epoch / self.epoch_seconds
        self.downsample_factor = max(1, round(self.source_fs / target_rate))

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one IBI sample per DREAMT recording of a patient.

        Args:
            patient: A patient from :class:`~pyhealth.datasets.DREAMTDataset`.

        Returns:
            Samples with ``signal`` of shape ``(T, 750)``, three-class
            ``label`` of shape ``(T,)`` and ``lengths == T``.

        Examples:
            >>> task = SleepStagingIBIDREAMT()
            >>> task.points_per_epoch
            750
        """
        from pyhealth.datasets.dreamt import load_dreamt_frame

        events = patient.get_events(event_type="dreamt_sleep")
        if not events:
            events = patient.get_events()
        file_attr = (
            "file_100hz" if self.signal_source == "psg_100hz" else "file_64hz"
        )
        samples: list[dict[str, Any]] = []
        for event in events:
            file_path = getattr(event, file_attr, None)
            if not file_path or not Path(str(file_path)).is_file():
                logger.warning(
                    "Skipping DREAMT recording of patient %s: missing %s",
                    patient.patient_id,
                    file_attr,
                )
                continue
            try:
                frame = load_dreamt_frame(
                    str(file_path), signal_cols=(self.signal_channel,)
                )
                signal = _clean_signal(frame[self.signal_channel].to_numpy())
                stages = frame["Sleep_Stage"].to_numpy(dtype=np.int64)

                samples_per_epoch = round(self.source_fs * self.epoch_seconds)
                changes = np.where(np.diff(stages) != 0)[0] + 1
                if changes.size:
                    counts = np.bincount(
                        changes % samples_per_epoch, minlength=samples_per_epoch
                    )
                    anchor = int(np.argmax(counts))
                else:
                    anchor = 0
                n_epochs = max((len(stages) - anchor) // samples_per_epoch, 0)
                if n_epochs == 0:
                    continue

                end = anchor + n_epochs * samples_per_epoch
                grid = stages[anchor:end].reshape(n_epochs, samples_per_epoch)
                first = grid[:, 0]
                consistent = np.all(grid == first[:, None], axis=1)
                known = first >= 0
                epoch_stages = np.full(n_epochs, -1, dtype=np.int64)
                epoch_stages[consistent & known] = first[consistent & known]
                waveforms = signal[anchor:end].reshape(
                    n_epochs, samples_per_epoch
                )
                rows = self._waveforms_to_rows(waveforms)
                samples.extend(
                    self._finalize(
                        rows,
                        epoch_stages,
                        patient.patient_id,
                        Path(str(file_path)).stem,
                    )
                )
            except (ValueError, OSError) as err:
                logger.warning(
                    "Skipping DREAMT recording of patient %s: %s",
                    patient.patient_id,
                    err,
                )
        return samples


class SleepStagingIBISHHS(_SleepStagingIBIBase):
    """Per-recording IBI sleep staging for NSRR SHHS recordings.

    Uses :class:`~pyhealth.datasets.NSRRSleepDataset` EDF/XML events. The
    default ECG channel is read at its native 125 Hz, IBI is strided by five,
    and profusion stages are mapped to ``{W: 0, NREM: 1, REM: 2}`` with
    ``-1`` for unknown epochs.

    Attributes:
        task_name: ``"SleepStagingIBISHHS"``.
        input_schema: Shared ``signal``/``lengths`` schema.
        output_schema: Long-tensor ``label`` schema.

    Args:
        signal_channel: EDF channel name, normally ``"ECG"``.
        source_fs: Sampling rate used for the epoch grid.
        downsample_factor: Integer stride from ``source_fs`` to 25 Hz.
        epoch_seconds: Seconds per epoch.
        points_per_epoch: Number of IBI samples per epoch.
        max_epochs: Maximum epochs per recording.
        min_epochs: Drop recordings with fewer epochs than this.
        max_ibi: Intervals at or above this many seconds become zero.
        peak_backend: Peak detector, or ``"auto"`` for the preferred
            installed backend with a scipy fallback.

    Examples:
        >>> from pyhealth.tasks import SleepStagingIBISHHS
        >>> task = SleepStagingIBISHHS()
        >>> task.signal_type
        'ecg'
    """

    task_name = "SleepStagingIBISHHS"

    def __init__(
        self,
        signal_channel: str = "ECG",
        source_fs: float = 125.0,
        downsample_factor: int = 5,
        epoch_seconds: float = 30.0,
        points_per_epoch: int = 750,
        max_epochs: int = 1100,
        min_epochs: int = 1,
        max_ibi: float = 2.0,
        peak_backend: str = "auto",
    ) -> None:
        if source_fs <= 0 or not np.isfinite(source_fs):
            raise ValueError("source_fs must be positive and finite")
        if downsample_factor <= 0:
            raise ValueError("downsample_factor must be positive")
        self.source_fs = float(source_fs)
        self.downsample_factor = int(downsample_factor)
        self.signal_channel = str(signal_channel)
        self.signal_type = (
            "ecg" if self.signal_channel.lower() in CHANNEL_ALIASES["ecg"] else "ppg"
        )
        aliases = CHANNEL_ALIASES[self.signal_type]
        aliases = (*aliases, self.signal_channel.lower())
        self.channel_aliases = tuple(dict.fromkeys(aliases))
        super().__init__(
            epoch_seconds=epoch_seconds,
            points_per_epoch=points_per_epoch,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_ibi=max_ibi,
            peak_backend=peak_backend,
        )

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one IBI sample per SHHS recording of a patient.

        Args:
            patient: A patient from
                :class:`~pyhealth.datasets.NSRRSleepDataset`.

        Returns:
            Samples with ``signal`` of shape ``(T, 750)``, three-class
            ``label`` of shape ``(T,)`` and ``lengths == T``.

        Examples:
            >>> task = SleepStagingIBISHHS()
            >>> task.task_name
            'SleepStagingIBISHHS'
        """
        return _nsrr_samples(self, patient)


class SleepStagingIBIMESA(_SleepStagingIBIBase):
    """Per-recording IBI sleep staging for NSRR MESA recordings.

    Uses :class:`~pyhealth.datasets.NSRRSleepDataset` EDF/XML events. The
    default ``Pleth`` signal is Fourier-resampled from its native 256 Hz to
    100 Hz, IBI is strided by four to 750 samples per epoch, and profusion
    stages are mapped to ``{W: 0, NREM: 1, REM: 2}`` with ``-1`` unknown.

    Attributes:
        task_name: ``"SleepStagingIBIMESA"``.
        input_schema: Shared ``signal``/``lengths`` schema.
        output_schema: Long-tensor ``label`` schema.

    Args:
        signal_channel: EDF channel name, normally ``"Pleth"``.
        source_fs: Sampling rate used for the epoch grid.
        downsample_factor: Integer stride from ``source_fs`` to 25 Hz.
        epoch_seconds: Seconds per epoch.
        points_per_epoch: Number of IBI samples per epoch.
        max_epochs: Maximum epochs per recording.
        min_epochs: Drop recordings with fewer epochs than this.
        max_ibi: Intervals at or above this many seconds become zero.
        peak_backend: Peak detector, or ``"auto"`` for the preferred
            installed backend with a scipy fallback.

    Examples:
        >>> from pyhealth.tasks import SleepStagingIBIMESA
        >>> task = SleepStagingIBIMESA()
        >>> task.signal_type
        'ppg'
    """

    task_name = "SleepStagingIBIMESA"

    def __init__(
        self,
        signal_channel: str = "Pleth",
        source_fs: float = 100.0,
        downsample_factor: int = 4,
        epoch_seconds: float = 30.0,
        points_per_epoch: int = 750,
        max_epochs: int = 1100,
        min_epochs: int = 1,
        max_ibi: float = 2.0,
        peak_backend: str = "auto",
    ) -> None:
        if source_fs <= 0 or not np.isfinite(source_fs):
            raise ValueError("source_fs must be positive and finite")
        if downsample_factor <= 0:
            raise ValueError("downsample_factor must be positive")
        self.source_fs = float(source_fs)
        self.downsample_factor = int(downsample_factor)
        self.signal_channel = str(signal_channel)
        self.signal_type = (
            "ecg" if self.signal_channel.lower() in CHANNEL_ALIASES["ecg"] else "ppg"
        )
        aliases = CHANNEL_ALIASES[self.signal_type]
        aliases = (*aliases, self.signal_channel.lower())
        self.channel_aliases = tuple(dict.fromkeys(aliases))
        super().__init__(
            epoch_seconds=epoch_seconds,
            points_per_epoch=points_per_epoch,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_ibi=max_ibi,
            peak_backend=peak_backend,
        )

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one IBI sample per MESA recording of a patient.

        Args:
            patient: A patient from
                :class:`~pyhealth.datasets.NSRRSleepDataset`.

        Returns:
            Samples with ``signal`` of shape ``(T, 750)``, three-class
            ``label`` of shape ``(T,)`` and ``lengths == T``.

        Examples:
            >>> task = SleepStagingIBIMESA()
            >>> task.task_name
            'SleepStagingIBIMESA'
        """
        return _nsrr_samples(self, patient)
