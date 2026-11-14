"""Seizure detection task for the CHB-MIT Scalp EEG Database.

Author: Jathurshan Pradeepkumar (jp65@illinois.edu), UIUC
Description: 10-second bipolar windows and binary seizure labels, following
    the CHB-MIT preprocessing used by CBraMod and TFM-Tokenizer.

References:
    - Wang, J. et al. CBraMod: A Criss-Cross Brain Foundation Model for
      EEG Decoding. ICLR 2025. https://github.com/wjq-learning/CBraMod
      (``preprocessing/CHB-MIT/process1.py``, ``process2.py``,
      ``datasets/chb_dataset.py``)
    - Pradeepkumar, J. et al. Tokenizing Single-Channel EEG with
      Time-Frequency Motif Learning. ICLR 2026.
      https://github.com/Jathurshan0330/TFM-Tokenizer
      (``datasets_processing/CHB-MIT``, ``CHBMITloader``)
"""

import logging
import math
from typing import Any, ClassVar

import numpy as np
import torch
from scipy.signal import resample

from pyhealth.tasks import BaseTask
from pyhealth.tasks.eegbci import normalize_signal
from pyhealth.tasks.subject_ids import normalize_prefixed_subject
from pyhealth.tasks.temple_university_EEG_tasks import read_edf_uv

logger = logging.getLogger(__name__)

# Bipolar 10-20 channels kept by CBraMod, in that order. CHB-MIT already
# stores these differences in the EDF, so no re-referencing is done.
BIPOLAR_CHANNELS: tuple[str, ...] = (
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
)


def normalize_subject(subject: str) -> str:
    """Canonicalize a CHB-MIT subject id to ``chbXX``.

    ``"chb01"``, ``"CHB1"`` and ``"1"`` all become ``"chb01"``. Ids that are
    not a case number (for example a custom cohort name) are lower-cased
    and returned unchanged apart from a leading ``chb`` prefix.

    Args:
        subject: Subject id as stored on a sample or passed by the user.

    Returns:
        The canonical subject id.

    Examples:
        >>> from pyhealth.tasks.seizure_detection_chbmit import normalize_subject
        >>> [normalize_subject(s) for s in ("1", "CHB2", "chb10")]
        ['chb01', 'chb02', 'chb10']
    """
    return normalize_prefixed_subject(subject, "chb", 2)


def _channel_index(ch_names: list[str], name: str) -> int | None:
    """Find ``name`` in an EDF header, including MNE's duplicate suffix.

    MNE renames a repeated label ``T8-P8`` to ``T8-P8-0`` and ``T8-P8-1``.
    The first copy is the channel CBraMod keeps, so ``-0`` is accepted and
    ``-1`` is not.

    Args:
        ch_names: Channel names from the EDF, in file order.
        name: Wanted bipolar channel, for example ``"T8-P8"``.

    Returns:
        Index into ``ch_names``, or None when the channel is absent.
    """
    stripped = [ch.strip() for ch in ch_names]
    if name in stripped:
        return stripped.index(name)
    duplicate = f"{name}-0"
    if duplicate in stripped:
        return stripped.index(duplicate)
    return None


def select_bipolar_channels(
    raw_data: np.ndarray, ch_names: list[str]
) -> np.ndarray | None:
    """Keep the 16 CBraMod bipolar channels, in CBraMod order.

    A missing channel is filled with zeros, which is what CBraMod does when
    a recording does not carry the full montage. A recording that has none
    of the 16 channels (a purely referential file) is rejected so the task
    does not emit an all-zero signal.

    Args:
        raw_data: EEG array of shape ``(n_channels, n_samples)``, in uV.
        ch_names: Channel names aligned with the first axis of ``raw_data``.

    Returns:
        Array of shape ``(16, n_samples)``, or None if no wanted channel
        is present.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.tasks.seizure_detection_chbmit import (
        ...     select_bipolar_channels,
        ... )
        >>> raw = np.ones((1, 4))
        >>> out = select_bipolar_channels(raw, ["FP1-F7"])
        >>> out.shape
        (16, 4)
    """
    selected = np.zeros((len(BIPOLAR_CHANNELS), raw_data.shape[1]), dtype=float)
    found = 0
    for row, name in enumerate(BIPOLAR_CHANNELS):
        index = _channel_index(ch_names, name)
        if index is None:
            logger.warning("Channel %s is missing; filling it with zeros.", name)
            continue
        selected[row] = raw_data[index]
        found += 1
    if found == 0:
        return None
    return selected


def _parse_time_list(value: Any) -> list[int]:
    """Parse a metadata seizure-time field into seconds.

    The dataset stores times as a ``;``-separated string. Polars may also
    hand back a number, an empty string, or a missing value.

    Args:
        value: ``seizure_starts`` or ``seizure_ends`` attribute of one event.

    Returns:
        Seizure times in seconds. Missing or blank input gives an empty list.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return [int(value)]
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "<na>"}:
        return []
    times: list[int] = []
    for part in text.split(";"):
        part = part.strip()
        if part:
            times.append(int(float(part)))
    return times


def _seizure_intervals(
    starts: list[int],
    ends: list[int],
    sfreq: float,
    label_mode: str,
) -> list[tuple[int, int]]:
    """Convert seizure times in seconds to sample indices.

    ``cbramod`` uses CBraMod's index ``seconds * sfreq - 1``. ``overlap``
    uses ``seconds * sfreq``, so a window is positive when its time range
    overlaps the seizure.

    Args:
        starts: Seizure start times in seconds.
        ends: Seizure end times in seconds, paired with ``starts``.
        sfreq: Sampling rate of the recording in Hz.
        label_mode: ``"cbramod"`` or ``"overlap"``.

    Returns:
        ``(start_index, end_index)`` pairs. Pairs are truncated to the
        shorter of the two lists.
    """
    offset = 1 if label_mode == "cbramod" else 0
    intervals = []
    for start, end in zip(starts, ends):
        intervals.append(
            (round(start * sfreq) - offset, round(end * sfreq) - offset)
        )
    return intervals


def _window_is_seizure(
    start: int,
    end: int,
    intervals: list[tuple[int, int]],
    label_mode: str,
) -> bool:
    """Decide whether one window is a seizure window.

    ``cbramod`` is positive when a seizure start or a seizure end falls
    strictly inside the window, which is the rule in CBraMod ``process2.py``.
    ``overlap`` is positive when the window and the seizure share any sample.

    Args:
        start: First sample of the window.
        end: Sample just past the window.
        intervals: Seizure ``(start_index, end_index)`` pairs.
        label_mode: ``"cbramod"`` or ``"overlap"``.

    Returns:
        True when the window should be labeled seizure.
    """
    for seizure_start, seizure_end in intervals:
        if label_mode == "cbramod":
            if start < seizure_start < end or start < seizure_end < end:
                return True
        elif start < seizure_end and seizure_start < end:
            return True
    return False


class SeizureDetectionCHBMIT(BaseTask):
    """Binary seizure detection on CHB-MIT, in the CBraMod windowing.

    Works with :class:`~pyhealth.datasets.CHBMITDataset`.

    For each EDF this task:

    1. reads the file and keeps 16 bipolar channels (missing ones are zero)
    2. optionally band-pass and notch filters
    3. cuts non-overlapping ``window_sec`` windows and labels them
    4. optionally adds extra seizure windows at ``oversample_stride_sec``
    5. resamples each window to ``resample_rate`` and normalizes it

    ``label_mode="overlap"`` marks a window as seizure when it overlaps a
    seizure. ``label_mode="cbramod"`` reproduces CBraMod: the window is
    positive only when a seizure start or end falls inside it. Extra
    seizure windows (CBraMod's ``-s-`` segments) are tagged with
    ``is_augmented=1`` so they can be dropped from validation and test.

    Each sample contains:

    - ``signal``: float tensor, shape ``(16, resample_rate * window_sec)``
    - ``label``: 1 seizure, 0 background
    - ``is_augmented``: 1 for an extra seizure window, else 0
    - ``stft``: time-frequency tensor, when ``compute_stft`` is True

    The label schema is ``binary``, so models use a sigmoid output and the
    trainer reports binary metrics (``pr_auc``, ``roc_auc``, ``f1``).

    Each window is resampled on its own, as CBraMod does, so seizure
    boundaries stay in native-rate sample indices. TUAB and TUEV resample
    the whole recording before windowing, so values near window edges
    differ slightly between the two approaches.

    Args:
        resample_rate: Samples per second after resampling. CBraMod uses 200.
        window_sec: Window length in seconds. CBraMod uses 10.
        normalization: ``None``, ``"div_by_100"`` or ``"95th_percentile"``.
        bandpass_filter: ``(low, high)`` in Hz passed to MNE, or None to
            skip filtering (the CBraMod and TFM-Tokenizer default).
        notch_filter: Notch frequency in Hz, or None to skip it.
        compute_stft: When True, attach an STFT computed like TUAB/TUEV.
        oversample_seizures: When True, add the extra seizure windows
            CBraMod writes around each seizure.
        oversample_stride_sec: Stride of those extra windows. CBraMod uses 5.
        label_mode: ``"overlap"`` or ``"cbramod"``.
        subjects: Subject ids to process. None processes every subject.
            ``"chb01"`` and ``"1"`` are accepted.

    Raises:
        ValueError: If ``label_mode`` or ``normalization`` is unknown.

    Examples:
        >>> from pyhealth.datasets import CHBMITDataset
        >>> from pyhealth.tasks import SeizureDetectionCHBMIT
        >>> dataset = CHBMITDataset(
        ...     root="/srv/local/data/physionet.org/files/chbmit/1.0.0/",
        ... )
        >>> task = SeizureDetectionCHBMIT(
        ...     normalization="95th_percentile",
        ...     compute_stft=False,
        ... )
        >>> train, val, test = dataset.set_task(
        ...     task, split=dataset.default_subject_split(),
        ... )
        >>> train[0]["signal"].shape
    """

    task_name: str = "CHBMIT_seizure_detection"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor", "stft": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __init__(
        self,
        resample_rate: float = 200,
        window_sec: float = 10,
        normalization: str | None = None,
        bandpass_filter: tuple[float, float] | None = None,
        notch_filter: float | None = None,
        compute_stft: bool = True,
        oversample_seizures: bool = True,
        oversample_stride_sec: float = 5,
        label_mode: str = "overlap",
        subjects: list[str] | None = None,
    ) -> None:
        super().__init__()
        if label_mode not in ("overlap", "cbramod"):
            raise ValueError(
                "label_mode must be 'overlap' or 'cbramod', got " f"{label_mode!r}"
            )
        # Validate early so a bad name fails before any EDF is read.
        normalize_signal(np.zeros((1, 2)), normalization)
        # Plain floats keep round() returning an int for window lengths.
        self.resample_rate = float(resample_rate)
        self.window_sec = float(window_sec)
        self.normalization = normalization
        self.bandpass_filter = bandpass_filter
        self.notch_filter = notch_filter
        self.compute_stft = compute_stft
        self.oversample_seizures = oversample_seizures
        self.oversample_stride_sec = float(oversample_stride_sec)
        self.label_mode = label_mode
        # A sorted tuple, not a set: vars(task) feeds the task cache key, and
        # str(set) order changes with the per-process hash seed.
        self.subjects = (
            None
            if subjects is None
            else tuple(sorted({normalize_subject(s) for s in subjects}))
        )
        if not compute_stft:
            self.input_schema = {"signal": "tensor"}

    def read_edf(self, file_name: str) -> tuple[np.ndarray, float] | None:
        """Load one EDF and return the 16 bipolar channels in uV.

        Args:
            file_name: Path to an EDF recording.

        Returns:
            ``(signal, sampling_rate)`` with ``signal`` shaped
            ``(16, n_samples)``, or None if the file cannot be read or has
            none of the 16 bipolar channels.
        """
        try:
            data, ch_names, sfreq = read_edf_uv(
                file_name, self.bandpass_filter, self.notch_filter
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to read %s: %s", file_name, exc)
            return None

        selected = select_bipolar_channels(data, ch_names)
        if selected is None:
            logger.warning("No bipolar channels in %s; skipping the file.", file_name)
            return None
        return selected, sfreq

    def _windows(
        self, n_samples: int, sfreq: float, intervals: list[tuple[int, int]]
    ) -> list[tuple]:
        """List regular and extra-seizure windows for one recording.

        A window shorter than ``window_sec`` is dropped. Extra seizure
        windows run from one second before the seizure to one second after
        it, stepped by ``oversample_stride_sec``, and are always labeled 1.

        Args:
            n_samples: Length of the recording in samples.
            sfreq: Sampling rate in Hz.
            intervals: Seizure sample intervals from :func:`_seizure_intervals`.

        Returns:
            Tuples ``(start, end, label, is_augmented, segment_id)``.
            ``segment_id`` is the CBraMod-style suffix (``"0"``,
            ``"s-0-add-256"``).
        """
        win = round(self.window_sec * sfreq)
        if win <= 0:
            return []
        windows = []
        for start in range(0, n_samples - win + 1, win):
            end = start + win
            label = int(_window_is_seizure(start, end, intervals, self.label_mode))
            windows.append((start, end, label, 0, str(start)))

        if not self.oversample_seizures:
            return windows
        pad = round(sfreq)
        stride = round(self.oversample_stride_sec * sfreq)
        if stride <= 0:
            return windows
        for index, (seizure_start, seizure_end) in enumerate(intervals):
            first = max(0, seizure_start - pad)
            last = min(seizure_end + pad, n_samples)
            for start in range(first, last, stride):
                end = start + win
                if end > n_samples:
                    break
                windows.append((start, end, 1, 1, f"s-{index}-add-{start}"))
        return windows

    def _samples_from_event(self, patient_id: str, event: Any) -> list[dict[str, Any]]:
        """Turn one recording event into window samples.

        Args:
            patient_id: Subject id, for example ``"chb01"``.
            event: A ``recordings`` event with ``signal_file``,
                ``record_id``, ``seizure_starts`` and ``seizure_ends``.

        Returns:
            One sample dict per full window. Empty if the EDF is skipped.
        """
        loaded = self.read_edf(event.signal_file)
        if loaded is None:
            return []
        signal, sfreq = loaded
        starts = _parse_time_list(getattr(event, "seizure_starts", ""))
        ends = _parse_time_list(getattr(event, "seizure_ends", ""))
        if len(starts) != len(ends):
            logger.warning(
                "%s has %d seizure starts and %d ends; using the first %d.",
                event.signal_file,
                len(starts),
                len(ends),
                min(len(starts), len(ends)),
            )
            n_pairs = min(len(starts), len(ends))
            starts, ends = starts[:n_pairs], ends[:n_pairs]
        intervals = _seizure_intervals(starts, ends, sfreq, self.label_mode)
        record_id = getattr(event, "record_id", None) or event.signal_file

        samples: list[dict[str, Any]] = []
        target = round(self.resample_rate * self.window_sec)
        for start, end, label, augmented, segment_id in self._windows(
            signal.shape[1], sfreq, intervals
        ):
            window = signal[:, start:end]
            if window.shape[1] != target:
                window = resample(window, target, axis=1)
            window = normalize_signal(window, self.normalization)
            tensor = torch.FloatTensor(np.ascontiguousarray(window))
            sample: dict[str, Any] = {
                "patient_id": patient_id,
                "record_id": record_id,
                "signal_file": event.signal_file,
                "signal": tensor,
                "label": label,
                "segment_id": segment_id,
                "start_time": start,
                "end_time": end,
                "is_augmented": augmented,
            }
            if self.compute_stft:
                from pyhealth.models.tfm_tokenizer import get_stft_torch

                sample["stft"] = get_stft_torch(
                    tensor.unsqueeze(0),
                    resampling_rate=int(self.resample_rate),
                ).squeeze(0)
            samples.append(sample)
        return samples

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build window samples for one subject.

        Args:
            patient: A PyHealth patient whose ``recordings`` events are the
                rows of ``chbmit-pyhealth.csv``.

        Returns:
            Window samples for that subject. Empty when ``subjects`` was set
            and this subject is not in it.
        """
        patient_id = normalize_subject(patient.patient_id)
        if self.subjects is not None and patient_id not in self.subjects:
            return []
        samples: list[dict[str, Any]] = []
        for event in patient.get_events("recordings"):
            samples.extend(self._samples_from_event(patient_id, event))
        return samples
