"""Seizure onset zone localization on the HUP iEEG dataset.

Author: Axel Coiffard-Donaty

Each patient of the dataset (OpenNeuro ds004100) was implanted with
intracranial electrodes before epilepsy surgery. ``channels.tsv`` marks, for
every recording, the contacts of the clinically determined seizure onset
zone (``soz``) and those inside the resected or ablated tissue (``resect``).
This task cuts the interictal recordings into single-channel windows and
labels each window with the mark of its contact.

A contact belongs to one patient, and all its windows carry the same label,
so windows of one patient must never be spread over training and test data:
split by patient.

References:
    - Bernabei, J. M., Li, A. et al. Quantitative approaches to guide
      epilepsy surgery from intracranial EEG. Brain 146(6), 2248-2258, 2023.
      https://doi.org/10.1093/brain/awad007
    - Bernabei, J. M., Sinha, N. et al. Normative intracranial EEG maps
      epileptogenic tissues in focal epilepsy. Brain 145(6), 1949-1961,
      2022. https://doi.org/10.1093/brain/awab480
    - Dataset: https://doi.org/10.18112/openneuro.ds004100.v1.1.3
"""

import csv
import logging
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

import mne
import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, tf2sos

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

HUP_TARGETS: tuple[str, ...] = ("soz", "resect")
"""Marks of ``channels.tsv`` that can be the label."""

HUP_REFERENCES: tuple[str, ...] = ("average", "stored")

HUP_OUTCOMES: dict[str, str] = {
    "S": "seizure free after surgery",
    "F": "recurring seizures after surgery",
}
"""Outcome codes of ``participants.tsv``."""

# Quality factor of the notch filter (60 Hz / 30: about 2 Hz wide per pass).
_NOTCH_QUALITY = 30.0

# Order of the Butterworth band-pass filter (Bernabei et al., 2022).
_BANDPASS_ORDER = 1

# Largest numerator or denominator of the resampling ratio.
_MAX_RESAMPLING_TERM = 1000

_EMPTY = {"", "nan", "none", "n/a"}


def read_hup_channels(channels_file: str | Path) -> list[dict[str, Any]]:
    """Read the ``channels.tsv`` of one recording.

    Args:
        channels_file: Path to the BIDS ``*_channels.tsv`` file.

    Returns:
        One dict per channel, in the order of the file, with ``name``,
        ``type`` (``"SEEG"`` or ``"ECOG"``), ``good`` (``status`` is
        ``good``), ``soz`` and ``resect`` (the mark is listed in
        ``status_description``).

    Raises:
        ValueError: If the file lacks the ``name``, ``status`` or
            ``status_description`` column.

    Examples:
        >>> from pyhealth.tasks.soz_localization_hup import read_hup_channels
        >>> channels = read_hup_channels(
        ...     "/path/to/ds004100/sub-HUP060/ses-presurgery/ieeg/"
        ...     "sub-HUP060_ses-presurgery_task-interictal_acq-seeg_run-01"
        ...     "_channels.tsv"
        ... )
        >>> [c["name"] for c in channels if c["good"] and c["soz"]]
        ['RPFa3', 'RPFa4', 'RPFb3', 'RPFc3']
    """
    with open(channels_file, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames or []
        missing = [
            c for c in ("name", "status", "status_description") if c not in columns
        ]
        if missing:
            raise ValueError(
                f"{channels_file} has no {missing} column(s); its columns are "
                f"{columns}."
            )
        channels = []
        for row in reader:
            marks = {
                mark.strip().lower()
                for mark in (row["status_description"] or "").split(",")
            }
            channels.append(
                {
                    "name": (row["name"] or "").strip(),
                    "type": (row.get("type") or "").strip().upper(),
                    "good": (row["status"] or "").strip().lower() == "good",
                    "soz": "soz" in marks,
                    "resect": "resect" in marks,
                }
            )
    return channels


def read_hup_electrodes(
    electrodes_file: str | Path,
) -> dict[str, tuple[float, float, float]]:
    """Read the position of the contacts of one patient.

    The values are returned as written in ``electrodes.tsv``. They are
    millimetres, although ``coordsystem.json`` declares metres; the dataset
    gives their space as ``fsaverage`` in ``coordsystem.json`` and as
    ICBM152 MNI in its README (see
    :class:`~pyhealth.datasets.HUPiEEGDataset`). ``x`` is negative on the
    left.

    Args:
        electrodes_file: Path to the BIDS ``*_electrodes.tsv`` file.

    Returns:
        ``(x, y, z)`` for each electrode name; the three values are NaN when
        the file gives no position (``n/a``) or a value that is not a
        number. A repeated name keeps its first row, and rows without a
        name are left out.

    Raises:
        ValueError: If the file lacks the ``name``, ``x``, ``y`` or ``z``
            column.

    Examples:
        >>> from pyhealth.tasks.soz_localization_hup import read_hup_electrodes
        >>> positions = read_hup_electrodes(
        ...     "/path/to/ds004100/sub-HUP064/ses-presurgery/ieeg/"
        ...     "sub-HUP064_ses-presurgery_acq-ecog_space-fsaverage"
        ...     "_electrodes.tsv"
        ... )
        >>> positions["EEG LF 01-Ref"]
        (-1.46523, 56.6861, -20.215324097541)
    """
    with open(electrodes_file, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames or []
        missing = [c for c in ("name", "x", "y", "z") if c not in columns]
        if missing:
            raise ValueError(
                f"{electrodes_file} has no {missing} column(s); its columns are "
                f"{columns}."
            )
        positions: dict[str, tuple[float, float, float]] = {}
        for row in reader:
            name = (row["name"] or "").strip()
            if not name or name in positions:
                continue
            try:
                x, y, z = (float(row[axis]) for axis in ("x", "y", "z"))
            except (TypeError, ValueError):
                x = y = z = math.nan
            if not all(math.isfinite(value) for value in (x, y, z)):
                x = y = z = math.nan
            positions[name] = (x, y, z)
    return positions


def _is_positive_number(value: Any) -> bool:
    """Whether ``value`` is a finite number above zero (booleans excluded)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


class SOZLocalizationHUP(BaseTask):
    """Seizure onset zone localization on the HUP iEEG dataset.

    Runs on :class:`~pyhealth.datasets.HUPiEEGDataset`. For each interictal
    recording of a patient this task:

    1. reads ``channels.tsv`` and keeps the channels with status ``good``
       (the dataset marks as bad the ECG channels, almost all scalp channels
       and some intracranial contacts)
    2. reads these channels in microvolts and re-references them
       (``reference``)
    3. filters them (``notch``, ``bandpass``) and resamples them
       (``resample_rate``)
    4. cuts every channel into windows of ``window_seconds`` every
       ``stride_seconds``; each window is one sample, labelled 1 when its
       contact carries the ``target`` mark and 0 otherwise

    Labels: ``target="soz"`` (default) is the seizure onset zone that
    clinicians determined; ``"resect"`` is the contacts that overlap with the
    resected or ablated tissue. Neither is a verified epileptogenic zone:
    Bernabei, Li et al. (2023) note that this zone is not observed, and
    that training data are ideal when the clinically hypothesized regions
    lie well within the resected or ablated tissue. ``outcomes=("S",)``
    restricts the task to the patients who became seizure free (``S`` is
    seizure freedom and ``F`` recurring seizures in ``participants.tsv``).

    A contact is labelled 1 when it carries the mark in at least one
    interictal recording of the patient, so that its label is the same in
    every recording; a warning lists the contacts whose mark differs between
    recordings. In version 1.1.3 this concerns ``sub-HUP180``: its second
    interictal recording marks 2 seizure onset contacts and all its other
    recordings 8, and that file is the only ``channels.tsv`` of the patient
    that version 1.1.3, which corrected its seizure onset zone, did not
    update. A patient without any marked good contact is skipped with a
    warning, because the mark is then missing, not negative (no patient of
    version 1.1.3 is in this case).

    In version 1.1.3 the task covers 56 patients (``sub-HUP158`` has no
    interictal recording) and 6119 good contacts, of which 538 (8.8%) are in
    the seizure onset zone: the classes are unbalanced.

    Preprocessing defaults follow Bernabei, Sinha et al. (2022), who studied
    interictal recordings of the same hospital: a first-order Butterworth
    band-pass filter from 0.5 to 80 Hz, a 60 Hz notch filter, and resampling
    to 200 Hz. Differences from that study: it used a bipolar montage, which
    needs the layout of each electrode and is not built here;
    ``reference="average"`` subtracts the mean over the good channels of the
    recording instead, and ``"stored"`` keeps the referential signal of the
    files (their reference electrode is not documented). Choices where the
    paper is silent: filters run forward and backward (zero phase), the
    notch has a quality factor of 30, and the resampling filter continues
    the signal linearly past its two ends. Resampling is needed in any case,
    since the recordings come at 256, 500, 512 or 1024 Hz. That study used
    two 30 second clips per patient; the interictal recordings here last 5
    minutes, so the default 30 second windows give 10 samples per contact
    and recording.

    Limits: a sample holds one contact, so it carries no information on the
    other contacts. Bernabei, Sinha et al. (2022) separated resected seizure
    onset contacts from normal ones with an area under the curve of 0.65
    from the spectral power of each contact, and 0.77 once connectivity
    between contacts was added. The ictal recordings of the dataset are not
    used.

    Split by patient: use :func:`~pyhealth.datasets.split_by_patient` or
    :class:`~pyhealth.datasets.PatientSplit`, never a split by sample.

    Each sample contains:

    - ``signal``: float32 array, shape ``(1, window_seconds * rate)``,
      ``(1, 6000)`` by default
    - ``label``: 1 if the contact carries the ``target`` mark, else 0
    - ``channel``: name of the contact in the recording
    - ``channel_type``: ``"SEEG"`` or ``"ECOG"``
    - ``x``, ``y``, ``z``: position of the contact in millimetres, as
      stored in ``electrodes.tsv`` (see
      :class:`~pyhealth.datasets.HUPiEEGDataset` for what the dataset
      states about their unit and space), NaN when the file gives none
      (126 of the 6119 good contacts of version 1.1.3) or is missing, with
      a warning in the second case. They are not part of the model input
      by default.
    - ``run``: run number of the recording
    - ``start_second``: start of the window in the recording

    Args:
        target: Mark used as label, one of ``HUP_TARGETS``.
        window_seconds: Window length in seconds.
        stride_seconds: Shift between windows in seconds.
        resample_rate: Sampling rate of the output in Hz.
        bandpass: ``(low, high)`` edges in Hz of the band-pass filter, or
            None for no band-pass filter.
        notch: Frequency in Hz of the notch filter, or None for no notch
            filter. The recordings come from the United States (60 Hz).
        reference: ``"average"`` to subtract the mean over the good
            channels, or ``"stored"`` to keep the reference of the file.
        outcomes: Outcome codes of the patients to keep, among the keys of
            ``HUP_OUTCOMES``, for example ``("S",)``; None keeps every
            patient.

    Raises:
        ValueError: If ``target`` or ``reference`` is unknown, a duration or
            a rate isn't a positive number, ``bandpass`` isn't an increasing
            pair of positive numbers, or ``outcomes`` is empty or holds an
            unknown code. When a recording is read: if a filter edge isn't
            below half its sampling rate, if a window or the stride covers
            less than one sample, or if ``resample_rate`` cannot be reached
            from the recording's rate with a ratio of integers up to 1000.

    Examples:
        >>> from pyhealth.datasets import HUPiEEGDataset, split_by_patient
        >>> from pyhealth.tasks import SOZLocalizationHUP
        >>> dataset = HUPiEEGDataset(root="/path/to/ds004100")
        >>> task = SOZLocalizationHUP(target="soz")
        >>> samples = dataset.set_task(task)
        >>> samples[0]["signal"].shape
        torch.Size([1, 6000])
        >>> train, val, test = split_by_patient(samples, [0.6, 0.2, 0.2])
    """

    task_name: str = "SOZLocalizationHUP"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __init__(
        self,
        target: str = "soz",
        window_seconds: float = 30.0,
        stride_seconds: float = 30.0,
        resample_rate: float = 200.0,
        bandpass: tuple[float, float] | None = (0.5, 80.0),
        notch: float | None = 60.0,
        reference: str = "average",
        outcomes: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if target not in HUP_TARGETS:
            raise ValueError(f"target must be one of {HUP_TARGETS}, got {target}.")
        if reference not in HUP_REFERENCES:
            raise ValueError(
                f"reference must be one of {HUP_REFERENCES}, got {reference}."
            )
        for name, value in (
            ("window_seconds", window_seconds),
            ("stride_seconds", stride_seconds),
            ("resample_rate", resample_rate),
            ("notch", notch),
        ):
            if value is not None and not _is_positive_number(value):
                raise ValueError(f"{name} must be a positive number, got {value!r}.")
        if bandpass is not None:
            if not (
                isinstance(bandpass, (tuple, list))
                and len(bandpass) == 2
                and all(_is_positive_number(edge) for edge in bandpass)
                and bandpass[0] < bandpass[1]
            ):
                raise ValueError(
                    f"bandpass must be (low, high) with 0 < low < high, got "
                    f"{bandpass!r}."
                )
            bandpass = (float(bandpass[0]), float(bandpass[1]))
        if outcomes is not None:
            if isinstance(outcomes, str):
                outcomes = (outcomes,)
            if not isinstance(outcomes, (tuple, list, set, frozenset)):
                raise ValueError(
                    f"outcomes must be a tuple of codes or None, got {outcomes!r}."
                )
            outcomes = tuple(str(code).strip().upper() for code in outcomes)
            unknown = [code for code in outcomes if code not in HUP_OUTCOMES]
            if not outcomes or unknown:
                raise ValueError(
                    f"outcomes must hold codes among {tuple(HUP_OUTCOMES)}, got "
                    f"{outcomes}."
                )
        self.target = target
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.resample_rate = float(resample_rate)
        self.bandpass = bandpass
        self.notch = None if notch is None else float(notch)
        self.reference = reference
        self.outcomes = outcomes

    def preprocess(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """Re-reference, filter and resample the channels of one recording.

        Args:
            data: Array ``(channels, n_samples)`` in microvolts.
            sfreq: Sampling rate of ``data`` in Hz.

        Returns:
            Array ``(channels, n_samples * resample_rate / sfreq)``.

        Raises:
            ValueError: If a filter edge isn't below ``sfreq / 2``, or
                ``resample_rate`` cannot be reached from ``sfreq`` with a
                ratio of integers up to 1000.

        Examples:
            >>> import numpy as np
            >>> from pyhealth.tasks import SOZLocalizationHUP
            >>> task = SOZLocalizationHUP()
            >>> task.preprocess(np.random.randn(4, 5120), sfreq=512.0).shape
            (4, 2000)
        """
        if self.reference == "average":
            data = data - data.mean(axis=0, keepdims=True)
        nyquist = sfreq / 2
        if self.notch is not None:
            if self.notch >= nyquist:
                raise ValueError(
                    f"notch={self.notch} Hz must be below half the sampling "
                    f"rate ({sfreq} Hz)."
                )
            sos = tf2sos(*iirnotch(self.notch, _NOTCH_QUALITY, fs=sfreq))
            data = sosfiltfilt(sos, data, axis=-1)
        if self.bandpass is not None:
            if self.bandpass[1] >= nyquist:
                raise ValueError(
                    f"bandpass={self.bandpass} Hz must end below half the "
                    f"sampling rate ({sfreq} Hz)."
                )
            sos = butter(
                _BANDPASS_ORDER, self.bandpass, btype="bandpass", fs=sfreq, output="sos"
            )
            data = sosfiltfilt(sos, data, axis=-1)
        if self.resample_rate != sfreq:
            ratio = Fraction(self.resample_rate / sfreq).limit_denominator(
                _MAX_RESAMPLING_TERM
            )
            if (
                ratio <= 0
                or ratio.numerator > _MAX_RESAMPLING_TERM
                or not math.isclose(sfreq * ratio, self.resample_rate, rel_tol=1e-9)
            ):
                raise ValueError(
                    f"resample_rate={self.resample_rate} Hz cannot be reached from "
                    f"{sfreq} Hz with a ratio of integers up to "
                    f"{_MAX_RESAMPLING_TERM}."
                )
            # padtype="line" continues the signal past its two ends; the
            # default pads with zeros, which distorts the first and last
            # samples of a recording that is not centred on zero.
            data = resample_poly(
                data, ratio.numerator, ratio.denominator, axis=-1, padtype="line"
            )
        return data

    def _good_channels(
        self, patient_id: str, event: Any
    ) -> tuple[str, list[dict[str, Any]]] | None:
        """Read the good channels of one interictal recording.

        Args:
            patient_id: Participant id, for example ``"sub-HUP060"``.
            event: A ``recordings`` event with ``signal_file``,
                ``channels_file``, ``electrodes_file``, ``recording_type``
                and ``outcome``.

        Returns:
            ``(signal_file, channels)`` with the good channels of
            ``channels.tsv``, a repeated name kept once, each with its
            ``position`` from ``electrodes.tsv``; None when the
            recording is not interictal, the patient's outcome is not in
            ``outcomes``, or a file is missing.
        """
        recording_type = str(getattr(event, "recording_type", "") or "").strip()
        if recording_type.lower() != "interictal":
            return None
        if self.outcomes is not None:
            outcome = str(getattr(event, "outcome", "") or "").strip().upper()
            if outcome not in self.outcomes:
                return None
        signal_file = str(getattr(event, "signal_file", "") or "").strip()
        channels_file = str(getattr(event, "channels_file", "") or "").strip()
        if signal_file.lower() in _EMPTY or channels_file.lower() in _EMPTY:
            logger.warning(
                "Skipping a recording of %s, it has no signal or channels file.",
                patient_id,
            )
            return None
        channels, repeated = [], []
        for channel in read_hup_channels(channels_file):
            if not channel["good"]:
                continue
            if any(channel["name"] == kept["name"] for kept in channels):
                repeated.append(channel["name"])
                continue
            channels.append(channel)
        if repeated:
            logger.warning(
                "%s lists %s more than once; the first row of each is used.",
                channels_file,
                sorted(set(repeated)),
            )
        electrodes_file = str(getattr(event, "electrodes_file", "") or "").strip()
        positions = {}
        if electrodes_file.lower() in _EMPTY or not Path(electrodes_file).is_file():
            logger.warning(
                "No electrodes file for %s; the positions of its contacts are NaN.",
                signal_file,
            )
        else:
            positions = read_hup_electrodes(electrodes_file)
        for channel in channels:
            channel["position"] = positions.get(channel["name"], (math.nan,) * 3)
        return signal_file, channels

    def _samples_from_recording(
        self,
        patient_id: str,
        run: str,
        signal_file: str,
        channels: list[dict[str, Any]],
        marked: set[str],
    ) -> list[dict[str, Any]]:
        """Turn one recording into single-channel window samples.

        Args:
            patient_id: Participant id, for example ``"sub-HUP060"``.
            run: Run number of the recording.
            signal_file: Path to the EDF file.
            channels: Good channels of the recording, from
                ``_good_channels``.
            marked: Names of the contacts labelled 1 for this patient.

        Returns:
            One sample dict per good channel and window; empty when the
            recording is skipped.
        """
        raw = mne.io.read_raw_edf(signal_file, preload=False, verbose="error")
        absent = [c["name"] for c in channels if c["name"] not in raw.ch_names]
        if absent:
            logger.warning(
                "%s does not hold %d channel(s) that its channels.tsv marks "
                "good; they are left out: %s",
                signal_file,
                len(absent),
                absent,
            )
            channels = [c for c in channels if c["name"] in raw.ch_names]
        if self.reference == "average" and len(channels) < 2:
            logger.warning(
                "Skipping %s, an average reference needs at least 2 good "
                "channels and it has %d.",
                signal_file,
                len(channels),
            )
            return []
        if not channels:
            logger.warning("Skipping %s, it has no good channel.", signal_file)
            return []

        sfreq = float(raw.info["sfreq"])
        window_len = round(self.window_seconds * self.resample_rate)
        stride_len = round(self.stride_seconds * self.resample_rate)
        if window_len < 1 or stride_len < 1:
            raise ValueError(
                f"window_seconds={self.window_seconds} and "
                f"stride_seconds={self.stride_seconds} must each cover at least "
                f"one sample at {self.resample_rate} Hz."
            )
        seconds = raw.n_times / sfreq
        if round(seconds * self.resample_rate) < window_len:
            logger.warning(
                "Skipping %s, it is shorter (%.1f s) than one window (%.1f s).",
                signal_file,
                seconds,
                self.window_seconds,
            )
            return []

        data = raw.get_data(picks=[c["name"] for c in channels], units="uV")
        signal = self.preprocess(data, sfreq)

        samples: list[dict[str, Any]] = []
        for index, channel in enumerate(channels):
            x, y, z = channel["position"]
            for start in range(0, signal.shape[1] - window_len + 1, stride_len):
                window = signal[index : index + 1, start : start + window_len]
                samples.append(
                    {
                        "patient_id": patient_id,
                        "channel": channel["name"],
                        "channel_type": channel["type"],
                        "x": x,
                        "y": y,
                        "z": z,
                        "run": run,
                        "start_second": start / self.resample_rate,
                        "signal": window.astype(np.float32),
                        "label": int(channel["name"] in marked),
                    }
                )
        return samples

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build single-channel window samples for one patient.

        Args:
            patient: A PyHealth patient whose ``recordings`` events are the
                rows of ``hup_ieeg-pyhealth.csv``.

        Returns:
            Window samples of the interictal recordings of that patient;
            empty when none of its good contacts carries the mark.
        """
        patient_id = patient.patient_id
        recordings = []
        for event in patient.get_events("recordings"):
            loaded = self._good_channels(patient_id, event)
            if loaded is not None:
                run = str(getattr(event, "run", "") or "").strip()
                recordings.append((run, *loaded))
        if not recordings:
            return []

        marks = [
            {c["name"] for c in channels if c[self.target]}
            for _, _, channels in recordings
        ]
        marked = set().union(*marks)
        if not marked:
            logger.warning(
                "Skipping %s, none of its good channels is marked %r.",
                patient_id,
                self.target,
            )
            return []
        listed = [{c["name"] for c in channels} for _, _, channels in recordings]
        differing = sorted(
            name
            for name in marked
            if any(
                name in names and name not in mark for names, mark in zip(listed, marks)
            )
        )
        if differing:
            logger.warning(
                "The %r mark of %s differs between the interictal recordings "
                "of %s; these contacts are labelled 1 in all of them.",
                self.target,
                differing,
                patient_id,
            )

        samples: list[dict[str, Any]] = []
        for run, signal_file, channels in recordings:
            samples.extend(
                self._samples_from_recording(
                    patient_id, run, signal_file, channels, marked
                )
            )
        return samples
