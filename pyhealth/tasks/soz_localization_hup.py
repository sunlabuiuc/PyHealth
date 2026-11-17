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

# Quality factor of the notch filter: a 2 Hz wide notch at 60 Hz.
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


class SOZLocalizationHUP(BaseTask):
    """Seizure onset zone localization on the HUP iEEG dataset.

    Runs on :class:`~pyhealth.datasets.HUPiEEGDataset`. For each interictal
    recording of a patient this task:

    1. reads ``channels.tsv`` and keeps the channels with status ``good``
       (this drops the scalp and ECG channels and the contacts that the
       dataset authors marked bad)
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
    A recording in which no good channel carries the mark is skipped with a
    warning, because the mark is then missing, not negative (no recording of
    version 1.1.3 is in this case). In version 1.1.3, 532 of the 6119 good
    contacts are in the seizure onset zone, a median of 8% per patient, so
    the classes are unbalanced.

    Preprocessing defaults follow Bernabei, Sinha et al. (2022), who studied
    interictal recordings of the same hospital: a first-order Butterworth
    band-pass filter from 0.5 to 80 Hz, a 60 Hz notch filter, and resampling
    to 200 Hz. Differences from that study: it used a bipolar montage, which
    needs the layout of each electrode and is not built here;
    ``reference="average"`` subtracts the mean over the good channels of the
    recording instead, and ``"stored"`` keeps the referential signal of the
    files (their reference electrode is not documented). The paper does not
    give the direction of filtering or the width of the notch: filters run
    forward and backward (zero phase) and the notch has a quality factor of
    30. Resampling is needed in any case, since the recordings come at 256,
    500, 512 or 1024 Hz. The 30 second windows match the clips of that
    study (two 30 second clips per patient); the recordings here last 5
    minutes or more, so each contact gives several windows.

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
        outcomes: Outcome codes of the patients to keep, for example
            ``("S",)``, or None to keep every patient.

    Raises:
        ValueError: If ``target`` or ``reference`` is unknown, a duration or
            a rate isn't a positive number, ``bandpass`` isn't an increasing
            pair of positive numbers, or ``outcomes`` is empty. When a
            recording is read: if a filter edge isn't below half its
            sampling rate, if a window or the stride covers less than one
            sample, or if ``resample_rate`` cannot be reached from the
            recording's rate with a ratio of integers up to 1000.

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
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive number, got {value}.")
        if bandpass is not None:
            if len(bandpass) != 2 or not (
                all(math.isfinite(edge) for edge in bandpass)
                and 0 < bandpass[0] < bandpass[1]
            ):
                raise ValueError(
                    f"bandpass must be (low, high) with 0 < low < high, got {bandpass}."
                )
            bandpass = (float(bandpass[0]), float(bandpass[1]))
        if outcomes is not None:
            if isinstance(outcomes, str):
                outcomes = (outcomes,)
            outcomes = tuple(str(code).strip().upper() for code in outcomes)
            if not outcomes:
                raise ValueError("outcomes must hold at least one code, or be None.")
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
            if ratio <= 0 or not math.isclose(
                sfreq * ratio, self.resample_rate, rel_tol=1e-9
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

    def _samples_from_event(self, patient_id: str, event: Any) -> list[dict[str, Any]]:
        """Turn one recording into single-channel window samples.

        Args:
            patient_id: Participant id, for example ``"sub-HUP060"``.
            event: A ``recordings`` event with ``signal_file``,
                ``channels_file``, ``recording_type``, ``run`` and
                ``outcome``.

        Returns:
            One sample dict per good channel and window; empty when the
            recording is not interictal, the patient's outcome is not in
            ``outcomes``, or the recording is skipped.
        """
        recording_type = str(getattr(event, "recording_type", "") or "").strip()
        if recording_type.lower() != "interictal":
            return []
        if self.outcomes is not None:
            outcome = str(getattr(event, "outcome", "") or "").strip().upper()
            if outcome not in self.outcomes:
                return []

        signal_file = str(getattr(event, "signal_file", "") or "").strip()
        channels_file = str(getattr(event, "channels_file", "") or "").strip()
        if signal_file.lower() in _EMPTY or channels_file.lower() in _EMPTY:
            logger.warning(
                "Skipping a recording of %s, it has no signal or channels file.",
                patient_id,
            )
            return []

        channels = [c for c in read_hup_channels(channels_file) if c["good"]]
        raw = mne.io.read_raw_edf(signal_file, preload=False, verbose="error")
        absent = [c["name"] for c in channels if c["name"] not in raw.ch_names]
        if absent:
            logger.warning(
                "%s lists %d good channel(s) that %s does not hold; they are "
                "left out: %s",
                channels_file,
                len(absent),
                signal_file,
                absent,
            )
            channels = [c for c in channels if c["name"] in raw.ch_names]
        if not any(c[self.target] for c in channels):
            logger.warning(
                "Skipping %s, none of its good channels is marked %r in %s.",
                signal_file,
                self.target,
                channels_file,
            )
            return []

        sfreq = float(raw.info["sfreq"])
        data = raw.get_data(picks=[c["name"] for c in channels], units="uV")
        signal = self.preprocess(data, sfreq)

        window_len = round(self.window_seconds * self.resample_rate)
        stride_len = round(self.stride_seconds * self.resample_rate)
        if window_len < 1 or stride_len < 1:
            raise ValueError(
                f"window_seconds={self.window_seconds} and "
                f"stride_seconds={self.stride_seconds} must each cover at least "
                f"one sample at {self.resample_rate} Hz."
            )
        if signal.shape[1] < window_len:
            logger.warning(
                "Skipping %s, it is shorter (%.1f s) than one window (%.1f s).",
                signal_file,
                signal.shape[1] / self.resample_rate,
                self.window_seconds,
            )
            return []

        run = str(getattr(event, "run", "") or "").strip()
        samples: list[dict[str, Any]] = []
        for index, channel in enumerate(channels):
            for start in range(0, signal.shape[1] - window_len + 1, stride_len):
                window = signal[index : index + 1, start : start + window_len]
                samples.append(
                    {
                        "patient_id": patient_id,
                        "channel": channel["name"],
                        "channel_type": channel["type"],
                        "run": run,
                        "start_second": start / self.resample_rate,
                        "signal": window.astype(np.float32),
                        "label": int(channel[self.target]),
                    }
                )
        return samples

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build single-channel window samples for one patient.

        Args:
            patient: A PyHealth patient whose ``recordings`` events are the
                rows of ``hup_ieeg-pyhealth.csv``.

        Returns:
            Window samples of the interictal recordings of that patient.
        """
        samples: list[dict[str, Any]] = []
        for event in patient.get_events("recordings"):
            samples.extend(self._samples_from_event(patient.patient_id, event))
        return samples
