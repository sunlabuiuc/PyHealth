"""Dementia classification on the AHEPA resting-state EEG dataset.

Author: Axel Coiffard-Donaty

Each subject of the dataset (OpenNeuro ds004504) has one eyes-closed
recording and one diagnosis: Alzheimer's disease, frontotemporal dementia or
healthy control. This task cuts each recording into overlapping windows and
labels every window with the diagnosis of its subject, for one of the
classification problems studied on this dataset.

The label belongs to the subject, so windows of one subject must never be
spread over training and test data: split by patient. The dataset authors'
review of the studies on this dataset reports that weaker validation goes
with accuracies 7 to 10 points higher.

References:
    - Miltiadous, A. et al. A Dataset of Scalp EEG Recordings of Alzheimer's
      Disease, Frontotemporal Dementia and Healthy Subjects from Routine
      EEG. Data 8(6), 95, 2023. https://doi.org/10.3390/data8060095
    - Review of the studies on this dataset: Miltiadous, A. et al. The
      AHEPA EEG benchmark: setting the standard for machine learning in
      dementia diagnosis, a scoping review. Cognitive Neurodynamics 20(1),
      2026. https://doi.org/10.1007/s11571-026-10464-w
    - Reference electrodes: Bigdely-Shamlo, N. et al. The PREP pipeline:
      standardized preprocessing for large-scale EEG analysis. Frontiers in
      Neuroinformatics 9:16, 2015. https://doi.org/10.3389/fninf.2015.00016
"""

import logging
import math
from fractions import Fraction
from typing import Any, ClassVar

import mne
import numpy as np
from scipy.signal import resample_poly

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

AHEPA_CHANNELS: tuple[str, ...] = (
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
)  # fmt: skip
"""The 19 channels of every recording, in the order of the files."""

AHEPA_GROUPS: dict[str, str] = {
    "A": "Alzheimer's disease",
    "F": "frontotemporal dementia",
    "C": "healthy control",
}
"""Group codes of ``participants.tsv``."""

# Label of each group in each problem; a group that is absent is left out.
_PROBLEMS: dict[str, dict[str, int]] = {
    "ad_vs_cn": {"C": 0, "A": 1},
    "ftd_vs_cn": {"C": 0, "F": 1},
    "ad_vs_ftd": {"F": 0, "A": 1},
    "dementia_vs_cn": {"C": 0, "A": 1, "F": 1},
    "three_class": {"C": 0, "A": 1, "F": 2},
}

AHEPA_PROBLEMS: tuple[str, ...] = tuple(_PROBLEMS)

AHEPA_VARIANTS: tuple[str, ...] = ("derivatives", "raw")

AHEPA_REFERENCES: tuple[str, ...] = ("stored", "average")

# Largest numerator or denominator of the resampling ratio.
_MAX_RESAMPLING_TERM = 1000

_VARIANT_COLUMN: dict[str, str] = {
    "derivatives": "signal_file",
    "raw": "raw_signal_file",
}


class DementiaClassificationAHEPA(BaseTask):
    """Dementia classification on the AHEPA resting-state EEG dataset.

    Runs on :class:`~pyhealth.datasets.AHEPADementiaDataset`. For each
    subject this task:

    1. keeps the subject if its group belongs to ``problem``
    2. opens the artifact-cleaned or the unprocessed recording
       (``variant``) and reads the 19 channels in microvolts
    3. optionally resamples it (``resample_rate``)
    4. cuts windows of ``window_seconds`` every ``stride_seconds``, each
       labelled with the diagnosis of the subject

    Problems, with the label of each group:

    - ``"ad_vs_cn"`` (default): control 0, Alzheimer's disease 1
    - ``"ftd_vs_cn"``: control 0, frontotemporal dementia 1
    - ``"ad_vs_ftd"``: frontotemporal dementia 0, Alzheimer's disease 1
    - ``"dementia_vs_cn"``: control 0, either dementia 1
    - ``"three_class"``: control 0, Alzheimer's disease 1, frontotemporal
      dementia 2 (a ``multiclass`` label; the others are ``binary``)

    Defaults: the data descriptor (Miltiadous et al., 2023) benchmarks
    Alzheimer's disease against controls and frontotemporal dementia against
    controls on 4 second windows with 50% overlap (section 3.3), and
    encourages the use of the artifact-cleaned recordings (section 4). The
    authors' review (Miltiadous et al., 2026) counts Alzheimer's disease
    against controls as the most studied problem (40 studies). Window
    lengths in the reviewed studies range from 1 to 240 seconds, for
    example 30 seconds with 15 seconds of overlap for the authors' DICE-net
    model, so both are parameters.

    Reference: ``reference="stored"`` (the default) keeps the recordings as
    distributed, referenced to the mastoids A1-A2; ``reference="average"``
    subtracts the mean over the 19 channels at each time. The API page of
    this task gives the measurements and the limits of either choice.

    Split by patient: every window of a subject carries the same label, and
    the review asks for complete subject-level independence between
    training and test data; the data descriptor uses leave-one-subject-out
    validation.
    Use :func:`~pyhealth.datasets.split_by_patient` or
    :class:`~pyhealth.datasets.PatientSplit`, never a split by sample.

    A subject without the recording of ``variant``, or whose recording
    misses one of the 19 channels or is shorter than one window, is skipped
    with a warning. The end of a recording that does not fill a window is
    dropped. A binary problem needs subjects of both groups: with one group
    only, the binary label processor of ``set_task`` raises an error.

    Each sample contains:

    - ``patient_id``: participant id, for example ``"sub-001"``
    - ``signal``: float32 array, shape ``(19, window_seconds * rate)``,
      ``(19, 2000)`` by default (the recordings are 500 Hz)
    - ``label``: integer class index of ``problem``
    - ``group``: group code of the subject (``"A"``, ``"F"`` or ``"C"``)
    - ``start_second``: start of the window in the recording

    Args:
        problem: Classification problem, one of ``AHEPA_PROBLEMS``.
        window_seconds: Window length in seconds.
        stride_seconds: Shift between windows in seconds.
        variant: ``"derivatives"`` for the artifact-cleaned recordings
            (0.5-45 Hz, see :class:`~pyhealth.datasets.AHEPADementiaDataset`)
            or ``"raw"`` for the unprocessed ones.
        resample_rate: Sampling rate of the output in Hz, or None to keep
            the 500 Hz of the recordings.
        reference: ``"stored"`` to keep the reference of the file, or
            ``"average"`` to subtract the mean over the 19 channels.

    Raises:
        ValueError: If ``problem``, ``variant`` or ``reference`` is unknown, or
            ``window_seconds``, ``stride_seconds`` or ``resample_rate`` isn't
            a positive number. When a recording is read: if a window or the
            stride covers less than one sample, or if ``resample_rate``
            cannot be reached from the recording's rate with a ratio of
            integers up to 1000.

    Examples:
        >>> from pyhealth.datasets import AHEPADementiaDataset, split_by_patient
        >>> from pyhealth.tasks import DementiaClassificationAHEPA
        >>> dataset = AHEPADementiaDataset(root="/path/to/ds004504")
        >>> task = DementiaClassificationAHEPA(problem="ad_vs_cn")
        >>> samples = dataset.set_task(task)
        >>> samples[0]["signal"].shape
        torch.Size([19, 2000])
        >>> train, val, test = split_by_patient(samples, [0.6, 0.2, 0.2])
    """

    task_name: str = "DementiaClassificationAHEPA"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __init__(
        self,
        problem: str = "ad_vs_cn",
        window_seconds: float = 4.0,
        stride_seconds: float = 2.0,
        variant: str = "derivatives",
        resample_rate: float | None = None,
        reference: str = "stored",
    ) -> None:
        super().__init__()
        if problem not in _PROBLEMS:
            raise ValueError(f"problem must be one of {AHEPA_PROBLEMS}, got {problem}.")
        if variant not in AHEPA_VARIANTS:
            raise ValueError(f"variant must be one of {AHEPA_VARIANTS}, got {variant}.")
        if reference not in AHEPA_REFERENCES:
            raise ValueError(
                f"reference must be one of {AHEPA_REFERENCES}, got {reference}."
            )
        for name, value in (
            ("window_seconds", window_seconds),
            ("stride_seconds", stride_seconds),
            ("resample_rate", resample_rate),
        ):
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive number, got {value}.")
        self.problem = problem
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.variant = variant
        self.reference = reference
        self.resample_rate = None if resample_rate is None else float(resample_rate)
        self._labels = _PROBLEMS[problem]
        n_classes = len(set(self._labels.values()))
        if n_classes > 2:
            # Fixed classes: the labels keep their meaning when a group is
            # absent, for example after a partial download.
            self.output_schema = {"label": ("multiclass", {"labels": n_classes})}

    def read_recording(self, signal_file: str) -> tuple[np.ndarray, float] | None:
        """Load the 19 channels of one recording in microvolts.

        Args:
            signal_file: Path to the EEGLAB ``.set`` file.

        Returns:
            ``(data, sfreq)`` with ``data`` shaped ``(19, n_samples)`` in the
            order of ``AHEPA_CHANNELS`` and ``sfreq`` the output rate, or
            None if a channel is missing.
        """
        raw = mne.io.read_raw_eeglab(signal_file, preload=True, verbose="error")
        missing = [c for c in AHEPA_CHANNELS if c not in raw.ch_names]
        if missing:
            logger.warning(
                "Skipping %s, it has no %s channel(s). Channels are %s.",
                signal_file,
                missing,
                raw.ch_names,
            )
            return None
        data = raw.get_data(picks=list(AHEPA_CHANNELS), units="uV")
        sfreq = float(raw.info["sfreq"])
        if self.reference == "average":
            data = data - data.mean(axis=0, keepdims=True)
        if self.resample_rate is not None and self.resample_rate != sfreq:
            ratio = Fraction(self.resample_rate / sfreq).limit_denominator(
                _MAX_RESAMPLING_TERM
            )
            if ratio <= 0 or not math.isclose(
                sfreq * ratio, self.resample_rate, rel_tol=1e-9
            ):
                raise ValueError(
                    f"resample_rate={self.resample_rate} Hz cannot be reached from "
                    f"the {sfreq} Hz of {signal_file} with a ratio of integers up "
                    f"to {_MAX_RESAMPLING_TERM}."
                )
            # padtype="line" continues the signal past its two ends; the
            # default pads with zeros, which distorts the first and last
            # samples of a recording that is not centred on zero.
            data = resample_poly(
                data, ratio.numerator, ratio.denominator, axis=-1, padtype="line"
            )
            sfreq = self.resample_rate
        return data, sfreq

    def _samples_from_event(self, patient_id: str, event: Any) -> list[dict[str, Any]]:
        """Turn the recording of one subject into window samples.

        Args:
            patient_id: Participant id, for example ``"sub-001"``.
            event: A ``recordings`` event with ``group`` and the recording
                columns.

        Returns:
            One sample dict per window; empty when the subject's group is
            not part of ``problem``.
        """
        group = str(getattr(event, "group", "") or "").strip().upper()
        label = self._labels.get(group)
        if label is None:
            if group not in AHEPA_GROUPS:
                logger.warning(
                    "Skipping %s, its group %r is not one of %s.",
                    patient_id,
                    group,
                    tuple(AHEPA_GROUPS),
                )
            return []

        signal_file = getattr(event, _VARIANT_COLUMN[self.variant], "")
        signal_file = "" if signal_file is None else str(signal_file).strip()
        if signal_file in {"", "nan", "None"}:
            logger.warning(
                "Skipping %s, it has no %s recording.", patient_id, self.variant
            )
            return []
        loaded = self.read_recording(signal_file)
        if loaded is None:
            return []
        signal, sfreq = loaded

        window_len = round(self.window_seconds * sfreq)
        stride_len = round(self.stride_seconds * sfreq)
        if window_len < 1 or stride_len < 1:
            raise ValueError(
                f"window_seconds={self.window_seconds} and "
                f"stride_seconds={self.stride_seconds} must each cover at least "
                f"one sample at {sfreq} Hz."
            )
        if signal.shape[1] < window_len:
            logger.warning(
                "Skipping %s, its recording (%.1f s) is shorter than one window "
                "(%.1f s).",
                patient_id,
                signal.shape[1] / sfreq,
                self.window_seconds,
            )
            return []
        samples: list[dict[str, Any]] = []
        for start in range(0, signal.shape[1] - window_len + 1, stride_len):
            samples.append(
                {
                    "patient_id": patient_id,
                    "group": group,
                    "start_second": start / sfreq,
                    "signal": signal[:, start : start + window_len].astype(np.float32),
                    "label": label,
                }
            )
        return samples

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build window samples for one subject.

        Args:
            patient: A PyHealth patient whose ``recordings`` events are the
                rows of ``ahepa_dementia-pyhealth.csv``.

        Returns:
            Window samples for the recording of that subject.
        """
        samples: list[dict[str, Any]] = []
        for event in patient.get_events("recordings"):
            samples.extend(self._samples_from_event(patient.patient_id, event))
        return samples
