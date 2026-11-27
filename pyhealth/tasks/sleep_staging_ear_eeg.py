"""Sleep staging on EESM23 ear-EEG.

Author: Jathurshan Pradeepkumar (jp65@illinois.edu), UIUC
Description: 30-second epoch sleep staging on the four in-ear channels
    (RB, RT, LB, LT), following TFM-Tokenizer preprocessing for EESM23.

References:
    - Mikkelsen, K. B. et al. Ear-EEG sleep monitoring data sets.
      Scientific Data 12, 2025.
      https://www.nature.com/articles/s41597-025-04579-8
    - Rezai Tabar, Y. et al. Ear-EEG Sleep Monitoring 2023 (EESM23).
      OpenNeuro ds005178. https://doi.org/10.18112/openneuro.ds005178.v1.0.0
    - Pradeepkumar, J. et al. Tokenizing Single-Channel EEG with
      Time-Frequency Motif Learning. ICLR 2026.
      https://github.com/Jathurshan0330/TFM-Tokenizer
      (``datasets_processing/EarEEG/process.py``, ``EarEEGloader``)
"""

import logging
from typing import Any, ClassVar

import mne
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample

from pyhealth.tasks import BaseTask
from pyhealth.tasks.eegbci import normalize_signal
from pyhealth.tasks.subject_ids import normalize_prefixed_subject

logger = logging.getLogger(__name__)

# In-ear dry electrodes (average-referenced). Not a 10-20 scalp montage.
EAR_EEG_CHANNELS: tuple[str, ...] = ("RB", "RT", "LB", "LT")

# TFM-Tokenizer label order: scoring_idx 1..6 minus one.
SLEEP_STAGES: tuple[str, ...] = ("Wake", "REM", "N1", "N2", "N3", "Artefact")

ARTEFACT_SCORING_IDX = 6


def normalize_subject(subject: str) -> str:
    """Canonicalize an EESM23 subject id to ``sub-XXX``.

    ``"sub-001"``, ``"SUB1"`` and ``"1"`` all become ``"sub-001"``. Other
    strings are lower-cased and returned unchanged. A thin wrapper over
    :func:`~pyhealth.tasks.subject_ids.normalize_prefixed_subject`, which
    the CHB-MIT task shares.

    Args:
        subject: Subject id as stored on a sample or passed by the user.

    Returns:
        The canonical subject id.

    Examples:
        >>> from pyhealth.tasks.sleep_staging_ear_eeg import normalize_subject
        >>> [normalize_subject(s) for s in ("1", "SUB2", "sub-010")]
        ['sub-001', 'sub-002', 'sub-010']
    """
    return normalize_prefixed_subject(subject, "sub", 3, separator="-")


def _has_scoring(event: Any) -> bool:
    """Return True when the metadata row has manual sleep scoring.

    Args:
        event: A ``recordings`` event from ``eesm23-pyhealth.csv``.

    Returns:
        True when ``has_scoring`` is set or ``scoring_file`` is non-empty.
    """
    flag = str(getattr(event, "has_scoring", "0")).strip()
    if flag in {"1", "true", "True"}:
        return True
    scoring_file = str(getattr(event, "scoring_file", "") or "").strip()
    return scoring_file != "" and scoring_file.lower() not in {"nan", "none", "<na>"}


class SleepStagingEarEEG(BaseTask):
    """Multi-class sleep staging on EESM23 ear-EEG.

    Works with :class:`~pyhealth.datasets.EESM23Dataset`.

    For each scored night this task:

    1. reads the EEGLAB ``.set`` file and keeps RB, RT, LB and LT (ELE is
       never used)
    2. walks the 30 s epochs in the scoring TSV
    3. skips epochs with NaN samples (common around recording gaps)
    4. band-pass and notch filters each epoch (0.1-100 Hz, 50 Hz)
    5. resamples to ``resample_rate`` and normalizes

    Labels follow TFM-Tokenizer: ``scoring_idx - 1`` for six classes
    (Wake, REM, N1, N2, N3, Artefact). Set ``include_artefact=False`` to
    drop Artefact epochs and keep five AASM stages (labels 0-4).

    Filtering is applied per epoch because whole-recording filters would
    spread NaN gaps across the night.

    Each sample contains:

    - ``signal``: float tensor, shape ``(4, resample_rate * epoch_sec)``
    - ``label``: integer class index
    - ``stft``: time-frequency tensor, when ``compute_stft`` is True

    The label schema is ``multiclass`` (accuracy, ``f1_macro``, Cohen's
    kappa, and related metrics).

    Args:
        resample_rate: Samples per second after resampling. TFM-Tokenizer
            finetuning often uses 200.
        epoch_sec: Epoch length in seconds. EESM23 uses 30.
        normalization: ``None``, ``"div_by_100"`` or ``"95th_percentile"``.
        bandpass_filter: ``(low, high)`` in Hz passed to MNE per epoch, or
            None to skip filtering.
        notch_filter: Notch frequency in Hz per epoch, or None to skip it.
        compute_stft: When True, attach an STFT like TUAB/TUEV/CHB-MIT.
        include_artefact: When True (default), keep Artefact as class 5.
            When False, Artefact epochs are dropped.
        subjects: Subject ids to process. None processes every subject.

    Raises:
        ValueError: If ``normalization`` is unknown.

    Examples:
        >>> from pyhealth.datasets import EESM23Dataset
        >>> from pyhealth.tasks import SleepStagingEarEEG
        >>> dataset = EESM23Dataset(
        ...     root="/path/to/ds005178",
        ... )
        >>> task = SleepStagingEarEEG(
        ...     normalization="95th_percentile",
        ...     compute_stft=False,
        ... )
        >>> train, val, test = dataset.set_task(
        ...     task, split=dataset.default_subject_split(),
        ... )
        >>> train[0]["signal"].shape
    """

    task_name: str = "SleepStagingEarEEG"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor", "stft": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "multiclass"}

    def __init__(
        self,
        resample_rate: float = 200,
        epoch_sec: float = 30,
        normalization: str | None = "95th_percentile",
        bandpass_filter: tuple[float, float] | None = (0.1, 100.0),
        notch_filter: float | None = 50.0,
        compute_stft: bool = True,
        include_artefact: bool = True,
        subjects: list[str] | None = None,
    ) -> None:
        super().__init__()
        normalize_signal(np.zeros((1, 2)), normalization)
        self.resample_rate = float(resample_rate)
        self.epoch_sec = float(epoch_sec)
        self.normalization = normalization
        self.bandpass_filter = bandpass_filter
        self.notch_filter = notch_filter
        self.compute_stft = compute_stft
        self.include_artefact = include_artefact
        self.subjects = (
            None
            if subjects is None
            else tuple(sorted({normalize_subject(s) for s in subjects}))
        )
        if not compute_stft:
            self.input_schema = {"signal": "tensor"}

    def read_recording(
        self, file_name: str
    ) -> tuple[np.ndarray, float, list[str]] | None:
        """Load one EEGLAB file and return the four ear channels in uV.

        Args:
            file_name: Path to ``*_acq-earEEG_eeg.set``.

        Returns:
            ``(data, sfreq, ch_names)`` with ``data`` shaped
            ``(4, n_samples)``, or None if the file cannot be read or is
            missing any of RB, RT, LB or LT.
        """
        try:
            raw = mne.io.read_raw_eeglab(
                file_name, preload=True, verbose="error"
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to read %s: %s", file_name, exc)
            return None

        ch_names = list(raw.ch_names)
        missing = [name for name in EAR_EEG_CHANNELS if name not in ch_names]
        if missing:
            logger.warning(
                "Missing ear channels %s in %s; skipping the file.",
                missing,
                file_name,
            )
            raw.close()
            return None

        data = raw.get_data(picks=list(EAR_EEG_CHANNELS), units="uV")
        sfreq = float(raw.info["sfreq"])
        raw.close()
        return data, sfreq, list(EAR_EEG_CHANNELS)

    def _filter_epoch(self, epoch: np.ndarray, sfreq: float) -> np.ndarray:
        """Band-pass and notch filter one epoch.

        Args:
            epoch: Array shaped ``(n_channels, n_samples)``.
            sfreq: Sampling rate in Hz.

        Returns:
            Filtered epoch with the same shape.
        """
        out = epoch
        if self.bandpass_filter is not None:
            out = mne.filter.filter_data(
                out,
                sfreq=sfreq,
                l_freq=self.bandpass_filter[0],
                h_freq=self.bandpass_filter[1],
                verbose="error",
            )
        if self.notch_filter is not None:
            out = mne.filter.notch_filter(
                out, Fs=sfreq, freqs=[self.notch_filter], verbose="error"
            )
        return out

    def _samples_from_event(
        self, patient_id: str, event: Any
    ) -> list[dict[str, Any]]:
        """Turn one scored night into epoch samples.

        Args:
            patient_id: Subject id, for example ``"sub-001"``.
            event: A ``recordings`` event with ``signal_file`` and
                ``scoring_file``.

        Returns:
            One sample dict per valid epoch. Empty if the night is unscored
            or the recording is skipped.
        """
        if not _has_scoring(event):
            return []

        scoring_file = str(getattr(event, "scoring_file", "") or "").strip()
        if not scoring_file:
            return []

        loaded = self.read_recording(event.signal_file)
        if loaded is None:
            return []
        signal, sfreq, _ch_names = loaded

        try:
            scoring = pd.read_csv(scoring_file, sep="\t")
        except (OSError, ValueError) as exc:
            logger.warning("Failed to read %s: %s", scoring_file, exc)
            return []

        session_id = getattr(event, "session_id", "")
        target_len = round(self.epoch_sec * sfreq)
        resampled_len = round(self.resample_rate * self.epoch_sec)
        samples: list[dict[str, Any]] = []

        for row_idx, row in scoring.iterrows():
            scoring_idx = int(row["scoring_idx"])
            if not self.include_artefact and scoring_idx == ARTEFACT_SCORING_IDX:
                continue
            label = scoring_idx - 1

            start = round(float(row["onset"]) * sfreq)
            end = start + target_len
            if end > signal.shape[1]:
                continue
            epoch = signal[:, start:end]
            if epoch.shape[1] != target_len:
                continue
            if np.isnan(epoch).any():
                continue

            epoch = self._filter_epoch(epoch, sfreq)
            if resampled_len != target_len:
                epoch = resample(epoch, resampled_len, axis=1)
            epoch = normalize_signal(epoch, self.normalization)
            tensor = torch.FloatTensor(np.ascontiguousarray(epoch))
            sample: dict[str, Any] = {
                "patient_id": patient_id,
                "session_id": session_id,
                "epoch_index": int(row_idx),
                "signal_file": event.signal_file,
                "scoring_file": scoring_file,
                "signal": tensor,
                "label": label,
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
        """Build epoch samples for one subject.

        Args:
            patient: A PyHealth patient whose ``recordings`` events are the
                rows of ``eesm23-pyhealth.csv``.

        Returns:
            Epoch samples for that subject. Empty when ``subjects`` was set
            and this subject is not in it.
        """
        patient_id = normalize_subject(patient.patient_id)
        if self.subjects is not None and patient_id not in self.subjects:
            return []
        samples: list[dict[str, Any]] = []
        for event in patient.get_events("recordings"):
            samples.extend(self._samples_from_event(patient_id, event))
        return samples
