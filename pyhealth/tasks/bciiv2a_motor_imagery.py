"""
PyHealth task for BCI Competition IV dataset 2a motor imagery.

Dataset link:
    https://bnci-horizon-2020.eu/database/data-sets (entry 001-2014)

Dataset description:
    C. Brunner, R. Leeb, G. R. Muller-Putz, A. Schlogl, and G. Pfurtscheller.
    "BCI Competition 2008 - Graz data set A."
    https://www.bbci.de/competition/iv/desc_2a.pdf

Description:
    Implements ``BCIIV2aMotorImageryClassification`` (four-class motor
    imagery from one EEG epoch per trial) and the trial reader
    ``load_bciiv2a_trials``.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

#: Class names in label-index order (BNCI ``y`` = index + 1).
BCIIV2A_CLASSES: tuple[str, ...] = ("left_hand", "right_hand", "feet", "tongue")
#: Number of EEG channels; the 3 EOG channels follow them in ``X``.
N_EEG_CHANNELS = 22
SESSION_FOLD = {"T": 0, "E": 1}


def load_bciiv2a_trials(
    path: str,
    tmin: float = 2.0,
    tmax: float = 6.0,
    drop_artifacts: bool = True,
    include_eog: bool = False,
    l_freq: float | None = None,
    h_freq: float | None = None,
) -> list[dict[str, Any]]:
    """Cut the trials of one BNCI 001-2014 session file into epochs.

    Each run's ``trial`` field gives the 1-based sample of the trial start
    (fixation cross and warning tone, t = 0 s in desc_2a); the cue appears at
    t = 2 s and motor imagery lasts until t = 6 s. Runs without trials (the
    EOG calibration runs) are skipped.

    Args:
        path (str): Path to an ``A0sT.mat`` / ``A0sE.mat`` file.
        tmin (float): Window start in seconds after trial start.
        tmax (float): Window end (exclusive) in seconds after trial start.
        drop_artifacts (bool): Skip trials the experts marked as artifacts.
            Trials whose window holds NaN or infinite samples, or runs past
            the end of the run, are always skipped with a warning.
        include_eog (bool): Keep the 3 EOG channels (desc_2a says they must
            not be used for classification). Defaults to ``False``.
        l_freq (float | None): Optional band-pass low edge in Hz, applied to
            each continuous run with MNE before epoching.
        h_freq (float | None): Optional band-pass high edge in Hz.

    Returns:
        list[dict[str, Any]]: One dict per trial with ``run``, ``trial``,
        ``label`` (0-3), ``artifact`` and ``signal`` of shape
        ``(channels, round((tmax - tmin) * 250))`` in microvolts (float32).

    Raises:
        ValueError: If ``tmax <= tmin``.

    Examples:
        >>> trials = load_bciiv2a_trials("/data/bnci/A01T.mat")  # doctest: +SKIP
        >>> trials[0]["signal"].shape  # doctest: +SKIP
        (22, 1000)
    """
    if tmax <= tmin:
        raise ValueError(f"tmax must be greater than tmin, got {tmin}, {tmax}")
    from scipy.io import loadmat

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    runs = np.atleast_1d(mat["data"])
    n_channels = N_EEG_CHANNELS + (3 if include_eog else 0)
    out: list[dict[str, Any]] = []
    for run_index, run in enumerate(runs):
        onsets = np.atleast_1d(run.trial).astype(int)
        if onsets.size == 0:
            continue
        fs = float(run.fs)
        x = np.asarray(run.X, dtype=np.float64)[:, :n_channels].T
        if l_freq is not None or h_freq is not None:
            import mne

            x = mne.filter.filter_data(x, fs, l_freq, h_freq, verbose="error")
        labels = np.atleast_1d(run.y).astype(int)
        artifacts = np.atleast_1d(run.artifacts).astype(int)
        start_offset = round(tmin * fs)
        length = round((tmax - tmin) * fs)
        for trial_index, onset in enumerate(onsets):
            artifact = bool(artifacts[trial_index]) if artifacts.size else False
            if artifact and drop_artifacts:
                continue
            start = onset - 1 + start_offset
            if start < 0 or start + length > x.shape[1]:
                logger.warning(
                    "Trial %d of run %d exceeds the run", trial_index, run_index
                )
                continue
            signal = x[:, start : start + length].astype(np.float32)
            if not np.isfinite(signal).all():
                logger.warning(
                    "Trial %d of run %d of %s has non-finite samples; skipped",
                    trial_index,
                    run_index,
                    path,
                )
                continue
            out.append(
                {
                    "run": run_index,
                    "trial": trial_index,
                    "label": int(labels[trial_index]) - 1,
                    "artifact": artifact,
                    "signal": signal,
                }
            )
    return out


class BCIIV2aMotorImageryClassification(BaseTask):
    """Four-class motor imagery classification on BCI Competition IV 2a.

    Works with :class:`~pyhealth.datasets.BCICompetitionIV2aDataset`.

    One sample per trial: a 22-channel EEG epoch (microvolts, 250 Hz) and its
    class, ``0`` left hand, ``1`` right hand, ``2`` feet, ``3`` tongue
    (:data:`BCIIV2A_CLASSES`). The default window [2, 6) s after trial start
    is the cue-to-end imagery period of the paradigm (desc_2a: fixation cross
    at t = 0 s, cue at t = 2 s, imagery until the cross disappears at
    t = 6 s), giving ``(22, 1000)`` epochs.

    Each sample also carries ``session`` (``"T"``/``"E"``), ``split``
    (``"train"``/``"test"``, the official protocol) and ``session_fold``
    (0/1, for :func:`~pyhealth.datasets.split_by_strat_fold`).

    Deviations from CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB):

    - CLIMB epochs [0, 4) s after the ``trial`` marker, which is the trial
      start, so its window is 2 s of fixation plus the first 2 s of imagery.
      The default here is [2, 6) s; ``tmin=0, tmax=4`` reproduces CLIMB.
    - CLIMB's config says a 0.5-50 Hz band-pass but its code applies 4-40 Hz.
      No filter is applied by default; pass ``l_freq`` / ``h_freq``.
    - Artifact trials are dropped from both sessions by default, as in CLIMB
      and as in the competition's scoring (desc_2a builds the confusion
      matrices from artifact-free trials). ``drop_artifacts=False`` keeps
      the artifact-marked trials too (288 per session on the released
      files). Trials whose window contains NaN or infinite samples, or that
      run past the end of their run, are always skipped with a warning,
      whatever ``drop_artifacts`` is.
    - CLIMB's per-subject train-set normalisation is model-side and is not
      applied here.

    Args:
        tmin (float): Window start, seconds after trial start. Default 2.0.
        tmax (float): Window end, seconds after trial start. Default 6.0.
        drop_artifacts (bool): Skip artifact-marked trials. Default ``True``.
        include_eog (bool): Append the 3 EOG channels. Default ``False``.
        l_freq (float | None): Optional band-pass low edge (Hz).
        h_freq (float | None): Optional band-pass high edge (Hz).

    Examples:
        >>> task = BCIIV2aMotorImageryClassification()
        >>> task.output_schema
        {'label': ('multiclass', {'labels': 4})}
        >>> from pyhealth.datasets import BCICompetitionIV2aDataset
        >>> ds = BCICompetitionIV2aDataset(root="/data/bnci/001-2014")  # doctest: +SKIP
        >>> samples = ds.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "BCIIV2aMotorImageryClassification"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, Any]] = {
        "label": ("multiclass", {"labels": len(BCIIV2A_CLASSES)})
    }

    def __init__(
        self,
        tmin: float = 2.0,
        tmax: float = 6.0,
        drop_artifacts: bool = True,
        include_eog: bool = False,
        l_freq: float | None = None,
        h_freq: float | None = None,
    ) -> None:
        if tmax <= tmin:
            raise ValueError(f"tmax must be greater than tmin, got {tmin}, {tmax}")
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.drop_artifacts = drop_artifacts
        self.include_eog = include_eog
        self.l_freq = l_freq
        self.h_freq = h_freq
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per trial of each of the patient's sessions."""
        samples: list[dict[str, Any]] = []
        for event in patient.get_events(event_type="sessions"):
            session = str(event.session)
            trials = load_bciiv2a_trials(
                str(event.signal_file),
                tmin=self.tmin,
                tmax=self.tmax,
                drop_artifacts=self.drop_artifacts,
                include_eog=self.include_eog,
                l_freq=self.l_freq,
                h_freq=self.h_freq,
            )
            for trial in trials:
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": str(event.record_id),
                        "session": session,
                        "split": str(event.split),
                        "session_fold": SESSION_FOLD[session],
                        "run": trial["run"],
                        "trial": trial["trial"],
                        "signal": trial["signal"],
                        "label": trial["label"],
                    }
                )
        return samples
