"""Sleep staging on the Haaglanden Medisch Centrum (HMC) database.

Author: Axel Coiffard-Donaty

Each HMC recording is scored in 30 second epochs with the AASM stages, and
its ``SNxxx_sleepscoring.txt`` file lists one row per annotation:
``Date, Time, Recording onset, Duration, Annotation, Linked channel``. The
annotations are ``Sleep stage W``, ``N1``, ``N2``, ``N3`` and ``R``, plus one
``Lights off`` and one ``Lights on`` marker. This task cuts the recording
into epochs, keeps a set of channels (the four EEG derivations by default)
and labels each epoch with its stage.

References:
    - Alvarez-Estevez, D., Rijsman, R. M. Inter-database validation of a
      deep learning approach for automatic sleep scoring. PLoS ONE 16(8):
      e0256111, 2021. https://doi.org/10.1371/journal.pone.0256111
"""

import csv
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import mne
import numpy as np

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

HMC_EEG_CHANNELS: tuple[str, ...] = (
    "EEG F4-M1",
    "EEG C4-M1",
    "EEG O2-M1",
    "EEG C3-M2",
)
"""The four EEG derivations of every HMC recording, as named in the EDF."""

HMC_PSG_CHANNELS: tuple[str, ...] = HMC_EEG_CHANNELS + (
    "EOG E1-M2",
    "EOG E2-M2",
    "EMG chin",
)
"""EEG, EOG and chin EMG: the channels AASM scoring relies on."""

SLEEP_STAGES: tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")

_STAGE_TO_LABEL: dict[str, int] = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
}

_SCORED_EPOCH_SECONDS = 30


@dataclass
class HMCScoring:
    """Content of one ``SNxxx_sleepscoring.txt`` file.

    Attributes:
        onsets: Onset in seconds of each scored epoch, from the start of the
            recording.
        stages: Annotation text of each scored epoch, for example
            ``"Sleep stage N2"``.
        lights_off: Onset of the ``Lights off`` marker, or None.
        lights_on: Onset of the ``Lights on`` marker, or None.

    Examples:
        >>> from pyhealth.tasks.sleep_staging_hmc import HMCScoring
        >>> scoring = HMCScoring(onsets=[0.0, 30.0], stages=["Sleep stage W"] * 2)
        >>> scoring.lights_off is None
        True
    """

    onsets: list[float] = field(default_factory=list)
    stages: list[str] = field(default_factory=list)
    lights_off: float | None = None
    lights_on: float | None = None


def read_hmc_scoring(label_file: str) -> HMCScoring:
    """Read one HMC ``_sleepscoring.txt`` file.

    Rows whose annotation starts with ``Sleep stage`` are kept as scored
    epochs, whatever the stage, so that unexpected stages can be reported by
    the caller. Other annotations, apart from the lights markers, are
    ignored.

    Args:
        label_file: Path to ``SNxxx_sleepscoring.txt``.

    Returns:
        The scored epochs and the lights markers.

    Examples:
        >>> import os, tempfile
        >>> from pyhealth.tasks.sleep_staging_hmc import read_hmc_scoring
        >>> text = (
        ...     "Date, Time, Recording onset, Duration, Annotation, Linked channel\\n"
        ...     "01.01.01, 23.59.30, 0, 30, Sleep stage W, \\n"
        ...     "01.01.01, 23.59.45, 15.5, 0, Lights off, EEG F4-A1\\n"
        ...     "02.01.01, 00.00.00, 30, 30, Sleep stage N1, \\n"
        ... )
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = os.path.join(tmp, "SN001_sleepscoring.txt")
        ...     with open(path, "w") as f:
        ...         _ = f.write(text)
        ...     scoring = read_hmc_scoring(path)
        >>> scoring.onsets, scoring.stages, scoring.lights_off
        ([0.0, 30.0], ['Sleep stage W', 'Sleep stage N1'], 15.5)
    """
    scoring = HMCScoring()
    with open(label_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 5:
                continue
            onset, annotation = float(row[2]), row[4].strip()
            if annotation.startswith("Sleep stage"):
                scoring.onsets.append(onset)
                scoring.stages.append(annotation)
            elif annotation == "Lights off":
                scoring.lights_off = onset
            elif annotation == "Lights on":
                scoring.lights_on = onset
    return scoring


class SleepStagingHMC(BaseTask):
    """Multi-class sleep staging on the HMC database.

    Runs on :class:`~pyhealth.datasets.HMCDataset`. For each recording this
    task:

    1. reads the hypnogram and the lights markers from
       ``SNxxx_sleepscoring.txt``
    2. opens the EDF file and keeps ``channels``, in microvolts
    3. cuts each scored 30 second epoch into ``30 / epoch_seconds`` epochs

    Lights period: PhysioNet describes the recordings as clipped to the
    scoring interval between lights off and lights on, but in version 1.1
    the first two or three scored epochs (all wake) start before the
    ``Lights off`` marker, and in all but one recording the last one or two
    end after ``Lights on``. With ``lights_period_only=True`` (the default)
    only scored epochs that lie entirely between the two markers are kept:
    the AASM manual counts the recording time from lights off to lights
    on. Set it to False to keep every scored epoch. A recording
    without both markers keeps every scored epoch, with a warning.

    Labels are 0 W, 1 N1, 2 N2, 3 N3 and 4 REM, see ``SLEEP_STAGES``.
    Epochs with another ``Sleep stage`` annotation are dropped with a
    warning, as are scored epochs that run past the end of the signal.
    Recordings missing one of the channels are skipped with a warning.

    Each sample contains:

    - ``patient_id``: recording id, for example ``"SN001"``
    - ``signal``: float32 array, shape ``(len(channels), 256 * epoch_seconds)``
      for HMC's 256 Hz recordings
    - ``label``: integer class index
    - ``epoch_index``: position of the epoch in the recording, counting
      every ``epoch_seconds`` long epoch from the start of the scoring

    Args:
        channels: EDF channel names to keep. Defaults to
            ``HMC_EEG_CHANNELS``, the four EEG derivations;
            ``HMC_PSG_CHANNELS`` adds the two EOG and the chin EMG.
        epoch_seconds: Epoch length in whole seconds. It has to divide 30.
        lights_period_only: Keep only scored epochs between the lights off
            and lights on markers. Defaults to True.

    Raises:
        ValueError: If ``epoch_seconds`` isn't a whole number that divides 30
            or ``channels`` is empty.

    Examples:
        >>> from pyhealth.datasets import HMCDataset
        >>> from pyhealth.tasks import SleepStagingHMC
        >>> dataset = HMCDataset(root="/path/to/hmc-sleep-staging/1.1")
        >>> samples = dataset.set_task(SleepStagingHMC())
        >>> samples[0]["signal"].shape
        torch.Size([4, 7680])
    """

    task_name: str = "SleepStagingHMC"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "multiclass"}

    def __init__(
        self,
        channels: Sequence[str] = HMC_EEG_CHANNELS,
        epoch_seconds: int = 30,
        lights_period_only: bool = True,
    ) -> None:
        super().__init__()
        if (
            epoch_seconds <= 0
            or epoch_seconds != int(epoch_seconds)
            or _SCORED_EPOCH_SECONDS % epoch_seconds != 0
        ):
            raise ValueError(
                "epoch_seconds must be a whole number of seconds that divides "
                f"{_SCORED_EPOCH_SECONDS}, got {epoch_seconds}."
            )
        if len(channels) == 0:
            raise ValueError("channels must name at least one channel.")
        self.channels = tuple(channels)
        self.epoch_seconds = int(epoch_seconds)
        self.lights_period_only = lights_period_only

    def read_recording(self, signal_file: str) -> tuple[np.ndarray, float] | None:
        """Load the selected channels of one HMC recording in microvolts.

        Args:
            signal_file: Path to ``SNxxx.edf``.

        Returns:
            ``(data, sfreq)`` with ``data`` shaped
            ``(len(channels), n_samples)``, or None if a channel is missing.
        """
        raw = mne.io.read_raw_edf(signal_file, preload=False, verbose="error")
        missing = [c for c in self.channels if c not in raw.ch_names]
        if missing:
            logger.warning(
                "Skipping %s, it has no %s channel(s). Channels are %s.",
                signal_file,
                missing,
                raw.ch_names,
            )
            raw.close()
            return None
        data = raw.get_data(picks=list(self.channels), units="uV")
        sfreq = float(raw.info["sfreq"])
        raw.close()
        return data, sfreq

    def _in_lights_period(self, scoring: HMCScoring, signal_file: str) -> list[bool]:
        """Flag the scored epochs that lie between lights off and lights on.

        Args:
            scoring: The recording's scoring.
            signal_file: Path used in the warning.

        Returns:
            One flag per scored epoch.
        """
        if not self.lights_period_only:
            return [True] * len(scoring.onsets)
        if scoring.lights_off is None or scoring.lights_on is None:
            logger.warning(
                "%s has no lights off or lights on marker; keeping every scored epoch.",
                signal_file,
            )
            return [True] * len(scoring.onsets)
        return [
            onset >= scoring.lights_off
            and onset + _SCORED_EPOCH_SECONDS <= scoring.lights_on
            for onset in scoring.onsets
        ]

    def _samples_from_event(self, patient_id: str, event: Any) -> list[dict[str, Any]]:
        """Turn one scored recording into epoch samples.

        Args:
            patient_id: Recording id, for example ``"SN001"``.
            event: A ``recordings`` event with ``signal_file`` and
                ``label_file``.

        Returns:
            One sample dict per kept epoch.
        """
        loaded = self.read_recording(event.signal_file)
        if loaded is None:
            return []
        signal, sfreq = loaded
        scoring = read_hmc_scoring(event.label_file)
        keep = self._in_lights_period(scoring, event.signal_file)

        epoch_len = round(self.epoch_seconds * sfreq)
        per_score = _SCORED_EPOCH_SECONDS // self.epoch_seconds
        n_samples = signal.shape[1]
        samples: list[dict[str, Any]] = []
        unknown, past_end = 0, 0
        for i, (onset, stage) in enumerate(zip(scoring.onsets, scoring.stages)):
            if not keep[i]:
                continue
            label = _STAGE_TO_LABEL.get(stage)
            if label is None:
                unknown += 1
                continue
            first = round(onset * sfreq)
            if first + per_score * epoch_len > n_samples:
                past_end += 1
                continue
            for j in range(per_score):
                start = first + j * epoch_len
                samples.append(
                    {
                        "patient_id": patient_id,
                        "epoch_index": i * per_score + j,
                        "signal": signal[:, start : start + epoch_len].astype(
                            np.float32
                        ),
                        "label": label,
                    }
                )
        if unknown:
            logger.warning(
                "Dropped %d epochs of %s with a stage other than W, N1, N2, N3 and R.",
                unknown,
                event.label_file,
            )
        if past_end:
            logger.warning(
                "Dropped %d scored epochs of %s that run past the end of the signal.",
                past_end,
                event.signal_file,
            )
        return samples

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build epoch samples for one recording.

        Args:
            patient: A PyHealth patient whose ``recordings`` events are the
                rows of ``hmc-pyhealth.csv``.

        Returns:
            Epoch samples for every scored recording of that patient.
        """
        samples: list[dict[str, Any]] = []
        for event in patient.get_events("recordings"):
            samples.extend(self._samples_from_event(patient.patient_id, event))
        return samples
