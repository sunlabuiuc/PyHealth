"""Sleep staging on ISRUC-Sleep.

Author: Sameh Aissa

Each ISRUC recording is scored in 30 second epochs by two experts with
AASM stages, stored one integer per line in ``<id>_1.txt`` and
``<id>_2.txt``: 0 W, 1 N1, 2 N2, 3 N3 and 5 REM. This task cuts the
recording into epochs, keeps a set of channels (the six EEG channels
F3, F4, C3, C4, O1 and O2 by default) and labels each epoch with the
scoring of one expert. ``SleepStagingSequenceISRUC`` groups the epochs of
one channel into sequences, for sequence models such as DeepSleepNet.

References:
    - Khalighi, S. et al. ISRUC-Sleep: A comprehensive public dataset for
      sleep researchers. Computer Methods and Programs in Biomedicine 124,
      2016. https://doi.org/10.1016/j.cmpb.2015.10.013
"""

import logging
from collections.abc import Sequence
from typing import Any, ClassVar

import mne
import numpy as np
import torch

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

ISRUC_EEG_CHANNELS: tuple[str, ...] = ("F3", "F4", "C3", "C4", "O1", "O2")

SLEEP_STAGES: tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")

# ISRUC scores 0, 1, 2, 3 and 5. Score 4 (the old R&K S4 stage) is unused.
_SCORE_TO_LABEL: dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4}

_SCORED_EPOCH_SECONDS = 30

_REFERENCE_SUFFIXES: tuple[str, ...] = ("-A1", "-A2", "-M1", "-M2")


def base_channel_name(name: str) -> str:
    """Strip the mastoid reference from an ISRUC channel name.

    ISRUC names a channel after its electrode and reference, and the
    reference is written as A1/A2 in some recordings and M1/M2 in others.

    Args:
        name: Channel name as stored in the recording.

    Returns:
        The electrode name without the reference.

    Examples:
        >>> from pyhealth.tasks.sleep_staging_isruc import base_channel_name
        >>> [base_channel_name(c) for c in ("F3-A2", "C4-M1", "X1")]
        ['F3', 'C4', 'X1']
    """
    name = name.strip()
    for suffix in _REFERENCE_SUFFIXES:
        if name.upper().endswith(suffix):
            return name[: -len(suffix)]
    return name


def read_isruc_scores(label_file: str) -> np.ndarray:
    """Read one ISRUC scoring file.

    Args:
        label_file: Path to ``<id>_1.txt`` or ``<id>_2.txt``.

    Returns:
        Integer array with one score per 30 second epoch.

    Examples:
        >>> import tempfile, os
        >>> from pyhealth.tasks.sleep_staging_isruc import read_isruc_scores
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = os.path.join(tmp, "1_1.txt")
        ...     with open(path, "w") as f:
        ...         _ = f.write("0\\n0\\n5\\n2\\n")
        ...     read_isruc_scores(path).tolist()
        [0, 0, 5, 2]
    """
    return np.loadtxt(label_file, dtype=int, ndmin=1)


class SleepStagingISRUC(BaseTask):
    """Multi-class sleep staging on ISRUC-Sleep.

    Runs on :class:`~pyhealth.datasets.ISRUCDataset`. For each recording
    this task:

    1. reads the scoring of one expert (``annotator`` 1 or 2)
    2. opens the EDF file and keeps ``channels``, matched without their
       A1/A2 or M1/M2 reference, in microvolts
    3. cuts the signal into ``epoch_seconds`` long epochs, so a 30 second
       score labels 30 / ``epoch_seconds`` epochs in a row

    Recordings missing one of the channels are skipped with a warning. When
    the scoring and the signal disagree on the number of 30 second epochs,
    only the epochs present in both are kept. Epochs with a score other
    than 0, 1, 2, 3 or 5 are dropped.

    Labels are 0 W, 1 N1, 2 N2, 3 N3 and 4 REM, see ``SLEEP_STAGES``.

    Each sample contains:

    - ``signal``: float32 array, shape ``(len(channels), sfreq * epoch_seconds)``
    - ``label``: integer class index

    Args:
        channels: Channel names to keep, without the reference. Defaults to
            the six EEG channels F3, F4, C3, C4, O1 and O2.
        epoch_seconds: Epoch length in whole seconds. It has to divide 30.
        annotator: Which expert's scoring to use, 1 or 2.

    Raises:
        ValueError: If ``epoch_seconds`` isn't a whole number that divides 30,
            ``annotator`` isn't 1 or 2, or ``channels`` is empty.

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset
        >>> from pyhealth.tasks import SleepStagingISRUC
        >>> dataset = ISRUCDataset(root="/path/to/ISRUC-Sleep-I")
        >>> samples = dataset.set_task(SleepStagingISRUC(annotator=1))
        >>> samples[0]["signal"].shape
        torch.Size([6, 6000])
    """

    task_name: str = "SleepStagingISRUC"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "multiclass"}

    def __init__(
        self,
        channels: Sequence[str] = ISRUC_EEG_CHANNELS,
        epoch_seconds: int = 30,
        annotator: int = 1,
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
        if annotator not in (1, 2):
            raise ValueError(f"annotator must be 1 or 2, got {annotator}.")
        if len(channels) == 0:
            raise ValueError("channels must name at least one channel.")
        self.channels = tuple(channels)
        self.epoch_seconds = int(epoch_seconds)
        self.annotator = annotator

    def read_recording(self, signal_file: str) -> tuple[np.ndarray, float] | None:
        """Load the selected channels of one ISRUC recording in microvolts.

        Args:
            signal_file: Path to the EDF file.

        Returns:
            ``(data, sfreq)`` with ``data`` shaped
            ``(len(channels), n_samples)``, or None if a channel is missing.
        """
        raw = mne.io.read_raw_edf(signal_file, preload=False, verbose="error")
        by_base: dict[str, str] = {}
        for name in raw.ch_names:
            by_base.setdefault(base_channel_name(name), name)
        missing = [c for c in self.channels if c not in by_base]
        if missing:
            logger.warning(
                "Skipping %s, it has no %s channel(s). Channels are %s.",
                signal_file,
                missing,
                raw.ch_names,
            )
            raw.close()
            return None
        picks = [by_base[c] for c in self.channels]
        data = raw.get_data(picks=picks, units="uV")
        sfreq = float(raw.info["sfreq"])
        raw.close()
        return data, sfreq

    def _samples_from_event(self, patient_id: str, event: Any) -> list[dict[str, Any]]:
        """Turn one scored recording into epoch samples.

        Args:
            patient_id: Subject id, for example ``"1"``.
            event: A ``recordings`` event with ``session_id``,
                ``signal_file`` and the scoring file columns.

        Returns:
            One sample dict per kept epoch.
        """
        label_file = str(getattr(event, f"label_file_{self.annotator}", "") or "")
        if label_file.strip() in {"", "nan", "None"}:
            logger.warning(
                "Skipping %s, it has no scoring from annotator %d.",
                event.signal_file,
                self.annotator,
            )
            return []

        loaded = self.read_recording(event.signal_file)
        if loaded is None:
            return []
        signal, sfreq = loaded
        scores = read_isruc_scores(label_file)

        scored_len = round(_SCORED_EPOCH_SECONDS * sfreq)
        epoch_len = round(self.epoch_seconds * sfreq)
        per_score = _SCORED_EPOCH_SECONDS // self.epoch_seconds
        n_scored = min(len(scores), signal.shape[1] // scored_len)
        if n_scored != len(scores):
            logger.warning(
                "%s has %d scored epochs but %d full epochs of signal, "
                "keeping the first %d.",
                event.signal_file,
                len(scores),
                signal.shape[1] // scored_len,
                n_scored,
            )

        session_id = str(getattr(event, "session_id", ""))
        samples: list[dict[str, Any]] = []
        unknown = 0
        for i in range(n_scored):
            label = _SCORE_TO_LABEL.get(int(scores[i]))
            if label is None:
                unknown += 1
                continue
            for j in range(per_score):
                start = i * scored_len + j * epoch_len
                epoch = signal[:, start : start + epoch_len]
                samples.append(
                    {
                        "patient_id": patient_id,
                        "session_id": session_id,
                        "epoch_index": i * per_score + j,
                        "signal": epoch.astype(np.float32),
                        "label": label,
                    }
                )
        if unknown:
            logger.warning(
                "Dropped %d epochs of %s with a score outside 0, 1, 2, 3 and 5.",
                unknown,
                label_file,
            )
        return samples

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build epoch samples for one subject.

        Args:
            patient: A PyHealth patient whose ``recordings`` events are the
                rows of ``isruc-pyhealth.csv``.

        Returns:
            Epoch samples for every scored recording of that subject.
        """
        samples: list[dict[str, Any]] = []
        for event in patient.get_events("recordings"):
            samples.extend(self._samples_from_event(patient.patient_id, event))
        return samples


class SleepStagingSequenceISRUC(SleepStagingISRUC):
    """Sequence-to-sequence sleep staging on ISRUC-Sleep, one EEG channel.

    Runs on :class:`~pyhealth.datasets.ISRUCDataset`. Each sample is
    ``seq_len`` consecutive 30 s epochs of one channel, labeled with the
    stage of every epoch: 0 W, 1 N1, 2 N2, 3 N3 and 4 REM. This is the input
    of sequence models such as :class:`~pyhealth.models.DeepSleepNet` and of
    :class:`~pyhealth.models.SleepKD` (Liang et al., IJCAI 2023), which use
    20-epoch sequences of F3-A2 resampled from 200 Hz to 100 Hz on ISRUC-III.
    It is the ISRUC counterpart of
    :class:`~pyhealth.tasks.SleepStagingSequenceSleepEDF`.

    Epochs are read as in :class:`SleepStagingISRUC`. A window never spans
    an epoch with an unknown score: windows are cut from runs of
    consecutive kept epochs, and the end of a run shorter than a window is
    dropped. ``signal`` is in microvolts.

    Args:
        seq_len: Number of 30 s epochs in a sample. Default is 20.
        stride: Number of epochs between the starts of two windows. Default
            is ``seq_len`` (non-overlapping windows).
        channel: The channel to use, without the reference. Default is
            ``"F3"``.
        sfreq: Rate in Hz to resample the signal to, or None to keep the
            recording's rate (200 Hz for the EEG). Default is 100.
        annotator: Which expert's scoring to use, 1 or 2.

    Raises:
        ValueError: If ``seq_len``, ``stride`` or ``sfreq`` isn't positive,
            or ``annotator`` isn't 1 or 2.

    Attributes:
        task_name (str): The name of the task, "SleepStagingSequenceISRUC".
        input_schema (dict[str, str]): ``{"signal": "tensor"}``, a float array
            of shape ``(seq_len, 1, 30 * sfreq)``.
        output_schema (dict[str, Any]): ``{"label": ("tensor", {"dtype":
            torch.long})}``, the ``(seq_len,)`` stage of each epoch, as in
            :class:`~pyhealth.tasks.SleepStagingSequenceSleepEDF`.

    Examples:
        >>> from pyhealth.datasets import ISRUCDataset
        >>> from pyhealth.tasks import SleepStagingSequenceISRUC
        >>> dataset = ISRUCDataset(root="/path/to/ISRUC-Sleep-III")
        >>> samples = dataset.set_task(SleepStagingSequenceISRUC(seq_len=20))
        >>> samples[0]["signal"].shape
        torch.Size([20, 1, 3000])
    """

    task_name: str = "SleepStagingSequenceISRUC"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, Any]] = {
        "label": ("tensor", {"dtype": torch.long})
    }

    def __init__(
        self,
        seq_len: int = 20,
        stride: int | None = None,
        channel: str = "F3",
        sfreq: int | None = 100,
        annotator: int = 1,
    ) -> None:
        super().__init__(channels=(channel,), annotator=annotator)
        stride = seq_len if stride is None else stride
        if seq_len < 1:
            raise ValueError(f"seq_len must be at least 1, got {seq_len}.")
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}.")
        if sfreq is not None and sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {sfreq}.")
        self.seq_len = seq_len
        self.stride = stride
        self.sfreq = sfreq

    def read_recording(self, signal_file: str) -> tuple[np.ndarray, float] | None:
        """Load the channel of one recording, resampled to ``sfreq``.

        Args:
            signal_file: Path to the EDF file.

        Returns:
            ``(data, sfreq)`` with ``data`` shaped ``(1, n_samples)``, or None
            if the channel is missing.
        """
        loaded = super().read_recording(signal_file)
        if loaded is None or self.sfreq is None or self.sfreq == loaded[1]:
            return loaded
        data, sfreq = loaded
        data = mne.filter.resample(data, up=self.sfreq, down=sfreq, verbose="error")
        return data, float(self.sfreq)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Cut each recording of the subject into ``seq_len``-epoch samples.

        Args:
            patient: A PyHealth patient whose ``recordings`` events are the
                rows of ``isruc-pyhealth.csv``.

        Returns:
            samples: a list of samples, each a dict with patient_id,
                session_id, epoch_index (of the first epoch of the window),
                signal (a float32 array of shape ``(seq_len, 1, 30 * sfreq)``)
                and label (an int64 array of ``seq_len`` stages).
        """
        samples: list[dict[str, Any]] = []
        for event in patient.get_events("recordings"):
            epochs = self._samples_from_event(patient.patient_id, event)
            index = np.array([e["epoch_index"] for e in epochs], dtype=int)
            # Runs of consecutive epochs: a gap means a dropped epoch.
            breaks = np.flatnonzero(np.diff(index) != 1) + 1
            for run in np.split(np.arange(len(epochs)), breaks):
                for i in range(0, len(run) - self.seq_len + 1, self.stride):
                    window = [epochs[j] for j in run[i : i + self.seq_len]]
                    samples.append(
                        {
                            "patient_id": patient.patient_id,
                            "session_id": window[0]["session_id"],
                            "epoch_index": window[0]["epoch_index"],
                            "signal": np.stack([e["signal"] for e in window]),
                            "label": np.array(
                                [e["label"] for e in window], dtype=np.int64
                            ),
                        }
                    )
        return samples
