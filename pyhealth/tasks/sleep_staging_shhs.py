"""Sleep staging on SHHS polysomnography.

Each 30 s epoch of the two EEG channels is one sample, labelled with the stage
assigned by the SHHS reading centre: wake, N1, N2, N3, N4 or REM. Movement
epochs (code 9) are dropped because they are not a class.

The paired dataset is :class:`~pyhealth.datasets.SHHSDataset`, which builds the
``shhs1`` and ``shhs2`` tables from the NSRR ``polysomnography`` release.

Authors:
    - PyHealth Contributors

References:
    - Quan, S. F. et al. The Sleep Heart Health Study: design, rationale, and
      methods. Sleep 20, 1077-1085 (1997).
    - NSRR: https://sleepdata.org/datasets/shhs
"""

import logging
from typing import Any, ClassVar

import mne
import numpy as np
import torch

from pyhealth.tasks import BaseTask
from pyhealth.tasks.eegbci import normalize_signal

logger = logging.getLogger(__name__)

#: Sampling rate of the released SHHS EDF files, in Hz.
SHHS_SAMPLE_RATE = 125

#: Epoch lengths the released annotations support. The reading centre scored
#: every 30 s, so only divisors of 30 keep the signal and the labels aligned.
_EPOCH_SEC_OPTIONS = (1, 2, 3, 5, 6, 10, 15, 30, 60, 90, 120, 300)

#: Channels the 1.x loader took positionally when it could not match names.
_LEGACY_CHANNEL_POSITIONS = (2, 7)


def _normalize_channel_name(name: str) -> str:
    """Lowercase a channel name and strip the spacing EDF writers vary on."""
    return name.strip().lower().replace(" ", "")


def _read_sleep_stages(annotation_file: str) -> list[int]:
    """Read the Profusion stage codes, reusing the dataset-level parser.

    Imported lazily because :mod:`pyhealth.datasets` imports
    :mod:`pyhealth.tasks` at module load time.
    """
    from pyhealth.datasets.shhs import read_sleep_stages

    return read_sleep_stages(annotation_file)


def _valid_sleep_stages() -> frozenset[int]:
    """Return the stage codes that count as classes."""
    from pyhealth.datasets.shhs import VALID_SLEEP_STAGES

    return VALID_SLEEP_STAGES


class SleepStagingSHHS(BaseTask):
    """Epoch-level sleep staging for the Sleep Heart Health Study.

    One sample is one scored 30 s epoch: ``signal`` holds the selected EEG
    channels (in uV) and ``label`` is the stage code ``0``-``5``. Epochs
    scored as movement (``9``) are dropped, as are epochs with no matching
    signal window.

    Args:
        epoch_sec: Epoch length in seconds. Defaults to ``30``, the length the
            reading centre scored. Must divide 30 so signal and labels stay
            aligned.
        channels: EEG channel names to keep, for example ``["EEG"]``.
            Defaults to every channel whose name starts with ``EEG``.
        normalization: Passed to
            :func:`~pyhealth.tasks.eegbci.normalize_signal`, for example
            ``"95th_percentile"``. Defaults to ``None`` (no normalization).
        subjects: Patient ids to keep. Defaults to ``None`` (all of them).

    Attributes:
        task_name (str): ``"SleepStagingSHHS"``.
        input_schema (dict): ``{"signal": "tensor"}``.
        output_schema (dict): ``{"label": "multiclass"}``.

    Examples:
        >>> from pyhealth.datasets import SHHSDataset
        >>> from pyhealth.tasks import SleepStagingSHHS
        >>> dataset = SHHSDataset(  # doctest: +SKIP
        ...     root="/path/to/shhs/polysomnography",
        ...     visits=("shhs1",),
        ... )
        >>> sample_dataset = dataset.set_task(SleepStagingSHHS())  # doctest: +SKIP
        >>> sample_dataset[0]["signal"].shape  # doctest: +SKIP
        torch.Size([2, 3750])
    """

    task_name: str = "SleepStagingSHHS"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "multiclass"}

    def __init__(
        self,
        epoch_sec: float = 30,
        channels: list[str] | None = None,
        normalization: str | None = None,
        subjects: list[str] | None = None,
    ) -> None:
        super().__init__()
        if float(epoch_sec) not in _EPOCH_SEC_OPTIONS:
            raise ValueError(
                f"epoch_sec must be one of {_EPOCH_SEC_OPTIONS}, got {epoch_sec!r}; "
                "SHHS annotations are scored every 30 s."
            )
        normalize_signal(np.zeros((1, 2)), normalization)
        self.epoch_sec = float(epoch_sec)
        self.channels = None if channels is None else tuple(channels)
        self.normalization = normalization
        self.subjects = None if subjects is None else tuple(str(s) for s in subjects)

    def read_recording(
        self, file_name: str
    ) -> tuple[np.ndarray, float, list[str]] | None:
        """Load one EDF recording and keep the requested EEG channels.

        Args:
            file_name: Path to a ``shhs1-<pid>.edf`` or ``shhs2-<pid>.edf``.

        Returns:
            ``(data, sfreq, ch_names)`` with ``data`` in uV and shaped
            ``(n_channels, n_samples)``, or ``None`` when the file cannot be
            read or none of the requested channels are present.
        """
        try:
            raw = mne.io.read_raw_edf(file_name, preload=True, verbose="error")
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to read %s: %s", file_name, exc)
            return None

        ch_names = list(raw.ch_names)
        picks = self._channel_indices(ch_names, file_name)
        if picks is None:
            raw.close()
            return None
        data = raw.get_data(picks=picks, units="uV")
        sfreq = float(raw.info["sfreq"])
        raw.close()
        return data, sfreq, [ch_names[i] for i in picks]

    def _channel_indices(self, ch_names: list[str], source: str) -> list[int] | None:
        """Resolve channel names to indices, falling back to 1.x positions."""
        normalized = [_normalize_channel_name(name) for name in ch_names]

        if self.channels is None:
            picks = [i for i, name in enumerate(normalized) if name.startswith("eeg")]
            if not picks:
                picks = [i for i in _LEGACY_CHANNEL_POSITIONS if i < len(ch_names)]
                logger.warning(
                    "No EEG channels named in %s; falling back to positions %s.",
                    source,
                    picks,
                )
            return picks or None

        picks = []
        for wanted in self.channels:
            key = _normalize_channel_name(wanted)
            match = [i for i, name in enumerate(normalized) if name == key]
            if not match:
                match = [i for i, name in enumerate(normalized) if name.startswith(key)]
            if not match:
                logger.warning("Channel %r not found in %s", wanted, source)
                return None
            picks.append(match[0])
        return sorted(set(picks)) or None

    def _samples_from_event(self, patient_id: str, event: Any) -> list[dict[str, Any]]:
        """Turn one scored night into epoch samples.

        Args:
            patient_id: The participant id, for example ``"200001"``.
            event: A ``shhs1`` or ``shhs2`` event carrying ``signal_file`` and
                ``annotation_file``.

        Returns:
            One sample dict per usable epoch, empty when the night is
            unreadable or ``subjects`` excludes this participant.
        """
        signal_file = str(getattr(event, "signal_file", "") or "").strip()
        annotation_file = str(getattr(event, "annotation_file", "") or "").strip()
        if not signal_file or not annotation_file:
            return []

        visit = (
            getattr(event, "visit", None) or getattr(event, "event_type", "") or "shhs"
        )
        record_id = getattr(event, "record_id", "") or f"{visit}-{patient_id}"

        loaded = self.read_recording(signal_file)
        if loaded is None:
            return []
        signal, sfreq, _ch_names = loaded

        try:
            labels = _read_sleep_stages(annotation_file)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to read %s: %s", annotation_file, exc)
            return []

        valid = _valid_sleep_stages()
        epoch_len = round(sfreq * self.epoch_sec)
        if epoch_len <= 0:
            return []

        n_epochs = min(signal.shape[1] // epoch_len, len(labels))
        samples: list[dict[str, Any]] = []
        unscored = 0
        for epoch_index in range(n_epochs):
            label = labels[epoch_index]
            if label not in valid:
                unscored += 1
                continue
            start = epoch_index * epoch_len
            epoch = signal[:, start : start + epoch_len]
            if epoch.shape[1] != epoch_len:
                continue
            if np.isnan(epoch).any():
                continue
            epoch = normalize_signal(epoch, self.normalization)
            samples.append(
                {
                    "patient_id": patient_id,
                    "visit": visit,
                    "record_id": f"{record_id}-{epoch_index}",
                    "epoch_index": int(epoch_index),
                    "signal": torch.FloatTensor(np.ascontiguousarray(epoch)),
                    "label": int(label),
                }
            )

        if unscored:
            logger.debug(
                "Dropped %d of %d epochs without a valid stage in %s",
                unscored,
                n_epochs,
                record_id,
            )
        return samples

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build epoch samples for one participant.

        Args:
            patient: A PyHealth patient whose events are the rows of
                ``shhs1-pyhealth.csv`` and ``shhs2-pyhealth.csv``.

        Returns:
            Epoch samples for that participant. Empty when ``subjects`` was
            set and this participant is not in it.
        """
        patient_id = str(patient.patient_id)
        if self.subjects is not None and patient_id not in self.subjects:
            return []
        samples: list[dict[str, Any]] = []
        for event in patient.get_events():
            samples.extend(self._samples_from_event(patient_id, event))
        return samples
