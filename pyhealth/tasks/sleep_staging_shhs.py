"""Sleep staging on SHHS polysomnography.

The SHHS reading centre scored every 30 s of the night: wake, N1, N2, N3, N4 or
REM, with code 9 for movement. By default one scored epoch of the two EEG
channels is one sample. A shorter epoch length cuts each score into several
samples, so ``epoch_seconds=10`` yields three 10 s samples per score. Movement
epochs are dropped because they are not a class.

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
import math
from typing import Any, ClassVar
from xml.etree import ElementTree as ET

import mne
import numpy as np
import torch

from pyhealth.tasks import BaseTask
from pyhealth.tasks.eegbci import normalize_signal

logger = logging.getLogger(__name__)

#: Sampling rate of the released SHHS EDF files, in Hz. Epoch boundaries are
#: recomputed from each file's own ``sfreq``, so a recording is never cut in the
#: wrong place; anything other than this rate is unexpected enough to warn
#: about.
SHHS_SAMPLE_RATE = 125

#: Length of one epoch in the released annotations, in seconds. The reading
#: centre scored every 30 s, so ``epoch_seconds`` has to divide this evenly for
#: a score to line up with a whole number of signal windows.
_SCORED_EPOCH_SECONDS = 30

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

    The reading centre scored one stage per 30 s of the night. With the default
    ``epoch_seconds=30`` one sample is one scored epoch: ``signal`` holds the
    selected EEG channels (in uV) and ``label`` is the stage code ``0``-``5``.
    Epochs scored as movement (``9``) are dropped, as are epochs with no
    matching signal window.

    A shorter ``epoch_seconds`` cuts each scored epoch into
    ``30 / epoch_seconds`` samples that share its label, exactly as
    :class:`~pyhealth.tasks.SleepStagingISRUC` does, so ``epoch_seconds=10``
    yields three 10 s samples per score. Lengths that do not divide 30 are
    rejected: a score would otherwise cover a fractional number of windows and
    the labels would drift out of step with the signal.

    Args:
        epoch_seconds: Epoch length in whole seconds. Defaults to ``30``, the
            length the reading centre scored. Must divide 30 so signal and
            labels stay aligned.
        channels: EEG channel names to keep, for example ``["EEG"]``.
            Defaults to every channel whose name starts with ``EEG``. The rows
            of ``signal`` follow the order given here, not the order the EDF
            lists them, so a montage can be named directly.
        normalization: Passed to
            :func:`~pyhealth.tasks.eegbci.normalize_signal`, for example
            ``"95th_percentile"``. Defaults to ``None`` (no normalization).
        subjects: Patient ids to keep. Defaults to ``None`` (all of them).

    Raises:
        ValueError: If ``epoch_seconds`` is not a positive whole number of
            seconds that divides 30, if ``channels`` is empty, or if
            ``normalization`` is not a mode
            :func:`~pyhealth.tasks.eegbci.normalize_signal` knows.

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
        epoch_seconds: int = 30,
        channels: list[str] | None = None,
        normalization: str | None = None,
        subjects: list[str] | None = None,
    ) -> None:
        super().__init__()
        try:
            seconds = float(epoch_seconds)
            whole_seconds = int(seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "epoch_seconds must be a whole number of seconds that divides "
                f"{_SCORED_EPOCH_SECONDS}, got {epoch_seconds!r}."
            ) from exc
        if (
            seconds <= 0
            or seconds != whole_seconds
            or _SCORED_EPOCH_SECONDS % whole_seconds != 0
        ):
            raise ValueError(
                "epoch_seconds must be a whole number of seconds that divides "
                f"{_SCORED_EPOCH_SECONDS}, got {epoch_seconds!r}."
            )
        if channels is not None and not channels:
            raise ValueError("channels must name at least one channel.")
        normalize_signal(np.zeros((1, 2)), normalization)
        self.epoch_seconds = whole_seconds
        self.channels = None if channels is None else tuple(channels)
        # Normalizing the requested names once keeps the per-recording loop to
        # a single pass over the channels the file happens to carry.
        self._channel_lookup = (
            None
            if channels is None
            else tuple((name, _normalize_channel_name(name)) for name in channels)
        )
        if self._channel_lookup is not None:
            requested = [key for _name, key in self._channel_lookup]
            if len(set(requested)) != len(requested):
                logger.warning(
                    "channels names a channel twice (%s); it is kept once, so "
                    "signal has fewer rows than channels.",
                    requested,
                )
        self.normalization = normalization
        # A set, because every participant is looked up once per pass.
        self.subjects = (
            None if subjects is None else frozenset(str(s) for s in subjects)
        )

    def read_recording(
        self, file_name: str
    ) -> tuple[np.ndarray, float, list[str]] | None:
        """Load one EDF recording and keep the requested EEG channels.

        The file is not preloaded: ``get_data`` reads only the requested
        channels, so a fourteen channel night never has to sit in memory in
        full.

        Args:
            file_name: Path to a ``shhs1-<pid>.edf`` or ``shhs2-<pid>.edf``.

        Returns:
            ``(data, sfreq, ch_names)`` with ``data`` in uV and shaped
            ``(n_channels, n_samples)``, or ``None`` when the file cannot be
            read or none of the requested channels are present.
        """
        try:
            raw = mne.io.read_raw_edf(file_name, preload=False, verbose="error")
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to read %s: %s", file_name, exc)
            return None

        ch_names = list(raw.ch_names)
        picks = self._channel_indices(ch_names, file_name)
        if picks is None:
            raw.close()
            return None

        sfreq = float(raw.info["sfreq"])
        try:
            # Without preloading the data, a truncated or corrupt EDF surfaces
            # here rather than inside read_raw_edf.
            data = raw.get_data(picks=picks, units="uV")
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Failed to read %s: %s", file_name, exc)
            return None
        finally:
            raw.close()

        if not math.isclose(sfreq, SHHS_SAMPLE_RATE, rel_tol=1e-6):
            logger.warning(
                "Expected %s Hz for %s, got %s Hz; cutting epochs at the "
                "file's own sampling rate.",
                SHHS_SAMPLE_RATE,
                file_name,
                sfreq,
            )
        return data, sfreq, [ch_names[i] for i in picks]

    def _channel_indices(self, ch_names: list[str], source: str) -> list[int] | None:
        """Resolve channel names to indices, falling back to 1.x positions.

        The result follows the order the channels were requested in, not the
        order the file lists them, so ``signal`` rows line up with
        ``channels``. Without that, a caller building a montage would silently
        train on permuted rows.
        """
        normalized = [_normalize_channel_name(name) for name in ch_names]

        if self._channel_lookup is None:
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
        for wanted, key in self._channel_lookup:
            exact = [i for i, name in enumerate(normalized) if name == key]
            match = exact or [i for i, n in enumerate(normalized) if n.startswith(key)]
            if not match:
                logger.warning("Channel %r not found in %s", wanted, source)
                return None
            if not exact and len(match) > 1:
                logger.warning(
                    "Channel %r matches %s in %s; keeping the first.",
                    wanted,
                    [ch_names[i] for i in match],
                    source,
                )
            picks.append(match[0])
        # dict.fromkeys dedupes while keeping the requested order; a sorted set
        # would silently reorder the rows.
        return list(dict.fromkeys(picks)) or None

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

        # Parse the annotation first: it is a small XML file, so a night that
        # is going to be dropped never pays for reading its EDF.
        try:
            labels = _read_sleep_stages(annotation_file)
        except (OSError, ValueError, ET.ParseError) as exc:
            # ET.ParseError is a SyntaxError, so it is not covered by
            # ValueError: a truncated download or an HTML error page would
            # otherwise abort the whole run instead of dropping one night.
            logger.warning("Failed to read %s: %s", annotation_file, exc)
            return []
        if not labels:
            logger.warning("No scored epochs in %s", annotation_file)
            return []

        loaded = self.read_recording(signal_file)
        if loaded is None:
            return []
        signal, sfreq, _ch_names = loaded

        valid = _valid_sleep_stages()
        scored_len = round(_SCORED_EPOCH_SECONDS * sfreq)
        epoch_len = round(self.epoch_seconds * sfreq)
        if scored_len <= 0 or epoch_len <= 0:
            return []
        # One score labels this many consecutive epochs of epoch_len samples.
        per_score = _SCORED_EPOCH_SECONDS // self.epoch_seconds

        n_scored = min(len(labels), signal.shape[1] // scored_len)
        if n_scored < len(labels):
            logger.warning(
                "%s has %d scored epochs but only %d complete %d s epochs of "
                "signal, keeping the first %d.",
                signal_file,
                len(labels),
                signal.shape[1] // scored_len,
                _SCORED_EPOCH_SECONDS,
                n_scored,
            )

        samples: list[dict[str, Any]] = []
        unscored = 0
        for scored_index in range(n_scored):
            label = labels[scored_index]
            if label not in valid:
                unscored += 1
                continue
            for sub_index in range(per_score):
                epoch_index = scored_index * per_score + sub_index
                start = scored_index * scored_len + sub_index * epoch_len
                # Stop at the end of this score's window. Rounding a
                # fractional sfreq can make the sub-epochs overshoot it, and an
                # epoch reaching into the next window would carry that score's
                # label. The length check below then drops the short remainder
                # rather than returning a window with a borrowed label.
                stop = min(start + epoch_len, (scored_index + 1) * scored_len)
                epoch = signal[:, start:stop]
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
                "Dropped %d of %d scored epochs without a valid stage in %s",
                unscored,
                n_scored,
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
