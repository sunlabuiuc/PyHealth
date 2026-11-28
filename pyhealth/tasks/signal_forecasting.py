"""Biosignal forecasting on PTB-XL (ECG) and Sleep-EDF (EEG).

The setup of the MIRA evaluation (Li et al., NeurIPS 2025, "MIRA: Medical
Time Series Foundation Model for Real-World Health Data",
https://arxiv.org/abs/2506.07584; ``model_eval.py`` in
https://github.com/microsoft/MIRA): each channel is a univariate series,
z-scored per record, and the model forecasts the ``horizon`` samples that
follow a ``context_length`` window. ``model_eval.py`` runs the (context,
horizon) pairs (48, 24), (72, 36), (96, 48) and (128, 64).

The samples have the fields of
:class:`~pyhealth.tasks.PhysioNet2012Forecasting`, so
:class:`~pyhealth.models.MIRA` and :class:`~pyhealth.models.APN` run on them
unchanged.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from .base_task import BaseTask

PTBXL_LEADS = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
)
"""The 12 PTB-XL leads, in the order of the WFDB records."""

SLEEPEDF_CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")
"""The two Sleep-EDF EEG channels (100 Hz) used by default."""

_SLEEP_STAGES = {
    "Sleep stage 1", "Sleep stage 2", "Sleep stage 3", "Sleep stage 4",
    "Sleep stage R",
}


class _SignalForecasting(BaseTask):
    """Shared windowing of :class:`PTBXLForecasting` and :class:`SleepEDFForecasting`."""

    input_schema: ClassVar[dict[str, str]] = {
        "x": "tensor", "x_time": "tensor", "y_time": "tensor"
    }
    output_schema: ClassVar[dict[str, str]] = {"y": "tensor"}

    def __init__(
        self,
        context_length: int,
        horizon: int,
        num_windows: int,
        drop_rate: float,
        seed: int,
    ) -> None:
        super().__init__()
        if context_length < 2 or horizon < 1 or num_windows < 1:
            raise ValueError(
                "context_length must be at least 2, horizon and num_windows at least 1."
            )
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError("drop_rate must be in [0, 1).")
        self.context_length = int(context_length)
        self.horizon = int(horizon)
        self.num_windows = int(num_windows)
        self.drop_rate = float(drop_rate)
        self.seed = int(seed)

    def _windows(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        record_key: str,
        first: int = 0,
        last: int | None = None,
    ) -> list[dict[str, Any]]:
        """Cuts ``signal`` (``[channels, time]``) into forecasting samples.

        ``num_windows`` windows of ``context_length + horizon`` samples start
        evenly spaced in ``[first, last - window]`` (one window starts at
        ``first``). Each channel is z-scored with the mean and standard
        deviation of ``signal[:, first:last]``.
        """
        signal = np.asarray(signal, dtype=np.float64)
        last = signal.shape[1] if last is None else min(last, signal.shape[1])
        window = self.context_length + self.horizon
        if last - first < window:
            return []
        part = signal[:, first:last]
        mean = part.mean(axis=1, keepdims=True)
        std = part.std(axis=1, keepdims=True)
        std = np.where(std > 0, std, 1.0)
        if self.num_windows == 1:
            starts = [first]
        else:
            starts = np.linspace(first, last - window, self.num_windows)
            starts = sorted({int(s) for s in starts})
        rng = np.random.default_rng([self.seed, zlib.crc32(record_key.encode())])
        x_time = np.arange(self.context_length, dtype=np.float32) / sampling_rate
        y_time = (
            np.arange(self.context_length, window, dtype=np.float32) / sampling_rate
        )
        samples = []
        for index, start in enumerate(starts):
            values = ((signal[:, start:start + window] - mean) / std).T
            x = values[: self.context_length].astype(np.float32)
            if self.drop_rate > 0:
                x[rng.random(x.shape) < self.drop_rate] = np.nan
            samples.append({
                "window": index,
                "start": int(start),
                "x": x,
                "x_time": x_time,
                "y_time": y_time,
                "y": values[self.context_length:].astype(np.float32),
            })
        return samples


class PTBXLForecasting(_SignalForecasting):
    """Forecast the next samples of each PTB-XL ECG lead.

    Works with :class:`~pyhealth.datasets.PTBXLDataset` (100 Hz or 500 Hz
    records). Every lead is z-scored with the mean and standard deviation of
    its 10-second record; as in ``model_eval.py``, these statistics include
    the forecast targets. Each window gives one sample: ``x`` holds the
    ``context_length`` samples of the selected leads (``[context_length,
    leads]``), ``y`` the next ``horizon`` samples (``[horizon, leads]``), and
    ``x_time`` / ``y_time`` their times in seconds from the window start.

    By default one window per record starts at the first sample, as
    ``model_eval.py`` in the MIRA repository does. ``drop_rate`` removes
    context values at random (NaN), the way MIRA's evaluation makes regular
    series irregular; the targets are always complete. Samples carry
    ``strat_fold`` for :func:`~pyhealth.datasets.split_by_strat_fold`.

    Args:
        context_length: Samples given as input.
        horizon: Samples to forecast.
        leads: Leads to use, from :data:`PTBXL_LEADS`; all 12 by default.
        num_windows: Windows per record, evenly spaced over the record.
        drop_rate: Share of context values set to NaN.
        seed: Seed of the dropped values (fixed per record).

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> from pyhealth.tasks import PTBXLForecasting
        >>> dataset = PTBXLDataset(root="/data/ptb-xl/1.0.3")  # doctest: +SKIP
        >>> samples = dataset.set_task(
        ...     PTBXLForecasting(context_length=96, horizon=48)
        ... )  # doctest: +SKIP
    """

    task_name: str = "PTBXLForecasting"

    def __init__(
        self,
        context_length: int = 96,
        horizon: int = 48,
        leads: Sequence[str] | None = None,
        num_windows: int = 1,
        drop_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__(context_length, horizon, num_windows, drop_rate, seed)
        self.leads = list(PTBXL_LEADS if leads is None else leads)
        unknown = [lead for lead in self.leads if lead not in PTBXL_LEADS]
        if unknown or not self.leads:
            raise ValueError(f"leads must be a non-empty subset of {PTBXL_LEADS}.")
        self.lead_index = [PTBXL_LEADS.index(lead) for lead in self.leads]

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        from pyhealth.datasets.ptbxl import load_ptbxl_record

        samples = []
        for event in patient.get_events(event_type="records"):
            signal = load_ptbxl_record(event.signal_file)[self.lead_index]
            record_id = str(event.record_id)
            for sample in self._windows(
                signal, float(event.sampling_rate), record_id
            ):
                samples.append({
                    "patient_id": patient.patient_id,
                    "record_id": record_id,
                    "strat_fold": int(float(event.strat_fold)),
                    **sample,
                })
        return samples


class SleepEDFForecasting(_SignalForecasting):
    """Forecast the next samples of Sleep-EDF EEG channels.

    Works with :class:`~pyhealth.datasets.SleepEDFDataset` (cassette or
    telemetry). Windows are taken from the sleep period of each night, from
    the onset of the first to the end of the last scored sleep stage (1-4 or
    R) in the hypnogram, because the recordings start and end with hours of
    wake. ``num_windows`` windows start evenly spaced over that period, and
    each channel is z-scored with its mean and standard deviation over it
    (which, as in ``model_eval.py``, include the forecast targets).
    The fields are those of :class:`PTBXLForecasting`, with ``night``
    instead of ``strat_fold``; times are in seconds from the window start.

    Args:
        context_length: Samples given as input.
        horizon: Samples to forecast.
        channels: Channels to use; :data:`SLEEPEDF_CHANNELS` (both EEG
            channels, 100 Hz) by default. MNE reads every channel at the
            file's highest rate, so a 1 Hz channel such as the airflow is
            upsampled.
        num_windows: Windows per night.
        drop_rate: Share of context values set to NaN.
        seed: Seed of the dropped values (fixed per night).

    Raises:
        ValueError: If a night lacks one of the channels.

    Examples:
        >>> from pyhealth.datasets import SleepEDFDataset
        >>> from pyhealth.tasks import SleepEDFForecasting
        >>> dataset = SleepEDFDataset(root="/data/sleep-edfx/1.0.0")  # doctest: +SKIP
        >>> samples = dataset.set_task(SleepEDFForecasting())  # doctest: +SKIP
    """

    task_name: str = "SleepEDFForecasting"

    def __init__(
        self,
        context_length: int = 96,
        horizon: int = 48,
        channels: Sequence[str] = SLEEPEDF_CHANNELS,
        num_windows: int = 20,
        drop_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__(context_length, horizon, num_windows, drop_rate, seed)
        self.channels = list(channels)
        if not self.channels:
            raise ValueError("channels must not be empty.")

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        import mne

        samples = []
        for event in patient.get_events():
            raw = mne.io.read_raw_edf(event.signal_file, preload=False, verbose="error")
            missing = [c for c in self.channels if c not in raw.ch_names]
            if missing:
                raise ValueError(f"{event.signal_file} has no channel {missing}.")
            sfreq = float(raw.info["sfreq"])
            first, last = 0, raw.n_times
            if event.label_file:
                # Attaching the hypnogram aligns its onsets with the PSG
                # (orig_time) and drops stages outside the recording; onsets
                # are then relative to raw.first_time.
                raw.set_annotations(
                    mne.read_annotations(event.label_file), emit_warning=False
                )
                ann = raw.annotations
                sleep = [
                    (onset, onset + duration)
                    for onset, duration, label in zip(
                        ann.onset, ann.duration, ann.description
                    )
                    if label in _SLEEP_STAGES
                ]
                if not sleep:
                    continue
                first = round((sleep[0][0] - raw.first_time) * sfreq)
                last = round((max(end for _, end in sleep) - raw.first_time) * sfreq)
            signal = raw.get_data(picks=self.channels, start=first, stop=last)
            night = str(event.night)
            for sample in self._windows(
                signal, sfreq, f"{patient.patient_id}-{night}"
            ):
                sample["start"] += first
                samples.append({
                    "patient_id": patient.patient_id,
                    "record_id": f"{patient.patient_id}-{night}",
                    "night": night,
                    **sample,
                })
        return samples
