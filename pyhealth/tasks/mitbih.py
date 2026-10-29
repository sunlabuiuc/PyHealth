"""MIT-BIH ECG anomaly detection windows with per-step abnormal-beat labels.

Contributors: Thi Van Trinh Tran and Zexi Li.
Reference: MedTsLLM, https://arxiv.org/abs/2408.07773.
Data: https://physionet.org/content/mitdb/1.0.0/.

Two-lead windows resampled from 360 Hz to 125 Hz; every abnormal beat is
labeled 150 ms before and after it, following MedTsLLM.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, ClassVar, Literal

import numpy as np

from pyhealth.data import Patient
from pyhealth.datasets.mitbih import (
    MITBIH_BEAT_SYMBOLS,
    MITBIH_EVENT_TYPE,
    MITBIH_NORMAL_SYMBOL,
    _require_wfdb,
)

from .base_task import BaseTask


def anomaly_mask(
    beat_samples: np.ndarray,
    symbols: list[str],
    n_samples: int,
    fs: float,
    half_window_ms: float = 150.0,
) -> np.ndarray:
    """Builds a per-sample anomaly mask from beat annotations.

    Every abnormal beat (a beat annotation other than ``"N"``) marks the
    samples within ``half_window_ms`` before and after it. Non-beat
    annotations are ignored.

    Args:
        beat_samples: Sample index of each annotation.
        symbols: WFDB symbol of each annotation.
        n_samples: Length of the signal.
        fs: Sampling frequency of ``beat_samples`` in Hz.
        half_window_ms: Window on each side of an abnormal beat, in ms.

    Returns:
        np.ndarray: ``uint8`` array of shape ``(n_samples,)``, 1 = anomalous.

    Examples:
        >>> anomaly_mask(np.array([2, 7]), ["N", "V"], 10, fs=1000,
        ...              half_window_ms=1).tolist()
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    """
    mask = np.zeros(n_samples, dtype=np.uint8)
    half = round(fs * half_window_ms / 1000.0)
    for idx, sym in zip(beat_samples, symbols):
        if sym in MITBIH_BEAT_SYMBOLS and sym != MITBIH_NORMAL_SYMBOL:
            mask[max(0, int(idx) - half) : min(n_samples, int(idx) + half + 1)] = 1
    return mask


class MITBIHAnomalyDetection(BaseTask):
    """Windowed ECG anomaly detection on MIT-BIH (MedTsLLM setup).

    For each recording, the two-lead signal (in mV) is resampled from 360 Hz
    to ``target_fs`` with a polyphase filter and cut into windows of
    ``window_len`` samples every ``step_size`` samples. By default the label
    is a per-time-step binary anomaly mask aligned with the window, as in
    MedTsLLM. With ``label_level="window"``, the label is instead one binary
    value per window (1 if any time step is anomalous), so that standard
    PyHealth classifiers can be trained on the task.

    Each sample contains:

    - ``signal``: ``float32`` array of shape ``(window_len, 2)``
    - ``label``: ``int64`` array of shape ``(window_len,)`` with
      ``label_level="step"``, or an ``int`` with ``label_level="window"``;
      1 = anomalous
    - ``patient_id`` / ``record_id``: record name, e.g. ``"208"``
    - ``window_start``: index of the first sample in the resampled signal
    - ``description``: header comments (age, sex, medications, notes), usable
      as a text prompt by multimodal models such as MedTsLLM

    The per-step labels are intended for evaluation: reconstruction-based
    detectors such as GPT4TS or MedTsLLM train on the signal only.

    Args:
        target_fs: Sampling frequency after resampling, in Hz. Defaults to
            125.0 (MedTsLLM).
        window_len: Samples per window. Defaults to 128 (MedTsLLM).
        step_size: Stride between windows. Defaults to ``window_len``
            (non-overlapping windows).
        half_window_ms: Anomaly window on each side of an abnormal beat, in
            ms. Defaults to 150.0 (MedTsLLM).
        label_level: ``"step"`` for a per-time-step label (``"tensor"``
            output schema) or ``"window"`` for one label per window
            (``"binary"`` output schema). Defaults to ``"step"``.

    Raises:
        ValueError: If a length is not positive, ``half_window_ms`` is
            negative, or ``label_level`` is unknown.

    Examples:
        >>> from pyhealth.datasets import MITBIHDataset  # doctest: +SKIP
        >>> from pyhealth.tasks import MITBIHAnomalyDetection  # doctest: +SKIP
        >>> ds = MITBIHDataset(root="/data/mitdb/1.0.0")  # doctest: +SKIP
        >>> train, test = ds.set_task(  # doctest: +SKIP
        ...     MITBIHAnomalyDetection(), split=ds.anomaly_ratio_split()
        ... )
        >>> test[0]["label"].shape  # doctest: +SKIP
        torch.Size([128])
    """

    task_name: str = "MITBIHAnomalyDetection"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    # Set per instance from label_level ("tensor" or "binary").
    output_schema: dict[str, str]

    def __init__(
        self,
        target_fs: float = 125.0,
        window_len: int = 128,
        step_size: int | None = None,
        half_window_ms: float = 150.0,
        label_level: Literal["step", "window"] = "step",
    ) -> None:
        if target_fs <= 0:
            raise ValueError("target_fs must be positive.")
        if window_len <= 0 or (step_size is not None and step_size <= 0):
            raise ValueError("window_len and step_size must be positive.")
        if half_window_ms < 0:
            raise ValueError("half_window_ms must be non-negative.")
        if label_level not in ("step", "window"):
            raise ValueError(
                f"Unknown label_level {label_level!r}; use 'step' or 'window'."
            )
        super().__init__()
        self.target_fs = float(target_fs)
        self.window_len = int(window_len)
        self.step_size = int(step_size) if step_size is not None else self.window_len
        self.half_window_ms = float(half_window_ms)
        self.label_level = label_level
        self.output_schema = {"label": "tensor" if label_level == "step" else "binary"}

    def _resample(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """Resamples ``signal`` along time from ``fs`` to ``target_fs``.

        Args:
            signal: Array of shape ``(n_samples, n_leads)``.
            fs: Original sampling frequency in Hz.

        Returns:
            np.ndarray: ``float32`` array of shape ``(n_resampled, n_leads)``.

        Examples:
            >>> MITBIHAnomalyDetection()._resample(np.zeros((360, 2)), 360).shape
            (125, 2)
        """
        if fs == self.target_fs:
            return signal.astype(np.float32)
        from scipy.signal import resample_poly

        ratio = Fraction(self.target_fs / fs).limit_denominator(10_000)
        out = resample_poly(signal, ratio.numerator, ratio.denominator, axis=0)
        return out.astype(np.float32)

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Generates windowed samples for one recording.

        Args:
            patient: Patient with one ``records`` event.

        Returns:
            list[dict[str, Any]]: One sample per window; empty if the patient
            has no record event or the recording is shorter than a window.

        Examples:
            >>> samples = MITBIHAnomalyDetection()(patient)  # doctest: +SKIP
        """
        events = patient.get_events(event_type=MITBIH_EVENT_TYPE)
        if not events:
            return []
        event = events[0]
        wfdb = _require_wfdb()

        record = wfdb.rdrecord(event["signal_file"])
        ann = wfdb.rdann(event["signal_file"], "atr")
        fs = float(record.fs)
        raw = record.p_signal
        signal = self._resample(raw, fs)

        # Label at the original rate, then map each resampled time step to the
        # nearest original sample, so label and signal lengths always agree.
        mask = anomaly_mask(
            ann.sample, list(ann.symbol), len(raw), fs, self.half_window_ms
        )
        nearest = np.round(np.linspace(0, len(raw) - 1, len(signal))).astype(int)
        label = mask[nearest].astype(np.int64)

        record_id = str(event["record_id"])
        # Empty CSV fields are read back as missing values.
        comments = event["comments"]
        description = comments if isinstance(comments, str) else ""
        samples = []
        for start in range(0, len(signal) - self.window_len + 1, self.step_size):
            end = start + self.window_len
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": record_id,
                    "window_start": start,
                    "signal": signal[start:end],
                    "label": (
                        label[start:end]
                        if self.label_level == "step"
                        else int(label[start:end].any())
                    ),
                    "description": description,
                }
            )
        return samples
