"""ECG delineation (P / QRS / T segmentation) task for LUDB.

Reproduced paper:
    Park, J., Park, T., Kwon, J. & Jo, Y.. (2025). Benchmarking ECG
    Delineation using Deep Neural Network-based Semantic Segmentation
    Models. Proceedings of the sixth Conference on Health, Inference, and
    Learning, in Proceedings of Machine Learning Research 287:63-88
    Available from https://proceedings.mlr.press/v287/park25a.html.

Authors:
    Mateo Ignacio (Mateo.Ignacio@gatech.edu), Minwoo Jo (mjo38@gatech.edu)

Description:
    Implements ``ECGDelineationLUDB``. Each lead of each LUDB record becomes
    one sample: the Butterworth-filtered, z-scored signal and a three-channel
    per-timestep mask marking the P wave, the QRS complex and the T wave.
    The filter can be switched off or reduced to its high- or low-pass half
    for the preprocessing ablation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import numpy as np

from .base_task import BaseTask

FilterMode = Literal["bandpass", "highpass", "lowpass", "none"]

#: Mask channel order; channel ``i`` is 1 where wave ``WAVES[i]`` is present.
WAVES: tuple[str, ...] = ("P", "QRS", "T")

_FILTER_MODES: tuple[str, ...] = ("bandpass", "highpass", "lowpass", "none")


def filter_ecg(
    signal: np.ndarray,
    mode: FilterMode = "bandpass",
    sampling_rate: float = 500.0,
    low_cut_hz: float = 0.05,
    high_cut_hz: float = 150.0,
    order: int = 2,
) -> np.ndarray:
    """Apply a zero-phase Butterworth filter along the last axis.

    Park et al. use a 0.05 Hz high-pass filter against baseline wander and a
    150 Hz low-pass filter against noise, but do not give the filter order.
    The filter runs forwards and backwards (``sosfiltfilt``) so it does not
    shift waves in time, which would misalign them with the mask.

    Args:
        signal: ECG of shape ``(..., n_samples)``.
        mode: ``"bandpass"`` (both cut-offs), ``"highpass"`` (``low_cut_hz``
            only), ``"lowpass"`` (``high_cut_hz`` only) or ``"none"``.
        sampling_rate: Sampling rate in Hz.
        low_cut_hz: High-pass cut-off in Hz.
        high_cut_hz: Low-pass cut-off in Hz.
        order: Butterworth order of each filter half.

    Returns:
        The filtered signal as float32, with the shape of ``signal``.

    Raises:
        ValueError: If ``mode`` is unknown.

    Examples:
        >>> x = np.zeros((2, 1000), dtype=np.float32)
        >>> filter_ecg(x, mode="bandpass").shape
        (2, 1000)
    """
    if mode not in _FILTER_MODES:
        raise ValueError(f"mode must be one of {_FILTER_MODES}, got {mode!r}")
    if mode == "none":
        return np.asarray(signal, dtype=np.float32)

    from scipy.signal import butter, sosfiltfilt

    if mode == "bandpass":
        sos = butter(
            order, [low_cut_hz, high_cut_hz], "bandpass", fs=sampling_rate, output="sos"
        )
    elif mode == "highpass":
        sos = butter(order, low_cut_hz, "highpass", fs=sampling_rate, output="sos")
    else:
        sos = butter(order, high_cut_hz, "lowpass", fs=sampling_rate, output="sos")
    return sosfiltfilt(sos, signal, axis=-1).astype(np.float32)


def segments_to_mask(
    segments: Sequence[tuple[str, int, int]], n_samples: int
) -> np.ndarray:
    """Turn ``(wave, onset, offset)`` segments into a per-timestep mask.

    Args:
        segments: Output of :func:`pyhealth.datasets.ludb.parse_wave_annotations`;
            ``onset`` and ``offset`` are inclusive sample indices.
        n_samples: Signal length.

    Returns:
        A float32 array of shape ``(3, n_samples)`` with channels in
        :data:`WAVES` order (P, QRS, T); 1 inside a wave, 0 elsewhere.
        Segments partly outside the signal are clipped.

    Examples:
        >>> mask = segments_to_mask([("QRS", 2, 4)], 6)
        >>> mask[1].tolist()
        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    """
    mask = np.zeros((len(WAVES), n_samples), dtype=np.float32)
    for wave, onset, offset in segments:
        mask[WAVES.index(wave), max(onset, 0) : max(offset + 1, 0)] = 1.0
    return mask


class ECGDelineationLUDB(BaseTask):
    """Segment the P wave, QRS complex and T wave in single ECG leads.

    This is the LUDB task of Park et al. (CHIL 2025). Every lead of a
    10-second LUDB record is one sample, as in the paper's 2400 lead
    waveforms. The input is the filtered lead, and the target marks, for
    each of its samples, whether it lies inside a P wave, a QRS complex or
    a T wave. The paper trains on these three channels with a binary
    cross-entropy loss and scores them with per-wave IoU and mIoU.

    The output is a ``"tensor"`` target, so the model computes its own
    segmentation loss. Each sample also carries ``record_id``, ``lead``,
    ``has_p_wave``, ``rhythm`` and ``electric_axis`` for stratified splits
    and per-label evaluation; all leads of a record share one patient, so
    patient-level splits keep a record in a single subset.

    Args:
        filter_mode: Preprocessing filter, see :func:`filter_ecg`. The paper
            uses ``"bandpass"``; the other modes are the ablation.
        low_cut_hz: High-pass cut-off in Hz. Defaults to 0.05 (paper).
        high_cut_hz: Low-pass cut-off in Hz. Defaults to 150 (paper).
        filter_order: Butterworth order; not given in the paper.
        normalize: Z-score each lead (zero mean, unit variance) after
            filtering. The demo inputs released with the paper are per-lead
            z-scores scaled by 0.1.
        leads: LUDB lead names to use, e.g. ``["ii", "v1"]`` or ``"ii"``.
            Defaults to all 12.

    Raises:
        ValueError: If ``filter_mode``, the cut-offs, the order or the leads
            are invalid (unknown, repeated or empty).

    Examples:
        >>> from pyhealth.datasets import LUDBDataset
        >>> from pyhealth.tasks import ECGDelineationLUDB
        >>> dataset = LUDBDataset(root="/data/ludb")  # doctest: +SKIP
        >>> samples = dataset.set_task(ECGDelineationLUDB())  # doctest: +SKIP
        >>> samples[0]["signal"].shape, samples[0]["mask"].shape  # doctest: +SKIP
        (torch.Size([1, 5000]), torch.Size([3, 5000]))
    """

    task_name: str = "ECGDelineationLUDB"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"mask": "tensor"}

    def __init__(
        self,
        filter_mode: FilterMode = "bandpass",
        low_cut_hz: float = 0.05,
        high_cut_hz: float = 150.0,
        filter_order: int = 2,
        normalize: bool = True,
        leads: Sequence[str] | None = None,
    ) -> None:
        from pyhealth.datasets.ludb import LUDB_LEADS, LUDB_SAMPLING_RATE

        if filter_mode not in _FILTER_MODES:
            raise ValueError(
                f"filter_mode must be one of {_FILTER_MODES}, got {filter_mode!r}"
            )
        if not 0 < low_cut_hz < high_cut_hz < LUDB_SAMPLING_RATE / 2:
            raise ValueError(
                "Need 0 < low_cut_hz < high_cut_hz < "
                f"{LUDB_SAMPLING_RATE / 2} Hz (Nyquist)."
            )
        if filter_order < 1:
            raise ValueError(f"filter_order must be >= 1, got {filter_order}")
        if isinstance(leads, str):
            leads = (leads,)
        leads = tuple(leads) if leads is not None else LUDB_LEADS
        unknown = [lead for lead in leads if lead not in LUDB_LEADS]
        if unknown or not leads or len(set(leads)) != len(leads):
            raise ValueError(
                f"leads must be a non-empty subset of {LUDB_LEADS} without repeats"
            )

        self.filter_mode = filter_mode
        self.low_cut_hz = low_cut_hz
        self.high_cut_hz = high_cut_hz
        self.filter_order = filter_order
        self.normalize = normalize
        self.leads = leads
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per lead of the patient's LUDB record.

        Args:
            patient: A patient of :class:`~pyhealth.datasets.LUDBDataset`.

        Returns:
            One dict per lead with ``patient_id``, ``record_id``, ``lead``,
            ``has_p_wave``, ``rhythm``, ``electric_axis``, ``signal``
            (float32, ``(1, n_samples)``) and ``mask`` (float32,
            ``(3, n_samples)``, channels P / QRS / T).
        """
        from pyhealth.datasets.ludb import LUDB_SAMPLING_RATE, load_ludb_record

        samples: list[dict[str, Any]] = []
        for event in patient.get_events(event_type="recordings"):
            signal, annotations = load_ludb_record(event.signal_file, self.leads)
            signal = filter_ecg(
                signal,
                mode=self.filter_mode,
                sampling_rate=LUDB_SAMPLING_RATE,
                low_cut_hz=self.low_cut_hz,
                high_cut_hz=self.high_cut_hz,
                order=self.filter_order,
            )
            if self.normalize:
                mean = signal.mean(axis=-1, keepdims=True)
                std = signal.std(axis=-1, keepdims=True)
                signal = ((signal - mean) / (std + 1e-8)).astype(np.float32)
            for index, lead in enumerate(self.leads):
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": str(event.record_id),
                        "lead": lead,
                        "has_p_wave": int(event.has_p_wave),
                        "rhythm": str(event.rhythm),
                        "electric_axis": str(event.electric_axis),
                        "signal": signal[index][None, :],
                        "mask": segments_to_mask(annotations[lead], signal.shape[-1]),
                    }
                )
        return samples
