# Author: Anton Barchukov
# Paper: Chan et al., "MedTsLLM: Leveraging LLMs for Multimodal
#     Medical Time Series Analysis", MLHC 2024
# Description: R-peak boundary detection task for the MIT-BIH dataset.
#     Detects beat boundaries (R-peaks) in ECG signals.

from typing import Any, Dict

import numpy as np

from pyhealth.tasks import BaseTask

# Normal beat types (all others are anomalies)
_NORMAL_BEATS = {"N", "L", "R", "e", "j"}


class ECGBoundaryDetection(BaseTask):
    """R-peak boundary detection on MIT-BIH ECG signals.

    Binary classification of each time point as a beat boundary
    (R-peak) or not. Signal decoding, downsampling, and optional
    trimming are handled at the dataset level via
    :class:`MITBIHDataset`'s ``preprocess``, ``downsample_factor``,
    and ``trim`` kwargs.

    Args:
        window_size: Number of time points per window. Default 256.
        step_size: Stride between consecutive windows. Default 256.

    Attributes:
        task_name (str): ``"ECGBoundaryDetection"``
        input_schema (Dict[str, str]): ``{"signal": "tensor"}``
        output_schema (Dict[str, str]): ``{"label": "tensor"}``

    Examples:
        >>> from pyhealth.datasets import MITBIHDataset
        >>> dataset = MITBIHDataset(
        ...     root="/path/to/mitdb/", preprocess=True
        ... )
        >>> sample_dataset = dataset.set_task(ECGBoundaryDetection())
    """

    task_name: str = "ECGBoundaryDetection"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "tensor"}

    def __init__(
        self,
        window_size: int = 256,
        step_size: int = 256,
    ):
        self.window_size = window_size
        self.step_size = step_size
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Process a single patient into windowed samples.

        Args:
            patient: A Patient object from MITBIHDataset.

        Returns:
            List of sample dicts with signal and binary label arrays.
        """
        pid = patient.patient_id
        events = patient.get_events()

        samples = []
        for event in events:
            result = _load_signal_and_annotations(event)
            if result is None:
                continue
            signal, ann_sample, _ = result

            # Build binary R-peak mask.
            labels = np.zeros(len(signal), dtype=np.int64)
            for s in ann_sample:
                idx = int(s)
                if 0 <= idx < len(signal):
                    labels[idx] = 1

            description = _build_description(event)
            split = getattr(event, "split", "") or ""

            for start in range(
                0, len(signal) - self.window_size + 1, self.step_size
            ):
                end = start + self.window_size
                samples.append({
                    "patient_id": pid,
                    "signal": signal[start:end],
                    "label": labels[start:end].astype(np.float32),
                    "description": description,
                    "split": split,
                })

        return samples


def _load_signal_and_annotations(
    event,
    default_downsample: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (signal, ann_sample, ann_symbol) from cache or wfdb.

    Cache path is preferred. Wfdb fallback downsamples by
    ``default_downsample`` but does not trim — trim only applies to
    cached arrays built via ``MITBIHDataset(preprocess=True, trim=...)``.
    """
    import os

    processed_file = getattr(event, "processed_file", "") or ""
    if processed_file and os.path.exists(processed_file):
        with np.load(processed_file, allow_pickle=False) as npz:
            return (
                np.asarray(npz["signal"], dtype=np.float32),
                np.asarray(npz["ann_sample"], dtype=np.int64),
                np.asarray(npz["ann_symbol"]),
            )

    import wfdb

    try:
        record = wfdb.rdrecord(event.signal_file)
        ann = wfdb.rdann(event.signal_file, extension=event.annotation_file)
    except FileNotFoundError:
        return None

    signal = record.p_signal.astype(np.float32)
    if default_downsample > 1:
        signal = signal[::default_downsample]
    ds_samples = np.asarray(ann.sample) // default_downsample
    ann_symbols = np.asarray(ann.symbol)
    in_bounds = (ds_samples >= 0) & (ds_samples < len(signal))
    return (
        signal,
        ds_samples[in_bounds].astype(np.int64),
        ann_symbols[in_bounds],
    )


def _build_description(event) -> str:
    """Compose per-patient description from event demographics."""
    parts: list[str] = []
    if getattr(event, "age", ""):
        parts.append(f"age: {event.age}")
    if getattr(event, "sex", ""):
        parts.append(f"sex: {event.sex}")
    if getattr(event, "medications", ""):
        parts.append(f"medications: {event.medications}")
    return ", ".join(parts)
