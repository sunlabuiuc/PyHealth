# Author: Anton Barchukov
# Paper: Chan et al., "MedTsLLM: Leveraging LLMs for Multimodal
#     Medical Time Series Analysis", MLHC 2024
# Description: Beat-level anomaly detection task for the MIT-BIH
#     Arrhythmia Database. Each timestep is labeled 1 if it falls
#     within a beat interval annotated as an abnormal rhythm type,
#     and 0 otherwise (including non-beat gaps). Training is
#     reconstruction-style: MedTsLLM (task="anomaly_detection")
#     learns to reconstruct the ECG and anomalous beats are flagged
#     at eval time by elevated reconstruction error.

from typing import Any, Dict

import numpy as np

from pyhealth.tasks import BaseTask
from pyhealth.tasks.ecg_boundary_detection import (
    _load_signal_and_annotations,
)

# Rhythm annotation symbols considered "normal" — everything else
# is an abnormal beat in the MedTsLLM paper's setup.
_NORMAL_BEATS = {"N", "L", "R", "e", "j"}


class ECGAnomalyDetection(BaseTask):
    """Beat-level arrhythmia anomaly detection on MIT-BIH ECG.

    Produces per-timestep binary labels over a downsampled 2-channel
    ECG. A timestep is labeled ``1`` when it falls inside the
    interval of an abnormal-type beat annotation (all symbols other
    than ``{"N", "L", "R", "e", "j"}`` and the rhythm-change marker
    ``"+"``) and ``0`` otherwise.

    Signal decoding, downsampling, and optional trimming are handled
    at the dataset level via :class:`MITBIHDataset`'s ``preprocess``,
    ``downsample_factor``, and ``trim`` kwargs.

    Args:
        window_size: Number of time points per window. Default 128.
        step_size: Stride between consecutive windows. Default 128.

    Attributes:
        task_name (str): ``"ECGAnomalyDetection"``
        input_schema (Dict[str, str]): ``{"signal": "tensor"}``
        output_schema (Dict[str, str]): ``{"label": "tensor"}``

    Examples:
        >>> from pyhealth.datasets import MITBIHDataset
        >>> from pyhealth.tasks import ECGAnomalyDetection
        >>> dataset = MITBIHDataset(
        ...     root="/path/to/mitdb/",
        ...     preprocess=True,
        ...     paper_split="abnormal_sorted",
        ... )
        >>> sample_ds = dataset.set_task(ECGAnomalyDetection())
    """

    task_name: str = "ECGAnomalyDetection"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "tensor"}

    def __init__(
        self,
        window_size: int = 128,
        step_size: int = 128,
    ):
        self.window_size = window_size
        self.step_size = step_size
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Process a single patient into windowed samples."""
        pid = patient.patient_id
        events = patient.get_events()

        samples = []
        for event in events:
            result = _load_signal_and_annotations(event)
            if result is None:
                continue
            signal, ann_sample, ann_symbol = result

            labels = _build_anomaly_mask(
                ann_sample, ann_symbol, len(signal)
            )

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


def _build_anomaly_mask(
    ann_sample: np.ndarray,
    ann_symbol: np.ndarray,
    signal_len: int,
) -> np.ndarray:
    """Label each timestep 1 if inside an abnormal beat interval.

    Rhythm-change markers (``"+"``) are ignored. For beat ``i`` with
    sample ``s_i``, the interval ``[s_i, s_{i+1})`` is marked 1 when
    ``symbol_i`` is an abnormal beat type. The last beat extends to
    ``signal_len``.
    """
    labels = np.zeros(signal_len, dtype=np.int64)
    # Filter out rhythm-change markers to match the paper's labeling.
    beat_mask = np.array(
        [str(s) != "+" for s in ann_symbol], dtype=bool
    )
    beats_sample = np.asarray(ann_sample, dtype=np.int64)[beat_mask]
    beats_symbol = np.asarray(ann_symbol)[beat_mask]

    for i, symbol in enumerate(beats_symbol):
        if str(symbol) in _NORMAL_BEATS:
            continue
        start = max(0, int(beats_sample[i]))
        if i + 1 < len(beats_sample):
            end = int(beats_sample[i + 1])
        else:
            end = signal_len
        end = min(end, signal_len)
        if start < end:
            labels[start:end] = 1

    return labels


def _build_description(event) -> str:
    """Compose per-patient description from event demographics."""
    parts: list[str] = []
    for attr in ("age", "sex", "medications"):
        value = getattr(event, attr, "")
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        parts.append(f"{attr}: {text}")
    return ", ".join(parts)
