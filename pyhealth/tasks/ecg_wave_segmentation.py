# Author: Anton Barchukov
# Paper: Chan et al., "MedTsLLM: Leveraging LLMs for Multimodal
#     Medical Time Series Analysis", MLHC 2024
# Paper link: https://arxiv.org/abs/2408.07773
# Description: Per-timestep ECG wave segmentation task for the LUDB
#     dataset. Classifies each sample as background (0), P wave (1),
#     QRS complex (2), or T wave (3).

from typing import Any, Dict

import numpy as np

from pyhealth.tasks import BaseTask


# wfdb annotation symbol -> class index
_WAVE_LABELS = {"p": 1, "N": 2, "t": 3}


class ECGWaveSegmentation(BaseTask):
    """Per-timestep ECG wave segmentation on LUDB.

    Classifies each time point in a 12-lead ECG as one of:
    background (0), P wave (1), QRS complex (2), or T wave (3).

    The raw signal is windowed into fixed-length chunks. Each chunk
    produces a sample with a signal tensor and a per-timestep label
    array of the same length.

    Trim/decode are controlled at the dataset level via
    :class:`LUDBDataset`'s ``preprocess`` and ``trim`` kwargs; this
    task only handles windowing and emission.

    Args:
        window_size: Number of time points per window. Default 512.
        step_size: Stride between consecutive windows. Default 256.

    Attributes:
        task_name (str): ``"ECGWaveSegmentation"``
        input_schema (Dict[str, str]): ``{"signal": "tensor"}``
        output_schema (Dict[str, str]): ``{"label": "tensor"}``

    Examples:
        >>> from pyhealth.datasets import LUDBDataset
        >>> dataset = LUDBDataset(root="/path/to/ludb/", preprocess=True)
        >>> sample_dataset = dataset.set_task(ECGWaveSegmentation())
        >>> sample_dataset.samples[0].keys()
        dict_keys(['patient_id', 'lead', 'signal', 'label', ...])
    """

    task_name: str = "ECGWaveSegmentation"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "tensor"}

    def __init__(
        self,
        window_size: int = 512,
        step_size: int = 256,
    ):
        self.window_size = window_size
        self.step_size = step_size
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Process a single patient into windowed samples.

        Args:
            patient: A Patient object from LUDBDataset. Each event
                represents one ECG lead with ``signal_file`` (wfdb
                record path), ``label_file`` (annotation extension),
                and optionally ``processed_file`` (cached ``.npz``).

        Returns:
            List of sample dicts, each containing:
                - ``patient_id``: str
                - ``lead``: str, ECG lead name
                - ``signal``: np.ndarray, shape ``(window_size,)``
                - ``label``: np.ndarray of int, shape ``(window_size,)``
        """
        pid = patient.patient_id
        events = patient.get_events()

        samples = []
        for event in events:
            result = _load_signal_and_labels(event)
            if result is None:
                continue
            signal, labels = result

            description = _build_description(event)
            split = getattr(event, "split", "") or ""
            lead = event.lead

            # Window into fixed-length chunks
            for start in range(
                0, len(signal) - self.window_size + 1, self.step_size
            ):
                end = start + self.window_size
                samples.append({
                    "patient_id": pid,
                    "lead": lead,
                    "signal": signal[start:end],
                    "label": labels[start:end],
                    "description": description,
                    "split": split,
                })

        return samples


def _load_signal_and_labels(event) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (signal, labels) for one event, preferring the ``.npz`` cache."""
    processed_file = getattr(event, "processed_file", "") or ""
    if processed_file:
        import os

        if os.path.exists(processed_file):
            with np.load(processed_file, allow_pickle=False) as npz:
                return (
                    np.asarray(npz["signal"], dtype=np.float32),
                    np.asarray(npz["labels"], dtype=np.int64),
                )

    # Fallback: decode wfdb on demand.
    import wfdb

    try:
        record = wfdb.rdrecord(event.signal_file)
    except FileNotFoundError:
        return None
    try:
        lead_idx = record.sig_name.index(event.lead)
    except ValueError:
        return None
    signal = record.p_signal[:, lead_idx].astype(np.float32)

    try:
        ann = wfdb.rdann(event.signal_file, extension=event.label_file)
    except FileNotFoundError:
        return None

    labels = np.zeros(len(signal), dtype=np.int64)
    i = 0
    while i < len(ann.symbol):
        sym = ann.symbol[i]
        if sym == "(" and i + 2 < len(ann.symbol):
            wave_type = ann.symbol[i + 1]
            onset = ann.sample[i]
            offset = (
                ann.sample[i + 2]
                if ann.symbol[i + 2] == ")"
                else ann.sample[i + 1]
            )
            if wave_type in _WAVE_LABELS:
                labels[onset : offset + 1] = _WAVE_LABELS[wave_type]
            i += 3
        else:
            i += 1

    return signal, labels


def _build_description(event) -> str:
    """Compose a per-patient description string from event attributes.

    Returns a comma-separated ``"age: X, sex: Y, diagnoses: Z"``
    string built from the demographics attached to each LUDB event.
    NaN values (from pandas reading empty-string cells) and missing
    attributes are skipped.
    """
    parts: list[str] = []
    for attr in ("age", "sex", "diagnoses"):
        value = getattr(event, attr, "")
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        parts.append(f"{attr}: {text}")
    return ", ".join(parts)
