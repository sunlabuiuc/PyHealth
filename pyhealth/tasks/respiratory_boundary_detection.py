# Author: Anton Barchukov
# Paper: Chan et al., "MedTsLLM: Leveraging LLMs for Multimodal
#     Medical Time Series Analysis", MLHC 2024
# Description: Breath boundary detection task for the BIDMC dataset.
#     Detects breath boundaries in respiratory impedance signals.

from typing import Any, Dict

import numpy as np

from pyhealth.tasks import BaseTask

# RESP, PLETH, and ECG lead II — the 3 channels used in the paper.
_TARGET_CHANNELS = ["RESP,", "PLETH,", "II,"]


class RespiratoryBoundaryDetection(BaseTask):
    """Breath boundary detection on BIDMC respiratory signals.

    Binary classification of each time point as a breath boundary
    or not. The model is trained on 3 channels (RESP, PLETH, II)
    and predicts boundary locations.

    Args:
        window_size: Number of time points per window. Default 256.
        step_size: Stride between consecutive windows. Default 128.
        annotator: Which annotator's labels to use (1 or 2).
            Default 1.

    Attributes:
        task_name (str): ``"RespiratoryBoundaryDetection"``
        input_schema (Dict[str, str]): ``{"signal": "tensor"}``
        output_schema (Dict[str, str]): ``{"label": "tensor"}``

    Examples:
        >>> from pyhealth.datasets import BIDMCDataset
        >>> dataset = BIDMCDataset(root="/path/to/bidmc/")
        >>> sample_dataset = dataset.set_task(
        ...     RespiratoryBoundaryDetection()
        ... )
    """

    task_name: str = "RespiratoryBoundaryDetection"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "tensor"}

    def __init__(
        self,
        window_size: int = 256,
        step_size: int = 128,
        annotator: int = 1,
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.annotator = annotator
        super().__init__()

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Process a single patient into windowed samples.

        Args:
            patient: A Patient object from BIDMCDataset.

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
            signal, ann_sample, ann_aux = result
            if signal.shape[1] != 3:
                continue

            labels = np.zeros(len(signal), dtype=np.int64)
            ann_tag = f"ann{self.annotator}"
            for s, aux in zip(ann_sample, ann_aux):
                if str(aux) == ann_tag and 0 <= int(s) < len(signal):
                    labels[int(s)] = 1

            # Build patient description
            desc_parts = []
            if event.age:
                desc_parts.append(f"age: {event.age}")
            if event.sex:
                desc_parts.append(f"sex: {event.sex}")
            if event.location:
                desc_parts.append(f"location: {event.location}")
            description = ", ".join(desc_parts)
            split = getattr(event, "split", "") or ""

            # Window into fixed-length chunks
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (signal, ann_sample, ann_aux) from ``.npz`` cache or wfdb."""
    import os

    processed_file = getattr(event, "processed_file", "") or ""
    if processed_file and os.path.exists(processed_file):
        with np.load(processed_file, allow_pickle=False) as npz:
            return (
                np.asarray(npz["signal"], dtype=np.float32),
                np.asarray(npz["ann_sample"], dtype=np.int64),
                np.asarray(npz["ann_aux"]),
            )

    import wfdb

    try:
        record = wfdb.rdrecord(event.signal_file)
        ann = wfdb.rdann(event.signal_file, extension=event.annotation_file)
    except FileNotFoundError:
        return None

    col_idx = [
        record.sig_name.index(ch)
        for ch in _TARGET_CHANNELS
        if ch in record.sig_name
    ]
    signal = record.p_signal[:, col_idx].astype(np.float32)
    return (
        signal,
        np.asarray(ann.sample, dtype=np.int64),
        np.asarray(ann.aux_note),
    )
