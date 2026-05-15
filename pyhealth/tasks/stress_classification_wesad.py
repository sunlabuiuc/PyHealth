"""Stress classification task for the WESAD dataset.

This task implements stress detection from wrist-worn physiological
signals collected in the WESAD study. It supports binary (baseline
vs stress) and multi-class (baseline vs stress vs amusement) modes.

Reference:
    Toye, Gomez, Kleinberg, "Simulation of Health Time Series with
    Nonstationarity", CHIL 2024.

    Schmidt et al., "Introducing WESAD, a Multimodal Dataset for
    Wearable Stress and Affect Detection", ICMI 2018.

The signal is segmented into fixed-length non-overlapping windows,
and optional statistical features (mean, std, min, max) are extracted.
"""

import logging
import os
import pickle
from typing import Any, Dict, List

import numpy as np

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

WESAD_LABEL_MAP = {
    0: "not_defined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

WESAD_DEVICE_RATES = {
    "wrist": 4,    # Empatica E4: EDA at 4 Hz
    "chest": 700,  # RespiBAN: all signals at 700 Hz
}


class StressClassificationWESAD(BaseTask):
    """Stress classification from physiological signals on WESAD.

    Segments each subject's continuous signal into non-overlapping
    windows of ``window_size_sec`` seconds, optionally extracts
    statistical features, and assigns labels based on the majority
    condition label within each window.

    Supports both binary classification (baseline=0, stress=1) and
    multi-class classification (baseline=0, stress=1, amusement=2)
    via the ``include_amusement`` parameter.

    Attributes:
        task_name: ``"StressClassification"``
        input_schema: ``{"signal": "tensor"}``
        output_schema: ``{"label": "binary"}`` or
            ``{"label": "multiclass"}``

    Args:
        window_size_sec: Duration of each signal window in seconds.
            Default is 10.0 (matching Toye et al. CHIL 2024).
        use_features: If ``True``, extract statistical features
            (mean, std, min, max) per window. If ``False``, use raw
            signal values. Default is ``True``.
        signal_key: Which wrist signal to use. One of ``"EDA"``,
            ``"BVP"``, ``"ACC"``, ``"TEMP"``. Default is ``"EDA"``.
        device: Which device to use: ``"wrist"`` (Empatica E4) or
            ``"chest"`` (RespiBAN). Default is ``"wrist"``.
        include_amusement: If ``True``, include the amusement
            condition as a third class (label=2). Default is
            ``False`` (binary: baseline vs stress only).

    Examples:
        >>> from pyhealth.datasets import WESADDataset
        >>> from pyhealth.tasks import StressClassificationWESAD
        >>> dataset = WESADDataset(root="/path/to/WESAD/")
        >>> # Binary stress classification (default)
        >>> task = StressClassificationWESAD(window_size_sec=10.0)
        >>> sample_dataset = dataset.set_task(task)
        >>> sample_dataset[0]
        {'patient_id': 'S2', 'signal': tensor([...]), 'label': 0}
        >>> # Multi-class with amusement
        >>> task3 = StressClassificationWESAD(
        ...     include_amusement=True)
        >>> # Use BVP signal instead of EDA
        >>> task_bvp = StressClassificationWESAD(signal_key="BVP")
    """

    task_name: str = "StressClassification"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        window_size_sec: float = 10.0,
        use_features: bool = True,
        signal_key: str = "EDA",
        device: str = "wrist",
        include_amusement: bool = False,
    ) -> None:
        self.window_size_sec = window_size_sec
        self.use_features = use_features
        self.signal_key = signal_key
        self.device = device
        self.include_amusement = include_amusement

        if include_amusement:
            self.output_schema = {"label": "multiclass"}

        if device not in WESAD_DEVICE_RATES:
            raise ValueError(
                f"Unknown device '{device}'. "
                f"Expected one of {list(WESAD_DEVICE_RATES.keys())}."
            )

        super().__init__()

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient's data for stress classification.

        Loads the subject's pickle file, extracts the specified signal
        and condition labels, segments into windows, and returns
        samples.

        WESAD label mapping:
            0 = not defined / transient (skipped)
            1 = baseline  -> label 0
            2 = stress    -> label 1
            3 = amusement -> label 2 (if include_amusement=True)

        Args:
            patient: A PyHealth Patient object. Must have events with
                a ``signal_file`` attribute pointing to the WESAD
                pickle file.

        Returns:
            List of sample dicts, each containing ``patient_id``,
            ``signal`` (feature vector or raw window), and ``label``.
        """
        pid = patient.patient_id
        events = patient.get_events()

        samples: List[Dict[str, Any]] = []
        valid_labels = {1, 2}
        if self.include_amusement:
            valid_labels.add(3)

        label_remap = {1: 0, 2: 1, 3: 2}
        sampling_rate = WESAD_DEVICE_RATES[self.device]

        for event in events:
            signal_file = event.signal_file

            if not os.path.exists(signal_file):
                logger.warning(
                    "Signal file %s not found for patient %s, skipping.",
                    signal_file, pid,
                )
                continue

            with open(signal_file, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            try:
                signal = data["signal"][self.device][
                    self.signal_key
                ].flatten()
            except KeyError:
                available = list(data["signal"][self.device].keys())
                logger.warning(
                    "Signal key '%s' not found for patient %s. "
                    "Available keys: %s",
                    self.signal_key, pid, available,
                )
                continue

            raw_labels = data["label"].flatten()

            # Align labels to signal sampling rate
            if len(raw_labels) != len(signal):
                label_ratio = len(raw_labels) / len(signal)
                indices = np.round(
                    np.arange(len(signal)) * label_ratio
                ).astype(int)
                indices = np.clip(indices, 0, len(raw_labels) - 1)
                aligned_labels = raw_labels[indices]
            else:
                aligned_labels = raw_labels

            window_length = int(self.window_size_sec * sampling_rate)

            for start in range(
                0, len(signal) - window_length + 1, window_length
            ):
                window = signal[start : start + window_length]
                window_labels = aligned_labels[
                    start : start + window_length
                ]

                unique, counts = np.unique(
                    window_labels.astype(int), return_counts=True
                )
                majority_label = int(unique[np.argmax(counts)])

                if majority_label not in valid_labels:
                    continue

                binary_label = label_remap[majority_label]

                if self.use_features:
                    signal_out = np.array([
                        np.mean(window),
                        np.std(window),
                        np.min(window),
                        np.max(window),
                    ], dtype=np.float32)
                else:
                    signal_out = window.astype(np.float32)

                samples.append({
                    "patient_id": pid,
                    "signal": signal_out,
                    "label": binary_label,
                })

        return samples
