# Authors:     Paul Garcia (alanpg2), Rogelio Medina (orm9), Cesar Nava (can14)
# Paper:       Data Augmentation for Electrocardiograms (Raghu et al., CHIL 2022)
# Link:        https://proceedings.mlr.press/v174/raghu22a.html
# Description: Binary ECG classification task for PTB-XL supporting MI, HYP,
#              STTC, and CD diagnostic superclass labels.

"""Binary ECG classification tasks for PTB-XL.

Reference:
    Raghu et al. (2022). Data Augmentation for Electrocardiograms.
    Conference on Health, Inference, and Learning (CHIL), PMLR 174.
    https://proceedings.mlr.press/v174/raghu22a.html
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from pyhealth.data import Patient
from pyhealth.tasks import BaseTask

#: Supported diagnostic superclass labels.
SUPERCLASSES: Tuple[str, ...] = ("MI", "HYP", "STTC", "CD")
_LABEL_COL: Dict[str, str] = {s: f"{s.lower()}_label" for s in SUPERCLASSES}


class ECGBinaryClassification(BaseTask):
    """Binary ECG classification task for the PTB-XL dataset.

    Each ECG record produces one labelled sample.  The WFDB waveform is
    loaded from the path stored in the event, optionally resampled from
    *input_hz* to *output_hz* using anti-aliased Fourier resampling,
    per-lead z-score normalised, and padded or truncated to a fixed length
    along the time axis.

    The default configuration (500 Hz input, 250 Hz output, 2500-sample
    window) reproduces the setup of Raghu et al. (2022), who load the 500 Hz
    PTB-XL records and downsample to 250 Hz for a 10-second window.  Pair
    this task with ``PTBXLDataset(sampling_rate=500)`` (the dataset default).

    Args:
        task_label: Diagnostic superclass used as the binary target.
            One of ``"MI"``, ``"HYP"``, ``"STTC"``, or ``"CD"``.
            Default: ``"MI"``.
        target_length: Number of time steps per sample after resampling,
            padding, or truncation.  At 250 Hz the default of 2500 equals
            10 seconds (matching Raghu et al. 2022).
        input_hz: Sampling rate of the raw waveforms loaded from disk.
            Should match ``PTBXLDataset.sampling_rate``.  Default: ``500``.
        output_hz: Target sampling rate after resampling.  Set equal to
            *input_hz* to skip resampling.  Default: ``250``.

    Examples:
        >>> task = ECGBinaryClassification(task_label="MI")
        >>> samples = task(patient)
        >>> print(samples[0].keys())
        dict_keys(['patient_id', 'ecg_id', 'ecg', 'label'])
    """

    task_name: str = "ECGBinaryClassification"
    input_schema: Dict[str, str] = {"ecg": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        task_label: str = "MI",
        target_length: int = 2500,
        input_hz: int = 500,
        output_hz: int = 250,
    ) -> None:
        super().__init__()
        if task_label not in SUPERCLASSES:
            raise ValueError(
                f"task_label must be one of {SUPERCLASSES}, got {task_label!r}"
            )
        self.task_label = task_label
        self._label_col = _LABEL_COL[task_label]
        self.target_length = target_length
        self.input_hz = input_hz
        self.output_hz = output_hz

    def __call__(self, patient: Patient) -> List[Dict]:
        """Process one patient into a list of ECG classification samples.

        Args:
            patient: PyHealth ``Patient`` whose ``ecg_records`` events contain
                the attributes defined in ``configs/ptbxl.yaml``.

        Returns:
            List of sample dicts.  Each dict contains:

            * ``patient_id`` – string patient identifier
            * ``ecg_id`` – string ECG record identifier
            * ``ecg`` – float32 ndarray of shape ``(12, target_length)``
            * ``label`` – int (0 or 1)
        """
        samples: List[Dict] = []
        events = patient.get_events(event_type="ecg_records")

        for event in events:
            filename: str = event["filename"]
            label: int = int(event[self._label_col])

            signal = self._load_signal(filename)
            if signal is None:
                continue

            if self.input_hz != self.output_hz:
                signal = self._resample(signal, self.input_hz, self.output_hz)
            signal = self._normalize(signal)
            signal = self._pad_or_truncate(signal, self.target_length)

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "ecg_id": str(event["ecg_id"]),
                    "ecg": signal,
                    "label": label,
                }
            )

        return samples

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_signal(filename: str) -> Optional[np.ndarray]:
        """Load a WFDB record from disk and return a float32 signal array.

        Fails gracefully so that individual corrupt records are skipped
        rather than crashing the entire pipeline.

        Args:
            filename: Absolute path to the WFDB record *without* file
                extension (e.g. ``/data/ptb-xl/records100/00000/00001_lr``).

        Returns:
            Float32 ndarray of shape ``(leads, T)``, or ``None`` if the
            record cannot be read.
        """
        try:
            import wfdb  # lazy import — optional dependency

            record = wfdb.rdrecord(filename)
            return record.p_signal.T.astype(np.float32)  # (leads, T)
        except Exception:
            return None

    @staticmethod
    def _resample(signal: np.ndarray, input_hz: int, output_hz: int) -> np.ndarray:
        """Resample *signal* from *input_hz* to *output_hz* using Fourier resampling.

        Uses ``scipy.signal.resample`` which applies an anti-aliased FFT-based
        method, matching the approach used in Raghu et al. (2022) for 500→250 Hz
        downsampling.

        Args:
            signal: Float32 array of shape ``(leads, T)``.
            input_hz: Original sampling rate in Hz.
            output_hz: Target sampling rate in Hz.

        Returns:
            Resampled float32 array of shape ``(leads, T')``, where
            ``T' = round(T * output_hz / input_hz)``.
        """
        from scipy.signal import resample as scipy_resample  # lazy import

        T = signal.shape[1]
        target_samples = round(T * output_hz / input_hz)
        return scipy_resample(signal, target_samples, axis=1).astype(signal.dtype)

    @staticmethod
    def _normalize(signal: np.ndarray) -> np.ndarray:
        """Apply per-lead z-score normalisation (zero mean, unit std).

        Args:
            signal: Float32 array of shape ``(leads, T)``.

        Returns:
            Normalised array of the same shape.
        """
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True) + 1e-8
        return (signal - mean) / std

    @staticmethod
    def _pad_or_truncate(signal: np.ndarray, target: int) -> np.ndarray:
        """Truncate or right-zero-pad a signal to exactly *target* steps.

        Args:
            signal: Float32 array of shape ``(leads, T)``.
            target: Desired number of time steps.

        Returns:
            Float32 array of shape ``(leads, target)``.
        """
        T = signal.shape[1]
        if T >= target:
            return signal[:, :target]
        pad = np.zeros((signal.shape[0], target - T), dtype=signal.dtype)
        return np.concatenate([signal, pad], axis=1)
