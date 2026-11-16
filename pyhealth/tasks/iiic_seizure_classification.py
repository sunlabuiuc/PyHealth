# Author: Jathurshan Pradeepkumar
# NetID: jp65
# Paper Title: Tokenizing Single-Channel EEG with Time-Frequency Motif Learning
# Paper Link: https://arxiv.org/pdf/2502.16060
# Description: IIIC Seizure 6-class EEG classification task for PyHealth

import logging
import pickle
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from scipy.signal import resample

from pyhealth.tasks import BaseTask
from pyhealth.tasks.eegbci import normalize_signal

logger = logging.getLogger(__name__)

IIIC_CLASSES: dict[int, str] = {
    0: "Other",
    1: "SZ",
    2: "LPD",
    3: "GPD",
    4: "LRDA",
    5: "GRDA",
}


class EEGSeizureIIIC(BaseTask):
    """Multi-class EEG classification task for the IIIC Seizure (SPaRCNet) dataset.

    Works with :class:`~pyhealth.datasets.IIICDataset`.

    Classifies 10-second, 16-channel longitudinal bipolar (TCP) montage EEG
    segments into one of six Ictal-Interictal-Injury Continuum (IIIC) classes:
      - ``0``: Other
      - ``1``: Seizure (SZ)
      - ``2``: Lateralized Periodic Discharges (LPD)
      - ``3``: Generalized Periodic Discharges (GPD)
      - ``4``: Lateralized Rhythmic Delta Activity (LRDA)
      - ``5``: Generalized Rhythmic Delta Activity (GRDA)

    Each recording in the IIIC dataset is natively sampled at 200 Hz for 10
    seconds (shape ``(16, 2000)``). Following the TFM-Tokenizer and SPaRCNet
    processing pipelines, this task optionally resamples the signal along the
    time dimension, normalizes per channel (e.g., 95th-percentile scaling), and
    computes the Short-Time Fourier Transform (STFT) spectrogram when
    ``compute_stft=True``.

    Args:
        resample_rate: Target sampling rate in Hz. Default is ``200`` (native
            SPaRCNet rate; set to ``256`` to match the 256 Hz setting in
            ``Biosignal_Token``).
        signal_len: Segment duration in seconds. Default is ``10``.
        normalization: Signal normalization strategy:
            - ``"95th_percentile"`` (default): divides each channel by its 95th
              percentile absolute value
              ``np.quantile(np.abs(X), q=0.95, axis=-1) + 1e-8``
              (used in TFM-Tokenizer).
            - ``"div_by_100"``: divides signal values by ``100.0``.
            - ``None`` or ``"none"``: leaves raw signal amplitudes unchanged.
        compute_stft: Whether to compute and include the ``"stft"`` tensor
            using ``get_stft_torch`` for time-frequency models such as
            :class:`~pyhealth.models.TFMTokenizer`. Set to ``False`` when using
            time-domain models with a single feature key such as
            :class:`~pyhealth.models.SparcNet`. Default is ``True``.
        splits: Optional subset of splits to emit
            (e.g., ``("train", "val", "test")``).
            Default is ``("train", "val", "test")``.

    Examples:
        >>> from pyhealth.datasets import IIICDataset
        >>> from pyhealth.tasks import EEGSeizureIIIC
        >>> dataset = IIICDataset(root="/srv/local/data/jp65/IIIC/")  # doctest: +SKIP
        >>> sample_dataset = dataset.set_task(EEGSeizureIIIC())  # doctest: +SKIP
        >>> sample = sample_dataset[0]  # doctest: +SKIP
        >>> print(sample["signal"].shape)  # (16, 2000)  # doctest: +SKIP
    """

    task_name: str = "EEG_seizure_IIIC"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor", "stft": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "multiclass"}
    DEFAULT_SAMPLING_RATE: int = 200

    def __init__(
        self,
        resample_rate: int = 200,
        signal_len: int = 10,
        normalization: str | None = "95th_percentile",
        compute_stft: bool = True,
        splits: tuple[str, ...] | list[str] = ("train", "val", "test"),
    ) -> None:
        super().__init__()
        if normalization not in (None, "none", "95th_percentile", "div_by_100"):
            raise ValueError(
                "normalization must be one of "
                "{'95th_percentile', 'div_by_100', None, 'none'}"
            )
        self.resample_rate = int(resample_rate)
        self.signal_len = int(signal_len)
        self.normalization = None if normalization == "none" else normalization
        self.compute_stft = bool(compute_stft)
        self.splits = tuple(splits)

        if not self.compute_stft:
            self.input_schema = {"signal": "tensor"}
        else:
            self.input_schema = {"signal": "tensor", "stft": "tensor"}

    @staticmethod
    def load_segment(
        signal_file: str,
        fallback_label: Any | None = None,
    ) -> tuple[np.ndarray, int | None]:
        """Load a single EEG segment from a ``.pkl`` or ``.npy`` file.

        Supports:
          1) ``.pkl`` dictionary files containing ``{"signal": ndarray, "label": int}``
             (the format used in ``Biosignal_Token`` / ``/srv/local/data/jp65/IIIC/``).
          2) ``.npy`` files containing a 2D array ``(16, 2000)``, or a ``.npy::idx``
             indexed slice into a 3D memory-mapped array ``(N, 16, 2000)``.

        Args:
            signal_file: Path to the segment file (or ``path.npy::index``).
            fallback_label: Label from metadata table if not stored inside the file.

        Returns:
            Tuple of ``(signal_array, label_int)``.
        """
        file_str = str(signal_file)
        label: int | None = (
            int(fallback_label)
            if fallback_label is not None
            and str(fallback_label) != ""
            and not _is_na(fallback_label)
            else None
        )

        if "::" in file_str:
            npy_path, idx_str = file_str.rsplit("::", 1)
            arr = np.load(npy_path, mmap_mode="r")
            signal = np.asarray(arr[int(idx_str)], dtype=np.float32)
            return signal, label

        path = Path(file_str)
        suffix = path.suffix.lower()
        if suffix == ".pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                signal = np.asarray(data["signal"], dtype=np.float32)
                if label is None and "label" in data:
                    label = int(data["label"])
            else:
                signal = np.asarray(data, dtype=np.float32)
            return signal, label
        if suffix == ".npy":
            signal = np.asarray(np.load(path), dtype=np.float32)
            return signal, label
        raise ValueError(f"Unsupported signal file format: {signal_file}")

    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply resampling and amplitude normalization following ``IIICloader``.

        Args:
            signal: Raw 2D EEG segment of shape ``(n_channels, n_samples)``.

        Returns:
            Normalized 2D ``float32`` numpy array.
        """
        X = np.asarray(signal, dtype=np.float32)
        target_len = self.resample_rate * self.signal_len
        if (
            self.resample_rate != self.DEFAULT_SAMPLING_RATE
            or X.shape[-1] != target_len
        ):
            X = resample(X, target_len, axis=-1).astype(np.float32)

        X = normalize_signal(X, self.normalization)
        return X.astype(np.float32)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Process all IIIC EEG segments for a single patient.

        Args:
            patient: A PyHealth ``Patient`` object containing ``"signal"`` events
                (or ``"train"`` / ``"val"`` / ``"test"`` events).

        Returns:
            List of sample dictionaries with keys ``patient_id``, ``record_id``,
            ``signal_file``, ``split``, ``signal``, ``label``, and (when
            ``compute_stft=True``) ``stft``.
        """
        pid = patient.patient_id
        samples: list[dict[str, Any]] = []

        events = patient.get_events(event_type="signal")
        if not events:
            # Fallback if events were stored under split-named tables
            for sp in ("train", "val", "test", "eval"):
                sp_events = patient.get_events(event_type=sp)
                if sp_events:
                    events.extend(sp_events)

        for event in events:
            split = getattr(event, "split", "train")
            if self.splits and split not in self.splits:
                continue

            signal_file = getattr(event, "signal_file", None)
            if signal_file is None:
                continue

            event_label = getattr(event, "label", None)
            record_id = getattr(event, "record_id", Path(str(signal_file)).stem)

            try:
                raw_signal, label = self.load_segment(signal_file, event_label)
            except (
                OSError,
                pickle.UnpicklingError,
                ValueError,
                KeyError,
                EOFError,
            ) as exc:
                logger.warning(
                    "Skipping unreadable IIIC segment %s: %s",
                    signal_file,
                    exc,
                )
                continue

            if label is None:
                continue

            proc_signal = self.preprocess_signal(raw_signal)
            signal_tensor = torch.FloatTensor(proc_signal)

            sample: dict[str, Any] = {
                "patient_id": str(pid),
                "record_id": str(record_id),
                "signal_file": str(signal_file),
                "split": str(split),
                "signal": signal_tensor,
                "label": int(label),
            }

            if self.compute_stft:
                from pyhealth.models.tfm_tokenizer import get_stft_torch

                sample["stft"] = get_stft_torch(
                    signal_tensor.unsqueeze(0),
                    resampling_rate=self.resample_rate,
                ).squeeze(0)

            samples.append(sample)

        return samples


def _is_na(val: Any) -> bool:
    """Return True if value is NaN/None."""
    if val is None:
        return True
    try:
        return bool(np.isnan(val))
    except (TypeError, ValueError):
        return False


# Alias for convenience
IIICSeizureClassification = EEGSeizureIIIC
