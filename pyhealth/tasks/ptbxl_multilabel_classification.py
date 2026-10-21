"""Multi-label ECG classification task for the PTB-XL dataset."""

import os
import re
from typing import Any, ClassVar

import numpy as np

from .base_task import BaseTask

_LABEL_SETS = (
    "all",
    "diagnostic",
    "subdiagnostic",
    "superdiagnostic",
    "form",
    "rhythm",
)
_GAIN_PATTERN = re.compile(r"^([0-9.eE+-]+)(?:\(([-0-9]+)\))?(?:/.*)?$")


def read_wfdb_signal(record_path: str) -> np.ndarray:
    """Reads a WFDB record and returns its physical signal.

    The ``wfdb`` package is used when it is installed. Otherwise a small
    fallback reader that supports the 16-bit format used by PTB-XL is used, so
    that no extra dependency is required.

    Args:
        record_path: Path of the record without extension, e.g.
            ``".../records100/00000/00001_lr"``.

    Returns:
        A float32 array of shape ``(num_samples, num_leads)`` in millivolts.

    Raises:
        FileNotFoundError: If the header or data file does not exist.
        ValueError: If the record uses an unsupported WFDB storage format.

    Examples:
        >>> signal = read_wfdb_signal("records100/00000/00001_lr")  # doctest: +SKIP
        >>> signal.shape  # doctest: +SKIP
        (1000, 12)
    """
    try:
        import wfdb  # type: ignore

        return wfdb.rdrecord(record_path).p_signal.astype(np.float32)
    except ImportError:
        pass

    header_path = record_path + ".hea"
    if not os.path.isfile(header_path):
        raise FileNotFoundError(f"WFDB header not found: {header_path}")
    with open(header_path, "r", encoding="utf-8") as handle:
        lines = [ln.strip() for ln in handle if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith("#")]
    first = lines[0].split()
    num_leads, num_samples = int(first[1]), int(first[3])
    gains, baselines, files = [], [], set()
    for line in lines[1 : 1 + num_leads]:
        fields = line.split()
        files.add(fields[0])
        if fields[1].split("+")[0] != "16":
            raise ValueError(
                f"Unsupported WFDB format {fields[1]} in {header_path}; "
                "install `wfdb` to read this record."
            )
        match = _GAIN_PATTERN.match(fields[2])
        gain = float(match.group(1)) if match else 200.0
        gains.append(gain if gain != 0 else 200.0)
        baselines.append(
            int(match.group(2)) if match and match.group(2) else int(fields[4])
        )
    if len(files) != 1:
        raise ValueError("Only records with a single signal file are supported.")
    data_path = os.path.join(os.path.dirname(header_path), files.pop())
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"WFDB data file not found: {data_path}")
    digital = np.fromfile(data_path, dtype="<i2").reshape(-1, num_leads)
    digital = digital[:num_samples].astype(np.float32)
    return (digital - np.asarray(baselines, dtype=np.float32)) / np.asarray(
        gains, dtype=np.float32
    )


class PTBXLMultilabelClassification(BaseTask):
    """Multi-label classification of 12-lead ECGs from the PTB-XL dataset.

    Every ECG with at least one label in the chosen label set becomes one
    sample with a ``(12, 10 * sampling_rate)`` signal and a list of label
    codes. The label vocabulary is fitted by the ``multilabel`` processor.

    Args:
        label_set: Which labels to predict. One of ``"all"`` (all 71 SCP
            statements), ``"diagnostic"`` (44 diagnostic statements),
            ``"subdiagnostic"`` (23 aggregated sub-classes),
            ``"superdiagnostic"`` (5 aggregated super-classes), ``"form"`` or
            ``"rhythm"``. Defaults to ``"diagnostic"``.
        sampling_rate: 100 or 500 Hz. Defaults to 100.
        min_likelihood: Keep statements whose annotated likelihood is at
            least this value (0 keeps everything, including unknown-likelihood
            form/rhythm statements). Defaults to 0.0.
        standardize: ``"record"`` applies per-lead z-scoring within each
            record; ``None`` keeps raw millivolts. Defaults to ``"record"``.
        remove_baseline: If True, applies a 0.5 Hz high-pass Butterworth
            filter to remove baseline wander. Defaults to False.

    Attributes:
        task_name (str): Name of the task.
        input_schema (dict[str, str]): ``{"signal": "tensor"}``.
        output_schema (dict[str, str]): ``{"labels": "multilabel"}``.

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> from pyhealth.tasks import PTBXLMultilabelClassification
        >>> dataset = PTBXLDataset(root="/path/to/ptb-xl/1.0.3")
        >>> task = PTBXLMultilabelClassification(label_set="superdiagnostic")
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "PTBXLMultilabelClassification"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"labels": "multilabel"}

    def __init__(
        self,
        label_set: str = "diagnostic",
        sampling_rate: int = 100,
        min_likelihood: float = 0.0,
        standardize: str | None = "record",
        remove_baseline: bool = False,
    ) -> None:
        if label_set not in _LABEL_SETS:
            raise ValueError(f"label_set must be one of {_LABEL_SETS}")
        if sampling_rate not in (100, 500):
            raise ValueError("sampling_rate must be 100 or 500")
        if standardize not in ("record", None):
            raise ValueError("standardize must be 'record' or None")
        self.label_set = label_set
        self.sampling_rate = sampling_rate
        self.min_likelihood = min_likelihood
        self.standardize = standardize
        self.remove_baseline = remove_baseline
        super().__init__()

    @staticmethod
    def _parse_labels(text: Any, min_likelihood: float) -> list[str]:
        """Parses a ``"CODE:likelihood;CODE:likelihood"`` string.

        Args:
            text: The label string, or None / NaN when the set is empty.
            min_likelihood: Minimum likelihood to keep a label.

        Returns:
            The list of label codes that pass the likelihood threshold.
        """
        if not isinstance(text, str) or not text:
            return []
        labels = []
        for item in text.split(";"):
            code, _, likelihood = item.partition(":")
            if code and float(likelihood or 0.0) >= min_likelihood:
                labels.append(code)
        return labels

    def _prepare_signal(self, record_path: str) -> np.ndarray:
        """Loads, crops/pads and normalizes one ECG.

        Args:
            record_path: Record path without extension.

        Returns:
            A float32 array of shape ``(12, 10 * sampling_rate)``.
        """
        signal = read_wfdb_signal(record_path).T  # (leads, samples)
        length = 10 * self.sampling_rate
        if signal.shape[1] < length:
            pad = length - signal.shape[1]
            signal = np.pad(signal, ((0, 0), (0, pad)))
        signal = signal[:, :length]
        if self.remove_baseline:
            from scipy.signal import butter, filtfilt

            b, a = butter(1, 0.5 / (self.sampling_rate / 2), btype="high")
            signal = filtfilt(b, a, signal, axis=1)
        if self.standardize == "record":
            mean = signal.mean(axis=1, keepdims=True)
            std = signal.std(axis=1, keepdims=True)
            signal = (signal - mean) / (std + 1e-6)
        return np.ascontiguousarray(signal, dtype=np.float32)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one sample per labelled ECG of a patient.

        Args:
            patient: A patient object with ``ptbxl`` events.

        Returns:
            A list of dicts with ``patient_id``, ``ecg_id``, ``strat_fold``,
            ``signal`` (12 x T float32 array) and ``labels`` (list of codes).
            ECGs without any label in the chosen set are skipped.
        """
        samples = []
        for event in patient.get_events(event_type="ptbxl"):
            labels = self._parse_labels(
                getattr(event, f"labels_{self.label_set}", None),
                self.min_likelihood,
            )
            if not labels:
                continue
            path = (
                event.record_path_lr
                if self.sampling_rate == 100
                else event.record_path_hr
            )
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "ecg_id": int(event.ecg_id),
                    "strat_fold": int(event.strat_fold),
                    "signal": self._prepare_signal(path),
                    "labels": labels,
                }
            )
        return samples
