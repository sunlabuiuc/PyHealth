"""CLIMB's 12-lead ECG diagnosis task (Dai et al., ICML 2025).

CLIMB ("Data Foundations for Large Scale Multimodal Clinical Foundation
Models", https://arxiv.org/abs/2503.07667) groups the ECGs of PTB-XL, CPSC
2018, Georgia and Chapman-Shaoxing into seven diagnostic classes and splits
each source 80/20 into train and validation sets. Its split files
(``<source>_splits.csv`` in https://github.com/DDVD233/CLIMB) give each
recording's split and classes, so :func:`read_climb_ecg_splits` reads both from
them, and :class:`CLIMBECGClassification` turns the recordings into
fixed-length signals with those labels: PTB-XL, CPSC 2018 and Georgia from
:class:`~pyhealth.datasets.CardiologyDataset`, and Chapman-Shaoxing from
:class:`~pyhealth.datasets.ChapmanShaoxingDataset`.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample, resample_poly

from .base_task import BaseTask

#: CLIMB's seven ECG classes, in the column order of its split files.
CLIMB_ECG_CLASSES: tuple[str, ...] = ("normal", "cd", "mi", "sttc", "other", "afib", "hyp")

#: The 12 leads of the Challenge recordings, in file order.
ECG_LEADS: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
)

#: The eight independent leads ECG-JEPA reads (the other four are linear
#: combinations of I and II).
ECG_JEPA_LEADS: tuple[str, ...] = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


def read_climb_ecg_splits(path: str | Path) -> dict[str, tuple[str, list[str]]]:
    """Reads one of CLIMB's ECG split files.

    Each row of ``<source>_splits.csv`` has the recording name (column
    ``patient``), its split (``train`` or ``val``) and one 0/1 column per class
    of :data:`CLIMB_ECG_CLASSES`.

    Args:
        path: The split file, such as ``ptbxl_splits.csv``.

    Returns:
        A dict from recording name (such as ``"HR00001"``) to its split and
        the names of its classes (possibly none).

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from pyhealth.tasks.climb_ecg import read_climb_ecg_splits
        >>> path = Path(tempfile.mkdtemp()) / "ptbxl_splits.csv"
        >>> _ = path.write_text(
        ...     ",file_name,normal,cd,mi,sttc,other,afib,hyp,patient,split,label\\n"
        ...     "0,x/HR00001.hea,1,0,0,0,1,0,0,HR00001,train,0\\n"
        ... )
        >>> read_climb_ecg_splits(path)
        {'HR00001': ('train', ['normal', 'other'])}
    """
    splits = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            classes = [c for c in CLIMB_ECG_CLASSES if int(float(row[c]))]
            splits[row["patient"]] = (row["split"], classes)
    return splits


class CLIMBECGClassification(BaseTask):
    """Multi-label ECG diagnosis with CLIMB's seven classes.

    Works with :class:`~pyhealth.datasets.CardiologyDataset` and
    :class:`~pyhealth.datasets.ChapmanShaoxingDataset` (both read the
    Challenge ``.mat`` files). For each recording that has an entry in
    ``labels`` it, as CLIMB does:

    1. reads the ``.mat`` signal and resamples it to ``fs`` (500 Hz),
    2. keeps the first ``duration`` seconds, padding shorter recordings with
       zeros at the end,
    3. drops the recording if its first 15 samples are zero on every lead,
    4. resamples the window to ``num_samples`` points (2,500 for ECG-JEPA,
       that is 10 s at 250 Hz) and keeps ``leads``, and
    5. scales it by its range, ``x / (max(x) - min(x))`` over all leads.

    Each sample contains:

    - ``signal``: float tensor, shape ``(len(leads), num_samples)``
    - ``label``: the recording's classes (a subset of
      :data:`CLIMB_ECG_CLASSES`, possibly empty); the label processor keeps
      the order of :data:`CLIMB_ECG_CLASSES`
    - ``record_id``, ``source`` (the dataset table) and ``split`` (CLIMB's
      ``train`` or ``val``)

    Args:
        labels: Split and classes of each recording, as returned by
            :func:`read_climb_ecg_splits`. Recordings without an entry are
            skipped.
        duration: Seconds of signal kept. Default is 10.
        fs: Sampling rate before the final resampling. Default is 500.
        num_samples: Length of the output signal. Default is 2,500.
        leads: Leads to keep, by name (see :data:`ECG_LEADS`). Default is
            :data:`ECG_JEPA_LEADS`.
        normalize: Scale each recording by its range. Default is True.

    Raises:
        ValueError: If a length or rate is not positive, or a lead is unknown.

    Examples:
        >>> from pyhealth.tasks import CLIMBECGClassification
        >>> task = CLIMBECGClassification({"HR00001": ("train", ["normal"])})
        >>> task.input_schema
        {'signal': 'tensor'}
        >>> task.output_schema["label"][1]["labels"]
        ['normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp']
    """

    task_name: str = "CLIMBECGClassification"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, Any]] = {
        "label": ("multilabel", {"labels": list(CLIMB_ECG_CLASSES)})
    }

    def __init__(
        self,
        labels: Mapping[str, tuple[str, Sequence[str]]],
        duration: float = 10.0,
        fs: float = 500.0,
        num_samples: int = 2500,
        leads: Sequence[str] = ECG_JEPA_LEADS,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if min(duration, fs, num_samples) <= 0:
            raise ValueError("duration, fs and num_samples must be positive.")
        unknown = [lead for lead in leads if lead not in ECG_LEADS]
        if unknown:
            raise ValueError(f"Unknown leads {unknown}; choose from {ECG_LEADS}.")
        self.labels = {k: (split, list(classes)) for k, (split, classes) in labels.items()}
        self.duration = float(duration)
        self.fs = float(fs)
        self.num_samples = int(num_samples)
        self.leads = tuple(leads)
        self.normalize = normalize

    def signal(self, signal_file: str, fs: float) -> np.ndarray | None:
        """The processed ``(len(leads), num_samples)`` signal, or ``None``.

        ``None`` means the recording starts with 15 all-zero samples, which
        CLIMB drops.
        """
        # ChapmanShaoxingDataset stores the record path without ".mat";
        # loadmat adds it.
        x = loadmat(signal_file)["val"].astype(np.float64)
        if fs != self.fs:
            ratio = Fraction(self.fs / fs).limit_denominator(1000)
            x = resample_poly(x, ratio.numerator, ratio.denominator, axis=1)
        width = round(self.duration * self.fs)
        x = x[:, :width]
        if x.shape[1] < width:
            x = np.pad(x, ((0, 0), (0, width - x.shape[1])))
        if not np.any(x[:, :15]):
            return None
        if self.num_samples != width:
            x = resample(x, self.num_samples, axis=1)
        x = x[[ECG_LEADS.index(lead) for lead in self.leads]]
        if self.normalize:
            spread = x.max() - x.min()
            x = x / spread if spread > 0 else x
        return x.astype(np.float32)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds the sample of each labeled recording of one patient.

        Args:
            patient: A patient of either dataset; each event is a recording.

        Returns:
            One sample per recording that is in ``labels`` and not dropped.
        """
        samples = []
        for event in patient.get_events():
            # A metadata CSV indexed before the parse_header fix may still
            # hold "E00001.mat"; match it on the bare record name.
            record_id = Path(str(event.record_id)).stem
            entry = self.labels.get(record_id)
            if entry is None:
                continue
            attrs = event.attr_dict
            fs = attrs["fs"] if "fs" in attrs else attrs["sampling_rate"]
            x = self.signal(attrs["signal_file"], float(fs))
            if x is None:
                continue
            split, classes = entry
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": record_id,
                    "source": event.event_type,
                    "split": split,
                    "signal": torch.from_numpy(x),
                    "label": classes,
                }
            )
        return samples
