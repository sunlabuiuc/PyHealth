"""CPSC 2018 classification from HeartLang heartbeat sentences.

Author: rayrayowo (port of the HeartLang CPSC2018 benchmark)

Paper:
    Jiarui Jin et al. "Reading Your Heart: Learning ECG Words and Sentences
    via Pre-training ECG Language Model." ICLR 2025.
    https://arxiv.org/abs/2502.10707

Reference code:
    https://github.com/PKUDigitalHealth/HeartLang (MIT License), commit
    a08afd8, ``datasets/dataset_preprocess/CPSC2018/make_cpsc_dataset.py``
    and ``scripts/finetune/CPSC2018/base_cpsc2018_linear_prob.sh``.

Description:
    Implements :class:`CPSC2018HeartLangClassification`, the 9-label
    CPSC2018 benchmark of HeartLang, on the ``cpsc_2018`` source of
    :class:`~pyhealth.datasets.CardiologyDataset` (the PhysioNet/CinC
    Challenge 2020 copy of CPSC 2018). It applies the same preprocessing as
    the reference code (first 5 s, zero padding to 10 s, min-max to [-3, 3],
    resampling to 100 Hz, no filter) and builds the inputs of
    :class:`~pyhealth.models.HeartLang`. :func:`split_by_heartlang_cpsc2018`
    reproduces HeartLang's train/validation/test split from its record lists.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

#: The 9 labels of HeartLang's CPSC2018 benchmark, in the column order of its
#: ``icbeb_train.csv`` (the multi-hot order of the published linear probe).
HEARTLANG_CPSC2018_LABELS: tuple[str, ...] = (
    "AFIB", "VPC", "NORM", "1AVB", "CRBBB", "STE", "PAC", "CLBBB", "STD",
)  # fmt: skip

#: SNOMED-CT code of each label in the Challenge 2020 ``cpsc_2018`` headers.
#: The record counts of these codes in the Challenge's ``dx_mapping`` tables
#: (CPSC column) equal the label counts of HeartLang's three CSV files, for
#: example 1,857 for ``CRBBB`` and 700 for ``VPC``. HeartLang keeps the
#: PTB-XL benchmark's names: ``CRBBB`` and ``CLBBB`` are the plain right and
#: left bundle branch blocks of CPSC 2018, and ``VPC`` is coded as
#: "ventricular ectopics".
HEARTLANG_CPSC2018_CODES: dict[str, str] = {
    "164889003": "AFIB",  # atrial fibrillation
    "164884008": "VPC",  # ventricular ectopics
    "426783006": "NORM",  # sinus rhythm
    "270492004": "1AVB",  # 1st degree AV block
    "59118001": "CRBBB",  # right bundle branch block
    "164931005": "STE",  # ST elevation
    "284470004": "PAC",  # premature atrial contraction
    "164909002": "CLBBB",  # left bundle branch block
    "429622005": "STD",  # ST depression
}

_SPLITS = ("train", "val", "test")
_SOURCE_TABLE = "cpsc_2018"


def heartlang_cpsc2018_labels(dx: Any) -> list[str]:
    """Maps a record's SNOMED-CT codes to HeartLang CPSC2018 labels.

    Codes other than the nine of :data:`HEARTLANG_CPSC2018_CODES` are
    ignored. CPSC 2018 only uses these nine, so in practice every record
    gets one to three labels.

    Args:
        dx: The ``dx`` attribute of a
            :class:`~pyhealth.datasets.CardiologyDataset` record (codes
            separated by ``,``).

    Returns:
        The positive labels, in :data:`HEARTLANG_CPSC2018_LABELS` order.

    Examples:
        >>> from pyhealth.tasks.cpsc2018_heartlang import heartlang_cpsc2018_labels
        >>> heartlang_cpsc2018_labels("59118001,164889003")
        ['AFIB', 'CRBBB']
        >>> heartlang_cpsc2018_labels("713427006")
        []
    """
    codes = {c.strip() for c in str(dx or "").replace(";", ",").split(",")}
    present = {
        HEARTLANG_CPSC2018_CODES[c] for c in codes if c in HEARTLANG_CPSC2018_CODES
    }
    return [label for label in HEARTLANG_CPSC2018_LABELS if label in present]


def preprocess_heartlang_cpsc2018(
    signal: np.ndarray,
    sampling_rate: float = 500,
    target_rate: float = 100,
    keep_seconds: float = 5,
    total_seconds: float = 10,
) -> np.ndarray:
    """Applies HeartLang's CPSC2018 preprocessing to one ECG.

    The steps of the reference ``make_cpsc_dataset.py``, in order:

    1. keep the first ``keep_seconds`` (2,500 samples at 500 Hz)
    2. pad with zeros to ``total_seconds`` (5,000 samples), the length of
       the pre-training ECGs
    3. min-max the whole padded record, all leads together, to [-3, 3]
    4. resample to ``target_rate`` with an FFT resampler (1,000 samples)

    Unlike the CSN benchmark there is no band-pass filter: the reference
    defines an FIR filter but never calls it. Two consequences worth
    knowing when comparing with other CPSC 2018 results: half of every input
    is padding, and the zeros take part in the min-max, so the padded half
    becomes a constant that is usually not 0 and its edge rings after the
    FFT resampling. Min-max makes the step insensitive to the unit (the
    original release is in mV, the Challenge 2020 copy in µV).

    Args:
        signal: ECG of shape ``(leads, samples)``.
        sampling_rate: Rate of ``signal`` in Hz (500 in CPSC 2018).
        target_rate: Output rate in Hz (100 for the published model).
        keep_seconds: Seconds kept from the start of the record.
        total_seconds: Length after zero padding, in seconds.

    Returns:
        float32 array of shape ``(leads, total_seconds * target_rate)``.

    Raises:
        ValueError: If ``signal`` is not 2-D, holds NaN or infinite values,
            or ``keep_seconds`` is longer than ``total_seconds``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.tasks.cpsc2018_heartlang import (
        ...     preprocess_heartlang_cpsc2018)
        >>> ecg = np.random.default_rng(0).normal(size=(12, 7500))
        >>> preprocess_heartlang_cpsc2018(ecg).shape
        (12, 1000)
    """
    from scipy.signal import resample

    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"signal must be (leads, samples), got shape {x.shape}")
    if keep_seconds > total_seconds:
        raise ValueError("keep_seconds must not exceed total_seconds")
    if not np.isfinite(x).all():
        raise ValueError("signal holds NaN or infinite values")
    keep = round(keep_seconds * sampling_rate)
    total = round(total_seconds * sampling_rate)
    x = x[:, :keep]
    x = np.pad(x, ((0, 0), (0, total - x.shape[1])), constant_values=0)
    low, high = x.min(), x.max()
    if high > low:
        x = 6.0 * (x - low) / (high - low) - 3.0
    else:
        x = np.zeros_like(x)
    n_target = int(x.shape[1] * (target_rate / sampling_rate))
    x = resample(x, n_target, axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


class CPSC2018HeartLangClassification(BaseTask):
    """HeartLang's 9-label CPSC2018 benchmark.

    Works with the ``cpsc_2018`` source of
    :class:`~pyhealth.datasets.CardiologyDataset` and builds the inputs of
    :class:`~pyhealth.models.HeartLang`, reproducing the CPSC2018
    linear-probe setup of HeartLang (Jin et al., ICLR 2025). For each record
    it:

    1. reads the ``.mat`` waveform of the Challenge 2020 copy
    2. applies :func:`preprocess_heartlang_cpsc2018`: first 5 s, zero
       padding to 10 s, min-max to [-3, 3], 100 Hz
    3. splits the ECG into heartbeat words around the QRS complexes of the
       first selected lead
       (:func:`~pyhealth.models.heartlang.ecg_to_heartbeat_sentence`)
    4. labels it with :func:`heartlang_cpsc2018_labels` (multi-label)

    Records of other sources are ignored. Use
    :func:`split_by_heartlang_cpsc2018` for HeartLang's
    train/validation/test split of the 6,877 records.

    Args:
        leads: Lead indices to use (I, II, III, aVR, aVL, aVF, V1-V6 order).
            Defaults to all 12. QRS complexes are detected on the first one.
        max_len: Words per sentence (256 for the published model).
        token_len: Samples per word (96 for the published model).
        drop_empty_labels: Skip records with none of the nine labels.
            Defaults to ``False``; CPSC 2018 has no such record.

    Attributes:
        task_name (str): ``"CPSC2018HeartLangClassification"``.
        input_schema (dict): ``ecg_sentence``, ``lead_ids`` and ``time_ids``,
            all tensors.
        output_schema (dict): ``labels``, multi-label over
            :data:`HEARTLANG_CPSC2018_LABELS`.

    Examples:
        >>> from pyhealth.tasks import CPSC2018HeartLangClassification
        >>> task = CPSC2018HeartLangClassification()
        >>> task.output_schema["labels"][1]["labels"][:3]
        ['AFIB', 'VPC', 'NORM']
        >>> from pyhealth.datasets import CardiologyDataset
        >>> from pyhealth.tasks.cpsc2018_heartlang import (
        ...     split_by_heartlang_cpsc2018)
        >>> dataset = CardiologyDataset(  # doctest: +SKIP
        ...     root="/data/challenge-2020/1.0.2/training", sources=["cpsc_2018"])
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> train, val, test = split_by_heartlang_cpsc2018(  # doctest: +SKIP
        ...     samples, "HeartLang/datasets/dataset_preprocess/CPSC2018")
        >>> samples[0]["ecg_sentence"].shape  # doctest: +SKIP
        torch.Size([256, 96])
    """

    task_name: str = "CPSC2018HeartLangClassification"
    input_schema: ClassVar[dict[str, str]] = {
        "ecg_sentence": "tensor",
        "lead_ids": "tensor",
        "time_ids": "tensor",
    }
    output_schema: ClassVar[dict[str, Any]] = {
        "labels": ("multilabel", {"labels": list(HEARTLANG_CPSC2018_LABELS)})
    }

    def __init__(
        self,
        leads: Sequence[int] | None = None,
        max_len: int = 256,
        token_len: int = 96,
        drop_empty_labels: bool = False,
    ) -> None:
        self.leads = None if leads is None else [int(i) for i in leads]
        self.max_len = max_len
        self.token_len = token_len
        self.drop_empty_labels = drop_empty_labels
        super().__init__()

    @staticmethod
    def _read_mat(signal_file: str) -> np.ndarray:
        """Reads one Challenge 2020 ``.mat`` waveform as ``(leads, samples)``."""
        from scipy.io import loadmat

        return np.asarray(loadmat(signal_file)["val"], dtype=np.float64)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one sample per CPSC 2018 record of a patient.

        Args:
            patient: A patient from
                :class:`~pyhealth.datasets.CardiologyDataset`.

        Returns:
            Dicts with ``patient_id``, ``record_id``, ``ecg_sentence``
            (float32 ``(max_len, token_len)``), ``lead_ids`` and ``time_ids``
            (``(max_len,)``) and ``labels``.
        """
        from pyhealth.models.heartlang import ecg_to_heartbeat_sentence

        samples: list[dict[str, Any]] = []
        for event in patient.get_events(event_type=_SOURCE_TABLE):
            labels = heartlang_cpsc2018_labels(event.dx)
            if not labels and self.drop_empty_labels:
                continue
            raw = self._read_mat(event.signal_file)
            signal = preprocess_heartlang_cpsc2018(raw, sampling_rate=float(event.fs))
            if not np.isfinite(signal).all():
                logger.warning("Skipping %s: non-finite samples", event.record_id)
                continue
            words, lead_ids, time_ids = ecg_to_heartbeat_sentence(
                signal,
                sampling_rate=100,
                max_len=self.max_len,
                token_len=self.token_len,
                leads=self.leads,
            )
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": str(event.record_id),
                    "ecg_sentence": words,
                    "lead_ids": lead_ids,
                    "time_ids": time_ids,
                    "labels": labels,
                }
            )
        return samples


def load_heartlang_cpsc2018_split(
    csv_dir: str | os.PathLike,
) -> dict[str, set[str]]:
    """Reads HeartLang's CPSC2018 record lists.

    HeartLang ships ``icbeb_train.csv``, ``icbeb_val.csv`` and
    ``icbeb_test.csv`` (4,950 / 551 / 1,376 records, all 6,877 of CPSC 2018)
    in ``datasets/dataset_preprocess/CPSC2018/`` of
    https://github.com/PKUDigitalHealth/HeartLang. Only their ``filename``
    column (the record name, such as ``A0001``) is read. Their
    ``scp_codes`` column is empty (``{}``) in every row; the labels live in
    the one-hot columns only.

    Args:
        csv_dir: Folder holding the three CSV files.

    Returns:
        ``{"train": ids, "val": ids, "test": ids}``.

    Raises:
        FileNotFoundError: If a CSV file is missing.
        ValueError: If a record is in more than one split.

    Examples:
        >>> from pyhealth.tasks.cpsc2018_heartlang import (
        ...     load_heartlang_cpsc2018_split)
        >>> ids = load_heartlang_cpsc2018_split(  # doctest: +SKIP
        ...     "HeartLang/datasets/dataset_preprocess/CPSC2018")
        >>> len(ids["train"])  # doctest: +SKIP
        4950
    """
    import pandas as pd

    ids: dict[str, set[str]] = {}
    for split in _SPLITS:
        path = Path(csv_dir) / f"icbeb_{split}.csv"
        names = pd.read_csv(path, usecols=["filename"])["filename"].astype(str)
        ids[split] = {Path(name).name for name in names}
    for i, first in enumerate(_SPLITS):
        for second in _SPLITS[i + 1 :]:
            shared = ids[first] & ids[second]
            if shared:
                raise ValueError(
                    f"{len(shared)} records are in both {first} and {second}"
                )
    return ids


def split_by_heartlang_cpsc2018(
    dataset: Any,
    csv_dir: str | os.PathLike,
    record_ids: Sequence[str] | None = None,
):
    """Splits a sample dataset into HeartLang's CPSC2018 train/val/test sets.

    Samples whose record is in none of HeartLang's lists are left out, and
    their number is logged. An empty split is logged as a warning.

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` built with
            :class:`CPSC2018HeartLangClassification` (or any object with
            ``subset`` and samples carrying ``record_id``).
        csv_dir: Folder with HeartLang's CSV files, see
            :func:`load_heartlang_cpsc2018_split`.
        record_ids: Optional record id of every sample, in dataset order, to
            avoid decoding each sample.

    Returns:
        tuple: ``(train_dataset, val_dataset, test_dataset)`` subsets.

    Raises:
        ValueError: If ``record_ids`` has the wrong length.

    Examples:
        >>> from pyhealth.tasks.cpsc2018_heartlang import (
        ...     split_by_heartlang_cpsc2018)
        >>> train, val, test = split_by_heartlang_cpsc2018(  # doctest: +SKIP
        ...     samples, "HeartLang/datasets/dataset_preprocess/CPSC2018")
    """
    ids = load_heartlang_cpsc2018_split(csv_dir)
    n_samples = len(dataset)
    if record_ids is None:
        record_ids = [str(dataset[i]["record_id"]) for i in range(n_samples)]
    elif len(record_ids) != n_samples:
        raise ValueError(
            f"record_ids has length {len(record_ids)} but dataset has "
            f"{n_samples} samples"
        )
    index: dict[str, list[int]] = {split: [] for split in _SPLITS}
    for i, record_id in enumerate(record_ids):
        for split in _SPLITS:
            if record_id in ids[split]:
                index[split].append(i)
                break
    unused = n_samples - sum(len(v) for v in index.values())
    if unused:
        logger.info("%d samples are in no HeartLang CPSC2018 split", unused)
    for split in _SPLITS:
        if not index[split]:
            logger.warning("HeartLang CPSC2018 %s split is empty", split)
    return tuple(dataset.subset(index[split]) for split in _SPLITS)
