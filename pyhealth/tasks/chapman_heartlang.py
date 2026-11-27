"""Chapman-Shaoxing (CSN) classification from HeartLang heartbeat sentences.

Author: rayrayowo (port of the HeartLang CSN benchmark)

Paper:
    Jiarui Jin et al. "Reading Your Heart: Learning ECG Words and Sentences
    via Pre-training ECG Language Model." ICLR 2025.
    https://arxiv.org/abs/2502.10707

Reference code:
    https://github.com/PKUDigitalHealth/HeartLang (MIT License), commit
    a08afd8, ``datasets/dataset_preprocess/CSN/make_csn_datasets.py`` and
    ``scripts/finetune/CSN/base_csn_linear_prob.sh``.

Description:
    Implements :class:`ChapmanShaoxingHeartLangClassification`, the 38-label
    CSN benchmark of HeartLang on
    :class:`~pyhealth.datasets.ChapmanShaoxingDataset`, with the same
    preprocessing as the reference code (min-max to [-3, 3], resampling to
    100 Hz, a 0.67-40 Hz band-pass) and the inputs of
    :class:`~pyhealth.models.HeartLang`. :func:`split_by_heartlang_csn`
    reproduces HeartLang's train/validation/test split from its record lists.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from pyhealth.tasks.base_task import BaseTask

# Shared on purpose: the SNOMED-CT code -> acronym table describes the
# Chapman-Shaoxing release itself, not one task. HeartLang's labels are those
# acronyms, so both Chapman tasks read the one table, and a fix to it (a new
# code in a later release, say) reaches both.
from pyhealth.tasks.chapman_shaoxing_classification import (
    CHAPMAN_SNOMED_CODES,
    split_dx_codes,
)

logger = logging.getLogger(__name__)

#: The 38 labels of HeartLang's CSN benchmark, in the column order of its
#: ``chapman_train.csv`` (the multi-hot order of the published linear probe).
#: Sinus bradycardia (SB) and 11 rarer acronyms of the release are not labels.
HEARTLANG_CSN_LABELS: tuple[str, ...] = (
    "AQW", "UW", "SR", "WPW", "2AVB", "AT", "VB", "ARS", "STTC", "SA",
    "STE", "VPB", "TWO", "STTU", "ALS", "APB", "2AVB1", "PRIE", "CCR", "CR",
    "AF", "AVB", "QTIE", "LBBB", "VEB", "SVT", "RBBB", "1AVB", "STDD", "MI",
    "AFIB", "TWC", "PWC", "ERV", "RVH", "LVH", "ST", "JEB",
)  # fmt: skip

_LABEL_MATCHES = ("heartlang", "exact")
_SPLITS = ("train", "val", "test")
_RECORD_ID = re.compile(r"JS\d+")


def heartlang_csn_labels(dx_codes: Any, match: str = "heartlang") -> list[str]:
    """Maps a record's ``#Dx`` SNOMED-CT codes to HeartLang CSN labels.

    Each code is first turned into its acronym with
    :data:`~pyhealth.tasks.chapman_shaoxing_classification.CHAPMAN_SNOMED_CODES`
    (codes outside that table are ignored). With ``match="heartlang"`` a
    label is positive when it is a substring of one of the record's
    acronyms, which is how HeartLang's label table was built. Every case
    where an acronym of the release contains another label:

    - ``AF`` (atrial flutter) fires for ``AFIB`` (atrial fibrillation)
    - ``ST`` (sinus tachycardia) fires for ``STTC``, ``STTU``, ``STE`` and
      ``STDD`` (ST-T changes, ST elevation and ST depression)
    - ``AVB`` and ``VB`` fire for ``1AVB``, ``2AVB``, ``3AVB``, ``2AVB1`` and
      ``2AVB2``; ``VB`` also fires for ``AVB``
    - ``2AVB`` fires for ``2AVB1`` and ``2AVB2``
    - ``CR`` fires for ``CCR``
    - ``SA`` (sinus arrhythmia) fires for ``LVQRSAL`` (low QRS voltage)

    This rule reproduces all 23,026 rows of HeartLang's released label table from
    its ``diagnose`` column. ``match="exact"`` keeps only whole acronyms,
    which is clinically correct but no longer comparable with the paper.

    Args:
        dx_codes: The ``dx_codes`` attribute of a record (codes separated by
            ``,`` or ``;``).
        match: ``"heartlang"`` (default) or ``"exact"``.

    Returns:
        The positive labels, in :data:`HEARTLANG_CSN_LABELS` order.

    Raises:
        ValueError: If ``match`` is not ``"heartlang"`` or ``"exact"``.

    Examples:
        >>> from pyhealth.tasks.chapman_heartlang import heartlang_csn_labels
        >>> heartlang_csn_labels("164889003")  # AFIB
        ['AF', 'AFIB']
        >>> heartlang_csn_labels("164889003", match="exact")
        ['AFIB']
    """
    if match not in _LABEL_MATCHES:
        raise ValueError(f"match must be one of {_LABEL_MATCHES}, got {match!r}")
    acronyms = [
        CHAPMAN_SNOMED_CODES[code]
        for code in split_dx_codes(dx_codes)
        if code in CHAPMAN_SNOMED_CODES
    ]
    if match == "exact":
        present = set(acronyms)
        return [label for label in HEARTLANG_CSN_LABELS if label in present]
    return [
        label
        for label in HEARTLANG_CSN_LABELS
        if any(label in acronym for acronym in acronyms)
    ]


def preprocess_heartlang_csn(
    signal: np.ndarray,
    sampling_rate: float = 500,
    target_rate: int = 100,
) -> np.ndarray:
    """Applies HeartLang's CSN preprocessing to one ECG.

    The steps follow ``make_csn_datasets.py`` of the reference code:

    1. min-max scaling of the whole record (all leads together) to [-3, 3];
       a constant record becomes all zeros instead of dividing by zero
    2. Fourier resampling (:func:`scipy.signal.resample`) to ``target_rate``
    3. an order-5 Butterworth band-pass of 0.67-40 Hz applied forward and
       backward (:func:`scipy.signal.filtfilt`)

    The reference scales the raw ADC values of the ``.mat`` files; min-max
    scaling does not change if the input is in mV instead, as long as every
    lead has the same gain (1000/mV in this release).

    Args:
        signal: ECG of shape ``(leads, samples)``.
        sampling_rate: Rate of ``signal`` in Hz (500 in this release).
        target_rate: Output rate in Hz (100 for the published model).

    Returns:
        float32 array of shape ``(leads, samples * target_rate /
        sampling_rate)``.

    Raises:
        ValueError: If ``signal`` is not 2-D, or too short to filter.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.tasks.chapman_heartlang import preprocess_heartlang_csn
        >>> ecg = np.random.default_rng(0).normal(size=(12, 5000))
        >>> preprocess_heartlang_csn(ecg).shape
        (12, 1000)
    """
    from scipy.signal import butter, filtfilt, resample

    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"signal must be (leads, samples), got shape {x.shape}")
    low, high = x.min(), x.max()
    if high > low:
        x = 6.0 * (x - low) / (high - low) - 3.0
    else:
        x = np.zeros_like(x)
    n_target = int(x.shape[1] * (target_rate / sampling_rate))
    x = resample(x, n_target, axis=1)
    nyquist = 0.5 * target_rate
    b, a = butter(5, [0.67 / nyquist, 40 / nyquist], btype="band")
    if x.shape[1] <= 3 * max(len(a), len(b)):
        raise ValueError(
            f"{x.shape[1]} samples at {target_rate} Hz are too few to filter"
        )
    x = filtfilt(b, a, x, axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


class ChapmanShaoxingHeartLangClassification(BaseTask):
    """HeartLang's 38-label CSN benchmark on Chapman-Shaoxing.

    Works with :class:`~pyhealth.datasets.ChapmanShaoxingDataset` and builds
    the inputs of :class:`~pyhealth.models.HeartLang`, reproducing the CSN
    linear-probe setup of HeartLang (Jin et al., ICLR 2025). For each record
    it:

    1. reads the raw ADC values with ``wfdb`` (the ``pyhealth[ptbxl]`` extra),
       as the reference reads the ``.mat`` files
    2. applies :func:`preprocess_heartlang_csn`: min-max to [-3, 3], 100 Hz,
       0.67-40 Hz band-pass
    3. splits the ECG into heartbeat words around the QRS complexes of the
       first selected lead
       (:func:`~pyhealth.models.heartlang.ecg_to_heartbeat_sentence`)
    4. labels it with :func:`heartlang_csn_labels`

    Records with no positive label are kept by default, as in HeartLang's
    record lists. Use :func:`split_by_heartlang_csn` for HeartLang's
    train/validation/test split, which covers 23,026 of the 45,152 records.

    Notes on the reference labels: they treat a label as positive when it is
    a substring of a diagnosis acronym (see :func:`heartlang_csn_labels`);
    ``label_match="exact"`` turns that off. Sinus bradycardia, the second
    most frequent rhythm of the release, is not one of the 38 labels.

    Args:
        label_match: ``"heartlang"`` (default, as in the paper) or
            ``"exact"``; see :func:`heartlang_csn_labels`.
        leads: Lead indices to use (I, II, III, aVR, aVL, aVF, V1-V6 order).
            Defaults to all 12. QRS complexes are detected on the first one.
        max_len: Words per sentence (256 for the published model).
        token_len: Samples per word (96 for the published model).
        drop_empty_labels: Skip records with no positive label. Defaults to
            ``False``.

    Attributes:
        task_name (str): ``"ChapmanShaoxingHeartLangClassification"``.
        input_schema (dict): ``ecg_sentence``, ``lead_ids`` and ``time_ids``,
            all tensors.
        output_schema (dict): ``labels``, multi-label over
            :data:`HEARTLANG_CSN_LABELS`.

    Examples:
        >>> from pyhealth.tasks import ChapmanShaoxingHeartLangClassification
        >>> task = ChapmanShaoxingHeartLangClassification()
        >>> len(task.output_schema["labels"][1]["labels"])
        38
        >>> from pyhealth.datasets import ChapmanShaoxingDataset
        >>> from pyhealth.tasks.chapman_heartlang import split_by_heartlang_csn
        >>> dataset = ChapmanShaoxingDataset(  # doctest: +SKIP
        ...     root="/data/ecg-arrhythmia/1.0.0")
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> train, val, test = split_by_heartlang_csn(  # doctest: +SKIP
        ...     samples, "HeartLang/datasets/dataset_preprocess/CSN")
        >>> samples[0]["ecg_sentence"].shape  # doctest: +SKIP
        torch.Size([256, 96])
    """

    task_name: str = "ChapmanShaoxingHeartLangClassification"
    input_schema: ClassVar[dict[str, str]] = {
        "ecg_sentence": "tensor",
        "lead_ids": "tensor",
        "time_ids": "tensor",
    }
    output_schema: ClassVar[dict[str, Any]] = {
        "labels": ("multilabel", {"labels": list(HEARTLANG_CSN_LABELS)})
    }

    def __init__(
        self,
        label_match: str = "heartlang",
        leads: Sequence[int] | None = None,
        max_len: int = 256,
        token_len: int = 96,
        drop_empty_labels: bool = False,
    ) -> None:
        if label_match not in _LABEL_MATCHES:
            raise ValueError(
                f"label_match must be one of {_LABEL_MATCHES}, got {label_match!r}"
            )
        self.label_match = label_match
        self.leads = None if leads is None else [int(i) for i in leads]
        self.max_len = max_len
        self.token_len = token_len
        self.drop_empty_labels = drop_empty_labels
        super().__init__()

    @staticmethod
    def _read_adc(signal_file: str) -> tuple[np.ndarray, float]:
        """Reads one record's raw ADC values as ``(leads, samples)``."""
        try:
            import wfdb
        except ImportError as exc:
            raise ImportError(
                "Reading Chapman-Shaoxing waveforms requires the optional "
                "'wfdb' package. Install it with: pip install pyhealth[ptbxl]"
            ) from exc
        record = wfdb.rdrecord(str(signal_file), physical=False)
        return np.asarray(record.d_signal, dtype=np.float64).T, float(record.fs)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one sample per ECG record of a patient.

        Args:
            patient: A patient from
                :class:`~pyhealth.datasets.ChapmanShaoxingDataset`.

        Returns:
            Dicts with ``patient_id``, ``record_id``, ``ecg_sentence``
            (float32 ``(max_len, token_len)``), ``lead_ids`` and ``time_ids``
            (``(max_len,)``) and ``labels``.
        """
        from pyhealth.models.heartlang import ecg_to_heartbeat_sentence

        samples: list[dict[str, Any]] = []
        for event in patient.get_events(event_type="records"):
            labels = heartlang_csn_labels(event.dx_codes, self.label_match)
            if not labels and self.drop_empty_labels:
                continue
            adc, rate = self._read_adc(event.signal_file)
            signal = preprocess_heartlang_csn(adc, sampling_rate=rate)
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


def load_heartlang_csn_split(csv_dir: str | os.PathLike) -> dict[str, set[str]]:
    """Reads HeartLang's CSN record lists.

    HeartLang ships ``chapman_train.csv``, ``chapman_val.csv`` and
    ``chapman_test.csv`` (16,546 / 1,860 / 4,620 records) in
    ``datasets/dataset_preprocess/CSN/`` of
    https://github.com/PKUDigitalHealth/HeartLang. Only their ``ecg_path``
    column is read; the record name (``JS00001``) is taken from it.

    Args:
        csv_dir: Folder holding the three CSV files.

    Returns:
        ``{"train": ids, "val": ids, "test": ids}``.

    Raises:
        FileNotFoundError: If a CSV file is missing.
        ValueError: If a record is in more than one split.

    Examples:
        >>> from pyhealth.tasks.chapman_heartlang import load_heartlang_csn_split
        >>> ids = load_heartlang_csn_split(  # doctest: +SKIP
        ...     "HeartLang/datasets/dataset_preprocess/CSN")
        >>> len(ids["train"])  # doctest: +SKIP
        16546
    """
    import pandas as pd

    ids: dict[str, set[str]] = {}
    for split in _SPLITS:
        path = Path(csv_dir) / f"chapman_{split}.csv"
        paths = pd.read_csv(path, usecols=["ecg_path"])["ecg_path"].astype(str)
        ids[split] = {m.group(0) for p in paths if (m := _RECORD_ID.search(p))}
    for i, first in enumerate(_SPLITS):
        for second in _SPLITS[i + 1 :]:
            shared = ids[first] & ids[second]
            if shared:
                raise ValueError(
                    f"{len(shared)} records are in both {first} and {second}"
                )
    return ids


def split_by_heartlang_csn(
    dataset: Any,
    csv_dir: str | os.PathLike,
    record_ids: Sequence[str] | None = None,
):
    """Splits a sample dataset into HeartLang's CSN train/val/test sets.

    Samples whose record is in none of HeartLang's lists are left out (the
    lists cover 23,026 of the 45,152 records), and their number is logged.

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` built with
            :class:`ChapmanShaoxingHeartLangClassification` (or any object
            with ``subset`` and samples carrying ``record_id``).
        csv_dir: Folder with HeartLang's CSV files, see
            :func:`load_heartlang_csn_split`.
        record_ids: Optional record id of every sample, in dataset order, to
            avoid decoding each sample.

    Returns:
        tuple: ``(train_dataset, val_dataset, test_dataset)`` subsets.

    Raises:
        ValueError: If ``record_ids`` has the wrong length.

    Examples:
        >>> from pyhealth.tasks.chapman_heartlang import split_by_heartlang_csn
        >>> train, val, test = split_by_heartlang_csn(  # doctest: +SKIP
        ...     samples, "HeartLang/datasets/dataset_preprocess/CSN")
    """
    ids = load_heartlang_csn_split(csv_dir)
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
        logger.info("%d samples are in no HeartLang CSN split", unused)
    return tuple(dataset.subset(index[split]) for split in _SPLITS)
