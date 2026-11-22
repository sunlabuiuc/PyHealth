"""Multi-class (normal / target diagnosis / other) classification task
for the PTB-XL dataset.

Task definition follows the multi-class experiment of *In-depth Benchmarking
of Deep Neural Network Architectures for ECG Diagnosis* (Nonaka & Seita,
2021), which reproduces the protocol of Rajpurkar et al. (2022): every
record is assigned to exactly one of three classes — normal, the chosen
target diagnosis, or "other" — on top of the full SCP-ECG ("all") label
set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .base_task import BaseTask

# Fixed class order; index 0/1/2 matches the benchmark paper's loss layout.
PTBXL_MULTICLASS_CLASSES: tuple[str, ...] = ("normal", "target", "other")

# The benchmark's "Normal" is the PTB-XL ``NORM`` statement (normal ECG):
# 9,528 records (43.63% of v1.0.3), matching the Normal count of the
# benchmark paper's Table 2. ``SR`` (sinus rhythm) is a rhythm statement
# and is deliberately not used here.
PTBXL_MULTICLASS_NORMAL_CODE = "NORM"

# Target-diagnosis name → PTB-XL SCP-ECG statement code. ``STD`` has no
# SCP acronym in PTB-XL and is annotated with its SNOMED CT code
# (ST depression, 429622001). STE is omitted: the benchmark skipped it on
# PTB-XL (the paired statement is not used there).
PTBXL_MULTICLASS_TARGET_DX: dict[str, str] = {
    "AF": "AFIB",
    "IAVB": "IAVB",
    "LBBB": "LBBB",
    "RBBB": "RBBB",
    "PAC": "PAC",
    "PVC": "PVC",
    "STD": "429622001",
}

# The benchmark's five split settings: split_number -> (val_fold,
# test_fold). All remaining folds are training. Pass to
# :func:`pyhealth.datasets.split_by_strat_fold` (train_folds = complement).
PTBXL_PAPER_FOLD_SPLITS: dict[int, tuple[int, int]] = {
    1: (9, 10),
    2: (7, 8),
    3: (5, 6),
    4: (3, 4),
    5: (1, 2),
}


class PTBXLMulticlassClassification(BaseTask):
    """3-way classification (normal / target dx / other) of PTB-XL ECGs.

    Compatible with :class:`~pyhealth.datasets.PTBXLDataset`. Each labelled
    record yields one sample labelled ``"normal"``, ``"target"``, or
    ``"other"``. A record matching the target statement is labelled
    ``"target"`` **even if** it also carries the normal statement
    (precedence verified against the benchmark's label transform); records
    with an empty ``scp_codes`` dict are dropped (the benchmark runs the
    multi-class experiment on the "all" subset, which excludes them).

    Per-lead train-statistics normalization (``standardize="train_stats"``)
    mirrors the benchmark protocol: compute statistics from the training
    folds with
    :func:`~pyhealth.datasets.compute_ptbxl_lead_stats` **before** calling
    ``set_task``, then pass the artifact as ``stats_path``. Class weights
    for the imbalanced 3-way loss (``[1, n_target/n_normal,
    n_other/n_normal]`` in the benchmark) are intentionally **not**
    produced here: they depend on the train split, so compute them from
    the training :class:`~pyhealth.datasets.SampleDataset`.

    Args:
        target_dx: Target diagnosis name. One of ``"AF"``, ``"IAVB"``,
            ``"LBBB"``, ``"RBBB"``, ``"PAC"``, ``"PVC"``, ``"STD"`` (see
            :data:`PTBXL_MULTICLASS_TARGET_DX`).
        standardize: ``"record"`` per-lead z-scoring within each record;
            ``"train_stats"`` per-lead z-scoring with mean/std fitted on
            the training folds only (requires ``stats_path``); ``None``
            keeps raw millivolts. Defaults to ``"record"``.
        stats_path: Path to a per-lead train-statistics ``.npz`` artifact
            from :func:`~pyhealth.datasets.compute_ptbxl_lead_stats`.
            Required when ``standardize="train_stats"``, ignored otherwise.

    Attributes:
        task_name (str): ``"PTBXLMulticlassClassification"``.
        input_schema (dict): ``{"signal": "tensor"}``.
        output_schema (dict): ``{"label": ("multiclass", {"labels":
            ["normal", "target", "other"]})}`` — fixed class order, so the
            processor index is stable across runs.

    Note:
        ``sampling_rate`` is read from ``event.sampling_rate``; there is no
        constructor argument. Signals are cropped / zero-padded to
        ``10 * sampling_rate`` samples. Random 2.5 s subsampling (train)
        and sliding-window evaluation are model-side transforms and are not
        applied here.

    Examples:
        >>> from pyhealth.datasets import (  # doctest: +SKIP
        ...     PTBXLDataset, compute_ptbxl_lead_stats, split_by_strat_fold)
        >>> from pyhealth.tasks import PTBXLMulticlassClassification
        >>> dataset = PTBXLDataset(
        ...     root="/data/ptb-xl/1.0.3", sampling_rate=500)
        >>> stats = compute_ptbxl_lead_stats(dataset, train_folds=(1, 2, 3,
        ...     4, 5, 6, 9, 10))  # paper split 1: val 9, test 10
        >>> task = PTBXLMulticlassClassification(
        ...     target_dx="AF", standardize="train_stats", stats_path=stats)
        >>> samples = dataset.set_task(task)
        >>> train, val, test = split_by_strat_fold(
        ...     samples, train_folds=(1, 2, 3, 4, 5, 6, 9, 10),
        ...     val_folds=(9,), test_folds=(10,))
    """

    task_name: str = "PTBXLMulticlassClassification"
    input_schema: ClassVar[dict[str, Any]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, Any]] = {
        "label": ("multiclass", {"labels": list(PTBXL_MULTICLASS_CLASSES)})
    }

    def __init__(
        self,
        target_dx: str,
        standardize: str | None = "record",
        stats_path: str | os.PathLike | None = None,
    ) -> None:
        if target_dx not in PTBXL_MULTICLASS_TARGET_DX:
            raise ValueError(
                f"target_dx must be one of "
                f"{sorted(PTBXL_MULTICLASS_TARGET_DX)}, got {target_dx!r}"
            )
        if standardize not in ("record", "train_stats", None):
            raise ValueError(
                "standardize must be 'record', 'train_stats', or None"
            )
        self.target_dx = target_dx
        self.target_code = PTBXL_MULTICLASS_TARGET_DX[target_dx]
        self.standardize = standardize
        self.stats_path = (
            str(Path(stats_path).resolve()) if stats_path is not None else None
        )
        self._lead_mean: np.ndarray | None = None
        self._lead_std: np.ndarray | None = None
        if standardize == "train_stats":
            if self.stats_path is None:
                raise ValueError(
                    "stats_path is required when "
                    "standardize='train_stats'. Compute it with "
                    "pyhealth.datasets.compute_ptbxl_lead_stats(dataset, "
                    "train_folds=...) before constructing the task."
                )
            from pyhealth.datasets.ptbxl import load_ptbxl_lead_stats

            self._lead_mean, self._lead_std = load_ptbxl_lead_stats(
                self.stats_path
            )
        super().__init__()

    def _prepare_signal(self, signal_file: str, sampling_rate: int) -> np.ndarray:
        """Load one record and crop/pad to ``10 * sampling_rate`` samples.

        Signal loading goes through
        :func:`~pyhealth.datasets.ptbxl.load_ptbxl_record` so all PTB-XL
        tasks share one decoding path.

        Args:
            signal_file: Record path without extension (as stored in the
                event).
            sampling_rate: Waveform sampling rate in Hz (100 or 500).

        Returns:
            Float32 array of shape ``(12, 10 * sampling_rate)``.
        """
        from pyhealth.datasets.ptbxl import load_ptbxl_record

        signal = load_ptbxl_record(signal_file)  # (leads, samples)
        length = 10 * sampling_rate
        if signal.shape[1] < length:
            signal = np.pad(signal, ((0, 0), (0, length - signal.shape[1])))
        signal = signal[:, :length]
        if self.standardize == "record":
            mean = signal.mean(axis=1, keepdims=True)
            std = signal.std(axis=1, keepdims=True)
            signal = (signal - mean) / (std + 1e-6)
        elif self.standardize == "train_stats":
            signal = (signal - self._lead_mean[:, None]) / (
                self._lead_std[:, None] + 1e-6
            )
        return np.ascontiguousarray(signal, dtype=np.float32)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one multi-class sample per labelled ECG of a patient.

        Args:
            patient: A patient object with ``records`` events from
                :class:`~pyhealth.datasets.PTBXLDataset`.

        Returns:
            A list of dicts with ``patient_id``, ``record_id``,
            ``strat_fold``, ``signal`` (float32 array ``(12, T)``) and
            ``label`` (one of ``"normal"``, ``"target"``, ``"other"``).
            Records with no SCP statements at all are skipped.
        """
        from pyhealth.datasets.ptbxl import parse_scp_codes

        samples = []
        for event in patient.get_events(event_type="records"):
            codes = parse_scp_codes(event.scp_codes)
            if not codes:
                continue
            if self.target_code in codes:
                label = "target"
            elif PTBXL_MULTICLASS_NORMAL_CODE in codes:
                label = "normal"
            else:
                label = "other"
            sampling_rate = int(event.sampling_rate)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": str(event.record_id),
                    "strat_fold": int(float(event.strat_fold)),
                    "signal": self._prepare_signal(
                        event.signal_file, sampling_rate
                    ),
                    "label": label,
                }
            )
        return samples
