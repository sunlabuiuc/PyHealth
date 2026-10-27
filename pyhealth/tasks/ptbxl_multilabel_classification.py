"""Multi-label ECG classification task for the PTB-XL dataset."""

import functools
import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from .base_task import BaseTask

_LABEL_SETS = (
    "all",
    "diagnostic",
    "subdiagnostic",
    "superdiagnostic",
    "form",
    "rhythm",
)


@functools.cache
def _load_scp_statements(path: str) -> pd.DataFrame:
    """Load and cache SCP statements CSV; ``path`` must be the resolved str."""
    return pd.read_csv(path, index_col=0)


def _aggregate_labels(
    scp_codes: dict[str, float],
    scp_statements: pd.DataFrame,
    label_set: str,
    min_likelihood: float,
) -> list[str]:
    """Aggregate SCP codes to the chosen label set with likelihood filtering.

    Generalises :func:`~pyhealth.tasks.ptbxl.aggregate_diagnostic_superclasses`
    (which only maps diagnostic codes to the 5 ``diagnostic_class`` superclasses
    and ignores the per-code likelihood field) to all six SCP-ECG label sets
    (``all``, ``diagnostic``, ``subdiagnostic``, ``superdiagnostic``, ``form``,
    ``rhythm``) and applies per-code ``min_likelihood`` filtering. A direct
    call-through isn't viable because the public helper doesn't take a label
    set or a likelihood threshold, so this reimplementation lives next to the
    task.

    For ``"subdiagnostic"`` and ``"superdiagnostic"`` the max likelihood of
    all contributing codes is used per parent class (matching the original
    PTB-XL paper protocol).

    Args:
        scp_codes: ``{code: likelihood}`` from a single ECG record.
        scp_statements: Loaded ``scp_statements.csv``, indexed by SCP code.
        label_set: One of the six sets in :data:`_LABEL_SETS`.
        min_likelihood: Labels whose effective likelihood is below this are
            dropped.

    Returns:
        Sorted list of unique label strings.
    """
    per_class: dict[str, float] = {}
    for code, likelihood in scp_codes.items():
        if code not in scp_statements.index:
            continue
        likelihood = float(likelihood)
        row = scp_statements.loc[code]
        if label_set == "all" or (label_set == "diagnostic" and row.get("diagnostic") == 1):
            per_class[code] = max(per_class.get(code, float("-inf")), likelihood)
        elif label_set == "subdiagnostic" and row.get("diagnostic") == 1:
            sub = row.get("diagnostic_subclass")
            if isinstance(sub, str) and sub:
                per_class[sub] = max(per_class.get(sub, float("-inf")), likelihood)
        elif label_set == "superdiagnostic" and row.get("diagnostic") == 1:
            sup = row.get("diagnostic_class")
            if isinstance(sup, str) and sup:
                per_class[sup] = max(per_class.get(sup, float("-inf")), likelihood)
        elif (label_set == "form" and row.get("form") == 1) or (
            label_set == "rhythm" and row.get("rhythm") == 1
        ):
            per_class[code] = max(per_class.get(code, float("-inf")), likelihood)
    return sorted(k for k, v in per_class.items() if v >= min_likelihood)


class PTBXLMultilabelClassification(BaseTask):
    """Multi-label classification of 12-lead ECGs from the PTB-XL dataset.

    Compatible with :class:`~pyhealth.datasets.PTBXLDataset`. Label
    aggregation is computed on-the-fly from the raw ``scp_codes`` stored in
    each event; ``scp_statements.csv`` is needed to map codes to the chosen
    label set. Use ``dataset.scp_statements_path`` to pass it.

    Compared to :class:`~pyhealth.tasks.PTBXLSuperclassClassification`, this
    task exposes all six SCP-ECG label sets (diagnostic, subdiagnostic,
    superdiagnostic, form, rhythm, all 71 statements) and the full signal
    processing pipeline used by ProtoECGNet (optional 0.5 Hz baseline removal,
    per-lead z-scoring).

    Args:
        scp_statements_path: Path to ``scp_statements.csv`` from the PTB-XL
            download. Use ``dataset.scp_statements_path``.
        label_set: Which labels to predict. One of ``"all"`` (71 SCP
            statements), ``"diagnostic"`` (44 statements),
            ``"subdiagnostic"`` (23 sub-classes), ``"superdiagnostic"``
            (5 super-classes, same as :class:`PTBXLSuperclassClassification`),
            ``"form"`` or ``"rhythm"``. Defaults to ``"diagnostic"``.
        min_likelihood: Keep statements whose annotated likelihood is at
            least this value (0 keeps everything, including
            unknown-confidence form/rhythm statements). Defaults to 0.0.
        standardize: ``"record"`` applies per-lead z-scoring within each
            record; ``None`` keeps raw millivolts. Defaults to ``"record"``.
        remove_baseline: If True, applies a 0.5 Hz high-pass Butterworth
            filter to remove baseline wander (used in the ProtoECGNet paper).
            Defaults to False.

    Attributes:
        task_name (str): ``"PTBXLMultilabelClassification"``.
        input_schema (dict): ``{"signal": "tensor"}``.
        output_schema (dict): ``{"labels": "multilabel"}``.

    Note:
        The ``sampling_rate`` is read from ``event.sampling_rate`` (set by
        :class:`~pyhealth.datasets.PTBXLDataset`), so it follows whatever rate
        the dataset was constructed with (100 or 500 Hz). There is no
        ``sampling_rate`` constructor argument.

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> from pyhealth.tasks import PTBXLMultilabelClassification
        >>> dataset = PTBXLDataset(root="/path/to/ptb-xl/1.0.3")  # doctest: +SKIP
        >>> task = PTBXLMultilabelClassification(  # doctest: +SKIP
        ...     scp_statements_path=dataset.scp_statements_path,
        ...     label_set="superdiagnostic",
        ... )
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> print(samples[0]["signal"].shape)  # doctest: +SKIP
        torch.Size([12, 1000])
    """

    task_name: str = "PTBXLMultilabelClassification"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"labels": "multilabel"}

    def __init__(
        self,
        scp_statements_path: str | os.PathLike,
        label_set: str = "diagnostic",
        min_likelihood: float = 0.0,
        standardize: str | None = "record",
        remove_baseline: bool = False,
    ) -> None:
        if label_set not in _LABEL_SETS:
            raise ValueError(f"label_set must be one of {_LABEL_SETS}")
        if standardize not in ("record", None):
            raise ValueError("standardize must be 'record' or None")
        # Resolve to a canonical string so the cache key is stable.
        self.scp_statements_path = str(Path(scp_statements_path).resolve())
        self.label_set = label_set
        self.min_likelihood = min_likelihood
        self.standardize = standardize
        self.remove_baseline = remove_baseline
        super().__init__()

    def _prepare_signal(self, signal_file: str, sampling_rate: int) -> np.ndarray:
        """Loads, crops/pads and normalizes one ECG.

        Signal loading goes through
        :func:`~pyhealth.datasets.ptbxl.load_ptbxl_record` — the canonical
        PTB-XL reader also used by
        :class:`~pyhealth.tasks.PTBXLSuperclassClassification` — so there is a
        single decoding path. ``wfdb`` is a hard requirement here; install
        the ``pyhealth[ptbxl]`` extra.

        Args:
            signal_file: Record path without extension (as stored in the event).
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
        if self.remove_baseline:
            from scipy.signal import butter, filtfilt

            b, a = butter(1, 0.5 / (sampling_rate / 2), btype="high")
            signal = filtfilt(b, a, signal, axis=1)
        if self.standardize == "record":
            mean = signal.mean(axis=1, keepdims=True)
            std = signal.std(axis=1, keepdims=True)
            signal = (signal - mean) / (std + 1e-6)
        return np.ascontiguousarray(signal, dtype=np.float32)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one sample per labelled ECG of a patient.

        For the ``superdiagnostic`` label set with ``min_likelihood == 0`` the
        aggregation delegates to
        :func:`~pyhealth.tasks.ptbxl.aggregate_diagnostic_superclasses` (used
        by :class:`~pyhealth.tasks.PTBXLSuperclassClassification`), so the
        labels produced by both tasks are identical for that configuration.
        Other label sets or non-zero likelihood thresholds use
        :func:`_aggregate_labels`.

        Args:
            patient: A patient object with ``records`` events from
                :class:`~pyhealth.datasets.PTBXLDataset`.

        Returns:
            A list of dicts with ``patient_id``, ``strat_fold``,
            ``signal`` (float32 array ``(12, T)``) and ``labels`` (list of
            label codes). ECGs with no label in the chosen set are skipped.
        """
        from pyhealth.datasets.ptbxl import parse_scp_codes

        if self.label_set == "superdiagnostic" and self.min_likelihood == 0.0:
            from pyhealth.tasks.ptbxl import (
                aggregate_diagnostic_superclasses,
                load_diagnostic_class_map,
            )

            class_map = load_diagnostic_class_map(self.scp_statements_path)

            def aggregate(scp_codes_raw: Any) -> list[str]:
                return aggregate_diagnostic_superclasses(scp_codes_raw, class_map)
        else:
            statements = _load_scp_statements(self.scp_statements_path)

            def aggregate(scp_codes_raw: Any) -> list[str]:
                return _aggregate_labels(
                    parse_scp_codes(scp_codes_raw),
                    statements,
                    self.label_set,
                    self.min_likelihood,
                )

        samples = []
        for event in patient.get_events(event_type="records"):
            labels = aggregate(event.scp_codes)
            if not labels:
                continue
            sampling_rate = int(event.sampling_rate)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    # int(float(...)) matches PTBXLSuperclassClassification and
                    # tolerates "9.0" strings from the CSV.
                    "strat_fold": int(float(event.strat_fold)),
                    "signal": self._prepare_signal(event.signal_file, sampling_rate),
                    "labels": labels,
                }
            )
        return samples
