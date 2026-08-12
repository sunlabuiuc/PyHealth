"""
PyHealth tasks for PTB-XL multi-label ECG diagnosis.

Dataset link:
    https://physionet.org/content/ptb-xl/1.0.3/

Dataset paper: (please cite if you use this dataset)
    Patrick Wagner, Nils Strodthoff, Ralf-Dieter Bousseljot, Dieter Kreiseler,
    Fatima I. Lunze, Wojciech Samek, and Tobias Schaeffter. "PTB-XL, a large
    publicly available electrocardiography dataset." Scientific Data 7, 154
    (2020).

Dataset paper link:
    https://www.nature.com/articles/s41597-020-0495-6

Author:
    AxelNoun (GitHub: @AxelNoun) — external contributor, no NetID

Description:
    Implements the official 5-diagnostic-superclass multi-label task and
    aggregation helpers. Separable for a future ``pyhealth.benchmarks``
    package. Official fold splitting lives in
    ``pyhealth.datasets.split_by_strat_fold``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from pyhealth.data import Patient
from pyhealth.tasks.base_task import BaseTask

# Official diagnostic superclasses (duplicated here to avoid importing the
# dataset package at module import time — that would circular-import via
# ``datasets.__init__`` → ``BaseDataset`` → ``tasks``).
PTBXL_DIAGNOSTIC_SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")

# Scientific Data (Wagner et al., Table 9): after aggregating diagnostic
# statements to the 5 superclasses, 407 of 21,799 records have an empty
# label set (mainly pacemaker ECGs with form/rhythm-only annotations).
# Default drop_empty_labels=True matches the common PTB-XL benchmarking
# practice and avoids all-zero multi-hot targets that break most
# conformal / nonconformity scores used in CPBench.
PTBXL_EMPTY_SUPERCLASS_COUNT = 407
PTBXL_TOTAL_RECORDS = 21799


def load_diagnostic_class_map(
    scp_statements_path: str | Path,
) -> dict[str, str]:
    """Map SCP acronym → ``diagnostic_class`` for diagnostic statements only.

    Called when constructing :class:`PTBXLSuperclassClassification`.

    Args:
        scp_statements_path (str | Path): Path to official ``scp_statements.csv``.

    Returns:
        dict[str, str]: Mapping used for 5-superclass aggregation.

    Examples:
        >>> # doctest: +SKIP
        >>> mapping = load_diagnostic_class_map("/data/ptb-xl/1.0.3/scp_statements.csv")
        >>> mapping["NORM"]
        'NORM'
    """
    path = Path(scp_statements_path)
    if not path.is_file():
        raise FileNotFoundError(f"scp_statements.csv not found: {path}")
    df = pd.read_csv(path, index_col=0)
    if "diagnostic" not in df.columns or "diagnostic_class" not in df.columns:
        raise ValueError(
            f"{path} must contain 'diagnostic' and 'diagnostic_class' columns"
        )
    diag = df[df["diagnostic"] == 1]
    mapping: dict[str, str] = {}
    for code, row in diag.iterrows():
        cls = row["diagnostic_class"]
        if pd.isna(cls):
            continue
        mapping[str(code)] = str(cls)
    return mapping


def aggregate_diagnostic_superclasses(
    scp_codes: Any,
    diagnostic_class_map: Mapping[str, str],
) -> list[str]:
    """Aggregate stringified ``scp_codes`` to unique diagnostic superclasses.

    Includes every SCP key present in the dict (likelihood ``0`` = unknown
    confidence still counts). Only codes with ``diagnostic == 1`` in
    ``scp_statements.csv`` contribute a superclass.

    Args:
        scp_codes (Any): Raw / stringified SCP dict from PTB-XL metadata.
        diagnostic_class_map (Mapping[str, str]): From
            :func:`load_diagnostic_class_map`.

    Returns:
        list[str]: Sorted unique superclass labels (subset of
        NORM/MI/STTC/CD/HYP).

    Examples:
        >>> aggregate_diagnostic_superclasses(
        ...     "{'IMI': 80.0, 'SR': 0.0}", {"IMI": "MI"}
        ... )
        ['MI']
    """
    from pyhealth.datasets.ptbxl import parse_scp_codes

    codes = parse_scp_codes(scp_codes)
    labels = {
        diagnostic_class_map[code]
        for code in codes
        if code in diagnostic_class_map
    }
    return sorted(labels)


class PTBXLSuperclassClassification(BaseTask):
    """5-superclass multi-label classification on PTB-XL.

    Labels are the official diagnostic superclasses
    ``NORM``, ``MI``, ``STTC``, ``CD``, ``HYP`` obtained by mapping diagnostic
    SCP statements via ``scp_statements.csv``.

    Empty label sets
    ----------------
    After aggregation, **407 / 21,799** records have no diagnostic superclass
    (Scientific Data Table 9; mainly pacemaker ECGs). By default these are
    **dropped** (``drop_empty_labels=True``) because an all-zero multi-hot
    target breaks typical multi-label nonconformity scores used in CPBench,
    and matches common PTB-XL literature practice.

    Each sample also carries ``strat_fold``, ``site``, ``device``, ``age``,
    ``sex``, ``age_is_censored``, and ``age_is_missing`` for official splits
    and downstream shift evaluations.

    Note:
        71-SCP multi-label classification and age regression are intentionally
        not implemented here; add them in this module in a follow-up PR.

    Args:
        scp_statements_path (str | Path | None): Path to ``scp_statements.csv``.
            Required unless set later via the dataset's ``default_task``.
        drop_empty_labels (bool): Drop records with no superclass after
            aggregation. Defaults to ``True``.
        diagnostic_superclasses (Sequence[str]): Label vocabulary order
            (defaults to the official five).

    Examples:
        >>> # doctest: +SKIP
        >>> from pyhealth.datasets import PTBXLDataset
        >>> from pyhealth.tasks import PTBXLSuperclassClassification
        >>> ds = PTBXLDataset(root="/data/ptb-xl/1.0.3")
        >>> samples = ds.set_task(PTBXLSuperclassClassification(
        ...     scp_statements_path="/data/ptb-xl/1.0.3/scp_statements.csv"
        ... ))
    """

    task_name: str = "PTBXLSuperclassClassification"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"labels": "multilabel"}

    def __init__(
        self,
        scp_statements_path: str | Path | None = None,
        drop_empty_labels: bool = True,
        diagnostic_superclasses: Sequence[str] = PTBXL_DIAGNOSTIC_SUPERCLASSES,
    ) -> None:
        self.scp_statements_path = (
            Path(scp_statements_path) if scp_statements_path is not None else None
        )
        self.drop_empty_labels = drop_empty_labels
        self.diagnostic_superclasses = tuple(diagnostic_superclasses)
        self._diagnostic_class_map: dict[str, str] | None = None
        super().__init__()

    def _class_map(self) -> dict[str, str]:
        if self._diagnostic_class_map is None:
            if self.scp_statements_path is None:
                raise ValueError(
                    "scp_statements_path is required. Pass it to "
                    "PTBXLSuperclassClassification(...) or use "
                    "PTBXLDataset.default_task."
                )
            self._diagnostic_class_map = load_diagnostic_class_map(
                self.scp_statements_path
            )
        return self._diagnostic_class_map

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Build one sample per ECG record for the patient."""
        from pyhealth.datasets.ptbxl import load_ptbxl_record

        class_map = self._class_map()
        samples: list[dict[str, Any]] = []
        for event in patient.get_events(event_type="records"):
            labels = aggregate_diagnostic_superclasses(event.scp_codes, class_map)
            if not labels and self.drop_empty_labels:
                continue

            signal = load_ptbxl_record(event.signal_file)
            age_raw = event.age
            age_missing = str(getattr(event, "age_is_missing", "0")) in {
                "1",
                "True",
                "true",
            }
            if age_missing or age_raw is None or str(age_raw).strip() == "":
                age: Any = None
            else:
                age = int(float(age_raw))

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": str(event.record_id),
                    "signal": signal,
                    "labels": labels,
                    "strat_fold": int(float(event.strat_fold)),
                    "site": event.site,
                    "device": event.device,
                    "age": age,
                    "age_is_censored": str(
                        getattr(event, "age_is_censored", "0")
                    )
                    in {"1", "True", "true"},
                    "age_is_missing": age_missing,
                    # sex: 0 = female, 1 = male
                    "sex": int(float(event.sex)),
                }
            )
        return samples
