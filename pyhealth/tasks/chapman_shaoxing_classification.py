"""
PyHealth tasks for the Chapman-Shaoxing / Ningbo 12-lead ECG database.

Dataset link:
    https://physionet.org/content/ecg-arrhythmia/1.0.0/

Dataset paper: (please cite if you use this dataset)
    Jianwei Zheng, Hangyuan Guo, and Huimin Chu. "A large scale 12-lead
    electrocardiogram database for arrhythmia study" (version 1.0.0).
    PhysioNet (2022). https://doi.org/10.13026/wgex-er52

Description:
    Implements two multi-label tasks over the ``#Dx`` SNOMED-CT codes of each
    record: the seven CLIMB superclasses
    (:class:`ChapmanShaoxingSuperclassClassification`) and the raw SNOMED-CT
    codes of the release's ``ConditionNames_SNOMED-CT.csv``
    (:class:`ChapmanShaoxingSNOMEDClassification`). Both pin their label
    vocabulary, so the multi-hot layout does not depend on which labels occur
    in a subset.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np

from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

#: The seven CLIMB superclasses, in multi-hot index order.
CHAPMAN_SUPERCLASSES: tuple[str, ...] = (
    "normal",
    "cd",
    "mi",
    "sttc",
    "other",
    "afib",
    "hyp",
)

#: The unique SNOMED-CT codes of ``ConditionNames_SNOMED-CT.csv`` (release
#: 1.0.0), with the first acronym the file gives for each. Several acronyms
#: share a code (e.g. LBBB / LBBBB / LFBBB -> 164909002; the MI variants ->
#: 164865005; IDC / IVB -> 698252002; WAVN / SAAWR -> 195101003).
CHAPMAN_SNOMED_CODES: dict[str, str] = {
    "270492004": "1AVB",
    "195042002": "2AVB",
    "54016002": "2AVB1",
    "28189009": "2AVB2",
    "27885002": "3AVB",
    "251173003": "ABI",
    "39732003": "ALS",
    "284470004": "APB",
    "164917005": "AQW",
    "47665007": "ARS",
    "233917008": "AVB",
    "251199005": "CCR",
    "251198002": "CR",
    "428417006": "ERV",
    "164942001": "FQRS",
    "698252002": "IDC",
    "426995002": "JEB",
    "251164006": "JPT",
    "164909002": "LBBB",
    "164873001": "LVH",
    "251146004": "LVQRSAL",
    "251148003": "LVQRSCL",
    "251147008": "LVQRSLL",
    "164865005": "MI",
    "164947007": "PRIE",
    "164912004": "PWC",
    "111975006": "QTIE",
    "446358003": "RAH",
    "59118001": "RBBB",
    "89792004": "RVH",
    "429622005": "STDD",
    "164930006": "STE",
    "428750005": "STTC",
    "164931005": "STTU",
    "164934002": "TWC",
    "59931005": "TWO",
    "164937009": "UW",
    "11157007": "VB",
    "75532003": "VEB",
    "13640000": "VFW",
    "17338001": "VPB",
    "195060002": "VPE",
    "251180001": "VET",
    "195101003": "WAVN",
    "74390002": "WPW",
    "426177001": "SB",
    "426783006": "SR",
    "164889003": "AFIB",
    "427084000": "ST",
    "164890007": "AF",
    "427393009": "SA",
    "426761007": "SVT",
    "713422000": "AT",
    "233896004": "AVNRT",
    "233897008": "AVRT",
}

#: SNOMED-CT code -> CLIMB superclass, restating CLIMB's grouping as data.
#: RBBB (59118001) is added to ``cd`` here (see ``include_rbbb_in_cd``).
#: Codes absent from this table (including codes found in the full 45k
#: release but not in ``ConditionNames_SNOMED-CT.csv``) are ignored.
CHAPMAN_SNOMED_TO_SUPERCLASS: dict[str, str] = {
    # normal: sinus bradycardia, sinus rhythm, sinus tachycardia
    "426177001": "normal",
    "426783006": "normal",
    "427084000": "normal",
    # cd: conduction disturbances (CLIMB also puts some rhythms here)
    "270492004": "cd",
    "195042002": "cd",
    "233917008": "cd",
    "251166008": "cd",
    "713422000": "cd",
    "27885002": "cd",
    "164909002": "cd",
    "54016002": "cd",
    "698252002": "cd",
    "426995002": "cd",
    "251164006": "cd",
    "164947007": "cd",
    "251146004": "cd",  # CLIMB: "PRWP"; ConditionNames: low QRS voltage
    "164912004": "cd",
    "17366009": "cd",
    "426761007": "cd",
    "75532003": "cd",
    "251180001": "cd",
    "17338001": "cd",
    "195060002": "cd",
    "195101003": "cd",
    "74390002": "cd",
    "59118001": "cd",  # RBBB; CLIMB drops it through a spelling mismatch
    # mi
    "164865005": "mi",
    # sttc
    "429622005": "sttc",
    "164930006": "sttc",
    "428750005": "sttc",
    "164931005": "sttc",
    "164934002": "sttc",
    "59931005": "sttc",
    # afib: atrial fibrillation and flutter
    "164890007": "afib",
    "164889003": "afib",
    # hyp: hypertrophy, plus CLIMB's axis shifts, rotation and AVRT
    "39732003": "hyp",
    "47665007": "hyp",
    "251198002": "hyp",
    "164873001": "hyp",
    "55827005": "hyp",
    "446358003": "hyp",
    "233897008": "hyp",
    "89792004": "hyp",
    # other
    "251173003": "other",
    "284470004": "other",
    "164917005": "other",
    "428417006": "other",
    "164942001": "other",
    "111975006": "other",
    "164937009": "other",
    "11157007": "other",
}

RBBB_SNOMED = "59118001"


def split_dx_codes(dx_codes: Any) -> list[str]:
    """Split a ``;``- or ``,``-joined ``#Dx`` string into SNOMED codes.

    Args:
        dx_codes (Any): The ``dx_codes`` event attribute, e.g.
            ``"164889003;59118001"``. Missing values give ``[]``.

    Returns:
        list[str]: The codes, stripped, in the original order.

    Examples:
        >>> split_dx_codes("164889003;59118001")
        ['164889003', '59118001']
        >>> split_dx_codes(None)
        []
    """
    if isinstance(dx_codes, (int, np.integer)):
        return [str(int(dx_codes))]
    if isinstance(dx_codes, (float, np.floating)):
        return [] if np.isnan(dx_codes) else [str(int(dx_codes))]
    if not isinstance(dx_codes, str):
        return []
    text = dx_codes.replace(",", ";")
    return [code.strip() for code in text.split(";") if code.strip()]


def map_dx_to_superclasses(
    dx_codes: Any,
    include_rbbb_in_cd: bool = True,
    table: Mapping[str, str] | None = None,
) -> list[str]:
    """Map a record's SNOMED codes to its CLIMB superclasses.

    Called by :class:`ChapmanShaoxingSuperclassClassification`.

    Args:
        dx_codes (Any): ``;``-joined SNOMED codes from the metadata CSV.
        include_rbbb_in_cd (bool): Map RBBB (59118001) to ``cd``. CLIMB meant
            to but does not; ``False`` reproduces CLIMB's labels.
        table (Mapping[str, str] | None): Code -> superclass table. Defaults
            to :data:`CHAPMAN_SNOMED_TO_SUPERCLASS`.

    Returns:
        list[str]: Superclasses in :data:`CHAPMAN_SUPERCLASSES` order. Codes
        not in the table are ignored.

    Examples:
        >>> map_dx_to_superclasses("164889003;59118001;164934002")
        ['cd', 'sttc', 'afib']
        >>> map_dx_to_superclasses("164889003;59118001", include_rbbb_in_cd=False)
        ['afib']
    """
    table = CHAPMAN_SNOMED_TO_SUPERCLASS if table is None else table
    found = set()
    for code in split_dx_codes(dx_codes):
        if code == RBBB_SNOMED and not include_rbbb_in_cd:
            continue
        if code in table:
            found.add(table[code])
    return [name for name in CHAPMAN_SUPERCLASSES if name in found]


def count_unmapped_codes(
    dx_code_strings: Iterable[Any],
    known: Iterable[str] | None = None,
) -> Counter:
    """Count SNOMED codes that a label table does not cover.

    The full 45k release uses codes that ``ConditionNames_SNOMED-CT.csv`` and
    CLIMB's table omit (for example 713427006, complete RBBB). The tasks
    ignore them; this helper reports how many there are.

    Args:
        dx_code_strings (Iterable[Any]): ``dx_codes`` values, one per record.
        known (Iterable[str] | None): Codes the table covers. Defaults to the
            keys of :data:`CHAPMAN_SNOMED_TO_SUPERCLASS`.

    Returns:
        Counter: Unmapped code -> number of records carrying it.

    Examples:
        >>> count_unmapped_codes(["426783006;713427006", "713427006"])
        Counter({'713427006': 2})
    """
    known_set = set(CHAPMAN_SNOMED_TO_SUPERCLASS if known is None else known)
    counts: Counter = Counter()
    for value in dx_code_strings:
        for code in set(split_dx_codes(value)):
            if code not in known_set:
                counts[code] += 1
    return counts


def load_chapman_record(signal_file: str, standardize: bool = True) -> np.ndarray:
    """Load one WFDB record as a ``(12, n_samples)`` float32 array in mV.

    Args:
        signal_file (str): WFDB record base path (no ``.hea``/``.mat``).
        standardize (bool): Z-score each lead over time (a lead with zero
            variance is only centred).

    Returns:
        np.ndarray: Signal of shape ``(n_leads, n_samples)``.

    Raises:
        ImportError: If the optional ``wfdb`` package is not installed.

    Examples:
        >>> x = load_chapman_record("/data/WFDBRecords/01/010/JS00001")  # doctest: +SKIP
        >>> x.shape  # doctest: +SKIP
        (12, 5000)
    """
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "Reading Chapman-Shaoxing waveforms requires the optional 'wfdb' "
            "package. Install it with: pip install wfdb"
        ) from exc
    signals, _ = wfdb.rdsamp(str(signal_file))
    x = np.asarray(signals, dtype=np.float32).T
    if standardize:
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        x = (x - mean) / np.where(std > 0, std, 1.0)
    return x.astype(np.float32)


class _ChapmanShaoxingTask(BaseTask):
    """Shared sample building for the Chapman-Shaoxing tasks."""

    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}

    def __init__(self, standardize: bool = True, drop_empty_labels: bool = True):
        self.standardize = standardize
        self.drop_empty_labels = drop_empty_labels
        super().__init__()

    def _labels(self, dx_codes: Any) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per ECG record of the patient."""
        samples: list[dict[str, Any]] = []
        for event in patient.get_events(event_type="records"):
            labels = self._labels(event.dx_codes)
            if not labels and self.drop_empty_labels:
                logger.debug(
                    "Skipping %s: no mapped label in %r",
                    event.record_id,
                    event.dx_codes,
                )
                continue
            signal = load_chapman_record(event.signal_file, self.standardize)
            if not np.isfinite(signal).all():
                logger.warning("Skipping %s: non-finite samples", event.record_id)
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": str(event.record_id),
                    "signal": signal,
                    "labels": labels,
                }
            )
        return samples


class ChapmanShaoxingSuperclassClassification(_ChapmanShaoxingTask):
    """Seven-superclass multi-label ECG classification (CLIMB grouping).

    Works with :class:`~pyhealth.datasets.ChapmanShaoxingDataset`.

    Each record's ``#Dx`` SNOMED-CT codes are mapped to the CLIMB superclasses
    ``normal, cd, mi, sttc, other, afib, hyp`` through
    :data:`CHAPMAN_SNOMED_TO_SUPERCLASS`. The vocabulary is pinned to that
    order, so the multi-hot vector always has seven entries even when a class
    (often ``mi``) is absent from a subset. Codes outside the table are
    ignored; :class:`~pyhealth.datasets.ChapmanShaoxingDataset` logs how many
    there are when it builds its metadata. The input is the 12 x 5000 (10 s,
    500 Hz) signal in mV, z-scored per lead by default.

    Deviations from CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB):

    - RBBB (59118001) goes to ``cd``. CLIMB lists ``"Nonspecific BBB"`` under
      ``cd`` while its code table spells it ``"nonspecific BBB"``, so CLIMB
      never assigns RBBB. ``include_rbbb_in_cd=False`` reproduces CLIMB.
    - Only multi-label targets are produced. CLIMB's extra single ``label``
      (a random choice among the positive classes) is not reproduced.
    - CLIMB scales raw ADC values by ``x / (max - min)``; here the signal is
      in physical units and z-scored per lead (``standardize=False`` keeps mV).
    - CLIMB's groupings are kept as is and are clinically questionable in
      places: 251146004 (low QRS voltage) is in ``cd``, AVRT (233897008) and
      the axis shifts / rotation are in ``hyp``, QT prolongation (111975006)
      is in ``other``. CCR (251199005), RAHV and VFW (13640000) belong to no
      class, as in CLIMB.

    Args:
        include_rbbb_in_cd (bool): Map RBBB to ``cd``. Defaults to ``True``.
        standardize (bool): Z-score each lead. Defaults to ``True``.
        drop_empty_labels (bool): Skip records with no mapped superclass.
            Defaults to ``True``.

    Examples:
        >>> task = ChapmanShaoxingSuperclassClassification()
        >>> task.output_schema["labels"][1]["labels"]
        ['normal', 'cd', 'mi', 'sttc', 'other', 'afib', 'hyp']
        >>> from pyhealth.datasets import ChapmanShaoxingDataset
        >>> ds = ChapmanShaoxingDataset(root="/data/ecg-arrhythmia/1.0.0")  # doctest: +SKIP
        >>> samples = ds.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "ChapmanShaoxingSuperclassClassification"
    output_schema: ClassVar[dict[str, Any]] = {
        "labels": ("multilabel", {"labels": list(CHAPMAN_SUPERCLASSES)})
    }

    def __init__(
        self,
        include_rbbb_in_cd: bool = True,
        standardize: bool = True,
        drop_empty_labels: bool = True,
    ) -> None:
        self.include_rbbb_in_cd = include_rbbb_in_cd
        super().__init__(standardize=standardize, drop_empty_labels=drop_empty_labels)

    def _labels(self, dx_codes: Any) -> list[str]:
        return map_dx_to_superclasses(dx_codes, self.include_rbbb_in_cd)


class ChapmanShaoxingSNOMEDClassification(_ChapmanShaoxingTask):
    """Multi-label classification over raw SNOMED-CT diagnosis codes.

    Works with :class:`~pyhealth.datasets.ChapmanShaoxingDataset`.

    The labels are the record's ``#Dx`` codes themselves. By default the
    vocabulary is pinned to the 55 unique codes of the release's
    ``ConditionNames_SNOMED-CT.csv`` (:data:`CHAPMAN_SNOMED_CODES`), in that
    file's order; pass ``codes`` to choose another set (e.g. the most frequent
    codes). Codes outside the vocabulary are ignored.

    Args:
        codes (Sequence[str] | None): SNOMED-CT codes to predict, in
            multi-hot order. Defaults to :data:`CHAPMAN_SNOMED_CODES`.
        standardize (bool): Z-score each lead. Defaults to ``True``.
        drop_empty_labels (bool): Skip records with no code in ``codes``.
            Defaults to ``True``.

    Examples:
        >>> task = ChapmanShaoxingSNOMEDClassification(codes=["426783006", "164889003"])
        >>> task.output_schema["labels"][1]["labels"]
        ['426783006', '164889003']
        >>> task._labels("164889003;713427006")
        ['164889003']
    """

    task_name: str = "ChapmanShaoxingSNOMEDClassification"
    output_schema: ClassVar[dict[str, Any]] = {
        "labels": ("multilabel", {"labels": list(CHAPMAN_SNOMED_CODES)})
    }

    def __init__(
        self,
        codes: Sequence[str] | None = None,
        standardize: bool = True,
        drop_empty_labels: bool = True,
    ) -> None:
        self.codes = [str(c) for c in (codes if codes is not None else CHAPMAN_SNOMED_CODES)]
        if len(set(self.codes)) != len(self.codes):
            raise ValueError("codes must be unique")
        self.output_schema = {"labels": ("multilabel", {"labels": list(self.codes)})}
        super().__init__(standardize=standardize, drop_empty_labels=drop_empty_labels)

    def _labels(self, dx_codes: Any) -> list[str]:
        present = set(split_dx_codes(dx_codes))
        return [code for code in self.codes if code in present]
