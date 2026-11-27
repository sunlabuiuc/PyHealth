"""Next-visit CCS diagnosis prediction for EviCare on MIMIC-III and MIMIC-IV.

Contributor: Kavish Shah (GitHub: thekavishshah)

Paper: Hengyu Zhang, Xuyun Zhang, Pengxiang Zhan, Linhao Luo, Hang Lv,
Yanchao Tan, Shirui Pan, Carl Yang. EviCare: Enhancing Diagnosis Prediction
with Deep Model-Guided Evidence for In-Context Reasoning. KDD 2026.
https://arxiv.org/abs/2604.10455, https://doi.org/10.1145/3770854.3780257

Official code: https://github.com/zhyccc/EviCare (no license file; this is an
independent re-implementation written from the paper).

Description:
    Given the diagnosis codes of a patient's past admissions, predict the CCS
    categories recorded at the next admission. ICD codes are mapped to
    single-level CCS categories. Each sample keeps, besides the CCS input of
    the guide model, the raw ICD codes, their names, their parent CCS and the
    time from each past admission to the predicted one, which
    :class:`~pyhealth.models.EviCare` turns into prompt evidence.

    Two settings, as in the paper:

    - ``"overall"``: the label is every CCS code of the next admission.
    - ``"novel"``: the label keeps only CCS codes that appear in none of the
      patient's earlier admissions. A target admission without a novel code
      gives no sample.

    By default every admission after the first is a target, predicted from
    all earlier admissions (``sample_mode="all_prefixes"``). This is the
    sampling of the TRANS preprocessing that EviCare's data pipeline builds
    on, and it matches the paper's visit-level P@k, averaged over all test
    visits. ``sample_mode="last"`` keeps only the last admission of each
    patient.

    EviCare's co-occurrence graph needs every CCS code of each training
    patient, which the samples do not always hold (novel targets drop seen
    codes, and some patients give no sample). The tasks therefore also
    return per-patient code sets read from the patients themselves
    (``patient_code_sets``).
"""

import csv
import hashlib
import math
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .base_task import BaseTask

Setting = Literal["overall", "novel"]
SETTINGS: tuple[str, ...] = ("overall", "novel")
SampleMode = Literal["all_prefixes", "last"]
SAMPLE_MODES: tuple[str, ...] = ("all_prefixes", "last")

_Visit = tuple[str, datetime, list[tuple[str, int, str]]]


def normalize_icd(code: Any) -> str | None:
    """Returns an ICD code without dots or surrounding spaces, upper case.

    MIMIC stores codes without dots (``"4280"``) while code tables often use
    them (``"428.0"``); both normalise to the same key.

    Args:
        code: An ICD code, or None.

    Returns:
        The normalised code, or None if the code is missing or empty.

    Examples:
        >>> normalize_icd(" 428.0 ")
        '4280'
        >>> normalize_icd("i50.9")
        'I509'
        >>> normalize_icd(None) is None
        True
    """
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return None
    text = str(code).replace(".", "").strip().upper()
    return text or None


def _normalize_version(version: Any) -> int | None:
    """Returns 9 or 10 for an ICD version value such as ``"9"`` or ``10.0``."""
    try:
        value = int(float(str(version).strip()))
    except (TypeError, ValueError):
        return None
    return value if value in (9, 10) else None


def _normalize_id(value: Any) -> str | None:
    """Returns an identifier as a string (``100.0`` -> ``"100"``), or None."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            value = int(value)
    text = str(value).strip()
    return text or None


class CCSMapping:
    """ICD-9/ICD-10 diagnosis code to CCS category mapping, with names.

    Holds plain dictionaries, so it pickles cheaply to ``set_task`` workers,
    and has a deterministic ``repr`` (a digest of its content), so PyHealth's
    task cache is keyed by the mapping actually used.

    Build one with :meth:`from_crossmap` (PyHealth's single-level CCS tables,
    the default of the EviCare tasks) or :meth:`from_csv` (any user-supplied
    table, for example CCSR built from the HCUP files).

    Args:
        icd9_to_ccs: ICD-9-CM code -> CCS code. Keys may contain dots.
        icd10_to_ccs: ICD-10-CM code -> CCS code. Keys may contain dots.
        ccs_names: CCS code -> category name.
        icd9_names: ICD-9-CM code -> description.
        icd10_names: ICD-10-CM code -> description.
        source: A short label of where the table came from, shown in the
            ``repr``.

    Examples:
        >>> mapping = CCSMapping(
        ...     icd9_to_ccs={"428.0": "108"},
        ...     ccs_names={"108": "Congestive heart failure; nonhypertensive"},
        ... )
        >>> mapping.map("4280")
        '108'
        >>> mapping.map("4280", version=10) is None
        True
        >>> mapping.ccs_name("108")
        'Congestive heart failure; nonhypertensive'
        >>> mapping.icd_name("4280")
        '4280'
    """

    def __init__(
        self,
        icd9_to_ccs: Mapping[str, str] | None = None,
        icd10_to_ccs: Mapping[str, str] | None = None,
        ccs_names: Mapping[str, str] | None = None,
        icd9_names: Mapping[str, str] | None = None,
        icd10_names: Mapping[str, str] | None = None,
        source: str = "custom",
    ) -> None:
        self.source = source
        self._to_ccs: dict[int, dict[str, str]] = {
            9: self._clean(icd9_to_ccs, value_is_code=True),
            10: self._clean(icd10_to_ccs, value_is_code=True),
        }
        self._icd_names: dict[int, dict[str, str]] = {
            9: self._clean(icd9_names),
            10: self._clean(icd10_names),
        }
        self._ccs_names = {
            str(k).strip(): str(v).strip() for k, v in (ccs_names or {}).items()
        }
        digest = hashlib.sha256()
        for part in (
            self._to_ccs[9],
            self._to_ccs[10],
            self._ccs_names,
            self._icd_names[9],
            self._icd_names[10],
        ):
            digest.update(repr(sorted(part.items())).encode("utf-8"))
        self._digest = digest.hexdigest()[:12]

    @staticmethod
    def _clean(
        table: Mapping[str, str] | None, value_is_code: bool = False
    ) -> dict[str, str]:
        """Normalises the ICD keys (and CCS values) of a table."""
        cleaned: dict[str, str] = {}
        for key, value in (table or {}).items():
            icd = normalize_icd(key)
            if icd is None or value is None:
                continue
            value = str(value).strip()
            if value:
                cleaned[icd] = value
        return cleaned

    @classmethod
    def from_csv(cls, path: str | Path) -> "CCSMapping":
        """Loads a mapping from a CSV file.

        Required columns: ``icd_code`` and ``ccs_code``. Optional columns:
        ``icd_version`` (``9`` or ``10``; rows without it count as ICD-9),
        ``ccs_name`` and ``icd_name``. ICD codes may be written with or
        without dots. Any CCS-like grouping works, for example CCSR
        categories (``CIR019``) built from the HCUP CCSR tables.

        Args:
            path: Path to the CSV file.

        Returns:
            The mapping.

        Raises:
            ValueError: If a required column is missing.

        Examples:
            >>> mapping = CCSMapping.from_csv(  # doctest: +SKIP
            ...     "test-resources/core/evicare/ccs_mapping_tiny.csv"
            ... )
            >>> mapping.map("I509", version=10)  # doctest: +SKIP
            '108'
        """
        to_ccs: dict[int, dict[str, str]] = {9: {}, 10: {}}
        icd_names: dict[int, dict[str, str]] = {9: {}, 10: {}}
        ccs_names: dict[str, str] = {}
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = {"icd_code", "ccs_code"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path} is missing required column(s): {sorted(missing)}."
                )
            for row in reader:
                version = _normalize_version(row.get("icd_version") or 9)
                if version is None:
                    continue
                icd, ccs = row["icd_code"], (row["ccs_code"] or "").strip()
                if not icd or not ccs:
                    continue
                to_ccs[version][icd] = ccs
                if row.get("icd_name"):
                    icd_names[version][icd] = row["icd_name"]
                if row.get("ccs_name"):
                    ccs_names[ccs] = row["ccs_name"]
        return cls(
            icd9_to_ccs=to_ccs[9],
            icd10_to_ccs=to_ccs[10],
            ccs_names=ccs_names,
            icd9_names=icd_names[9],
            icd10_names=icd_names[10],
            source=f"csv:{Path(path).name}",
        )

    @classmethod
    def from_crossmap(cls, versions: Iterable[int] = (9, 10)) -> "CCSMapping":
        """Builds the mapping from PyHealth's single-level CCS tables.

        Uses :class:`~pyhealth.medcode.CrossMap` (``ICD9CM`` / ``ICD10CM`` ->
        ``CCSCM``) and the code names of :class:`~pyhealth.medcode.InnerMap`.
        This is the ICD-9 to CCS table of the official EviCare MIMIC-III
        pipeline (285 CCS categories). The tables are downloaded on first use
        and cached by PyHealth.

        Args:
            versions: ICD versions to load. Default both.

        Returns:
            The mapping.

        Examples:
            >>> mapping = CCSMapping.from_crossmap(versions=(9,))  # doctest: +SKIP
            >>> mapping.map("4280")  # doctest: +SKIP
            '108'
        """
        from ..medcode import CrossMap, InnerMap

        vocabularies = {9: "ICD9CM", 10: "ICD10CM"}
        to_ccs: dict[int, dict[str, str]] = {9: {}, 10: {}}
        icd_names: dict[int, dict[str, str]] = {9: {}, 10: {}}
        for version in versions:
            vocabulary = vocabularies[version]
            cross = CrossMap.load(vocabulary, "CCSCM")
            graph = InnerMap.load(vocabulary).graph
            for icd, targets in cross.mapping.items():
                if not targets:
                    continue
                to_ccs[version][icd] = targets[0]
                if icd in graph.nodes:
                    name = graph.nodes[icd].get("name")
                    if name:
                        icd_names[version][icd] = name
        ccs_graph = InnerMap.load("CCSCM").graph
        ccs_names = {
            code: data["name"]
            for code, data in ccs_graph.nodes(data=True)
            if data.get("name")
        }
        return cls(
            icd9_to_ccs=to_ccs[9],
            icd10_to_ccs=to_ccs[10],
            ccs_names=ccs_names,
            icd9_names=icd_names[9],
            icd10_names=icd_names[10],
            source="crossmap:CCSCM",
        )

    def map(self, code: Any, version: int = 9) -> str | None:
        """Returns the CCS code of an ICD code, or None if it is unmapped.

        Args:
            code: The ICD code, with or without dots.
            version: 9 or 10.

        Returns:
            The CCS code, or None.
        """
        icd = normalize_icd(code)
        if icd is None:
            return None
        return self._to_ccs.get(version, {}).get(icd)

    def ccs_name(self, ccs: str) -> str:
        """Returns the name of a CCS category, or the code if it has none."""
        return self._ccs_names.get(str(ccs), str(ccs))

    def icd_name(self, code: Any, version: int = 9) -> str:
        """Returns the description of an ICD code, or the code if it has none."""
        icd = normalize_icd(code) or str(code)
        return self._icd_names.get(version, {}).get(icd, icd)

    @property
    def ccs_codes(self) -> list[str]:
        """All CCS codes that some ICD code maps to, sorted."""
        return sorted(set(self._to_ccs[9].values()) | set(self._to_ccs[10].values()))

    @property
    def ccs_names(self) -> dict[str, str]:
        """A copy of the CCS code -> name table."""
        return dict(self._ccs_names)

    def __len__(self) -> int:
        return len(self._to_ccs[9]) + len(self._to_ccs[10])

    def __repr__(self) -> str:
        return (
            f"CCSMapping(source={self.source!r}, icd9={len(self._to_ccs[9])}, "
            f"icd10={len(self._to_ccs[10])}, digest={self._digest})"
        )


def _resolve_mapping(
    ccs_mapping: "CCSMapping | str | Path | None", versions: tuple[int, ...]
) -> CCSMapping:
    """Turns the ``ccs_mapping`` task argument into a :class:`CCSMapping`."""
    if ccs_mapping is None:
        return CCSMapping.from_crossmap(versions=versions)
    if isinstance(ccs_mapping, CCSMapping):
        return ccs_mapping
    if isinstance(ccs_mapping, (str, Path)):
        return CCSMapping.from_csv(ccs_mapping)
    raise TypeError(
        "ccs_mapping must be None, a CSV path or a CCSMapping, "
        f"got {type(ccs_mapping).__name__}."
    )


class _EviCareDiagnosisPrediction(BaseTask):
    """Shared logic of the EviCare diagnosis prediction tasks.

    Subclasses say how to read the ICD code and version of one diagnosis
    event (:meth:`_event_code`) and which ICD versions they need.

    Args:
        setting: ``"overall"`` or ``"novel"``.
        ccs_mapping: None (PyHealth's single-level CCS tables), a path to a
            CSV file (see :meth:`CCSMapping.from_csv`) or a
            :class:`CCSMapping`.
        min_visits: Minimum number of admissions with at least one mapped
            diagnosis. At least 2 (one history visit and the target visit).
        sample_mode: ``"all_prefixes"`` (one sample per admission after the
            first) or ``"last"`` (one sample per patient).
    """

    _icd_versions: tuple[int, ...] = (9,)

    def __init__(
        self,
        setting: Setting = "overall",
        ccs_mapping: "CCSMapping | str | Path | None" = None,
        min_visits: int = 2,
        sample_mode: SampleMode = "all_prefixes",
    ) -> None:
        if setting not in SETTINGS:
            raise ValueError(f"setting must be one of {SETTINGS}, got {setting!r}.")
        if min_visits < 2:
            raise ValueError(f"min_visits must be at least 2, got {min_visits}.")
        if sample_mode not in SAMPLE_MODES:
            raise ValueError(
                f"sample_mode must be one of {SAMPLE_MODES}, got {sample_mode!r}."
            )
        self.setting = setting
        self.min_visits = min_visits
        self.sample_mode = sample_mode
        self.ccs_mapping = _resolve_mapping(ccs_mapping, self._icd_versions)
        self.input_schema = {"conditions": "nested_sequence"}
        self.output_schema = {"ccs_targets": "multilabel"}
        super().__init__()

    def _event_code(self, event: Any) -> tuple[str, int] | None:
        """Returns ``(icd_code, icd_version)`` of a diagnosis event, or None."""
        raise NotImplementedError

    def _visits(self, patient: Any) -> list[_Visit]:
        """Builds the patient's visits, oldest first.

        Each visit is ``(hadm_id, admittime, [(icd, version, ccs), ...])``.
        ICD codes without a CCS category are dropped, duplicate ICD codes in a
        visit keep their first occurrence, and admissions left without codes
        or without an admission time are dropped.

        Args:
            patient: A :class:`~pyhealth.data.Patient`.

        Returns:
            The visits sorted by admission time.
        """
        by_visit: dict[str, list[tuple[str, int, str]]] = {}
        for event in patient.get_events(event_type="diagnoses_icd"):
            parsed = self._event_code(event)
            hadm_id = _normalize_id(getattr(event, "hadm_id", None))
            if parsed is None or hadm_id is None:
                continue
            icd, version = parsed
            ccs = self.ccs_mapping.map(icd, version)
            if ccs is None:
                continue
            codes = by_visit.setdefault(hadm_id, [])
            if all((icd, version) != (c[0], c[1]) for c in codes):
                codes.append((icd, version, ccs))

        visits = []
        for admission in patient.get_events(event_type="admissions"):
            hadm_id = _normalize_id(getattr(admission, "hadm_id", None))
            codes = by_visit.get(hadm_id) if hadm_id is not None else None
            if codes and admission.timestamp is not None:
                visits.append((hadm_id, admission.timestamp, codes))
        visits.sort(key=lambda visit: visit[1])
        return visits

    def _make_sample(
        self, patient_id: str, history: list[_Visit], target: _Visit
    ) -> dict[str, Any] | None:
        """Builds the sample predicting ``target`` from ``history``.

        Args:
            patient_id: The patient identifier.
            history: The earlier visits, oldest first (at least one).
            target: The visit to predict.

        Returns:
            The sample, or None in the ``"novel"`` setting if the target has
            no CCS code absent from the history.
        """
        target_id, target_time, target_codes = target
        seen: set[str] = set()
        conditions = []
        for _, _, codes in history:
            visit_ccs = list(dict.fromkeys(ccs for _, _, ccs in codes))
            conditions.append(visit_ccs)
            seen.update(visit_ccs)

        targets = list(dict.fromkeys(ccs for _, _, ccs in target_codes))
        if self.setting == "novel":
            targets = [ccs for ccs in targets if ccs not in seen]
            if not targets:
                return None

        return {
            "patient_id": patient_id,
            "visit_id": target_id,
            "conditions": conditions,
            "ccs_targets": targets,
            "history_icd": [[icd for icd, _, _ in codes] for _, _, codes in history],
            "history_icd_names": [
                [self.ccs_mapping.icd_name(icd, ver) for icd, ver, _ in codes]
                for _, _, codes in history
            ],
            "history_ccs": [[ccs for _, _, ccs in codes] for _, _, codes in history],
            "days_ago": [
                max((target_time - time).total_seconds(), 0.0) / 86400.0
                for _, time, _ in history
            ],
        }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Generates the samples of one patient.

        With ``sample_mode="all_prefixes"`` every admission after the first
        is a target, predicted from all earlier admissions; with ``"last"``
        only the last admission is.

        Args:
            patient: A :class:`~pyhealth.data.Patient` with ``admissions`` and
                ``diagnoses_icd`` events.

        Returns:
            The samples, oldest target first, each with these fields:

            - ``patient_id`` and ``visit_id`` (the target admission).
            - ``conditions``: CCS codes per history visit (the guide model's
              input; schema ``nested_sequence``).
            - ``ccs_targets``: CCS codes of the target admission (all of them,
              or only the novel ones); schema ``multilabel``.
            - ``history_icd``, ``history_icd_names``, ``history_ccs``: per
              history visit, the ICD codes, their descriptions and their CCS
              parents, aligned element by element.
            - ``days_ago``: per history visit, the days from its admission to
              the target admission.

            The last four fields are outside the schema and are passed
            through unchanged; :class:`~pyhealth.models.EviCare` reads them.
            The list is empty if the patient has fewer than ``min_visits``
            usable admissions. In the ``"novel"`` setting, targets without a
            CCS code absent from their history give no sample.
        """
        visits = self._visits(patient)
        if len(visits) < self.min_visits:
            return []
        if self.sample_mode == "last":
            targets = [len(visits) - 1]
        else:
            targets = list(range(1, len(visits)))
        samples = []
        for t in targets:
            sample = self._make_sample(patient.patient_id, visits[:t], visits[t])
            if sample is not None:
                samples.append(sample)
        return samples

    def patient_codes(self, patient: Any) -> list[str]:
        """Returns every CCS code of a patient, over all of their admissions.

        This is the patient's row of the matrix ``A`` behind EviCare's
        co-occurrence graph ``G = A^T A``. It uses the same admissions as
        :meth:`__call__` (mapped diagnoses, an admission time, at least
        ``min_visits`` of them) but ignores ``setting`` and ``sample_mode``:
        a patient counts even if no target gives a sample, for example in
        the ``"novel"`` setting when no admission adds a new code.

        Args:
            patient: A :class:`~pyhealth.data.Patient`.

        Returns:
            The sorted CCS codes, or an empty list if the patient has fewer
            than ``min_visits`` usable admissions.
        """
        visits = self._visits(patient)
        if len(visits) < self.min_visits:
            return []
        return sorted({ccs for _, _, codes in visits for _, _, ccs in codes})

    def patient_code_sets(
        self,
        dataset: Any,
        patient_ids: Iterable[Any] | None = None,
        batch_size: int = 1000,
    ) -> dict[str, list[str]]:
        """Returns :meth:`patient_codes` for every eligible patient.

        Pass the result, restricted to the training patients, to
        :meth:`~pyhealth.models.EviCare.fit_cooccurrence`. It does not depend
        on the samples, so patients without any sample (possible in the
        ``"novel"`` setting) are included, and the result is the same for
        both settings and both sample modes.

        Args:
            dataset: A :class:`~pyhealth.datasets.BaseDataset` (such as
                :class:`~pyhealth.datasets.MIMIC3Dataset`) or an iterable of
                :class:`~pyhealth.data.Patient`.
            patient_ids: Only these patients. Default None (all).
            batch_size: Patients loaded per query when ``dataset`` is a
                dataset.

        Returns:
            Patient ID -> sorted CCS codes, for the patients with at least
            ``min_visits`` usable admissions, in patient ID order.

        Examples:
            >>> task = EviCareDiagnosisPredictionMIMIC3(
            ...     setting="novel"
            ... )  # doctest: +SKIP
            >>> codes = task.patient_code_sets(dataset)  # doctest: +SKIP
            >>> model.fit_cooccurrence(  # doctest: +SKIP
            ...     {p: codes[p] for p in train_patient_ids if p in codes}
            ... )
        """
        wanted = (
            None if patient_ids is None else {str(p) for p in patient_ids}
        )
        result: dict[str, list[str]] = {}
        for patient in self._iter_patients(dataset, wanted, batch_size):
            codes = self.patient_codes(patient)
            if codes:
                result[str(patient.patient_id)] = codes
        return dict(sorted(result.items()))

    @staticmethod
    def _iter_patients(
        dataset: Any, wanted: set[str] | None, batch_size: int
    ) -> Iterable[Any]:
        """Yields the patients of a dataset (or iterable), optionally a subset."""
        if not hasattr(dataset, "global_event_df"):
            for patient in dataset:
                if wanted is None or str(patient.patient_id) in wanted:
                    yield patient
            return

        import polars as pl

        from ..data import Patient

        events = dataset.global_event_df.filter(
            pl.col("event_type").is_in(["admissions", "diagnoses_icd"])
        )
        ids = [str(p) for p in dataset.unique_patient_ids]
        if wanted is not None:
            ids = [p for p in ids if p in wanted]
        for start in range(0, len(ids), batch_size):
            frames = (
                events.filter(pl.col("patient_id").is_in(ids[start : start + batch_size]))
                .collect(engine="streaming")
                .partition_by("patient_id", as_dict=True)
            )
            for (patient_id,), frame in frames.items():
                yield Patient(patient_id=patient_id, data_source=frame)


class EviCareDiagnosisPredictionMIMIC3(_EviCareDiagnosisPrediction):
    """EviCare next-visit CCS diagnosis prediction on MIMIC-III.

    Works with :class:`~pyhealth.datasets.MIMIC3Dataset` (tables
    ``diagnoses_icd``; admissions are always loaded). ICD-9 codes are mapped
    to single-level CCS categories, by default with PyHealth's
    ``ICD9CM -> CCSCM`` table, the same table the official EviCare code
    uses (285 categories). See the module docstring for the two settings.

    Args:
        setting: ``"overall"`` (default) or ``"novel"``.
        ccs_mapping: None (default, PyHealth's CCS tables), a CSV path or a
            :class:`CCSMapping`.
        min_visits: Minimum number of admissions with mapped diagnoses.
            Default 2, as in the paper.
        sample_mode: ``"all_prefixes"`` (default: every admission after the
            first is a target, as in the paper's pipeline) or ``"last"``.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import EviCareDiagnosisPredictionMIMIC3
        >>> task = EviCareDiagnosisPredictionMIMIC3(
        ...     setting="novel",
        ...     ccs_mapping="test-resources/core/evicare/ccs_mapping_tiny.csv",
        ... )  # doctest: +SKIP
        >>> task.input_schema, task.output_schema  # doctest: +SKIP
        ({'conditions': 'nested_sequence'}, {'ccs_targets': 'multilabel'})
        >>> dataset = MIMIC3Dataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-iii/1.4", tables=["diagnoses_icd"]
        ... )
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "EviCareDiagnosisPredictionMIMIC3"
    _icd_versions = (9,)

    def _event_code(self, event: Any) -> tuple[str, int] | None:
        """Returns ``(icd9_code, 9)`` of a MIMIC-III diagnosis event."""
        code = normalize_icd(getattr(event, "icd9_code", None))
        return (code, 9) if code else None


class EviCareDiagnosisPredictionMIMIC4(_EviCareDiagnosisPrediction):
    """EviCare next-visit CCS diagnosis prediction on MIMIC-IV.

    Works with :class:`~pyhealth.datasets.MIMIC4EHRDataset` (table
    ``diagnoses_icd``). Both ICD-9 and ICD-10 diagnoses are kept and mapped
    to single-level CCS by default (``ICD9CM`` / ``ICD10CM -> CCSCM``). For
    CCSR, pass a CSV built from the HCUP CCSR tables as ``ccs_mapping``. The
    paper reports 842 CCS codes on MIMIC-IV, which matches neither
    single-level CCS nor CCSR cleanly.

    Args:
        setting: ``"overall"`` (default) or ``"novel"``.
        ccs_mapping: None (default, PyHealth's CCS tables), a CSV path or a
            :class:`CCSMapping`.
        min_visits: Minimum number of admissions with mapped diagnoses.
            Default 2.
        sample_mode: ``"all_prefixes"`` (default) or ``"last"``.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import EviCareDiagnosisPredictionMIMIC4
        >>> task = EviCareDiagnosisPredictionMIMIC4()  # doctest: +SKIP
        >>> task.task_name  # doctest: +SKIP
        'EviCareDiagnosisPredictionMIMIC4'
        >>> dataset = MIMIC4EHRDataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-iv/2.2", tables=["diagnoses_icd"]
        ... )
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "EviCareDiagnosisPredictionMIMIC4"
    _icd_versions = (9, 10)

    def _event_code(self, event: Any) -> tuple[str, int] | None:
        """Returns ``(icd_code, icd_version)`` of a MIMIC-IV diagnosis event."""
        version = _normalize_version(getattr(event, "icd_version", None))
        code = normalize_icd(getattr(event, "icd_code", None))
        if version is None or not code:
            return None
        return code, version
