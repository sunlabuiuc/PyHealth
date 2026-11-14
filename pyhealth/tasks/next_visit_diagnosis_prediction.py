"""Next-visit diagnosis prediction on MIMIC-III and MIMIC-IV.

Contributor: Alon Levy (NetID: alonl2, alonl2@illinois.edu)
Paper: Leisheng Yu, Yanxiao Cai, Minxing Zhang, Xia Hu. "Self-Explaining
    Hypergraph Neural Networks for Diagnosis Prediction." CHIL 2025 (PMLR 287).
Paper link: https://arxiv.org/abs/2502.10689

Given the ICD-9 diagnosis codes of a patient's past hospital admissions,
predict the full set of ICD-9 codes recorded at the next admission
(multi-label over the code vocabulary). This is the task evaluated in the SHy
paper and in many earlier diagnosis-prediction papers (Doctor AI, RETAIN,
Dipole, GRAM, CGL, Chet).
"""

import math
from datetime import datetime
from itertools import pairwise
from typing import Any, Literal

from .base_task import BaseTask

CodeLevel = Literal["full", "3digit"]
SampleMode = Literal["last", "all_prefixes"]


def _normalize_id(value: Any) -> str | None:
    """Returns an identifier as a string, or None if it is missing.

    Numeric identifiers read as floats (``100.0``) become ``"100"``, so that
    the same ``hadm_id`` matches across tables whatever the column type.
    """
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            value = int(value)
    text = str(value).strip()
    return text or None


def truncate_icd9(code: str) -> str:
    """Truncates a dot-less ICD-9-CM diagnosis code to its 3-digit category.

    MIMIC stores ICD-9 codes without the dot. Numeric and V codes keep their
    first three characters; E codes keep their first four (``E`` plus 3 digits).

    Args:
        code: A dot-less ICD-9-CM diagnosis code such as ``"4280"``.

    Returns:
        The category code, for example ``"428"`` or ``"E880"``.

    Examples:
        >>> truncate_icd9("4280")
        '428'
        >>> truncate_icd9("E8809")
        'E880'
    """
    code = code.replace(".", "").strip()
    return code[:4] if code.startswith("E") else code[:3]


class _NextVisitDiagnosisPrediction(BaseTask):
    """Shared logic of the next-visit diagnosis prediction tasks.

    Subclasses only say how to read the diagnosis code of one event and
    which events to keep (:meth:`_event_code`).

    Args:
        min_visits: Minimum number of admissions with at least one kept code
            that a patient needs to produce a sample. Must be at least 2 (one
            history visit plus the target visit). Default 2, as in the paper.
        max_visits: If set, keep only the most recent ``max_visits`` history
            visits of each sample. Default None (keep all).
        code_level: ``"full"`` keeps the full ICD-9 code (paper setting);
            ``"3digit"`` truncates codes to their 3-digit category, in both
            the input and the label.
        sample_mode: ``"last"`` gives one sample per patient, with all
            admissions but the last as input and the last admission as label
            (paper setting). ``"all_prefixes"`` gives one sample per
            admission after the first, each predicting that admission from
            all earlier ones.
        include_visit_gaps: If True, adds a ``visit_gaps`` input feature: for
            each history visit, the days since the previous history visit
            (0.0 for the first). Used by the time-gap extension of SHy.
            Default False.
    """

    def __init__(
        self,
        min_visits: int = 2,
        max_visits: int | None = None,
        code_level: CodeLevel = "full",
        sample_mode: SampleMode = "last",
        include_visit_gaps: bool = False,
    ) -> None:
        if min_visits < 2:
            raise ValueError(f"min_visits must be at least 2, got {min_visits}.")
        if max_visits is not None and max_visits < 1:
            raise ValueError(f"max_visits must be positive, got {max_visits}.")
        if code_level not in ("full", "3digit"):
            raise ValueError(
                f"code_level must be 'full' or '3digit', got {code_level!r}."
            )
        if sample_mode not in ("last", "all_prefixes"):
            raise ValueError(
                f"sample_mode must be 'last' or 'all_prefixes', got {sample_mode!r}."
            )
        self.min_visits = min_visits
        self.max_visits = max_visits
        self.code_level = code_level
        self.sample_mode = sample_mode
        self.include_visit_gaps = include_visit_gaps
        # Instance-level schemas so that the optional feature does not leak
        # into other instances of the task.
        self.input_schema = {"conditions": "nested_sequence"}
        if include_visit_gaps:
            self.input_schema["visit_gaps"] = "tensor"
        self.output_schema = {"next_conditions": "multilabel"}
        super().__init__()

    def _event_code(self, event: Any) -> str | None:
        """Returns the dot-less ICD-9 code of a diagnosis event, or None to drop it.

        Args:
            event: A ``diagnoses_icd`` event.

        Returns:
            The code, or None if the event should be ignored.
        """
        raise NotImplementedError

    def _visits(self, patient: Any) -> list[tuple[datetime, list[str]]]:
        """Builds the patient's visits as (admission time, codes), oldest first.

        Diagnosis events are grouped by ``hadm_id``. Admissions that end up
        with no kept code, or have no admission time, are dropped, and
        duplicate codes within a visit are removed (first occurrence kept).

        Args:
            patient: A :class:`~pyhealth.data.Patient`.

        Returns:
            A list of ``(admittime, codes)`` tuples sorted by admission time.
        """
        codes_by_visit: dict[str, list[str]] = {}
        for event in patient.get_events(event_type="diagnoses_icd"):
            code = self._event_code(event)
            hadm_id = _normalize_id(event.hadm_id)
            if not code or hadm_id is None:
                continue
            if self.code_level == "3digit":
                code = truncate_icd9(code)
            visit_codes = codes_by_visit.setdefault(hadm_id, [])
            if code not in visit_codes:
                visit_codes.append(code)

        visits = []
        for admission in patient.get_events(event_type="admissions"):
            codes = codes_by_visit.get(_normalize_id(admission.hadm_id))
            if codes and admission.timestamp is not None:
                visits.append((admission.timestamp, codes))
        visits.sort(key=lambda visit: visit[0])
        return visits

    def _make_sample(
        self,
        patient_id: str,
        history: list[tuple[datetime, list[str]]],
        target: list[str],
    ) -> dict[str, Any]:
        """Builds one sample from a history of visits and the target codes.

        Args:
            patient_id: The patient identifier.
            history: The input visits, oldest first.
            target: The codes of the visit to predict.

        Returns:
            The sample dictionary.
        """
        if self.max_visits is not None:
            history = history[-self.max_visits :]
        sample: dict[str, Any] = {
            "patient_id": patient_id,
            "conditions": [codes for _, codes in history],
            "next_conditions": list(target),
        }
        if self.include_visit_gaps:
            gaps = [0.0]
            for (prev_time, _), (time, _) in pairwise(history):
                gaps.append(max((time - prev_time).total_seconds(), 0.0) / 86400.0)
            sample["visit_gaps"] = gaps
        return sample

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Generates next-visit diagnosis samples for one patient.

        Args:
            patient: A :class:`~pyhealth.data.Patient` with ``admissions`` and
                ``diagnoses_icd`` events.

        Returns:
            A list of samples, each with ``patient_id``, ``conditions`` (list
            of visits, each a list of codes), ``next_conditions`` (list of
            codes) and, if enabled, ``visit_gaps``. Empty if the patient has
            fewer than ``min_visits`` usable admissions.
        """
        visits = self._visits(patient)
        if len(visits) < self.min_visits:
            return []
        if self.sample_mode == "last":
            return [self._make_sample(patient.patient_id, visits[:-1], visits[-1][1])]
        return [
            self._make_sample(patient.patient_id, visits[:t], visits[t][1])
            for t in range(1, len(visits))
        ]


class NextVisitDiagnosisPredictionMIMIC3(_NextVisitDiagnosisPrediction):
    """Next-visit diagnosis prediction on MIMIC-III (ICD-9 codes).

    Works with :class:`~pyhealth.datasets.MIMIC3Dataset`.

    Input ``conditions`` is the patient's admissions except the last, each a
    list of ICD-9 diagnosis codes; label ``next_conditions`` is the set of
    codes of the last admission. This is the setting of the SHy paper (Yu et
    al., CHIL 2025): one sample per patient, patients with at least two
    admissions.

    Args:
        min_visits: Minimum number of admissions with diagnoses. Default 2.
        max_visits: Keep only the most recent history visits. Default None.
        code_level: ``"full"`` (default) or ``"3digit"`` codes.
        sample_mode: ``"last"`` (default, paper) or ``"all_prefixes"``.
        include_visit_gaps: Add the ``visit_gaps`` feature (days between
            consecutive history visits). Default False.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import NextVisitDiagnosisPredictionMIMIC3
        >>> dataset = MIMIC3Dataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd"],
        ... )
        >>> task = NextVisitDiagnosisPredictionMIMIC3()
        >>> task.input_schema, task.output_schema
        ({'conditions': 'nested_sequence'}, {'next_conditions': 'multilabel'})
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "NextVisitDiagnosisPredictionMIMIC3"

    def _event_code(self, event: Any) -> str | None:
        """Returns the MIMIC-III ``icd9_code`` of a diagnosis event.

        Args:
            event: A ``diagnoses_icd`` event.

        Returns:
            The stripped code, or None if missing.
        """
        code = event.icd9_code
        if code is None:
            return None
        code = str(code).strip()
        return code or None


class NextVisitDiagnosisPredictionMIMIC4(_NextVisitDiagnosisPrediction):
    """Next-visit diagnosis prediction on MIMIC-IV, ICD-9 codes only.

    Works with :class:`~pyhealth.datasets.MIMIC4EHRDataset`.

    Same task as :class:`NextVisitDiagnosisPredictionMIMIC3`. MIMIC-IV mixes
    ICD-9 and ICD-10 codes; like the SHy authors' preprocessing, only ICD-9
    rows (``icd_version == 9``) are kept. Admissions left without codes are
    dropped (the authors drop such patients entirely).

    Args:
        min_visits: Minimum number of admissions with ICD-9 diagnoses.
            Default 2.
        max_visits: Keep only the most recent history visits. Default None.
        code_level: ``"full"`` (default) or ``"3digit"`` codes.
        sample_mode: ``"last"`` (default, paper) or ``"all_prefixes"``.
        include_visit_gaps: Add the ``visit_gaps`` feature. Default False.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import NextVisitDiagnosisPredictionMIMIC4
        >>> dataset = MIMIC4EHRDataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-iv/1.0",
        ...     tables=["diagnoses_icd"],
        ... )
        >>> task = NextVisitDiagnosisPredictionMIMIC4(max_visits=10)
        >>> task.task_name
        'NextVisitDiagnosisPredictionMIMIC4'
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "NextVisitDiagnosisPredictionMIMIC4"

    def _event_code(self, event: Any) -> str | None:
        """Returns the code of an ICD-9 diagnosis event, None for ICD-10.

        Args:
            event: A ``diagnoses_icd`` event with ``icd_code`` and
                ``icd_version``.

        Returns:
            The stripped ICD-9 code, or None.
        """
        if str(event.icd_version).strip() not in ("9", "9.0"):
            return None
        code = event.icd_code
        if code is None:
            return None
        code = str(code).strip()
        return code or None
