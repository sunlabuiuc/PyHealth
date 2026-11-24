"""Rare-condition ICU prediction tasks for KnowRare on MIMIC-IV.

KnowRare (Zhu et al., "Bridging data gaps of rare conditions in ICU: a
multi-disease adaptation approach for clinical prediction", npj Digital
Medicine 2025, https://arxiv.org/abs/2507.06432) predicts outcomes for ICU
patients whose primary condition is rare. Every sample therefore carries,
besides the model inputs and the label, the patient's condition (the
3-character category of the admission's primary diagnosis) and the
admission-level diagnoses and drugs that the condition knowledge graph is
built from (:class:`~pyhealth.models.knowrare.ConditionKnowledgeGraph`).

The paper evaluates on MIMIC-III and eICU. This module ports its eICU ICU
mortality setting (first 24 hours of the ICU stay, hourly bins) to MIMIC-IV,
reusing the YAIB cohort and feature extraction of
:mod:`pyhealth.tasks.yaib_mimic4`. :func:`select_rare_conditions` applies the
paper's rule for picking the evaluated rare conditions.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

import numpy as np
import polars as pl

from .yaib_mimic4 import YAIBMortalityMIMIC4

__all__ = [
    "KnowRareICUMortalityMIMIC4",
    "icd_category",
    "select_rare_conditions",
]


def icd_category(code: str, version: int | str) -> str:
    """Returns the 3-character ICD category of a code, tagged with its version.

    The paper groups patients by the first three characters ("level 3") of
    their ICD-9-CM primary diagnosis. MIMIC-IV mixes ICD-9 and ICD-10 codes,
    so the version is kept in the name (``"ICD9:410"``, ``"ICD10:I21"``).
    ICD-9 external-cause codes (``E800``-``E999``) keep four characters, as
    their category has three digits after the ``E``.

    Args:
        code: The ICD code without a dot, e.g. ``"41071"`` or ``"I214"``.
        version: The ICD version, 9 or 10.

    Returns:
        The tagged category.

    Examples:
        >>> icd_category("41071", 9)
        'ICD9:410'
        >>> icd_category("E8889", "9")
        'ICD9:E888'
        >>> icd_category("I214", 10)
        'ICD10:I21'
    """
    code = str(code).strip().upper().replace(".", "")
    version = int(version)
    length = 4 if version == 9 and code.startswith("E") else 3
    return f"ICD{version}:{code[:length]}"


def select_rare_conditions(
    conditions: Sequence[str],
    labels: Sequence[Any],
    num_patients: int | None = None,
    patient_ids: Sequence[str] | None = None,
    n: int = 10,
    max_prevalence: float = 1 / 2000,
    min_patients: int = 10,
    required: Iterable[Sequence[str]] = (),
) -> list[str]:
    """Picks the rare conditions to evaluate, following the paper.

    A condition qualifies when it has at least ``min_patients`` patients
    (the paper drops conditions with fewer than ten), fewer than
    ``max_prevalence * num_patients`` patients (rare: under one case per
    2,000 patients), and at least one positive label. Qualifying conditions
    are ranked from least to most prevalent and the first ``n`` are
    returned (ties broken by name, for a reproducible choice).

    Args:
        conditions: The condition of every sample.
        labels: The binary label of every sample (anything with truthiness).
        num_patients: Patients in the whole dataset, the denominator of the
            prevalence. Defaults to the number of distinct ``patient_ids``.
        patient_ids: The patient of every sample; prevalence counts
            patients, not samples. Defaults to one patient per sample.
        n: How many conditions to return.
        max_prevalence: Upper bound on the share of patients with the
            condition; ``None`` disables it (e.g. on a small demo).
        min_patients: Lower bound on the patients with the condition.
        required: Further sequences of conditions (e.g. the conditions of
            the training and test splits); a condition must appear in each.

    Returns:
        Up to ``n`` condition names, least prevalent first.

    Examples:
        >>> conditions = ["a", "a", "b", "b", "b", "c"]
        >>> labels = [1, 0, 0, 1, 0, 1]
        >>> select_rare_conditions(conditions, labels, n=2,
        ...                        max_prevalence=None, min_patients=2)
        ['a', 'b']
    """
    if patient_ids is None:
        patient_ids = [str(i) for i in range(len(conditions))]
    patients: dict[str, set] = {}
    positives: Counter = Counter()
    for condition, label, pid in zip(conditions, labels, patient_ids):
        patients.setdefault(condition, set()).add(pid)
        positives[condition] += int(bool(label))
    if num_patients is None:
        num_patients = len(set(patient_ids))
    present = [set(r) for r in required]
    eligible = []
    for condition, pids in patients.items():
        count = len(pids)
        if count < min_patients or positives[condition] == 0:
            continue
        if max_prevalence is not None and count >= max_prevalence * num_patients:
            continue
        if any(condition not in seen for seen in present):
            continue
        eligible.append((count, condition))
    return [condition for _, condition in sorted(eligible)[:n]]


def _fill_both_ways(values: np.ndarray) -> np.ndarray:
    """Forward-fills, then backward-fills, each column (NaN = missing)."""
    values = values.copy()
    for column in range(values.shape[1]):
        series = values[:, column]
        observed = np.flatnonzero(~np.isnan(series))
        if observed.size == 0:
            continue
        # Index of the last observation at or before each step (forward
        # fill); steps before the first observation take the first one.
        last = np.maximum.accumulate(
            np.where(~np.isnan(series), np.arange(len(series)), -1)
        )
        last[last < 0] = observed[0]
        values[:, column] = series[last]
    return values


class KnowRareICUMortalityMIMIC4(YAIBMortalityMIMIC4):
    """ICU mortality from the first 24 hours, grouped by primary condition.

    Works with :class:`~pyhealth.datasets.MIMIC4EHRDataset`.

    The MIMIC-IV counterpart of KnowRare's eICU ICU-mortality task. The
    cohort, label and features are those of
    :class:`~pyhealth.tasks.YAIBMortalityMIMIC4` (adults; ICU stays of at
    least 30 hours; death in the ICU), restricted to stays whose admission
    has a primary diagnosis. Each sample holds:

    - ``dynamic``: ``[24, 48]`` hourly YAIB variables (labs, vitals, urine
      output) for hours 0 to 23 of the stay. As in the paper, each variable
      is forward-filled and then backward-filled within the stay here;
      the processor then fills variables never measured in the stay with
      the training mean and z-scores every variable with training
      statistics (use ``set_task(task, split=PatientSplit(...))`` so the
      statistics come from the training patients only).
    - ``static``: age, sex, height and weight, standardized, with
      missing-value indicators.
    - ``mortality``: 1 if the patient died in the ICU.
    - ``condition``: the category (:func:`icd_category`) of the admission's
      primary diagnosis (``seq_num`` 1), the domain KnowRare adapts to.
    - ``diagnoses`` and ``drugs``: the admission's diagnosis categories and
      prescribed drug names, used only to build the condition knowledge
      graph. They are recorded during the whole admission, so they must
      not be used as model inputs for a 24-hour prediction.

    Differences from the paper's eICU task: MIMIC-IV instead of eICU (single
    centre); YAIB's exclusions (stays under 30 hours, deaths in the first 30
    hours) instead of excluding stays under 24 hours; the 48 YAIB variables
    instead of the paper's list (Supplementary Table S1); height and weight
    instead of race among the static variables; ICD-9 and ICD-10 categories
    side by side.

    Requires the ``labevents``, ``chartevents``, ``outputevents``,
    ``transfers``, ``diagnoses_icd`` and ``prescriptions`` tables.

    Args:
        icu_wards: Care units counted as ICU wards, as in
            :class:`~pyhealth.tasks.YAIBMortalityMIMIC4`.
        observation_hours: Hourly bins in ``dynamic`` (24 in the paper).

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset, PatientSplit
        >>> from pyhealth.tasks import KnowRareICUMortalityMIMIC4
        >>> dataset = MIMIC4EHRDataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["labevents", "chartevents", "outputevents",
        ...             "transfers", "diagnoses_icd", "prescriptions"],
        ... )
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     KnowRareICUMortalityMIMIC4(),
        ...     split=PatientSplit(ratios=(0.67, 0.16, 0.17), seed=0),
        ... )
    """

    task_name: str = "KnowRareICUMortalityMIMIC4"
    input_schema: ClassVar[dict[str, Any]] = {
        "dynamic": ("tensor", {"normalization": "zscore", "impute_strategy": "zero"}),
        "static": (
            "tensor",
            {
                "normalization": "zscore",
                "missing_indicators": True,
                "impute_strategy": "zero",
            },
        ),
    }
    output_schema: ClassVar[dict[str, Any]] = {"mortality": "binary"}
    _EXTRA_TABLES: ClassVar[tuple[str, ...]] = (
        "transfers",
        "diagnoses_icd",
        "prescriptions",
    )

    def __init__(
        self,
        icu_wards: Sequence[str] | None = None,
        observation_hours: int = 24,
    ) -> None:
        super().__init__(icu_wards=icu_wards)
        if not 1 <= observation_hours <= self.OBSERVATION_HOURS + 1:
            raise ValueError(
                f"observation_hours must be in [1, {self.OBSERVATION_HOURS + 1}]"
            )
        self.observation_hours = observation_hours

    @staticmethod
    def _admissions(patient: Any) -> dict[int, dict[str, Any]]:
        """Primary condition, diagnosis categories and drugs per admission."""
        info: dict[int, dict[str, Any]] = {}
        diagnoses = patient.get_events(event_type="diagnoses_icd", return_df=True)
        if diagnoses.height:
            rows = (
                diagnoses.select(
                    pl.col("diagnoses_icd/hadm_id").cast(pl.Int64).alias("hadm_id"),
                    pl.col("diagnoses_icd/icd_code").alias("code"),
                    pl.col("diagnoses_icd/icd_version").alias("version"),
                    pl.col("diagnoses_icd/seq_num")
                    .cast(pl.Int64, strict=False)
                    .alias("seq_num"),
                )
                .drop_nulls(["hadm_id", "code", "version"])
                .sort("hadm_id", "seq_num", nulls_last=True)
            )
            for hadm_id, code, version, _ in rows.iter_rows():
                entry = info.setdefault(
                    hadm_id, {"condition": None, "diagnoses": set(), "drugs": set()}
                )
                category = icd_category(code, version)
                if entry["condition"] is None:  # lowest seq_num comes first
                    entry["condition"] = category
                entry["diagnoses"].add(category)
        drugs = patient.get_events(event_type="prescriptions", return_df=True)
        if drugs.height:
            for hadm_id, drug in (
                drugs.select(
                    pl.col("prescriptions/hadm_id").cast(pl.Int64).alias("hadm_id"),
                    pl.col("prescriptions/drug").str.strip_chars().str.to_lowercase(),
                )
                .drop_nulls()
                .iter_rows()
            ):
                if hadm_id in info and drug:
                    info[hadm_id]["drugs"].add(drug)
        return info

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        samples = super().__call__(patient)
        if not samples:
            return []
        stay_to_hadm = dict(
            self._stays(patient).select("stay_id", "hadm_id").iter_rows()
        )
        admissions = self._admissions(patient)
        out = []
        for sample in samples:
            hadm_id = stay_to_hadm.get(sample["stay_id"])
            info = admissions.get(hadm_id)
            if info is None or info["condition"] is None:
                continue
            values = np.asarray(sample["dynamic"]["values"], dtype=np.float64)
            dynamic = _fill_both_ways(values[: self.observation_hours])
            out.append(
                {
                    **sample,
                    "hadm_id": hadm_id,
                    "dynamic": dynamic.astype(np.float32),
                    # The YAIB tasks emit static as a t = 0 series; KnowRare
                    # reads it as one flat vector.
                    "static": list(sample["static"]["values"][0]),
                    "condition": info["condition"],
                    "diagnoses": sorted(info["diagnoses"]),
                    "drugs": sorted(info["drugs"]),
                }
            )
        return out
