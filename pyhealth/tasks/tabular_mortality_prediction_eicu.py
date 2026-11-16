"""Tabular ICU mortality prediction on eICU-CRD.

Contributors:
    Alex Castillo (acast23@illinois.edu)
    Srikar Lunke (srikarl2@illinois.edu)
    Maheswara Vatti (mvatti2@illinois.edu)

Paper being reproduced:
    Lin, Yu and Lee, "A Case Study Exploring the Current Landscape of
    Synthetic Medical Record Generation with Commercial LLMs", CHIL 2025
    (PMLR 287), https://proceedings.mlr.press/v287/lin25a.html

AI assistance:
    Generated with Claude Fable 5.2 (Anthropic).

The paper prompts commercial LLMs to generate synthetic eICU records and
scores them with an XGBoost ICU mortality classifier. Its real cohort is a
flat table with one row per ICU stay: demographics, the first and last value
of each laboratory test in an early window of the stay, and whether the
patient died in the ICU. This module builds that table from
:class:`~pyhealth.datasets.eICUDataset`, so real and synthetic rows can be
evaluated with the same PyHealth models.

The other eICU mortality tasks in PyHealth differ in input and label.
:class:`~pyhealth.tasks.MortalityPredictionEICU` and
:class:`~pyhealth.tasks.MortalityPredictionEICU2` read diagnosis, treatment
and medication codes and predict death at the next stay.
:class:`~pyhealth.tasks.YAIBMortalityEICU` reads hourly time series. This
task gives one fixed-width numeric vector per stay.
"""

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import polars as pl

from .base_task import BaseTask

__all__ = ["TabularMortalityPredictionEICU"]

# Feature stem -> ``labname`` in the eICU ``lab`` table. The stems are the
# laboratory feature names of the paper's Figure 5.
_LAB_NAMES: dict[str, str] = {
    "albumin": "albumin",
    "bands": "-bands",
    "bg_baseexcess": "Base Excess",
    "bg_paco2": "paCO2",
    "bg_pao2": "paO2",
    "bg_ph": "pH",
    "bicarbonate": "bicarbonate",
    "bilirubin": "total bilirubin",
    "bun": "BUN",
    "calcium": "calcium",
    "creatinine": "creatinine",
    "glucose": "glucose",
    "hematocrit": "Hct",
    "hemoglobin": "Hgb",
    "inr": "PT - INR",
    "lactate": "lactate",
    "platelet": "platelets x 1000",
    "potassium": "potassium",
    "sodium": "sodium",
    "wbc": "WBC x 1000",
}
_AGGREGATIONS: tuple[str, ...] = ("first", "last")
_STATIC_FEATURES: tuple[str, ...] = ("age", "is_female")
# Lab feature name -> (labname, aggregation), e.g.
# "lactate_last_early" -> ("lactate", "last").
_LAB_FEATURES: dict[str, tuple[str, str]] = {
    f"{stem}_{aggregation}_early": (labname, aggregation)
    for stem, labname in _LAB_NAMES.items()
    for aggregation in _AGGREGATIONS
}
_LABELS: dict[str, int] = {"Alive": 0, "Expired": 1}


def _parse_age(age: Any) -> float:
    """Returns an eICU age in years, or NaN if it is missing.

    eICU reports ages above 89 as ``"> 89"``; these become 90.

    Args:
        age: The ``age`` value of a ``patient`` row.

    Returns:
        The age in years, or NaN.

    Examples:
        >>> _parse_age("> 89")
        90.0
        >>> _parse_age("61")
        61.0
    """
    if age is None:
        return math.nan
    text = str(age).strip()
    if text.startswith(">"):
        return 90.0
    try:
        return float(text)
    except ValueError:
        return math.nan


def _is_female(gender: Any) -> float:
    """Encodes an eICU gender as 1 for female, 0 for male and NaN otherwise.

    Args:
        gender: The ``gender`` value of a ``patient`` row.

    Returns:
        1.0, 0.0 or NaN.

    Examples:
        >>> _is_female("Female")
        1.0
    """
    return {"Female": 1.0, "Male": 0.0}.get(str(gender).strip(), math.nan)


class TabularMortalityPredictionEICU(BaseTask):
    """ICU mortality from one row of early features per eICU stay.

    Works with :class:`~pyhealth.datasets.eICUDataset`. Each ICU stay
    (``patientunitstayid``) with a known ICU discharge status gives one
    sample: a fixed-width numeric vector and a binary label, 1 if the patient
    died in the ICU (``unitdischargestatus`` is ``"Expired"``) and 0 if the
    patient left it alive. Stays with any other status are skipped.

    Features are chosen by name:

    - ``"age"``: age in years (``"> 89"`` becomes 90).
    - ``"is_female"``: 1 for female, 0 for male.
    - ``"<lab>_first_early"`` and ``"<lab>_last_early"``: the first and the
      last numeric result of a laboratory test in the window from ICU
      admission to ``window_hours`` later (both ends included). ``<lab>`` is
      one of albumin, bands, bg_baseexcess, bg_paco2, bg_pao2, bg_ph,
      bicarbonate, bilirubin, bun, calcium, creatinine, glucose, hematocrit,
      hemoglobin, inr, lactate, platelet, potassium, sodium and wbc.

    A feature without a value is NaN: results outside the window (including
    those before ICU admission) and non-numeric results are ignored. NaN is
    left in the vector, which tree models such as
    :class:`~pyhealth.models.XGBoostModel` read as missing. For a model that
    needs finite inputs, pass a ``TensorProcessor`` with
    ``impute_strategy="zero"`` to ``set_task``.

    The default features are the ten highest-ranked features of the paper's
    Figure 5 that the ``patient`` and ``lab`` tables supply. Hospital length
    of stay is left out because it is only known after the outcome.

    Each sample also carries the stay's ``gender`` and ``ethnicity`` as plain
    strings, outside the input schema, for group-wise analysis and
    group-conditioned generation.

    Requires the ``lab`` table unless only ``"age"`` and ``"is_female"`` are
    used.

    Attributes:
        task_name: Name of the task.
        input_schema: ``{"features": "tensor"}``.
        output_schema: ``{"mortality": "binary"}``.
        DEFAULT_FEATURES: Feature names used when ``features`` is not given.
        SUPPORTED_FEATURES: Every feature name the task accepts.
        features: Feature names, in the order of the vector's columns.
        window_hours: Length of the early window, in hours from ICU admission.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import TabularMortalityPredictionEICU
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["lab"],
        ... )  # doctest: +SKIP
        >>> task = TabularMortalityPredictionEICU()
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> len(task.features)
        10
        >>> task = TabularMortalityPredictionEICU(
        ...     features=["age", "lactate_first_early"], window_hours=6
        ... )
        >>> task.features
        ['age', 'lactate_first_early']
    """

    task_name: str = "TabularMortalityPredictionEICU"
    input_schema: ClassVar[dict[str, str]] = {"features": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"mortality": "binary"}

    DEFAULT_FEATURES: ClassVar[tuple[str, ...]] = (
        "lactate_last_early",
        "bun_last_early",
        "age",
        "albumin_first_early",
        "bilirubin_last_early",
        "wbc_last_early",
        "bicarbonate_last_early",
        "creatinine_last_early",
        "platelet_first_early",
        "bg_ph_last_early",
    )
    SUPPORTED_FEATURES: ClassVar[tuple[str, ...]] = (
        *_STATIC_FEATURES,
        *_LAB_FEATURES,
    )

    def __init__(
        self,
        features: Sequence[str] | None = None,
        window_hours: float = 24.0,
    ) -> None:
        """Initializes the task.

        Args:
            features: Names of the features, in the order of the vector's
                columns. Defaults to :attr:`DEFAULT_FEATURES`.
            window_hours: Length of the early window in hours, counted from
                ICU admission. Laboratory results outside it are ignored.

        Raises:
            TypeError: If ``features`` is a single string.
            ValueError: If ``features`` is empty, repeats a name or holds a
                name outside :attr:`SUPPORTED_FEATURES`, or if
                ``window_hours`` is not a positive number.
        """
        super().__init__()
        if features is None:
            features = self.DEFAULT_FEATURES
        if isinstance(features, str):
            raise TypeError(
                "features must be a sequence of feature names, not a string."
            )
        features = list(features)
        if not features:
            raise ValueError("features must hold at least one feature name.")
        unknown = [name for name in features if name not in self.SUPPORTED_FEATURES]
        if unknown:
            raise ValueError(
                f"Unknown features {unknown}. Supported features: "
                f"{list(self.SUPPORTED_FEATURES)}."
            )
        if len(set(features)) != len(features):
            raise ValueError(f"features holds a name more than once: {features}.")
        if (
            isinstance(window_hours, bool)
            or not isinstance(window_hours, (int, float))
            or not math.isfinite(window_hours)
            or window_hours <= 0
        ):
            raise ValueError(
                f"window_hours must be a positive number, got {window_hours!r}."
            )
        self.features: list[str] = features
        self.window_hours: float = float(window_hours)

    def _lab_names(self) -> list[str]:
        """Returns the eICU ``labname`` values the chosen features read.

        Returns:
            The lab names, sorted and without repeats.
        """
        return sorted(
            {_LAB_FEATURES[name][0] for name in self.features if name in _LAB_FEATURES}
        )

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Keeps the rows the task reads and checks the ``lab`` table is loaded.

        Only ``patient`` rows and the ``lab`` rows of the chosen laboratory
        tests are kept, so the task does not load unused events.

        Args:
            df: The dataset's event table.

        Returns:
            The event table without the rows the task does not read.

        Raises:
            ValueError: If a laboratory feature is used but the dataset was
                loaded without the ``lab`` table.
        """
        lab_names = self._lab_names()
        keep = pl.col("event_type") == "patient"
        if lab_names:
            if "lab/labname" not in df.collect_schema().names():
                raise ValueError(
                    f"{self.task_name} requires the eICU lab table for its "
                    "laboratory features; add 'lab' to the dataset's `tables`."
                )
            keep = keep | (
                (pl.col("event_type") == "lab") & pl.col("lab/labname").is_in(lab_names)
            )
        return df.filter(keep)

    def _lab_values(self, patient: Any) -> dict[tuple[int, str, str], float]:
        """Finds the first and last result of each lab test in the window.

        Args:
            patient: A patient of :class:`~pyhealth.datasets.eICUDataset`.

        Returns:
            The value for each ``(stay ID, labname, aggregation)`` that has a
            numeric result in the window.
        """
        lab_names = self._lab_names()
        if not lab_names:
            return {}
        events = patient.get_events(event_type="lab", return_df=True)
        if events.height == 0 or "lab/labname" not in events.columns:
            return {}
        results = (
            events.select(
                pl.col("lab/patientunitstayid")
                .cast(pl.Int64, strict=False)
                .alias("stay_id"),
                pl.col("lab/labresultoffset")
                .cast(pl.Float64, strict=False)
                .alias("offset"),
                pl.col("lab/labname").alias("labname"),
                pl.col("lab/labresult").cast(pl.Float64, strict=False).alias("value"),
            )
            .drop_nulls()
            .filter(
                pl.col("labname").is_in(lab_names)
                & pl.col("value").is_finite()
                & pl.col("offset").is_between(0, self.window_hours * 60)
            )
            .sort("offset", maintain_order=True)
            .group_by("stay_id", "labname", maintain_order=True)
            .agg(
                pl.col("value").first().alias("first"),
                pl.col("value").last().alias("last"),
            )
        )
        values: dict[tuple[int, str, str], float] = {}
        for stay_id, labname, first, last in results.iter_rows():
            values[(stay_id, labname, "first")] = first
            values[(stay_id, labname, "last")] = last
        return values

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one sample per ICU stay of a patient.

        Args:
            patient: A patient of :class:`~pyhealth.datasets.eICUDataset`.

        Returns:
            One dict per ICU stay with a known ICU discharge status, holding
            ``patient_id``, ``stay_id`` (the ``patientunitstayid``),
            ``features`` (a list of floats in the order of :attr:`features`,
            NaN where missing), ``mortality`` (1 for death in the ICU, else
            0), ``gender`` and ``ethnicity``. Empty if the patient has no
            such stay.
        """
        stays = patient.get_events(event_type="patient", return_df=True)
        if stays.height == 0:
            return []
        lab_values = self._lab_values(patient)
        samples = []
        for stay_id, status, age, gender, ethnicity in stays.select(
            pl.col("patient/patientunitstayid").cast(pl.Int64, strict=False),
            pl.col("patient/unitdischargestatus"),
            pl.col("patient/age"),
            pl.col("patient/gender"),
            pl.col("patient/ethnicity"),
        ).iter_rows():
            label = _LABELS.get(str(status).strip())
            if stay_id is None or label is None:
                continue
            static = {"age": _parse_age(age), "is_female": _is_female(gender)}
            vector = [
                static[name]
                if name in static
                else lab_values.get((stay_id, *_LAB_FEATURES[name]), math.nan)
                for name in self.features
            ]
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "stay_id": stay_id,
                    "features": vector,
                    "mortality": label,
                    "gender": "" if gender is None else str(gender),
                    "ethnicity": "" if ethnicity is None else str(ethnicity),
                }
            )
        return samples
