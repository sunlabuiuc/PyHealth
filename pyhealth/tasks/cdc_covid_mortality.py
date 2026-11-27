# Contributor: Arun Vishnu (anv5@illinois.edu)
# Paper: Evaluating Model Performance in Medical Datasets Over Time
# https://proceedings.mlr.press/v209/zhou23a.html
# Task: Case-level mortality prediction with leakage-aware feature sets.

"""Case-level mortality task for CDC COVID-19 surveillance records.

Hospitalization and ICU fields are intentionally excluded to reduce outcome
leakage. Unknown, missing and privacy-suppressed mortality labels are excluded.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from pyhealth.data import Patient

from .base_task import BaseTask
from .task_card import FeatureInfo, TaskCard

_MISSING = {"", "missing", "unknown", "na", "nan", "none", "<na>"}
_UNRECOGNIZED = "Unrecognized"
_CATEGORIES: dict[str, tuple[str, ...]] = {
    "age_group": (
        "0 - 17 years",
        "18 to 49 years",
        "50 to 64 years",
        "65+ years",
    ),
    "sex": ("Female", "Male", "Other"),
    "race": (
        "American Indian/Alaska Native",
        "Asian",
        "Black",
        "Multiple/Other",
        "Native Hawaiian/Other Pacific Islander",
        "White",
    ),
    "ethnicity": ("Hispanic/Latino", "Non-Hispanic/Latino"),
    "exposure_yn": ("Yes",),
    "current_status": ("Laboratory-confirmed case", "Probable Case"),
    "symptom_status": ("Asymptomatic", "Symptomatic"),
    "underlying_conditions_yn": ("No", "Yes"),
}
_DEMOGRAPHICS = ("age_group", "sex", "race", "ethnicity")
_PREOUTCOME = (
    *_DEMOGRAPHICS,
    "exposure_yn",
    "current_status",
    "symptom_status",
    "underlying_conditions_yn",
)
_NUMERIC = ("case_positive_specimen", "case_onset_interval")


def _clean(value: Any) -> str:
    """Return a stripped scalar string, treating null-like values as missing."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _MISSING else text


def _one_hot(value: Any, levels: tuple[str, ...]) -> list[float]:
    """Encode fixed CDC categories plus missing and unrecognized buckets."""
    text = _clean(value)
    category = text if text in levels else ("Missing" if not text else _UNRECOGNIZED)
    vocabulary = (*levels, "Missing", _UNRECOGNIZED)
    return [float(category == level) for level in vocabulary]


def _number(value: Any) -> tuple[float, float]:
    """Encode a finite numeric value and a missingness indicator."""
    text = _clean(value)
    if not text:
        return 0.0, 1.0
    try:
        number = float(text)
    except ValueError:
        return 0.0, 1.0
    if not math.isfinite(number):
        return 0.0, 1.0
    return number, 0.0


def feature_names(feature_set: str) -> list[str]:
    """Return ordered columns produced by a task feature set.

    Args:
        feature_set: ``"demographics"`` or ``"preoutcome"``.

    Returns:
        Human-readable feature names in tensor order.

    Raises:
        ValueError: ``feature_set`` is unsupported.

    Examples:
        >>> feature_names("demographics")[:2]
        ['age_group=0 - 17 years', 'age_group=18 to 49 years']
    """
    if feature_set not in {"demographics", "preoutcome"}:
        raise ValueError("feature_set must be 'demographics' or 'preoutcome'")
    fields = _DEMOGRAPHICS if feature_set == "demographics" else _PREOUTCOME
    names = [
        f"{field}={level}"
        for field in fields
        for level in (*_CATEGORIES[field], "Missing", _UNRECOGNIZED)
    ]
    if feature_set == "preoutcome":
        for field in _NUMERIC:
            names.extend((field, f"{field}_missing"))
    return names


class CDCCOVIDMortalityPrediction(BaseTask):
    """Predict recorded COVID-19 mortality from one surveillance case row.

    Works with :class:`~pyhealth.datasets.CDCCOVIDDataset`. ``death_yn=Yes``
    maps to 1 and ``No`` maps to 0; all other outcomes produce no sample.
    Each deidentified case is independent, so the task rejects a generated
    case identifier containing more than one event.

    Two fixed feature sets support an ablation without learned preprocessing:

    - ``demographics``: age group, sex, race and ethnicity.
    - ``preoutcome``: demographics plus exposure, case status, symptom status,
      underlying-condition flag and the two case-timing intervals.

    Hospitalization, ICU admission, death, county and process fields are never
    features. This reduces direct care-trajectory leakage but does not make the
    retrospective surveillance data prospectively available or unbiased.

    Args:
        feature_set: ``"demographics"`` or ``"preoutcome"``.

    Examples:
        >>> task = CDCCOVIDMortalityPrediction(feature_set="demographics")
        >>> task.output_schema
        {'mortality': 'binary'}
    """

    task_name: ClassVar[str] = "CDCCOVIDMortalityPrediction"

    def __init__(self, feature_set: str = "preoutcome") -> None:
        super().__init__()
        self.feature_set = feature_set
        self.columns = feature_names(feature_set)
        self.input_schema = {"features": "tensor"}
        self.output_schema = {"mortality": "binary"}

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Build zero or one labeled mortality sample for a case row.

        Args:
            patient: Generated one-row patient wrapper from CDCCOVIDDataset.

        Returns:
            One model-ready sample for Yes/No mortality, otherwise no sample.

        Raises:
            ValueError: A generated case identifier contains multiple rows or
                its case month is unavailable.
        """
        events = patient.get_events("cases")
        if not events:
            return []
        if len(events) != 1:
            raise ValueError("CDC COVID case identifiers must contain one event")
        event = events[0]
        label = _clean(event.death_yn).lower()
        if label not in {"yes", "no"}:
            return []
        if event.timestamp is None:
            raise ValueError("CDC COVID case_month is required")

        fields = _DEMOGRAPHICS if self.feature_set == "demographics" else _PREOUTCOME
        features = [
            value
            for field in fields
            for value in _one_hot(event[field], _CATEGORIES[field])
        ]
        if self.feature_set == "preoutcome":
            for field in _NUMERIC:
                features.extend(_number(event[field]))
        return [
            {
                "patient_id": patient.patient_id,
                "record_id": patient.patient_id,
                "case_month": event.timestamp.strftime("%Y-%m"),
                "features": features,
                "mortality": int(label == "yes"),
            }
        ]

    def task_card(self) -> TaskCard:
        """Describe the cohort, prediction time, outcome and leakage controls."""
        info = {
            name: FeatureInfo(
                meaning=name.replace("_", " "),
                group="demographics"
                if name.split("=", 1)[0] in _DEMOGRAPHICS
                else "surveillance",
            )
            for name in self.columns
        }
        return TaskCard(
            population=(
                "CDC public-use COVID-19 cases with Yes/No death_yn",
                "Unknown, missing and privacy-suppressed outcomes have no label.",
            ),
            prediction_time=(
                "Case surveillance record",
                "Only demographics and selected non-outcome surveillance fields are used.",
            ),
            outcome="Recorded death as a result of the illness",
            horizon="Unspecified in the public-use source",
            exclusions=[
                ("Unknown/Missing/NA death_yn", "A binary outcome is not observed."),
                (
                    "Hospitalization and ICU fields as predictors",
                    "They can occur on the care pathway to death and leak severity.",
                ),
            ],
            features=info,
            columns={"features": self.columns},
            notes=(
                "Rows are deidentified cases, not longitudinal people. Reporting "
                "missingness, delayed outcomes and changing surveillance practices "
                "limit prospective interpretation."
            ),
        )
