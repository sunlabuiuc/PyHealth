"""Baseline death/transplantation task for the author's PBC cohort.

Contributors: Amey Deotale (ameysd2), Anushka Shinde (ashinde4),
Abid Hossain (ahossa5).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://proceedings.mlr.press/v209/jeanselme23a.html
"""

from typing import Any

import numpy as np

from pyhealth.data import Event, Patient

from .base_task import BaseTask

PBC_CATEGORIES = {
    "drug": ("D-penicil", "placebo"),
    "sex": ("female", "male"),
    "ascites": ("No", "Yes"),
    "hepatomegaly": ("No", "Yes"),
    "spiders": ("No", "Yes"),
    "edema": ("No edema", "edema no diuretics", "edema despite diuretics"),
    "histologic": ("1", "2", "3", "4"),
}
PBC_NUMERIC = (
    "serBilir",
    "serChol",
    "albumin",
    "alkaline",
    "SGOT",
    "platelets",
    "prothrombin",
    "age",
)
PBC_FEATURES = (
    tuple(
        f"{name}={level}" for name, levels in PBC_CATEGORIES.items() for level in levels
    )
    + PBC_NUMERIC
)
PBC_MASK_INDICES = tuple(
    PBC_FEATURES.index(name) for name in ("serBilir", "albumin", "serChol")
)


def _missing(value: Any) -> bool:
    """Recognize empty source fields before encoding baseline features.

    Args:
        value: A scalar read from a PBC event, including CSV missing markers.

    Returns:
        Whether the field should stay missing until training-fold imputation.
    """
    return value is None or str(value).strip().lower() in {
        "",
        "na",
        "nan",
        "none",
        "<na>",
    }


def _features(row: Event) -> list[float]:
    """Encode a whole baseline event using the task's fixed clinical vocabulary.

    Args:
        row: Baseline PBC event with the required source attributes.

    Returns:
        Seventeen category indicators and eight numeric values in PBC_FEATURES
        order. Missing categories keep their entire indicator block missing.

    Raises:
        ValueError: A category is unknown or a numeric value contains infinity.
    """
    values: list[float] = []
    for name, levels in PBC_CATEGORIES.items():
        value = row[name]
        if _missing(value):
            values.extend([np.nan] * len(levels))
            continue
        value = str(value).strip()
        if name == "histologic":
            number = float(value)
            if np.isfinite(number) and number.is_integer():
                value = str(int(number))
        if value not in levels:
            raise ValueError(f"Unknown PBC category for {name}: {value}")
        values.extend(float(value == level) for level in levels)
    values.extend(
        np.nan if _missing(row[name]) else float(row[name]) for name in PBC_NUMERIC
    )
    if np.isinf(values).any():
        raise ValueError("PBC features cannot contain infinity")
    return values


class CompetingRisksPBC(BaseTask):
    """Extract one baseline first-event sample without future-value backfill.

    Status strings are alive=0 (right censoring), dead=1, transplanted=2.
    This task accepts the author's longitudinal pbc2.csv, not R's numeric-label
    pbc schema. Time is unshifted years from baseline. The fixed vocabulary
    yields 25 columns; missing category blocks remain NaN. Baseline age never
    includes eventual follow-up. Delayed entry is unsupported.

    Examples:
        >>> task = CompetingRisksPBC()
        >>> task.output_schema
        {'time': 'raw', 'event': 'raw'}
        >>> samples = cohort.set_task(task)  # doctest: +SKIP
    """

    task_name = "CompetingRisksPBC"

    def __init__(self) -> None:
        """Declare raw clinical/mask inputs and first-event time/cause labels."""
        self.input_schema = {"features": "raw", "missing_mask": "raw"}
        self.output_schema = {"time": "raw", "event": "raw"}
        super().__init__()

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Select the earliest whole baseline row and validate first endpoints.

        Args:
            patient: Patient with events loaded by PBCDataset.

        Returns:
            Zero samples for an empty patient, otherwise one sample containing
            patient_id, 25 features, a binary missing_mask, time, and event.

        Raises:
            ValueError: Entry, follow-up, repeated endpoints, category values,
                or conflicting baseline records violate the source contract.
        """
        rows = patient.get_events("pbc")
        if not rows:
            return []
        visits = np.asarray([float(row.year) for row in rows])
        if not np.isfinite(visits).all() or visits.min() != 0:
            raise ValueError("PBC requires finite visits and baseline entry at zero")
        baseline = [row for row, visit in zip(rows, visits) if visit == 0]
        row = baseline[0]
        end, status = float(row.years), str(row.status).strip()
        mapping = {"alive": 0, "dead": 1, "transplanted": 2}
        if status not in mapping or not np.isfinite(end) or end <= 0:
            raise ValueError("Invalid PBC status or follow-up")
        if any(
            float(item.years) != end or str(item.status).strip() != status
            for item in rows
        ):
            raise ValueError("PBC repeated endpoint fields disagree")
        if visits.max() > end:
            raise ValueError("PBC visit occurs after follow-up")
        features = _features(row)
        if any(
            not np.array_equal(features, _features(other), equal_nan=True)
            for other in baseline[1:]
        ):
            raise ValueError("Conflicting PBC baseline records")
        return [
            {
                "patient_id": patient.patient_id,
                "features": features,
                "missing_mask": np.isnan(features).astype(float).tolist(),
                "time": end,
                "event": mapping[status],
            }
        ]
