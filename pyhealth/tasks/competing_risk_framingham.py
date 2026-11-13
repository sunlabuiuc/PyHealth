"""Competing-risk survival prediction on the Framingham cohort."""

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar, Literal

from .base_task import BaseTask

logger = logging.getLogger(__name__)

FRAMINGHAM_FEATURES: tuple[str, ...] = (
    "sex",
    "current_smoker",
    "diabetes",
    "bp_meds",
    "education",
    "previous_chd",
    "previous_angina",
    "previous_mi",
    "previous_stroke",
    "previous_hypertension",
    "total_cholesterol",
    "age",
    "systolic_bp",
    "diastolic_bp",
    "cigarettes_per_day",
    "bmi",
    "heart_rate",
    "glucose",
)

TimeUnit = Literal["days", "years"]


class FraminghamCompetingRiskPrediction(BaseTask):
    """Predict cardiovascular disease and death from baseline covariates.

    Each participant produces one sample with a standardized 18-element
    feature vector, a positive duration, and an integer event cause:

    - 0: right censoring;
    - 1: cardiovascular disease;
    - 2: death without a prior cardiovascular event.

    Args:
        time_unit: Return durations in ``"days"`` or ``"years"``.
        horizon_days: Optional administrative censoring horizon. Events after
            the horizon are converted to censoring at the horizon.
        standardize: Z-score features using training-split statistics.

    Examples:
        >>> from pyhealth.datasets import FraminghamDataset
        >>> from pyhealth.tasks import FraminghamCompetingRiskPrediction
        >>> dataset = FraminghamDataset(  # doctest: +SKIP
        ...     root="/data/framingham", download=True
        ... )
        >>> samples = dataset.set_task(  # doctest: +SKIP
        ...     FraminghamCompetingRiskPrediction()
        ... )
    """

    task_name = "FraminghamCompetingRiskPrediction"
    output_schema: ClassVar[dict[str, Any]] = {
        "duration": "tensor",
        "event": "tensor",
    }

    def __init__(
        self,
        time_unit: TimeUnit = "years",
        horizon_days: float | None = None,
        standardize: bool = True,
    ) -> None:
        if time_unit not in {"days", "years"}:
            raise ValueError("time_unit must be 'days' or 'years'")
        if horizon_days is not None and not 0 < horizon_days < math.inf:
            raise ValueError("horizon_days must be positive and finite, or None")
        self.time_unit = time_unit
        self.horizon_days = None if horizon_days is None else float(horizon_days)
        self.standardize = bool(standardize)
        self.input_schema = {
            "features": (
                ("tensor", {"normalization": "zscore"})
                if standardize
                else "tensor"
            )
        }
        super().__init__()

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one competing-risk sample for a Framingham participant."""
        events = patient.get_events(event_type="framingham")
        if not events:
            return []
        if len(events) > 1:
            logger.warning(
                "Patient %s has %d Framingham rows; using the first",
                patient.patient_id,
                len(events),
            )
        row = events[0]
        features = [self._as_float(row[name]) for name in FRAMINGHAM_FEATURES]
        duration = self._as_float(row["duration_days"])
        cause = self._as_float(row["event"])
        if any(value is None for value in features):
            logger.warning("Patient %s has a missing feature", patient.patient_id)
            return []
        if duration is None or duration <= 0 or cause not in (0.0, 1.0, 2.0):
            logger.warning("Patient %s has an invalid outcome", patient.patient_id)
            return []
        if self.horizon_days is not None and duration > self.horizon_days:
            duration, cause = self.horizon_days, 0.0
        if self.time_unit == "years":
            duration /= 365.25
        return [
            {
                "patient_id": patient.patient_id,
                "features": features,
                "duration": [duration],
                "event": [int(cause)],
            }
        ]
