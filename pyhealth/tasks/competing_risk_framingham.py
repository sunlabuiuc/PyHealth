"""Competing-risk survival prediction on the Framingham cohort.

Contributors: Bin Zeng (bzeng48), Birkneh Tilahun Tadesse (btadesse7).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://arxiv.org/abs/2305.06703
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

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

    Each participant produces one sample with a raw 18-element baseline
    feature vector, a positive follow-up time, and an integer event cause:

    - 0: right censoring;
    - 1: cardiovascular disease;
    - 2: death without a prior cardiovascular event.

    The sample keys follow :class:`~pyhealth.tasks.CompetingRisksPBC`
    (``features``, ``time``, ``event``), so samples can be passed through
    :class:`~pyhealth.tasks.CompetingRisksPreprocessor` and
    :class:`~pyhealth.models.NeuralFineGray` unchanged. Feature scaling is
    left to the preprocessor so that it is fitted on training patients only.

    Args:
        time_unit: Return follow-up time in ``"days"`` or ``"years"``.
        horizon_days: Optional administrative censoring horizon. Follow-up
            beyond the horizon is censored at the horizon. Used by the
            example's task-level ablation.

    Examples:
        >>> from pyhealth.datasets import FraminghamDataset
        >>> from pyhealth.tasks import FraminghamCompetingRiskPrediction
        >>> dataset = FraminghamDataset(  # doctest: +SKIP
        ...     root="/data/framingham", download=True
        ... )
        >>> samples = dataset.set_task(  # doctest: +SKIP
        ...     FraminghamCompetingRiskPrediction()
        ... )
        >>> FraminghamCompetingRiskPrediction().output_schema
        {'time': 'raw', 'event': 'raw'}
    """

    task_name = "FraminghamCompetingRiskPrediction"

    def __init__(
        self,
        time_unit: TimeUnit = "years",
        horizon_days: float | None = None,
    ) -> None:
        """Validate the time unit and optional administrative horizon.

        Raises:
            ValueError: ``time_unit`` is unknown or ``horizon_days`` is not a
                positive finite number.
        """
        if time_unit not in {"days", "years"}:
            raise ValueError("time_unit must be 'days' or 'years'")
        if horizon_days is not None and not (
            isinstance(horizon_days, (int, float)) and 0 < horizon_days < math.inf
        ):
            raise ValueError("horizon_days must be positive and finite, or None")
        self.time_unit = time_unit
        self.horizon_days = None if horizon_days is None else float(horizon_days)
        self.input_schema = {"features": "raw"}
        self.output_schema = {"time": "raw", "event": "raw"}
        super().__init__()

    @staticmethod
    def _as_float(value: Any) -> float | None:
        """Parse a finite float, returning None for missing or invalid values."""
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one competing-risk sample for a Framingham participant.

        Args:
            patient: Patient whose ``framingham`` events come from
                :class:`~pyhealth.datasets.FraminghamDataset`.

        Returns:
            An empty list when the patient has no usable row, otherwise one
            sample with ``patient_id``, 18 ``features``, ``time`` and
            ``event`` (0=censored, 1=cardiovascular disease, 2=death).
        """
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
                "time": duration,
                "event": int(cause),
            }
        ]
