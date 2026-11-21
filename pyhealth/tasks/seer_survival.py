"""Fixed-horizon survival prediction for SEER tumor events.

Contributors:
    Parimala Palani (ppalani2), Kushaal Manchella (kushaal2),
    and Nick King (nrking2).

Paper:
    Reproducible Survival Prediction with SEER Cancer Data.
    https://proceedings.mlr.press/v85/hegselmann18a.html
"""

import json
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pyhealth.processors.seer_processor import SEERFeatureProcessor

from .base_task import BaseTask

if TYPE_CHECKING:
    from pyhealth.data import Patient


SEER_EXTENT_FEATURES = ("stage", "tumor_size", "extension", "lymph_nodes", "metastases")
SEER_FEATURES = (
    "diagnosis_year", "primary_site", "sex", "age_group", "histology", "grade",
    "laterality", "marital_status", *SEER_EXTENT_FEATURES,
)


class SEERSurvivalPrediction(BaseTask):
    """Build one binary survival sample per eligible tumor.

    Args:
        horizon_months: Positive survival horizon in months.
        encoding: Raw fields, one-hot encoding, or a numeric baseline.
            Pass a patient split to set_task when fitting either encoding.
        include_extent: Include stage, tumor size, extension, nodes, and metastases.

    Examples:
        >>> task = SEERSurvivalPrediction(horizon_months=12, include_extent=False)
        >>> task.output_schema
        {'label': 'binary'}
        >>> "stage" in task.feature_fields
        False
    """

    task_name = "SEERSurvivalPrediction"
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __init__(
        self,
        horizon_months: int = 12,
        *,
        encoding: Literal["raw", "one_hot", "numeric"] = "raw",
        include_extent: bool = True,
    ) -> None:
        if type(horizon_months) is not int or horizon_months < 1:
            raise ValueError("horizon_months must be a positive integer")
        if encoding not in {"raw", "one_hot", "numeric"}:
            raise ValueError("encoding must be raw, one_hot, or numeric")
        self.horizon_months = horizon_months
        self.feature_fields = tuple(
            field for field in SEER_FEATURES
            if include_extent or field not in SEER_EXTENT_FEATURES
        )
        self.input_schema = {"features": "raw"}
        if encoding != "raw":
            self.input_schema["features"] = (
                SEERFeatureProcessor,
                {"fields": self.feature_fields, "encoding": encoding},
            )
        super().__init__()

    def __call__(self, patient: "Patient") -> list[dict[str, Any]]:
        """Extract diagnosis-time fields and labels from one patient's events."""
        samples = []
        for tumor in patient.get_events(event_type="tumor"):
            months = tumor.survival_months
            cause = tumor.cause_of_death
            if months is None or cause in {"unknown", "not_applicable"}:
                continue
            survived = months >= self.horizon_months
            if not survived and cause != "cancer_death":
                continue
            features = {name: getattr(tumor, name) for name in self.feature_fields}
            samples.append({
                "patient_id": patient.patient_id,
                "record_id": json.dumps((patient.patient_id, tumor.record_number)),
                "features": features,
                "label": int(survived),
            })
        return samples
