"""Survival analysis task for the SEER cancer registry dataset.

The task predicts overall survival (time in months + event indicator) for
cancer patients in the SEER dataset.

Usage
-----
>>> from pyhealth.datasets import SEERDataset
>>> from pyhealth.tasks import SEERSurvivalTask
>>>
>>> dataset = SEERDataset(root="/path/to/seer")
>>> task    = SEERSurvivalTask()
>>> samples = dataset.set_task(task)
>>> samples[0].keys()
dict_keys(['patient_id', 'patient_features', 'tumour_features',
           'treatment_features', 'survival_months', 'vital_status'])
"""

import logging
from typing import Any, Dict, List, Optional

from .base_task import BaseTask

logger = logging.getLogger(__name__)

def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s.lower() not in ("nan", "none", "unknown", "") else None


def _age_bucket(age: float) -> str:
    if age < 30:
        return "under30"
    elif age < 45:
        return "30_44"
    elif age < 60:
        return "45_59"
    elif age < 75:
        return "60_74"
    else:
        return "75plus"


def _extract_patient_features(event: Any) -> List[str]:
    """Demographic and patient-level tokens."""
    features: List[str] = []

    age = _safe_float(getattr(event, "AGE_AT_DIAGNOSIS", None))
    if age is not None:
        features.append(f"age_{_age_bucket(age)}")

    sex = _safe_str(getattr(event, "SEX", None))
    if sex:
        features.append(f"sex_{sex.lower()}")

    race = _safe_str(getattr(event, "RACE", None))
    if race:
        features.append(f"race_{race.lower()}")

    year = _safe_float(getattr(event, "YEAR_OF_DIAGNOSIS", None))
    if year is not None:
        # Decade bucket
        decade = int(year // 10) * 10
        features.append(f"diagnosis_decade_{decade}")

    return features


def _extract_tumour_features(event: Any) -> List[str]:
    """Tumour characteristics tokens."""
    features: List[str] = []

    primary_site = _safe_str(getattr(event, "PRIMARY_SITE", None))
    if primary_site:
        features.append(f"site_{primary_site.lower()}")

    histology = _safe_str(getattr(event, "HISTOLOGY", None))
    if histology:
        features.append(f"hist_{histology.lower()}")

    stage = _safe_str(getattr(event, "STAGE", None))
    if stage:
        features.append(f"stage_{stage.lower()}")

    grade = _safe_str(getattr(event, "GRADE", None))
    if grade:
        features.append(f"grade_{grade.lower()}")

    laterality = _safe_str(getattr(event, "LATERALITY", None))
    if laterality:
        features.append(f"lat_{laterality.lower()}")

    tumor_size = _safe_float(getattr(event, "TUMOR_SIZE_MM", None))
    if tumor_size is not None and tumor_size >= 0:
        if tumor_size == 0:
            size_cat = "in_situ"
        elif tumor_size <= 10:
            size_cat = "le10mm"
        elif tumor_size <= 20:
            size_cat = "le20mm"
        elif tumor_size <= 50:
            size_cat = "le50mm"
        else:
            size_cat = "gt50mm"
        features.append(f"tumor_size_{size_cat}")

    nodes_pos = _safe_float(getattr(event, "REGIONAL_NODES_POSITIVE", None))
    if nodes_pos is not None and nodes_pos >= 0:
        node_cat = "none" if nodes_pos == 0 else ("low" if nodes_pos <= 3 else "high")
        features.append(f"nodes_pos_{node_cat}")

    return features


def _extract_treatment_features(event: Any) -> List[str]:
    """Treatment indicator tokens."""
    features: List[str] = []
    for field in ("SURGERY", "RADIATION", "CHEMOTHERAPY"):
        val = _safe_str(getattr(event, field, None))
        if val:
            features.append(f"{field.lower()}_{val.lower()}")
    return features

class SEERSurvivalTask(BaseTask):
    """Overall survival prediction task for the SEER cancer registry.

    Predicts overall survival for cancer patients using SEER clinical data:

    - **survival_months**: continuous time-to-death / censoring time in months
      (regression label).
    - **vital_status**: binary event indicator — 1 if the patient died (within
      the study window), 0 if alive at last follow-up (censored).

    Task Schema:
        Input:
            - patient_features: sequence of demographic / patient-level tokens
              (age bucket, sex, race, diagnosis decade).
            - tumour_features: sequence of tumour characteristic tokens
              (primary site, histology, AJCC stage, grade, tumour size,
              node status).
            - treatment_features: sequence of treatment indicator tokens
              (surgery, radiation, chemotherapy).
        Output:
            - survival_months: regression label (float, months ≥ 0).
            - vital_status: binary label (int 0/1).

    Args:
        min_survival_months: Minimum survival time to include (default 0).
            Patients with survival_months < this threshold are excluded.

    Examples:
        >>> from pyhealth.datasets import SEERDataset
        >>> from pyhealth.tasks import SEERSurvivalTask
        >>> dataset = SEERDataset(root="/path/to/seer")
        >>> samples = dataset.set_task(SEERSurvivalTask())
        >>> samples[0]['survival_months']
        84.0
        >>> samples[0]['vital_status']
        0
    """

    task_name: str = "SEERSurvivalTask"
    input_schema: Dict[str, str] = {
        "patient_features": "sequence",
        "tumour_features": "sequence",
        "treatment_features": "sequence",
    }
    output_schema: Dict[str, str] = {
        "survival_months": "regression",
        "vital_status": "binary",
    }

    def __init__(self, min_survival_months: float = 0.0):
        super().__init__()
        self.min_survival_months = min_survival_months

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        events = patient.get_events(event_type="seer")
        if not events:
            return []

        samples: List[Dict[str, Any]] = []
        for event in events:
            survival_months = _safe_float(getattr(event, "SURVIVAL_MONTHS", None))
            vital_status_raw = _safe_float(getattr(event, "VITAL_STATUS", None))

            if survival_months is None or vital_status_raw is None:
                continue
            if survival_months < self.min_survival_months:
                continue

            vital_status = int(vital_status_raw)
            if vital_status not in (0, 1):
                continue

            patient_features = _extract_patient_features(event)
            tumour_features = _extract_tumour_features(event)
            treatment_features = _extract_treatment_features(event)

            # Need at least patient and tumour features to form a useful sample
            if not patient_features or not tumour_features:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "patient_features": patient_features,
                    "tumour_features": tumour_features,
                    "treatment_features": treatment_features,
                    "survival_months": survival_months,
                    "vital_status": vital_status,
                }
            )

        return samples
