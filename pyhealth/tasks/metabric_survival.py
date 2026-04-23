"""Survival analysis tasks for the METABRIC breast cancer dataset.

Two ready-to-use task classes are provided:

- :class:`METABRICSurvivalOS` — overall survival (OS_MONTHS / OS_STATUS).
- :class:`METABRICSurvivalRFS` — relapse-free survival (RFS_MONTHS / RFS_STATUS).

Both share the same feature set and differ only in the survival endpoint.

Usage
-----
>>> from pyhealth.datasets import METABRICDataset
>>> from pyhealth.tasks import METABRICSurvivalOS
>>>
>>> dataset = METABRICDataset(root="/path/to/metabric")
>>> task    = METABRICSurvivalOS()
>>> samples = dataset.set_task(task)
>>> samples[0].keys()
dict_keys(['patient_id', 'clinical_features', 'treatment_features',
           'os_months', 'os_status'])
"""

import logging
from typing import Any, Dict, List, Optional

from .base_task import BaseTask

logger = logging.getLogger(__name__)

def _safe_float(value: Any) -> Optional[float]:
    """Return float(value) or None if conversion fails."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_str(value: Any) -> Optional[str]:
    """Return stripped string or None if empty."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def _extract_clinical_features(event: Any) -> List[str]:
    """Build a list of descriptive clinical feature tokens.

    Each token encodes both the feature name and its value so that the
    downstream :class:`~pyhealth.processors.SequenceProcessor` can learn
    an embedding per (name, value) combination.
    """
    features: List[str] = []

    # Continuous → discretised bucket tokens
    age = _safe_float(getattr(event, "AGE_AT_DIAGNOSIS", None))
    if age is not None:
        bucket = "young" if age < 45 else ("middle" if age < 65 else "elderly")
        features.append(f"age_group_{bucket}")

    tumor_size = _safe_float(getattr(event, "TUMOR_SIZE", None))
    if tumor_size is not None:
        size_cat = "small" if tumor_size < 20 else ("medium" if tumor_size < 50 else "large")
        features.append(f"tumor_size_{size_cat}")

    npi = _safe_float(getattr(event, "NPI", None))
    if npi is not None:
        npi_cat = "good" if npi < 3.4 else ("moderate" if npi < 5.4 else "poor")
        features.append(f"npi_{npi_cat}")

    grade = _safe_float(getattr(event, "GRADE", None))
    if grade is not None:
        features.append(f"grade_{int(grade)}")

    tumor_stage = _safe_float(getattr(event, "TUMOR_STAGE", None))
    if tumor_stage is not None:
        features.append(f"stage_{int(tumor_stage)}")

    # Categorical tokens (raw value)
    for field in (
        "INFERRED_MENOPAUSAL_STATE",
        "CELLULARITY",
        "ER_IHC",
        "HER2_SNP6",
        "INTCLUST",
        "ONCOTREE_CODE",
        "THREEGENE",
        "TYPE_OF_BREAST_SURGERY",
        "PR_STATUS",
        "HER2_STATUS",
    ):
        val = _safe_str(getattr(event, field, None))
        if val is not None:
            features.append(f"{field.lower()}_{val.lower()}")

    return features


def _extract_treatment_features(event: Any) -> List[str]:
    """Build treatment indicator tokens."""
    features: List[str] = []
    for field in ("CHEMOTHERAPY", "HORMONE_THERAPY", "RADIO_THERAPY"):
        val = _safe_str(getattr(event, field, None))
        if val is not None:
            features.append(f"{field.lower()}_{val.lower()}")
    return features

class METABRICSurvivalOS(BaseTask):
    """Overall survival prediction task for the METABRIC dataset.

    Predicts overall survival (OS) for breast cancer patients:

    - **os_months**: continuous time-to-event / censoring time in months
      (regression label).
    - **os_status**: binary event indicator — 1 if the patient died, 0 if
      alive at last follow-up (censored).

    Task Schema:
        Input:
            - clinical_features: sequence of tokenised clinical attributes
              (age group, tumour size, grade, stage, ER/HER2 status, etc.)
            - treatment_features: sequence of treatment indicator tokens
              (chemotherapy, hormone therapy, radiotherapy).
        Output:
            - os_months: regression label (float, months).
            - os_status: binary label (int 0/1).

    Examples:
        >>> from pyhealth.datasets import METABRICDataset
        >>> from pyhealth.tasks import METABRICSurvivalOS
        >>> dataset = METABRICDataset(root="/path/to/metabric")
        >>> samples = dataset.set_task(METABRICSurvivalOS())
        >>> samples[0]['os_months']
        85.6
        >>> samples[0]['os_status']
        0
    """

    task_name: str = "METABRICSurvivalOS"
    input_schema: Dict[str, str] = {
        "clinical_features": "sequence",
        "treatment_features": "sequence",
    }
    output_schema: Dict[str, str] = {
        "os_months": "regression",
        "os_status": "binary",
    }

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        events = patient.get_events(event_type="metabric")
        if not events:
            return []

        event = events[0]

        os_months = _safe_float(getattr(event, "OS_MONTHS", None))
        os_status_raw = _safe_float(getattr(event, "OS_STATUS", None))

        if os_months is None or os_status_raw is None:
            return []
        if os_months < 0:
            return []

        os_status = int(os_status_raw)
        if os_status not in (0, 1):
            return []

        clinical_features = _extract_clinical_features(event)
        treatment_features = _extract_treatment_features(event)

        # Require at least some clinical features
        if not clinical_features:
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "clinical_features": clinical_features,
                "treatment_features": treatment_features,
                "os_months": os_months,
                "os_status": os_status,
            }
        ]


class METABRICSurvivalRFS(BaseTask):
    """Relapse-free survival prediction task for the METABRIC dataset.

    Predicts relapse-free survival (RFS) for breast cancer patients:

    - **rfs_months**: continuous time-to-relapse / censoring time in months
      (regression label).
    - **rfs_status**: binary event indicator — 1 if relapse or death occurred,
      0 if relapse-free at last follow-up (censored).

    Task Schema:
        Input:
            - clinical_features: sequence of tokenised clinical attributes.
            - treatment_features: sequence of treatment indicator tokens.
        Output:
            - rfs_months: regression label (float, months).
            - rfs_status: binary label (int 0/1).

    Examples:
        >>> from pyhealth.datasets import METABRICDataset
        >>> from pyhealth.tasks import METABRICSurvivalRFS
        >>> dataset = METABRICDataset(root="/path/to/metabric")
        >>> samples = dataset.set_task(METABRICSurvivalRFS())
        >>> samples[0]['rfs_months']
        62.3
        >>> samples[0]['rfs_status']
        1
    """

    task_name: str = "METABRICSurvivalRFS"
    input_schema: Dict[str, str] = {
        "clinical_features": "sequence",
        "treatment_features": "sequence",
    }
    output_schema: Dict[str, str] = {
        "rfs_months": "regression",
        "rfs_status": "binary",
    }

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        events = patient.get_events(event_type="metabric")
        if not events:
            return []

        event = events[0]

        rfs_months = _safe_float(getattr(event, "RFS_MONTHS", None))
        rfs_status_raw = _safe_float(getattr(event, "RFS_STATUS", None))

        if rfs_months is None or rfs_status_raw is None:
            return []
        if rfs_months < 0:
            return []

        rfs_status = int(rfs_status_raw)
        if rfs_status not in (0, 1):
            return []

        clinical_features = _extract_clinical_features(event)
        treatment_features = _extract_treatment_features(event)

        if not clinical_features:
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "clinical_features": clinical_features,
                "treatment_features": treatment_features,
                "rfs_months": rfs_months,
                "rfs_status": rfs_status,
            }
        ]
