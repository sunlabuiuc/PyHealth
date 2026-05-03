import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_task import BaseTask


class TCGACancerTypeTask(BaseTask):
    """Cancer type classification from bulk RNA-seq token sequences.

    This task aims to predict the TCGA cancer cohort label (e.g. BRCA, LUAD)
    from a tokenized bulk RNA-seq expression profile. Supports both a 5-cohort
    restricted setting and the full 33-cohort pan-cancer setting.

    Attributes:
        task_name: Name of the task.
        input_schema: Input feature schema.
        output_schema: Output label schema.
        cohorts: Optional list of cohort labels to restrict classification.

    Args:
        cohorts: Optional list of cohort abbreviations to include

    Examples:
        >>> from pyhealth.tasks import TCGACancerTypeTask
        >>> task = TCGACancerTypeTask()
        >>> task_5 = TCGACancerTypeTask(
        ...     cohorts=["BRCA", "BLCA", "GBMLGG", "LUAD", "UCEC"]
        ... ) # 5-cohort restricted setting
    """

    task_name: str = "TCGACancerTypeTask"
    input_schema: Dict[str, str] = {"token_ids": "sequence"}
    output_schema: Dict[str, str] = {"cancer_type": "multiclass"}

    def __init__(self, cohorts: Optional[List[str]] = None) -> None:
        """Initialize task with optional cohort restriction.

        Args:
            cohorts: Optional list of cohort abbreviations to include.
                Default: all cohorts.
        """

        self.cohorts = set(cohorts) if cohorts is not None else None

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process one patient into a classification sample.

        Args:
            patient: PyHealth patient object with ``rnaseq`` and
                ``clinical`` events.

        Returns:
            List with a single sample dict containing ``token_ids`` and
            ``cancer_type``, or empty list if cohort label is missing or
            filtered out.
        """
        rnaseq_events = patient.get_events(event_type="rnaseq")
        if len(rnaseq_events) == 0:
            return []

        event = rnaseq_events[0]
        cohort = getattr(event, "cohort", None)

        if cohort is None or str(cohort) == "nan":
            return []

        if self.cohorts is not None and cohort not in self.cohorts:
            return []

        token_ids = _extract_token_ids(event)
        if len(token_ids) == 0:
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "token_ids": token_ids,
                "cancer_type": cohort,
            }
        ]


class TCGASurvivalTask(BaseTask):
    """Survival time prediction from bulk RNA-seq token sequences.

    This task aims to predict patient survival time (days) and event indicator
    for use with Cox proportional hazards models, as in BulkRNABert.

    Attributes:
        task_name: Name of the task.
        input_schema: Input feature schema.
        output_schema: Output label schema.
        cohorts: Optional list of cohort labels to restrict to.

    Args:
        cohorts: Optional list of cohort abbreviations to include.

    Examples:
        >>> from pyhealth.tasks import TCGASurvivalTask
        >>> task = TCGASurvivalTask()
        >>> task_blca = TCGASurvivalTask(cohorts=["BLCA"])
    """

    task_name: str = "TCGASurvivalTask"
    input_schema: Dict[str, str] = {"token_ids": "sequence"}
    output_schema: Dict[str, str] = {
        "survival_time": "regression",
        "event": "binary",
    }

    def __init__(self, cohorts: Optional[List[str]] = None) -> None:
        """Initialize task with optional cohort restriction.

        Args:
            cohorts: Optional list of cohort abbreviations to include. 
                Default: all cohorts.
        """

        self.cohorts = set(cohorts) if cohorts is not None else None

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process one patient into a survival prediction sample.
            Survival time is ``days_to_death`` for deceased patients and
            ``days_to_last_follow_up`` for censored patients.

        Args:
            patient: PyHealth patient object with ``rnaseq`` and
                ``clinical`` events.

        Returns:
            List with a single sample dict containing ``token_ids``,
            ``survival_time`` (float, days), and ``event`` (0 or 1),
            or empty list if required fields are missing.
        """
        rnaseq_events = patient.get_events(event_type="rnaseq")
        clinical_events = patient.get_events(event_type="clinical")

        if len(rnaseq_events) == 0 or len(clinical_events) == 0:
            return []

        event = rnaseq_events[0]
        clin = clinical_events[0]

        cohort = getattr(event, "cohort", None)
        if self.cohorts is not None:
            if cohort is None or cohort not in self.cohorts:
                return []

        vital_raw = getattr(clin, "vital_status", None)
        if vital_raw is None or str(vital_raw) == "nan":
            return []

        vital_lower = str(vital_raw).strip().lower()
        if vital_lower in ("dead", "deceased", "1"):
            event_indicator = 1
        elif vital_lower in ("alive", "living", "0"):
            event_indicator = 0
        else:
            return []

        if event_indicator == 1:
            days_raw = getattr(clin, "days_to_death", None)
        else:
            days_raw = getattr(clin, "days_to_last_follow_up", None)

        survival_time = _safe_float(days_raw)
        if survival_time is None or survival_time <= 0:
            return []

        token_ids = _extract_token_ids(event)
        if len(token_ids) == 0:
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "token_ids": token_ids,
                "survival_time": survival_time,
                "event": event_indicator,
                "cohort": cohort,
            }
        ]


# Helpers

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert a value to float.

    Args:
        value: Value to convert.
        default: Default to return on failure.

    Returns:
        Float value or default.
    """
    if value is None or str(value) == "nan":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _gene_sort_key(name: str) -> Tuple[int, str]:
    """Sort gene columns before cohort meta; numeric GENE* suffixes first."""
    m = re.match(r"(?i)gene(\d+)$", name)
    if m:
        return (0, int(m.group(1)))
    return (1, name.lower())


def _extract_token_ids(event: Any) -> List[int]:
    """Extract integer token IDs from an RNA-seq event object.

    Args:
        event: PyHealth event object with gene bin attributes.

    Returns:
        List of integer token IDs, one per gene, in stable gene-column order.
    """
    skip = {"cohort", "patient_id", "timestamp", "visit_id", "record_id"}
    pairs: List[Tuple[str, int]] = []

    attr_items = (
        event.attr_dict.items()
        if hasattr(event, "attr_dict")
        else (
            (k, getattr(event, k))
            for k in vars(event)
            if not k.startswith("_") and k not in skip
        )
    )

    for attr, val in attr_items:
        if attr.lower() in {s.lower() for s in skip}:
            continue
        if val is None or (isinstance(val, str) and str(val).strip() == ""):
            continue
        if isinstance(val, (int, np.integer)):
            pairs.append((attr, int(val)))
            continue
        if isinstance(val, float) and not np.isnan(val):
            pairs.append((attr, int(val)))
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if np.isnan(f):
            continue
        pairs.append((attr, int(f)))

    pairs.sort(key=lambda kv: _gene_sort_key(kv[0]))
    return [v for _, v in pairs]