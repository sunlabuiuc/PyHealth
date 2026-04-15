"""MIMIC-III Against-Medical-Advice (AMA) discharge prediction task.

Defines :class:`AMAPredictionMIMIC3` and helpers for Boag et al. 2018-style
demographic baselines (race / substance-use ablations).
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_task import BaseTask

_SUBSTANCE_PATTERN = re.compile(
    r"alcohol|opioid|opiate|heroin|cocaine|drug|withdrawal"
    r"|intoxication|overdose|substance|etoh",
    re.IGNORECASE,
)


def _normalize_race(ethnicity: Optional[str]) -> str:
    """Map MIMIC-III ethnicity strings to the race categories used by
    Boag et al. 2018.

    Args:
        ethnicity: Raw ethnicity value from MIMIC-III ``admissions``
            table.

    Returns:
        One of ``"White"``, ``"Black"``, ``"Hispanic"``,
        ``"Asian"``, ``"Native American"``, or ``"Other"``.
    """
    if ethnicity is None:
        return "Other"
    eth = str(ethnicity).upper()
    if "HISPANIC" in eth or "SOUTH AMERICAN" in eth:
        return "Hispanic"
    if "AMERICAN INDIAN" in eth:
        return "Native American"
    if "ASIAN" in eth:
        return "Asian"
    if "BLACK" in eth:
        return "Black"
    if "WHITE" in eth:
        return "White"
    return "Other"


def _normalize_insurance(insurance: Optional[str]) -> str:
    """Map MIMIC-III insurance strings to the categories used by
    Boag et al. 2018.

    Args:
        insurance: Raw insurance value from MIMIC-III ``admissions``
            table.

    Returns:
        ``"Public"`` for Medicare/Medicaid/Government, or the
        original value (typically ``"Private"`` or ``"Self Pay"``).
    """
    if insurance is None:
        return "Other"
    if insurance in ("Medicare", "Medicaid", "Government"):
        return "Public"
    return insurance


def _safe_parse_datetime(value: Any) -> datetime:
    """Parse a datetime string, trying common MIMIC-III formats."""
    if isinstance(value, datetime):
        return value
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except (ValueError, TypeError):
            continue
    raise ValueError(f"Cannot parse datetime: {value!r}")


def _has_substance_use(diagnosis: Any) -> int:
    """Detect substance-use related admission from the free-text
    ``DIAGNOSIS`` field in MIMIC-III ``ADMISSIONS``.

    Args:
        diagnosis: Raw ``diagnosis`` string from the admissions table.

    Returns:
        1 if a substance-use keyword is found, 0 otherwise.
    """
    if diagnosis is None:
        return 0
    return 1 if _SUBSTANCE_PATTERN.search(str(diagnosis)) else 0


class AMAPredictionMIMIC3(BaseTask):
    """Predict whether a patient leaves the hospital against medical advice.

    This task reproduces the AMA (Against Medical Advice) discharge
    prediction target from Boag et al. 2018, "Racial Disparities and
    Mistrust in End-of-Life Care."  A positive label indicates that the
    patient's ``discharge_location`` is ``"LEFT AGAINST MEDICAL ADVI"``
    in the MIMIC-III admissions table.

    The feature set supports three ablation baselines:

    * **BASELINE** -- ``demographics`` (gender, insurance),
      ``age``, and ``los``.  Select with
      ``feature_keys=["demographics", "age", "los"]``.

    * **BASELINE + RACE** -- adds ``race`` (normalized ethnicity).
      Select with
      ``feature_keys=["demographics", "age", "los", "race"]``.

    * **BASELINE + RACE + SUBSTANCE** -- adds ``has_substance_use``
      (derived from ``ADMISSIONS.DIAGNOSIS`` free-text field).
      Select with
      ``feature_keys=["demographics", "age", "los", "race",
      "has_substance_use"]``.

    These baselines can be toggled via the model's ``feature_keys``
    parameter without changing the task.

    Only administrative and demographic features are extracted; no
    clinical code tables (diagnoses, procedures, prescriptions) are
    required.

    Unlike mortality or readmission prediction, the label is a property
    of the **current** admission, so patients with only one visit are
    eligible.

    **Processor mapping (schemas):** Each value in ``input_schema`` and
    ``output_schema`` must be a processor string key understood by
    ``dataset.set_task`` (see the Tasks docs processor table).  Here,
    ``"multi_hot"`` feeds ``MultiHotProcessor`` (token lists for
    ``demographics`` and ``race``), ``"tensor"`` feeds ``TensorProcessor``
    (``age``, ``los``, ``has_substance_use``), and ``"binary"`` labels
    ``ama`` for the binary label path.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import AMAPredictionMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=[],
        ... )
        >>> task = AMAPredictionMIMIC3()
        >>> samples = dataset.set_task(task)

    Reference:
        Boag, W.; Suresh, H.; Celi, L. A.; Szolovits, P.; and Ghassemi, M.
        2018. Racial Disparities and Mistrust in End-of-Life Care. In Machine
        Learning for Healthcare Conference. PMLR.
    """

    AMA_DISCHARGE_LOCATION: str = "LEFT AGAINST MEDICAL ADVI"

    task_name: str = "AMAPredictionMIMIC3"
    input_schema: Dict[str, str] = {
        "demographics": "multi_hot",
        "age": "tensor",
        "los": "tensor",
        "race": "multi_hot",
        "has_substance_use": "tensor",
    }
    output_schema: Dict[str, str] = {"ama": "binary"}

    def __init__(self, exclude_newborns: bool = True) -> None:
        """Initializes the AMA prediction task.

        Args:
            exclude_newborns: If ``True``, admissions whose
                ``admission_type`` is ``"NEWBORN"`` are skipped.
                Defaults to ``True``.
        """
        self.exclude_newborns = exclude_newborns

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for AMA discharge prediction.

        Each non-newborn admission is emitted as a sample.  The binary
        label is derived from the admission's ``discharge_location``.

        Args:
            patient: A Patient object from ``MIMIC3Dataset``.

        Returns:
            A list of sample dictionaries.  Each dictionary contains
            the features described in the class docstring plus
            ``visit_id``, ``patient_id``, and the ``ama`` label.
        """
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) == 0:
            return []

        patients_events = patient.get_events(event_type="patients")
        if len(patients_events) == 0:
            return []
        patient_info = patients_events[0]

        gender = getattr(patient_info, "gender", None)
        dob_raw = getattr(patient_info, "dob", None)
        try:
            dob = _safe_parse_datetime(dob_raw)
        except (ValueError, TypeError):
            dob = None

        samples: List[Dict[str, Any]] = []

        for admission in admissions:
            if self.exclude_newborns:
                admission_type = getattr(
                    admission, "admission_type", None
                )
                if admission_type == "NEWBORN":
                    continue

            # --- Label ---
            discharge_location = getattr(
                admission, "discharge_location", None
            )
            ama_label = (
                1
                if discharge_location == self.AMA_DISCHARGE_LOCATION
                else 0
            )

            # --- BASELINE demographics (gender + insurance) ---
            insurance = getattr(admission, "insurance", None)

            demo_tokens: List[str] = []
            if gender:
                demo_tokens.append(f"gender:{gender}")
            demo_tokens.append(
                f"insurance:{_normalize_insurance(insurance)}"
            )

            # --- Race (separate feature for ablation) ---
            ethnicity = getattr(admission, "ethnicity", None)
            race_tokens: List[str] = [
                f"race:{_normalize_race(ethnicity)}"
            ]

            # --- Age (continuous) ---
            age_years = 0.0
            if dob is not None:
                admit_dt = admission.timestamp
                if isinstance(admit_dt, datetime):
                    age_years = (
                        admit_dt.year
                        - dob.year
                        - int(
                            (admit_dt.month, admit_dt.day)
                            < (dob.month, dob.day)
                        )
                    )
                    age_years = float(min(age_years, 90))

            # --- LOS (continuous, in days) ---
            los_days = 0.0
            dischtime_raw = getattr(admission, "dischtime", None)
            if dischtime_raw is not None:
                try:
                    dischtime = _safe_parse_datetime(dischtime_raw)
                    admit_dt = admission.timestamp
                    if isinstance(admit_dt, datetime):
                        los_days = max(
                            (dischtime - admit_dt).total_seconds()
                            / 86400.0,
                            0.0,
                        )
                except (ValueError, TypeError):
                    los_days = 0.0

            # --- Substance use (from ADMISSIONS.DIAGNOSIS) ---
            diagnosis_text = getattr(admission, "diagnosis", None)
            substance = float(_has_substance_use(diagnosis_text))

            samples.append(
                {
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "demographics": demo_tokens,
                    "age": [age_years],
                    "los": [los_days],
                    "race": race_tokens,
                    "has_substance_use": [substance],
                    "ama": ama_label,
                }
            )

        return samples
