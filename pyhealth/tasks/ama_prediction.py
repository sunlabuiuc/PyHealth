from datetime import datetime
from typing import Any, Dict, List

from .base_task import BaseTask


def _normalize_race(ethnicity: str) -> str:
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


def _normalize_insurance(insurance: str) -> str:
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


class AMAPredictionMIMIC3(BaseTask):
    """Predict whether a patient leaves the hospital against medical advice.

    This task reproduces the AMA (Against Medical Advice) discharge
    prediction target from Boag et al. 2018, "Racial Disparities and
    Mistrust in End-of-Life Care." A positive label indicates that the
    patient's ``discharge_location`` is ``"LEFT AGAINST MEDICAL ADVI"``
    in the MIMIC-III admissions table.

    The feature set follows the paper's **BASELINE+RACE**
    configuration:

    * **demographics** (multi-hot) -- gender, normalized race, and
      normalized insurance category.
    * **age** (tensor) -- patient age at admission in years.
    * **los** (tensor) -- length of hospital stay in days.
    * **conditions** (sequence) -- ICD-9 diagnosis codes.
    * **procedures** (sequence) -- ICD-9 procedure codes.
    * **drugs** (sequence) -- prescription drug names.

    The demographic and clinical-code features can be used
    independently or together via the model's ``feature_keys``
    parameter, enabling ablation studies that mirror the paper.

    Unlike mortality or readmission prediction, the label is a property
    of the **current** admission, so patients with only one visit are
    eligible.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import AMAPredictionMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
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
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
        "demographics": "multi_hot",
        "age": "tensor",
        "los": "tensor",
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

        Each admission with at least one diagnosis code, one procedure
        code, and one prescription is emitted as a sample.  The binary
        label is derived from the admission's ``discharge_location``.

        Args:
            patient: A Patient object from ``MIMIC3Dataset``.

        Returns:
            A list of sample dictionaries, each containing:
                - ``visit_id``: MIMIC-III ``hadm_id``.
                - ``patient_id``: MIMIC-III ``subject_id``.
                - ``conditions``: List of ICD-9 diagnosis codes.
                - ``procedures``: List of ICD-9 procedure codes.
                - ``drugs``: List of drug names from prescriptions.
                - ``demographics``: List of categorical tokens
                  (gender, race, insurance).
                - ``age``: Patient age at admission in years (float).
                - ``los``: Hospital length of stay in days (float).
                - ``ama``: Binary label (1 = AMA discharge, 0 = other).
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

            # --- Clinical codes ---
            hadm_filter = ("hadm_id", "==", admission.hadm_id)

            diagnoses = patient.get_events(
                event_type="diagnoses_icd", filters=[hadm_filter]
            )
            conditions = [event.icd9_code for event in diagnoses]
            if len(conditions) == 0:
                continue

            procedures = patient.get_events(
                event_type="procedures_icd", filters=[hadm_filter]
            )
            procedures_list = [event.icd9_code for event in procedures]
            if len(procedures_list) == 0:
                continue

            prescriptions = patient.get_events(
                event_type="prescriptions", filters=[hadm_filter]
            )
            drugs = [event.drug for event in prescriptions]
            if len(drugs) == 0:
                continue

            # --- Demographics (categorical) ---
            ethnicity = getattr(admission, "ethnicity", None)
            insurance = getattr(admission, "insurance", None)

            demo_tokens: List[str] = []
            if gender:
                demo_tokens.append(f"gender:{gender}")
            demo_tokens.append(f"race:{_normalize_race(ethnicity)}")
            demo_tokens.append(
                f"insurance:{_normalize_insurance(insurance)}"
            )

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

            samples.append(
                {
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "demographics": demo_tokens,
                    "age": [age_years],
                    "los": [los_days],
                    "ama": ama_label,
                }
            )

        return samples
