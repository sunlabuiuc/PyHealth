"""FAMEWS-style fairness-aware mortality prediction task for MIMIC-III.

Augments the standard :class:`pyhealth.tasks.MortalityPredictionMIMIC3` task
with seven demographic cohort attributes per sample (sex, age_group,
ethnicity_4, ethnicity_W, insurance_type, surgical_status, admission_type).
These cohort attributes are NOT used as model features — they are passed
through alongside the samples so downstream models can be audited for
cohort-level disparities via
:func:`pyhealth.tasks.fairness_utils.audit_predictions`.

Paper: Hoche, M., Mineeva, O., Burger, M., Blasimme, A., & Rätsch, G. (2024).
*FAMEWS: A Fairness Auditing Tool for Medical Early-Warning Systems.*
CHIL 2024, PMLR 248:297-311. https://proceedings.mlr.press/v248/hoche24a.html

Upstream FAMEWS code: https://github.com/ratschlab/famews

Contributor: Rahul Joshi (rahulpj2@illinois.edu)
"""
from typing import Any, Dict, List, Optional

from .base_task import BaseTask


def _bin_age(age_years: Optional[float]) -> str:
    """Bucket patient age into FAMEWS's canonical age groups.

    Matches ``config/group_mimic_complete.yaml`` from upstream FAMEWS.

    Args:
        age_years: Age at admission in years. MIMIC shifts >89 to pre-1900;
            values are clipped to ``[0, 90]``.

    Returns:
        One of ``"<50"``, ``"50-65"``, ``"65-75"``, ``"75-85"``, ``">85"``, or
        ``"unknown"`` if ``age_years`` is ``None``.
    """
    if age_years is None:
        return "unknown"
    age_years = max(0.0, min(float(age_years), 90.0))
    if age_years < 50:
        return "<50"
    if age_years < 65:
        return "50-65"
    if age_years < 75:
        return "65-75"
    if age_years < 85:
        return "75-85"
    return ">85"


def _normalize_ethnicity(ethnicity: Optional[str]) -> Dict[str, str]:
    """Map MIMIC-III's free-text ``ethnicity`` into two canonical groupings.

    Args:
        ethnicity: Free-text ethnicity string from MIMIC-III ADMISSIONS.

    Returns:
        Dict with keys:

        - ``ethnicity_4``: 4-class bucket
          ``{"WHITE", "BLACK", "OTHER", "UNK"}``
        - ``ethnicity_W``: binary bucket ``{"WHITE", "NON-WHITE"}``
    """
    e = (ethnicity or "").upper()
    if "WHITE" in e:
        eth4 = "WHITE"
    elif "BLACK" in e:
        eth4 = "BLACK"
    elif "UNKNOWN" in e or "UNABLE" in e or not e:
        eth4 = "UNK"
    else:
        eth4 = "OTHER"
    eth_w = "WHITE" if eth4 == "WHITE" else "NON-WHITE"
    return {"ethnicity_4": eth4, "ethnicity_W": eth_w}


def _surgical_status(first_careunit: Optional[str]) -> str:
    """Infer surgical vs non-surgical care from ``first_careunit``.

    SICU (surgical ICU) and CSRU (cardiac surgery recovery unit) are surgical;
    everything else is non-surgical.

    Args:
        first_careunit: ``first_careunit`` value from MIMIC-III ICUSTAYS.

    Returns:
        ``"Surgical"`` or ``"Non-surgical"``.
    """
    u = str(first_careunit or "").upper()
    return "Surgical" if ("SICU" in u or "CSRU" in u) else "Non-surgical"


def _normalize_admission_type(admission_type: Optional[str]) -> str:
    """Collapse MIMIC-III admission types into FAMEWS's two-class scheme.

    ``ELECTIVE`` stays elective; ``EMERGENCY`` and ``URGENT`` collapse to
    ``emergency``.

    Args:
        admission_type: MIMIC ``admission_type`` string (case-insensitive).

    Returns:
        ``"elective"`` or ``"emergency"``.
    """
    t = str(admission_type or "").lower()
    if "elect" in t:
        return "elective"
    return "emergency"


def _compute_age_years(patient: Any, admittime: Any) -> Optional[float]:
    """Compute patient age at admission in years using year-only math.

    MIMIC-III shifts DOB for elderly patients to pre-1677 to anonymize. Plain
    ``(admittime - dob).days / 365.25`` overflows pandas ``int64`` math on
    those records, so we use year subtraction (robust for 1-year resolution).

    Args:
        patient: PyHealth Patient-like object. Must have a ``.birth_datetime``
            attribute OR a ``dob`` event accessible via
            ``patient.get_events('patients')``.
        admittime: A datetime-like value with a ``.year`` attribute.

    Returns:
        Age in years, or ``None`` if either value is missing.
    """
    dob = getattr(patient, "birth_datetime", None)
    if dob is None:
        try:
            pat_events = patient.get_events(event_type="patients")
            if pat_events:
                dob = getattr(pat_events[0], "dob", None)
        except Exception:
            dob = None
    if dob is None or admittime is None:
        return None
    try:
        dob_year = int(getattr(dob, "year", None) or str(dob)[:4])
        adm_year = int(getattr(admittime, "year", None) or str(admittime)[:4])
    except (TypeError, ValueError):
        return None
    return float(adm_year - dob_year)


class MortalityPredictionWithFairnessMIMIC3(BaseTask):
    """Mortality prediction with FAMEWS-style cohort attributes for fairness audit.

    This task is a drop-in replacement for
    :class:`~pyhealth.tasks.MortalityPredictionMIMIC3` that additionally
    attaches seven demographic cohort attributes to each sample. The filter
    logic (predict mortality at next visit, drop visits missing any of
    conditions/procedures/drugs, drop final visit) is identical — the two
    tasks produce the same set of samples with the same labels, so they can
    be compared head-to-head.

    The extra cohort attributes do **not** appear in ``input_schema`` — they
    are pass-through fields consumed by
    :func:`pyhealth.tasks.fairness_utils.audit_predictions` after model
    inference, not by the model itself.

    Attributes:
        task_name: ``"MortalityPredictionWithFairnessMIMIC3"``.
        input_schema: ``{"conditions": "sequence", "procedures": "sequence",
            "drugs": "sequence"}`` — identical to ``MortalityPredictionMIMIC3``.
        output_schema: ``{"mortality": "binary"}``.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import (
        ...     MortalityPredictionWithFairnessMIMIC3,
        ...     audit_predictions,
        ... )
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        ... )
        >>> task = MortalityPredictionWithFairnessMIMIC3()
        >>> samples = dataset.set_task(task)
        >>> # ... train a model, collect test predictions in `y_prob` ...
        >>> audit = audit_predictions(samples, y_prob)
        >>> print(audit[audit["significantly_worse"]])

    References:
        Hoche, M., Mineeva, O., Burger, M., Blasimme, A., & Rätsch, G. (2024).
        *FAMEWS: A Fairness Auditing Tool for Medical Early-Warning Systems.*
        CHIL 2024, PMLR 248:297-311.
        https://proceedings.mlr.press/v248/hoche24a.html
    """

    task_name: str = "MortalityPredictionWithFairnessMIMIC3"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient, emitting one sample per eligible visit.

        Args:
            patient: PyHealth Patient with events for ``admissions``,
                ``diagnoses_icd``, ``procedures_icd``, ``prescriptions``, and
                (for demographics) ``patients`` / ``icustays``.

        Returns:
            List of sample dicts with keys:
            ``hadm_id``, ``patient_id``, ``conditions`` (List[str]),
            ``procedures`` (List[str]), ``drugs`` (List[str]),
            ``mortality`` (0|1), and seven cohort attributes
            (``sex``, ``age_group``, ``ethnicity_4``, ``ethnicity_W``,
            ``insurance_type``, ``surgical_status``, ``admission_type``).
        """
        samples: List[Dict[str, Any]] = []

        visits = patient.get_events(event_type="admissions")
        if len(visits) <= 1:
            return []

        # Patient-level demographics (gender from patients table)
        try:
            pat_events = patient.get_events(event_type="patients")
            gender_raw = getattr(pat_events[0], "gender", None) if pat_events else None
        except Exception:
            gender_raw = getattr(patient, "gender", None)
        sex = "F" if str(gender_raw or "M").upper().startswith("F") else "M"

        for i in range(len(visits) - 1):
            visit = visits[i]
            next_visit = visits[i + 1]

            if next_visit.hospital_expire_flag not in [0, 1, "0", "1"]:
                mortality_label = 0
            else:
                mortality_label = int(next_visit.hospital_expire_flag)

            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", visit.hadm_id)],
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", visit.hadm_id)],
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", visit.hadm_id)],
            )
            conditions = [e.icd9_code for e in diagnoses]
            procedures_list = [e.icd9_code for e in procedures]
            drugs = [e.ndc for e in prescriptions if getattr(e, "ndc", None)]
            if len(conditions) * len(procedures_list) * len(drugs) == 0:
                continue

            # Per-visit cohort attributes
            admittime = getattr(visit, "timestamp", None) or getattr(
                visit, "admittime", None
            )
            age = _compute_age_years(patient, admittime)
            eth = _normalize_ethnicity(getattr(visit, "ethnicity", None))

            # first_careunit requires an icustays lookup for this hadm
            first_careunit = None
            try:
                icu = patient.get_events(
                    event_type="icustays",
                    filters=[("hadm_id", "==", visit.hadm_id)],
                )
                if icu:
                    first_careunit = getattr(icu[0], "first_careunit", None)
            except Exception:
                pass

            samples.append(
                {
                    "hadm_id": visit.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "mortality": mortality_label,
                    "sex": sex,
                    "age_group": _bin_age(age),
                    "ethnicity_4": eth["ethnicity_4"],
                    "ethnicity_W": eth["ethnicity_W"],
                    "insurance_type": getattr(visit, "insurance", "UNK") or "UNK",
                    "surgical_status": _surgical_status(first_careunit),
                    "admission_type": _normalize_admission_type(
                        getattr(visit, "admission_type", None)
                    ),
                }
            )

        return samples
