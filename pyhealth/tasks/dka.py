# Description: DKA (Diabetic Ketoacidosis) prediction tasks for MIMIC-IV dataset

import math
from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import polars as pl

from .base_task import BaseTask


class DKAPredictionMIMIC4(BaseTask):
    """Task for predicting Diabetic Ketoacidosis (DKA) in the general patient population.

    This task creates PATIENT-LEVEL samples from ALL patients in the dataset,
    predicting whether they will develop DKA. Features are collected from
    admissions BEFORE the first DKA event to prevent data leakage.

    Target Population:
        - ALL patients in the dataset (no filtering)
        - Large pool of negative samples (patients without DKA)

    Label Definition:
        - Positive (1): Patient has any DKA diagnosis code (ICD-9 or ICD-10)
        - Negative (0): Patient has no DKA diagnosis codes

    Data Leakage Prevention:
        - Admissions are sorted chronologically
        - For DKA-positive patients: Only data from admissions BEFORE the
          first DKA admission is included (no data from DKA admission or after)
        - For DKA-negative patients: All admissions are included
        - Patients whose first admission has DKA are excluded (no pre-DKA data)

    Features:
        - icd_codes: Combined diagnosis + procedure ICD codes (stagenet format)
        - labs: 10-dimensional vectors with lab categories

    Args:
        padding: Additional padding for StageNet processor. Default: 0.

    Example:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import DKAPredictionMIMIC4
        >>>
        >>> dataset = MIMIC4Dataset(
        ...     root="/path/to/mimic4",
        ...     tables=["diagnoses_icd", "procedures_icd", "labevents", "admissions"],
        ... )
        >>> task = DKAPredictionMIMIC4()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "DKAPredictionMIMIC4"

    # ICD-9 codes for Diabetic Ketoacidosis
    DKA_ICD9_CODES: ClassVar[Set[str]] = {"25010", "25011", "25012", "25013"}

    # ICD-10 prefix for DKA (E10.1x, E11.1x, E13.1x codes cover T1D, T2D, other DKA)
    DKA_ICD10_PREFIXES: ClassVar[List[str]] = ["E101", "E111", "E131"]

    # Lab categories from mortality_prediction_stagenet_mimic4.py (verified item IDs)
    LAB_CATEGORIES: ClassVar[Dict[str, List[str]]] = {
        "Sodium": ["50824", "52455", "50983", "52623"],
        "Potassium": ["50822", "52452", "50971", "52610"],
        "Chloride": ["50806", "52434", "50902", "52535"],
        "Bicarbonate": ["50803", "50804"],
        "Glucose": ["50809", "52027", "50931", "52569"],
        "Calcium": ["50808", "51624"],
        "Magnesium": ["50960"],
        "Anion Gap": ["50868", "52500"],
        "Osmolality": ["52031", "50964", "51701"],
        "Phosphate": ["50970"],
    }

    LAB_CATEGORY_ORDER: ClassVar[List[str]] = [
        "Sodium", "Potassium", "Chloride", "Bicarbonate", "Glucose",
        "Calcium", "Magnesium", "Anion Gap", "Osmolality", "Phosphate",
    ]

    # Flat list of all lab item IDs for filtering
    LABITEMS: ClassVar[List[str]] = [
        item for items in LAB_CATEGORIES.values() for item in items
    ]

    def __init__(self, padding: int = 0):
        """Initialize task with optional padding.

        Args:
            padding: Additional padding for nested sequences. Default: 0.
        """
        self.padding = padding
        self.input_schema: Dict[str, Tuple[str, Dict[str, Any]]] = { # type: ignore
            "icd_codes": ("stagenet", {"padding": padding}),
            "labs": ("stagenet_tensor", {}),
        }
        self.output_schema: Dict[str, str] = {"label": "binary"} # type: ignore

    def _is_dka_code(self, code: str, version: Any) -> bool:
        """Check if an ICD code represents Diabetic Ketoacidosis."""
        if not code:
            return False
        normalized = code.replace(".", "").strip().upper()
        version_str = str(version) if version is not None else ""

        if version_str == "10":
            return any(normalized.startswith(p) for p in self.DKA_ICD10_PREFIXES)
        if version_str == "9":
            return normalized in self.DKA_ICD9_CODES
        return False

    def _build_lab_vector(self, lab_df: pl.DataFrame) -> List[float]:
        """Build a 10D lab feature vector from lab events DataFrame."""
        if lab_df.height == 0:
            return [math.nan] * len(self.LAB_CATEGORY_ORDER)

        # Filter to relevant lab items and cast
        filtered = (
            lab_df.with_columns([
                pl.col("labevents/itemid").cast(pl.Utf8),
                pl.col("labevents/valuenum").cast(pl.Float64),
            ])
            .filter(pl.col("labevents/itemid").is_in(self.LABITEMS))
            .filter(pl.col("labevents/valuenum").is_not_null())
        )

        if filtered.height == 0:
            return [math.nan] * len(self.LAB_CATEGORY_ORDER)

        # Build vector with one value per category (mean of observed values)
        vector: List[float] = []
        for category in self.LAB_CATEGORY_ORDER:
            itemids = self.LAB_CATEGORIES[category]
            cat_df = filtered.filter(pl.col("labevents/itemid").is_in(itemids))
            if cat_df.height > 0:
                values = cat_df["labevents/valuenum"].drop_nulls()
                vector.append(float(values.mean()) if len(values) > 0 else math.nan) # type: ignore
            else:
                vector.append(math.nan)
        return vector

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient to create DKA prediction samples.

        Iterates through sorted admissions, collecting features until DKA is found.
        Label is based on whether DKA occurs in any future admission.

        Args:
            patient: Patient object with get_events method.

        Returns:
            List with single sample, or empty list if insufficient data.
        """
        # Get admissions and sort by timestamp
        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []

        # Sort admissions chronologically by timestamp
        admissions = sorted(admissions, key=lambda x: x.timestamp)

        # Initialize aggregated data structures
        all_icd_codes: List[List[str]] = []
        all_icd_times: List[float] = []
        all_lab_values: List[List[float]] = []
        all_lab_times: List[float] = []

        previous_admission_time: Optional[datetime] = None
        has_dka = False

        # Iterate through admissions in chronological order
        for admission in admissions:
            # Parse admission times
            try:
                admission_time = admission.timestamp
                dischtime_str = getattr(admission, "dischtime", None)
                if dischtime_str:
                    admission_dischtime = datetime.strptime(
                        dischtime_str, "%Y-%m-%d %H:%M:%S"
                    )
                else:
                    admission_dischtime = None
            except (ValueError, AttributeError):
                continue

            # Get diagnoses for this admission
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )

            # Iterate through diagnoses - check for DKA and collect codes
            visit_codes: List[str] = []
            seen: Set[str] = set()

            for diag in diagnoses:
                code = getattr(diag, "icd_code", None)
                version = getattr(diag, "icd_version", None)
                if not code:
                    continue

                # Check for DKA - if found, stop everything
                if self._is_dka_code(code, version):
                    has_dka = True
                    break

                # Add diagnosis code if not seen
                normalized = f"D_{code.replace('.', '').upper()}"
                if normalized not in seen:
                    seen.add(normalized)
                    visit_codes.append(normalized)

            # If DKA found, don't append this visit's data and stop
            if has_dka:
                break

            # Get procedures for this admission
            procedures = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )

            for proc in procedures:
                code = getattr(proc, "icd_code", None)
                if not code:
                    continue
                normalized = f"P_{code.replace('.', '').upper()}"
                if normalized not in seen:
                    seen.add(normalized)
                    visit_codes.append(normalized)

            # Calculate time from previous admission (hours)
            if previous_admission_time is None:
                time_from_previous = 0.0
            else:
                time_from_previous = (
                    admission_time - previous_admission_time
                ).total_seconds() / 3600.0
            previous_admission_time = admission_time

            # Append this visit's codes
            if visit_codes:
                all_icd_codes.append(visit_codes)
                all_icd_times.append(time_from_previous)

            # Get lab events for this admission using hadm_id
            lab_df = patient.get_events(
                event_type="labevents",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )

            if lab_df.height > 0:
                all_lab_values.append(self._build_lab_vector(lab_df))
                all_lab_times.append(time_from_previous)

        # Skip if no pre-DKA data (DKA on first visit or no valid admissions)
        if not all_icd_codes:
            return []

        # Ensure we have lab data (use NaN vector if missing)
        if not all_lab_values:
            all_lab_values = [[math.nan] * len(self.LAB_CATEGORY_ORDER)]
            all_lab_times = [0.0]

        return [{
            "patient_id": patient.patient_id,
            "record_id": patient.patient_id,
            "icd_codes": (all_icd_times, all_icd_codes),
            "labs": (all_lab_times, all_lab_values),
            "label": int(has_dka),
        }]


class T1DDKAPredictionMIMIC4(BaseTask):
    """Task for predicting Diabetic Ketoacidosis (DKA) in Type 1 Diabetes patients.

    This task creates PATIENT-LEVEL samples by identifying patients with Type 1
    Diabetes Mellitus (T1DM) and predicting whether they will develop DKA within
    a specified time window. Features are collected from admissions BEFORE the
    first DKA event to prevent data leakage.

    Target Population:
        - Patients with Type 1 Diabetes (ICD-9 or ICD-10 codes)
        - Excludes patients without any T1DM diagnosis codes

    Label Definition:
        - Positive (1): Patient has DKA code within 90 days of T1DM diagnosis
        - Negative (0): Patient has T1DM but no DKA within the window

    Data Leakage Prevention:
        - Admissions are sorted chronologically
        - For DKA-positive patients: Only data from admissions BEFORE the
          first DKA admission is included (no data from DKA admission or after)
        - For DKA-negative patients: All admissions are included
        - Patients whose first admission has DKA are excluded (no pre-DKA data)

    Features:
        - icd_codes: Combined diagnosis + procedure ICD codes (stagenet format)
        - labs: 10-dimensional vectors with lab categories

    Args:
        dka_window_days: Number of days to consider for DKA occurrence after
            T1DM diagnosis. Default: 90.
        padding: Additional padding for StageNet processor. Default: 0.

    Example:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import T1DDKAPredictionMIMIC4
        >>>
        >>> dataset = MIMIC4Dataset(
        ...     root="/path/to/mimic4",
        ...     tables=["diagnoses_icd", "procedures_icd", "labevents", "admissions"],
        ... )
        >>> task = T1DDKAPredictionMIMIC4(dka_window_days=90)
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "T1DDKAPredictionMIMIC4"

    # ICD-10 prefix for Type 1 Diabetes Mellitus
    T1DM_ICD10_PREFIX: ClassVar[str] = "E10"

    # ICD-9 codes for Type 1 Diabetes Mellitus
    T1DM_ICD9_CODES: ClassVar[Set[str]] = {
        "25001", "25003", "25011", "25013", "25021", "25023",
        "25031", "25033", "25041", "25043", "25051", "25053",
        "25061", "25063", "25071", "25073", "25081", "25083",
        "25091", "25093",
    }

    # ICD-9 codes for Diabetic Ketoacidosis
    DKA_ICD9_CODES: ClassVar[Set[str]] = {"25010", "25011", "25012", "25013"}

    # ICD-10 prefix for DKA (E10.1x codes)
    DKA_ICD10_PREFIX: ClassVar[str] = "E101"

    # Lab categories from mortality_prediction_stagenet_mimic4.py
    LAB_CATEGORIES: ClassVar[Dict[str, List[str]]] = {
        "Sodium": ["50824", "52455", "50983", "52623"],
        "Potassium": ["50822", "52452", "50971", "52610"],
        "Chloride": ["50806", "52434", "50902", "52535"],
        "Bicarbonate": ["50803", "50804"],
        "Glucose": ["50809", "52027", "50931", "52569"],
        "Calcium": ["50808", "51624"],
        "Magnesium": ["50960"],
        "Anion Gap": ["50868", "52500"],
        "Osmolality": ["52031", "50964", "51701"],
        "Phosphate": ["50970"],
    }

    LAB_CATEGORY_ORDER: ClassVar[List[str]] = [
        "Sodium", "Potassium", "Chloride", "Bicarbonate", "Glucose",
        "Calcium", "Magnesium", "Anion Gap", "Osmolality", "Phosphate",
    ]

    LABITEMS: ClassVar[List[str]] = [
        item for items in LAB_CATEGORIES.values() for item in items
    ]

    def __init__(self, dka_window_days: int = 90, padding: int = 0):
        """Initialize task with configurable DKA window and padding."""
        self.dka_window_days = dka_window_days
        self.padding = padding
        self.input_schema: Dict[str, Tuple[str, Dict[str, Any]]] = { # type: ignore
            "icd_codes": ("stagenet", {"padding": padding}),
            "labs": ("stagenet_tensor", {}),
        } 
        self.output_schema: Dict[str, str] = {"label": "binary"} # type: ignore

    def _is_t1dm_code(self, code: str | None, version: Any) -> bool:
        """Check if an ICD code represents Type 1 Diabetes Mellitus."""
        if not code:
            return False
        normalized = code.replace(".", "").strip().upper()
        version_str = str(version) if version is not None else ""

        if version_str == "10":
            return normalized.startswith(self.T1DM_ICD10_PREFIX)
        if version_str == "9":
            return normalized in self.T1DM_ICD9_CODES
        return False

    def _is_dka_code(self, code: str, version: Any) -> bool:
        """Check if an ICD code represents Diabetic Ketoacidosis."""
        if not code:
            return False
        normalized = code.replace(".", "").strip().upper()
        version_str = str(version) if version is not None else ""

        if version_str == "10":
            return normalized.startswith(self.DKA_ICD10_PREFIX)
        if version_str == "9":
            return normalized in self.DKA_ICD9_CODES
        return False

    def _build_lab_vector(self, lab_df: pl.DataFrame) -> List[float]:
        """Build a 10D lab feature vector from lab events DataFrame."""
        if lab_df.height == 0:
            return [math.nan] * len(self.LAB_CATEGORY_ORDER)

        filtered = (
            lab_df.with_columns([
                pl.col("labevents/itemid").cast(pl.Utf8),
                pl.col("labevents/valuenum").cast(pl.Float64),
            ])
            .filter(pl.col("labevents/itemid").is_in(self.LABITEMS))
            .filter(pl.col("labevents/valuenum").is_not_null())
        )

        if filtered.height == 0:
            return [math.nan] * len(self.LAB_CATEGORY_ORDER)

        vector: List[float] = []
        for category in self.LAB_CATEGORY_ORDER:
            itemids = self.LAB_CATEGORIES[category]
            cat_df = filtered.filter(pl.col("labevents/itemid").is_in(itemids))
            if cat_df.height > 0:
                values = cat_df["labevents/valuenum"].drop_nulls()
                vector.append(float(values.mean()) if len(values) > 0 else math.nan) # type: ignore
            else:
                vector.append(math.nan)
        return vector

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Filter to keep only patients with Type 1 Diabetes codes."""
        if "diagnoses_icd/icd_code" not in df.collect_schema().names():
            return df

        collected_df = df.collect()
        mask_icd10 = collected_df["diagnoses_icd/icd_code"].str.starts_with(
            self.T1DM_ICD10_PREFIX
        )
        mask_icd9 = collected_df["diagnoses_icd/icd_code"].is_in(
            list(self.T1DM_ICD9_CODES)
        )
        t1dm_patients = collected_df.filter(mask_icd10 | mask_icd9)["patient_id"].unique()
        return collected_df.filter(collected_df["patient_id"].is_in(t1dm_patients)).lazy()

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient to create DKA prediction samples.

        First checks if patient has T1DM before processing admissions.
        Iterates through sorted admissions, collecting features until DKA is found.
        Label is based on whether DKA occurs within the time window of T1DM diagnosis.

        Args:
            patient: Patient object with get_events method.

        Returns:
            List with single sample, or empty list if patient lacks T1DM or pre-DKA data.
        """
        # First check: does this patient have T1DM? (quick scan before expensive ops)
        all_diagnoses = patient.get_events(event_type="diagnoses_icd")
        if not all_diagnoses:
            return []

        has_t1dm = False
        t1dm_times: List[datetime] = []

        for diag in all_diagnoses:
            code = getattr(diag, "icd_code", None)
            version = getattr(diag, "icd_version", None)
            if self._is_t1dm_code(code, version):
                has_t1dm = True
                diag_time = getattr(diag, "timestamp", None)
                if diag_time:
                    t1dm_times.append(diag_time)

        # Skip patients without T1DM diagnosis (early exit before sorting)
        if not has_t1dm:
            return []

        # Get admissions and sort by timestamp
        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []

        # Sort admissions chronologically by timestamp
        admissions = sorted(admissions, key=lambda x: x.timestamp)

        # Track earliest T1DM time to enforce the DKA window on history length
        window_start = min(t1dm_times) if t1dm_times else None
        window_end = (
            window_start + timedelta(days=self.dka_window_days)
            if window_start is not None
            else None
        )

        # Initialize tracking variables
        all_icd_codes: List[List[str]] = []
        all_icd_times: List[float] = []
        all_lab_values: List[List[float]] = []
        all_lab_times: List[float] = []

        previous_admission_time: Optional[datetime] = None
        has_dka = False
        dka_time: Optional[datetime] = None

        # Iterate through admissions in chronological order
        for admission in admissions:
            # Parse admission times
            try:
                admission_time = admission.timestamp
                dischtime_str = getattr(admission, "dischtime", None)
                if dischtime_str:
                    admission_dischtime = datetime.strptime(
                        dischtime_str, "%Y-%m-%d %H:%M:%S"
                    )
                else:
                    admission_dischtime = None
            except (ValueError, AttributeError):
                continue

            # Stop once we are past the allowed window to avoid leakage from long histories
            if window_end is not None and admission_time > window_end:
                break

            # Get diagnoses for this admission
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )

            # Iterate through diagnoses - check for DKA and collect codes
            visit_codes: List[str] = []
            seen: Set[str] = set()
            stop_processing = False

            for diag in diagnoses:
                code = getattr(diag, "icd_code", None)
                version = getattr(diag, "icd_version", None)
                if not code:
                    continue

                # Check for DKA - if found, record time and stop
                if self._is_dka_code(code, version):
                    candidate_dka_time = getattr(diag, "timestamp", admission_time)
                    # If DKA occurs outside the window, treat as negative and stop
                    if window_end is not None and candidate_dka_time > window_end:
                        stop_processing = True
                        has_dka = False
                        dka_time = None
                        break

                    has_dka = True
                    dka_time = candidate_dka_time
                    break

                # Add diagnosis code if not seen
                normalized = f"D_{code.replace('.', '').upper()}"
                if normalized not in seen:
                    seen.add(normalized)
                    visit_codes.append(normalized)

            if stop_processing:
                break

            # If DKA found, don't append this visit's data and stop
            if has_dka:
                break

            # Get procedures for this admission
            procedures = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )

            for proc in procedures:
                code = getattr(proc, "icd_code", None)
                if not code:
                    continue
                normalized = f"P_{code.replace('.', '').upper()}"
                if normalized not in seen:
                    seen.add(normalized)
                    visit_codes.append(normalized)

            # Calculate time from previous admission (hours)
            if previous_admission_time is None:
                time_from_previous = 0.0
            else:
                time_from_previous = (
                    admission_time - previous_admission_time
                ).total_seconds() / 3600.0
            previous_admission_time = admission_time

            # Append this visit's codes
            if visit_codes:
                all_icd_codes.append(visit_codes)
                all_icd_times.append(time_from_previous)

            # Get lab events for this admission using hadm_id
            lab_df = patient.get_events(
                event_type="labevents",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )

            if lab_df.height > 0:
                all_lab_values.append(self._build_lab_vector(lab_df))
                all_lab_times.append(time_from_previous)

        # Skip if no pre-DKA data
        if not all_icd_codes:
            return []

        # Determine label based on temporal relationship
        has_dka_within_window = False
        if has_dka and t1dm_times and dka_time:
            for t1dm_time in t1dm_times:
                delta = abs((dka_time - t1dm_time).days)
                if delta <= self.dka_window_days:
                    has_dka_within_window = True
                    break
        elif has_dka and not t1dm_times:
            # Fallback: if no temporal info, use has_dka
            has_dka_within_window = True

        # Ensure we have lab data
        if not all_lab_values:
            all_lab_values = [[math.nan] * len(self.LAB_CATEGORY_ORDER)]
            all_lab_times = [0.0]

        return [{
            "patient_id": patient.patient_id,
            "record_id": patient.patient_id,
            "icd_codes": (all_icd_times, all_icd_codes),
            "labs": (all_lab_times, all_lab_values),
            "label": int(has_dka_within_window),
        }]
