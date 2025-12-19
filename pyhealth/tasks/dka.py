# Description: DKA (Diabetic Ketoacidosis) prediction task for MIMIC-IV dataset

import math
from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import polars as pl

from .base_task import BaseTask


class DKAPredictionMIMIC4(BaseTask):
    """Task for predicting Diabetic Ketoacidosis (DKA) in Type 1 Diabetes patients.

    This task creates PATIENT-LEVEL samples by identifying patients with Type 1
    Diabetes Mellitus (T1DM) and predicting whether they will develop DKA within
    a specified time window. The task uses diagnosis codes and lab results across
    all admissions in StageNet format for temporal modeling.

    Target Population:
        - Patients with Type 1 Diabetes (ICD-9 or ICD-10 codes)
        - Excludes patients without any T1DM diagnosis codes

    Label Definition:
        - Positive (1): Patient has DKA code within 90 days of T1DM diagnosis
        - Negative (0): Patient has T1DM but no DKA within the window

    Time Calculation:
        - Diagnosis codes: Hours from previous admission (0 for first visit)
        - Labs: Hours from admission start (within-visit measurements)

    Lab Processing:
        - 6-dimensional vectors (one per lab category relevant to DKA)
        - Categories: glucose, bicarbonate, anion_gap, potassium, sodium, chloride
        - Multiple itemids per category → take mean of observed values
        - Missing categories → NaN in vector

    Args:
        dka_window_days: Number of days to consider for DKA occurrence after
            T1DM diagnosis. Default: 90.
        padding: Additional padding for StageNet processor. Default: 0.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, Tuple[str, Dict]]): The schema for input data:
            - diagnoses: Diagnosis ICD codes (stagenet format, nested by visit)
            - labs: Lab results (stagenet_tensor, 6D vectors per timestamp)
        output_schema (Dict[str, str]): The schema for output data:
            - label: Binary indicator (1 if DKA within window, 0 otherwise)

    Example:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import DKAPredictionMIMIC4
        >>>
        >>> dataset = MIMIC4Dataset(
        ...     root="/path/to/mimic4",
        ...     tables=["diagnoses_icd", "labevents", "admissions"],
        ... )
        >>> task = DKAPredictionMIMIC4(dka_window_days=90)
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "DKAPredictionMIMIC4"

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

    # Lab categories relevant to DKA monitoring
    # Each category maps to ONE dimension in the output vector
    LAB_CATEGORIES: ClassVar[Dict[str, List[str]]] = {
        "glucose": ["50809", "50931", "52027", "52569"],
        "bicarbonate": ["50803", "50804", "51084"],
        "anion_gap": ["50868"],
        "potassium": ["50822", "50971", "52452", "52510"],
        "sodium": ["50824", "50983", "52455"],
        "chloride": ["50806", "50902", "52434"],
    }

    # Ordered list of category names (defines vector dimension order)
    LAB_CATEGORY_ORDER: ClassVar[List[str]] = [
        "glucose",
        "bicarbonate",
        "anion_gap",
        "potassium",
        "sodium",
        "chloride",
    ]

    # Flat list of all lab item IDs for filtering
    ALL_LAB_ITEMIDS: ClassVar[List[str]] = sorted(
        {item for items in LAB_CATEGORIES.values() for item in items}
    )

    def __init__(self, dka_window_days: int = 90, padding: int = 0):
        """Initialize task with configurable DKA window and padding.

        Args:
            dka_window_days: Days after T1DM diagnosis to check for DKA. Default: 90.
            padding: Additional padding for nested sequences. Default: 0.
        """
        self.dka_window_days = dka_window_days
        self.dka_window = timedelta(days=dka_window_days)
        self.padding = padding

        # Use tuple format to pass kwargs to processor
        self.input_schema: Dict[str, Tuple[str, Dict[str, Any]]] = {
            "diagnoses": ("stagenet", {"padding": padding}),
            "labs": ("stagenet_tensor", {}),
        }
        self.output_schema: Dict[str, str] = {"label": "binary"}

    @staticmethod
    def _normalize_icd(code: Optional[str]) -> str:
        """Normalize ICD code by removing dots and standardizing format.

        Args:
            code: Raw ICD code string, may contain dots and varying case.

        Returns:
            Normalized uppercase code without dots, or empty string if None.
        """
        if code is None:
            return ""
        return code.replace(".", "").strip().upper()

    def _is_t1dm_code(self, code: Optional[str], version: Optional[object]) -> bool:
        """Check if an ICD code represents Type 1 Diabetes Mellitus.

        Args:
            code: ICD diagnosis code.
            version: ICD version (9 or 10).

        Returns:
            True if code represents T1DM, False otherwise.
        """
        normalized = self._normalize_icd(code)
        if not normalized:
            return False

        version_str = str(version) if version is not None else ""

        if version_str == "10":
            return normalized.startswith(self.T1DM_ICD10_PREFIX)
        if version_str == "9":
            return normalized in self.T1DM_ICD9_CODES

        return False

    def _is_dka_code(self, code: Optional[str], version: Optional[object]) -> bool:
        """Check if an ICD code represents Diabetic Ketoacidosis.

        Args:
            code: ICD diagnosis code.
            version: ICD version (9 or 10).

        Returns:
            True if code represents DKA, False otherwise.
        """
        normalized = self._normalize_icd(code)
        if not normalized:
            return False

        version_str = str(version) if version is not None else ""

        if version_str == "10":
            return normalized.startswith(self.DKA_ICD10_PREFIX)
        if version_str == "9":
            return normalized in self.DKA_ICD9_CODES

        return False

    @staticmethod
    def _safe_parse_datetime(value: Optional[object]) -> Optional[datetime]:
        """Safely parse a datetime value from various formats.

        Args:
            value: Datetime string or datetime object.

        Returns:
            Parsed datetime object, or None if parsing fails.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value

        text = str(value)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    @staticmethod
    def _deduplicate_preserve_order(values: List[str]) -> List[str]:
        """Remove duplicates from a list while preserving order.

        Args:
            values: List of strings with potential duplicates.

        Returns:
            Deduplicated list maintaining original order.
        """
        seen: Set[str] = set()
        ordered: List[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Filter to keep only patients with Type 1 Diabetes codes.

        This pre-filter reduces the dataset to patients who have at least
        one T1DM diagnosis code (ICD-9 or ICD-10), improving processing
        efficiency for the downstream task.

        Args:
            df: LazyFrame containing patient data with diagnosis codes.

        Returns:
            Filtered LazyFrame with only T1DM patients.
        """
        # Check if we have the diagnosis code column
        if "diagnoses_icd/icd_code" not in df.collect_schema().names():
            # Column not present, return unfiltered
            return df

        # Collect to DataFrame for filtering
        collected_df = df.collect()

        # Create masks for T1DM codes
        mask_icd10 = collected_df["diagnoses_icd/icd_code"].str.starts_with(
            self.T1DM_ICD10_PREFIX
        )
        mask_icd9 = collected_df["diagnoses_icd/icd_code"].is_in(
            list(self.T1DM_ICD9_CODES)
        )

        # Get patient IDs with T1DM codes
        t1dm_mask = mask_icd10 | mask_icd9
        t1dm_patients = collected_df.filter(t1dm_mask)["patient_id"].unique()

        # Keep only patients with T1DM codes
        filtered_df = collected_df.filter(
            collected_df["patient_id"].is_in(t1dm_patients)
        )

        return filtered_df.lazy()

    def _build_lab_vector(self, lab_df: Optional[pl.DataFrame]) -> List[float]:
        """Build a lab feature vector from lab events DataFrame.

        Creates a fixed-dimension vector with one value per lab category.
        Uses mean aggregation when multiple values exist for a category.

        Args:
            lab_df: DataFrame containing lab events with itemid and valuenum columns.

        Returns:
            List of floats with length equal to number of lab categories.
            Missing categories are represented as NaN.
        """
        feature_dim = len(self.LAB_CATEGORY_ORDER)

        if lab_df is None or lab_df.height == 0:
            return [math.nan] * feature_dim

        # Filter and cast columns
        filtered = (
            lab_df.with_columns([
                pl.col("labevents/itemid").cast(pl.Utf8),
                pl.col("labevents/valuenum").cast(pl.Float64),
            ])
            .filter(pl.col("labevents/itemid").is_in(self.ALL_LAB_ITEMIDS))
            .filter(pl.col("labevents/valuenum").is_not_null())
        )

        if filtered.height == 0:
            return [math.nan] * feature_dim

        # Build vector with one value per category
        vector: List[float] = []
        for category in self.LAB_CATEGORY_ORDER:
            itemids = self.LAB_CATEGORIES[category]
            cat_df = filtered.filter(pl.col("labevents/itemid").is_in(itemids))

            if cat_df.height == 0:
                vector.append(math.nan)
            else:
                values = cat_df["labevents/valuenum"].drop_nulls()
                if len(values) > 0:
                    vector.append(float(values.mean()))
                else:
                    vector.append(math.nan)

        return vector

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient to create DKA prediction samples.

        Creates ONE sample per patient with T1DM diagnosis. Aggregates all
        admissions and calculates whether DKA occurred within the specified
        time window of T1DM diagnosis.

        Args:
            patient: Patient object with get_events method.

        Returns:
            List with single sample containing patient_id, diagnosis sequences,
            lab sequences, and DKA label. Returns empty list if patient does
            not have T1DM or lacks required data.
        """
        # Get all diagnosis events
        diagnosis_events = patient.get_events(event_type="diagnoses_icd")
        if not diagnosis_events:
            return []

        # Identify T1DM and DKA occurrences with timestamps
        has_t1dm = False
        t1dm_times: List[datetime] = []
        dka_times: List[datetime] = []

        for event in diagnosis_events:
            version = getattr(event, "icd_version", None)
            code = getattr(event, "icd_code", None)
            event_time = self._safe_parse_datetime(getattr(event, "timestamp", None))

            if self._is_t1dm_code(code, version):
                has_t1dm = True
                if event_time is not None:
                    t1dm_times.append(event_time)

            if self._is_dka_code(code, version):
                if event_time is not None:
                    dka_times.append(event_time)

        # Skip patients without T1DM diagnosis
        if not has_t1dm:
            return []

        # Determine DKA label based on temporal relationship
        has_dka_within_window = False

        if dka_times and t1dm_times:
            for diagnosis_time in t1dm_times:
                for dka_time in dka_times:
                    if diagnosis_time is None or dka_time is None:
                        continue
                    delta = abs((dka_time - diagnosis_time).days)
                    if delta <= self.dka_window_days:
                        has_dka_within_window = True
                        break
                if has_dka_within_window:
                    break

        # Fallback: if no temporal info, check if patient has any DKA codes
        if not has_dka_within_window and not t1dm_times:
            has_dka_within_window = len(dka_times) > 0

        # Get admission information
        admissions = patient.get_events(event_type="admissions")
        admissions_info: Dict[str, Dict[str, Optional[datetime]]] = {}

        if admissions:
            for admission in admissions:
                hadm_id = getattr(admission, "hadm_id", None)
                admit_time = self._safe_parse_datetime(
                    getattr(admission, "timestamp", None)
                )
                discharge_time = self._safe_parse_datetime(
                    getattr(admission, "dischtime", None)
                )
                if hadm_id is not None and admit_time is not None:
                    admissions_info[str(hadm_id)] = {
                        "admit": admit_time,
                        "discharge": discharge_time,
                    }

        # Create dummy admission if none exist
        if not admissions_info:
            dummy_hadm_id = f"dummy_{patient.patient_id}"
            admissions_info[dummy_hadm_id] = {
                "admit": datetime.now(),
                "discharge": None,
            }

        # Build diagnosis code sequences per admission
        admission_codes: Dict[str, List[str]] = {
            hadm_id: [] for hadm_id in admissions_info
        }

        for event in diagnosis_events:
            code = getattr(event, "icd_code", None)
            normalized_code = self._normalize_icd(code)
            if not normalized_code:
                continue

            hadm_id = getattr(event, "hadm_id", None)
            admission_key = (
                str(hadm_id)
                if hadm_id is not None
                else list(admissions_info.keys())[0]
            )

            if admission_key in admission_codes:
                admission_codes[admission_key].append(normalized_code)

        # Sort admissions chronologically
        sorted_admissions = sorted(
            admissions_info.items(),
            key=lambda item: item[1]["admit"]
            if item[1]["admit"] is not None
            else datetime.min,
        )

        # Build sequences
        diagnoses_sequences: List[List[str]] = []
        diagnoses_times: List[float] = []
        lab_sequences: List[List[float]] = []
        lab_times: List[float] = []
        previous_admit: Optional[datetime] = None

        for hadm_id, info in sorted_admissions:
            admit_time = info["admit"]
            if admit_time is None:
                admit_time = datetime.now()

            # Get deduplicated codes for this admission
            codes = self._deduplicate_preserve_order(
                admission_codes.get(hadm_id, [])
            )
            if not codes:
                codes = ["UNKNOWN"]

            # Calculate time gap from previous admission (hours)
            time_gap = (
                0.0
                if previous_admit is None
                else (admit_time - previous_admit).total_seconds() / 3600.0
            )
            previous_admit = admit_time

            diagnoses_sequences.append(codes)
            diagnoses_times.append(time_gap)

            # Get lab data for this admission
            try:
                lab_df = patient.get_events(
                    event_type="labevents",
                    start=admit_time,
                    end=info.get("discharge"),
                    return_df=True,
                )
            except Exception:
                lab_df = None

            lab_sequences.append(self._build_lab_vector(lab_df))
            lab_times.append(time_gap)

        # Ensure we have at least one sequence entry
        if not diagnoses_sequences:
            diagnoses_sequences = [["UNKNOWN"]]
            diagnoses_times = [0.0]
            lab_sequences = [[math.nan] * len(self.LAB_CATEGORY_ORDER)]
            lab_times = [0.0]

        # Create sample in StageNet format
        sample: Dict[str, Any] = {
            "patient_id": patient.patient_id,
            "record_id": patient.patient_id,
            "diagnoses": (diagnoses_times, diagnoses_sequences),
            "labs": (lab_times, lab_sequences),
            "label": int(has_dka_within_window),
        }

        return [sample]

