from typing import Any, Dict, List, Literal, Union, Type
import polars as pl

from .base_task import BaseTask
from pyhealth.processors import (
    SequenceProcessor,
    TensorProcessor,
    RawProcessor,
)


class MIMIC4CKDSurvAnalysis(BaseTask):
    """CKD survival analysis task with simplified configuration.

        eGFR calculation methodology adapted from:
    - Original implementation: pkgs.data.utils.calculate_eGFR()
    - Formula source: pkgs.data.store.get_egfr_df()
        - Reference: CKD-EPI 2021 formula
            (https://pubmed.ncbi.nlm.nih.gov/34554658/)
    """

    # Private class variables for settings
    _SURVIVAL_SETTINGS = ["time_invariant", "time_variant", "heterogeneous"]
    _CKD_CODES = ["N183", "N184", "N185", "585.3", "585.4", "585.5"]
    _ESRD_CODES = ["N186", "Z992", "585.6", "V42.0"]
    _CREATININE_ITEMIDS = ["50912", "52546"]
    _PROTEIN_ITEMIDS = ["50976"]
    _ALBUMIN_ITEMIDS = ["50862"]

    # Gender constants (using MIMIC-IV native string values)
    _MALE_GENDER = "M"  # Male patients
    _FEMALE_GENDER = "F"  # Female patients

    # CKD-EPI 2021 equation constants (from pkgs.data.utils.calculate_eGFR)
    _BASE_COEFFICIENT = 141  # Original uses 142 in utils.py
    _AGE_FACTOR = 0.993  # Annual age decline factor
    _FEMALE_ADJUSTMENT = 1.018  # Female gender boost factor

    # Gender-specific creatinine thresholds and exponents
    _MALE_CREAT_THRESHOLD = 0.9  # mg/dL
    _FEMALE_CREAT_THRESHOLD = 0.7  # mg/dL
    _MALE_ALPHA_EXPONENT = -0.411  # For creatinine ≤ 0.9
    _FEMALE_ALPHA_EXPONENT = -0.329  # For creatinine ≤ 0.7
    _BETA_EXPONENT = -1.209  # For creatinine > threshold (both genders)

    def __init__(
        self,
        setting: Literal[
            "time_invariant", "time_variant", "heterogeneous"
        ] = "time_invariant",
        min_age: int = 18,
        prediction_window_days: int = 365 * 5,
    ):

        if setting not in self._SURVIVAL_SETTINGS:
            raise ValueError(f"Setting must be one of {self._SURVIVAL_SETTINGS}")

        self.setting = setting
        self.min_age = min_age
        self.prediction_window_days = prediction_window_days
        self.task_name = f"MIMIC4CKDSurvAnalysis_{self.setting}"
        self.input_schema, self.output_schema = self._configure_schemas()

    def _configure_schemas(
        self,
    ) -> tuple[Dict[str, Union[str, Type]], Dict[str, Union[str, Type]]]:
        """Configure schemas based on survival setting.

        Use registered processors:
        - "sequence" for categorical lists
          (e.g., demographics, gender, comorbidities)
        - "tensor" for numeric values (e.g., age, eGFR, durations, labels)
        """
        base_input: Dict[str, Union[str, Type]] = {
            "demographics": SequenceProcessor,
            "age": TensorProcessor,
            "gender": SequenceProcessor,
        }

        base_output: Dict[str, Union[str, Type]] = {
            "duration_days": TensorProcessor,
            "has_esrd": TensorProcessor,
        }

        if self.setting == "time_invariant":
            base_input.update(
                {
                    "baseline_egfr": TensorProcessor,
                    "comorbidities": SequenceProcessor,
                }
            )
        elif self.setting == "time_variant":
            # Use raw processor for time series list of dicts
            base_input.update({"lab_measurements": RawProcessor})
        else:  # heterogeneous
            # Raw lab measurements with sequence for missing indicators
            base_input.update(
                {
                    "lab_measurements": RawProcessor,
                    "missing_indicators": SequenceProcessor,
                }
            )

        return base_input, base_output

    def filter_patients(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Filter for CKD patients with required lab data."""
        ckd_patients = (
            df.filter(pl.col("event_type") == "diagnoses_icd")
            .filter(pl.col("diagnoses_icd/icd_code").is_in(self._CKD_CODES))
            .select("patient_id")
            .unique()
        )

        lab_patients = (
            df.filter(pl.col("event_type") == "labevents")
            .filter(pl.col("labevents/itemid").is_in(self._CREATININE_ITEMIDS))
            .select("patient_id")
            .unique()
        )

        valid_patients = ckd_patients.join(lab_patients, on="patient_id", how="inner")
        return df.filter(
            pl.col("patient_id").is_in(valid_patients.select("patient_id"))
        )

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process patient for survival analysis."""
        # Get demographics
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        demo = demographics[0]
        age = int(demo.anchor_age or 0)
        gender = (demo.gender or "").upper()

        if gender not in [self._MALE_GENDER, self._FEMALE_GENDER]:
            return []  # Skip patients with invalid/missing gender

        if age < self.min_age:
            return []

        # Get CKD baseline date
        ckd_diagnoses = patient.get_events(event_type="diagnoses_icd")
        ckd_events = [e for e in ckd_diagnoses if e.icd_code in self._CKD_CODES]

        if not ckd_events:
            return []

        baseline_date = min(e.timestamp for e in ckd_events)

        # Get ESRD outcome
        esrd_events = [e for e in ckd_diagnoses if e.icd_code in self._ESRD_CODES]

        if esrd_events:
            esrd_date = min(
                e.timestamp for e in esrd_events if e.timestamp > baseline_date
            )
            has_esrd = 1
            duration_days = (esrd_date - baseline_date).days
        else:
            has_esrd = 0
            # Get all events to find last observation
            # FIXED: filter out None timestamps
            all_events = patient.get_events()  # Get all events
            # Filter out events with None timestamps before finding max
            valid_events = [e for e in all_events if e.timestamp is not None]

            if valid_events:
                last_event = max(valid_events, key=lambda x: x.timestamp)
                duration_days = min(
                    (last_event.timestamp - baseline_date).days,
                    self.prediction_window_days,
                )
            else:
                # Fallback: use prediction window if no valid timestamps found
                duration_days = self.prediction_window_days

        if duration_days <= 0:
            return []

        # Process by setting
        if self.setting == "time_invariant":
            return self._process_time_invariant(
                patient, baseline_date, age, gender, duration_days, has_esrd
            )
        elif self.setting == "time_variant":
            return self._process_time_variant(
                patient, baseline_date, age, gender, duration_days, has_esrd
            )
        else:  # heterogeneous
            return self._process_heterogeneous(
                patient, baseline_date, age, gender, duration_days, has_esrd
            )

    def _process_time_invariant(
        self, patient, baseline_date, age, gender, duration_days, has_esrd
    ):
        """Process for time-invariant analysis."""
        lab_events = patient.get_events(event_type="labevents")
        creatinine_events = [
            e
            for e in lab_events
            if (
                e.itemid in self._CREATININE_ITEMIDS
                and e.valuenum is not None
                and e.timestamp >= baseline_date
            )
        ]

        if not creatinine_events:
            return []

        # Validate and find baseline creatinine
        valid_creatinine_events = []
        for e in creatinine_events:
            try:
                creatinine_value = float(e.valuenum)
                if creatinine_value > 0:
                    valid_creatinine_events.append((e, creatinine_value))
            except (ValueError, TypeError):
                continue

        if not valid_creatinine_events:
            return []

        # Closest to baseline
        _, baseline_creatinine_value = min(
            valid_creatinine_events,
            key=lambda x: abs((x[0].timestamp - baseline_date).days),
        )

        egfr = self._calculate_egfr(baseline_creatinine_value, age, gender)

        # Comorbidities before baseline
        diagnoses = patient.get_events(event_type="diagnoses_icd")
        comorbidities = [
            e.icd_code for e in diagnoses if e.timestamp <= baseline_date and e.icd_code
        ]

        # Race from admissions
        admissions = patient.get_events(event_type="admissions")
        race = admissions[0].race if admissions else "unknown"

        age_group = "elderly" if age >= 65 else "adult"
        gender_str = "male" if gender == self._MALE_GENDER else "female"

        sample = {
            "patient_id": patient.patient_id,
            "demographics": [age_group, gender_str, race],
            "baseline_egfr": egfr,
            "comorbidities": comorbidities,
            "age": float(age),
            "gender": [gender],
            "duration_days": float(duration_days),
            "has_esrd": has_esrd,
        }
        return [sample]

    def _process_time_variant(
        self, patient, baseline_date, age, gender, duration_days, has_esrd
    ):
        """Process for time-varying analysis."""
        lab_events = patient.get_events(event_type="labevents")
        creatinine_events = [
            e
            for e in lab_events
            if (
                e.itemid in self._CREATININE_ITEMIDS
                and e.valuenum is not None
                and e.timestamp >= baseline_date
            )
        ]

        if len(creatinine_events) < 2:
            return []

        # Chronological order
        creatinine_events.sort(key=lambda x: x.timestamp)
        lab_measurements = []

        for e in creatinine_events:
            try:
                creatinine_value = float(e.valuenum)
                if creatinine_value <= 0:
                    continue
            except (ValueError, TypeError):
                continue

            days_from_baseline = (e.timestamp - baseline_date).days
            egfr_value = self._calculate_egfr(creatinine_value, age, gender)
            lab_measurements.append(
                {
                    "timestamp": days_from_baseline,
                    "egfr": egfr_value,
                    "creatinine": creatinine_value,
                }
            )

        age_group = "elderly" if age >= 65 else "adult"
        gender_str = "male" if gender == self._MALE_GENDER else "female"

        sample = {
            "patient_id": patient.patient_id,
            "demographics": [age_group, gender_str],
            "lab_measurements": lab_measurements,
            "age": float(age),
            "gender": [gender],
            "duration_days": float(duration_days),
            "has_esrd": has_esrd,
        }
        return [sample]

    def _process_heterogeneous(
        self, patient, baseline_date, age, gender, duration_days, has_esrd
    ):
        """Process for heterogeneous analysis with missing indicators."""
        lab_events = patient.get_events(event_type="labevents")

        # Validator for lab values
        def validate(events, itemids):
            out = []
            for e in events:
                if e.itemid in itemids and e.valuenum is not None:
                    try:
                        v = float(e.valuenum)
                        if v > 0:
                            out.append((e, v))
                    except (ValueError, TypeError):
                        continue
            return out

        creatinine_events = validate(lab_events, self._CREATININE_ITEMIDS)
        protein_events = validate(lab_events, self._PROTEIN_ITEMIDS)
        albumin_events = validate(lab_events, self._ALBUMIN_ITEMIDS)

        if not creatinine_events:
            return []

        measurements_by_time: Dict[int, Dict[str, Any]] = {}

        # Creatinine/eGFR
        for e, creatinine_value in creatinine_events:
            if e.timestamp >= baseline_date:
                days = (e.timestamp - baseline_date).days
                egfr = self._calculate_egfr(creatinine_value, age, gender)
                if days not in measurements_by_time:
                    measurements_by_time[days] = {"timestamp": days}
                measurements_by_time[days].update(
                    {
                        "egfr": egfr,
                        "creatinine": creatinine_value,
                        "missing_egfr": False,
                    }
                )

        # Protein
        for e, protein_value in protein_events:
            if e.timestamp >= baseline_date:
                days = (e.timestamp - baseline_date).days
                if days not in measurements_by_time:
                    measurements_by_time[days] = {
                        "timestamp": days,
                        "missing_egfr": True,
                    }
                measurements_by_time[days].update(
                    {"protein": protein_value, "missing_protein": False}
                )

        # Albumin
        for e, albumin_value in albumin_events:
            if e.timestamp >= baseline_date:
                days = (e.timestamp - baseline_date).days
                if days not in measurements_by_time:
                    measurements_by_time[days] = {
                        "timestamp": days,
                        "missing_egfr": True,
                    }
                measurements_by_time[days].update(
                    {"albumin": albumin_value, "missing_albumin": False}
                )

        if len(measurements_by_time) < 2:
            return []

        # Sorted list and fill missing flags
        lab_measurements: List[Dict[str, Any]] = []
        for days in sorted(measurements_by_time.keys()):
            m = measurements_by_time[days]
            m.setdefault("missing_egfr", True)
            m.setdefault("missing_protein", True)
            m.setdefault("missing_albumin", True)
            m.setdefault("egfr", 0.0)
            m.setdefault("protein", 0.0)
            m.setdefault("albumin", 0.0)
            m.setdefault("creatinine", 0.0)
            lab_measurements.append(m)

        missing_indicators = set()
        for m in lab_measurements:
            for key, value in m.items():
                if key.startswith("missing_") and value:
                    missing_indicators.add(key)

        age_group = "elderly" if age >= 65 else "adult"
        gender_str = "male" if gender == self._MALE_GENDER else "female"

        sample = {
            "patient_id": patient.patient_id,
            "demographics": [age_group, gender_str],
            "lab_measurements": lab_measurements,
            "missing_indicators": list(missing_indicators),
            "age": float(age),
            "gender": [gender],
            "duration_days": float(duration_days),
            "has_esrd": has_esrd,
        }
        return [sample]

    def _calculate_egfr(self, creatinine: float, age: int, gender: str) -> float:
        """Calculate eGFR using simplified CKD-EPI equation.

        Implementation adapted from original MIMIC-IV analysis code:
        - Source file: pkgs.data.utils.calculate_eGFR()
        - Formula: CKD-EPI 2021 (https://pubmed.ncbi.nlm.nih.gov/34554658/)
        - Original coefficient: 142 (updated from 141 in this implementation)

        CKD-EPI Formula Constants (from original utils.py):
        - 0.9/0.7: Gender-specific creatinine thresholds (mg/dL)
        - 0.993: Age factor per year
        - 1.018: Female gender adjustment factor
        - -0.411/-0.329: Alpha exponents for creatinine ≤ threshold
        - -1.209: Beta exponent for creatinine > threshold (both genders)

        Args:
            creatinine: Serum creatinine in mg/dL (MIMIC-IV native units)
            age: Patient age in years
            gender: Gender string ('M' for male, 'F' for female)

        Returns:
            Estimated GFR in mL/min/1.73m²
        """
        # Validate inputs (following original validation)
        if creatinine <= 0:
            raise ValueError(f"Invalid creatinine value: {creatinine}")
        if gender not in [self._MALE_GENDER, self._FEMALE_GENDER]:
            raise ValueError(f"Invalid gender: {gender}")

        # Ensure creatinine is float for calculations
        creatinine = float(creatinine)

        if gender == self._MALE_GENDER:  # Male
            return (
                self._BASE_COEFFICIENT
                * min(creatinine / self._MALE_CREAT_THRESHOLD, 1)
                ** self._MALE_ALPHA_EXPONENT
                * max(creatinine / self._MALE_CREAT_THRESHOLD, 1) ** self._BETA_EXPONENT
                * self._AGE_FACTOR**age
            )
        else:  # Female (gender == self._FEMALE_GENDER)
            return (
                self._BASE_COEFFICIENT
                * min(creatinine / self._FEMALE_CREAT_THRESHOLD, 1)
                ** self._FEMALE_ALPHA_EXPONENT
                * max(creatinine / self._FEMALE_CREAT_THRESHOLD, 1)
                ** self._BETA_EXPONENT
                * self._AGE_FACTOR**age
                * self._FEMALE_ADJUSTMENT
            )
