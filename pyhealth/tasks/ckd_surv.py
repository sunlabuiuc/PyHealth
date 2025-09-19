from typing import Any, Dict, List, Literal, Union, Type
from datetime import timedelta
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
    _BASE_COEFFICIENT = 142  # Match original pipeline constant
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
            # Raw lab measurements; per-timestep missing flags inside each
            # measurement element
            base_input.update({"lab_measurements": RawProcessor})

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

        # Gather diagnoses
        ckd_diagnoses = patient.get_events(event_type="diagnoses_icd")
        ckd_events = [e for e in ckd_diagnoses if e.icd_code in self._CKD_CODES]
        if not ckd_events:
            return []
        esrd_events = [e for e in ckd_diagnoses if e.icd_code in self._ESRD_CODES]
        esrd_date = min((e.timestamp for e in esrd_events), default=None)

        # Collect lab events relevant to the scenario and validate
        lab_events = patient.get_events(event_type="labevents")

        def _valid_numeric(e):
            try:
                return (
                    e.valuenum is not None
                    and float(e.valuenum) > 0
                    and e.timestamp is not None
                )
            except (ValueError, TypeError):
                return False

        # Select labs per scenario
        if self.setting in ("time_invariant", "time_variant"):
            creatinine_events = [
                e
                for e in lab_events
                if e.itemid in self._CREATININE_ITEMIDS and _valid_numeric(e)
            ]
            if not creatinine_events:
                return []
            # t0 is first creatinine lab
            t0 = min(e.timestamp for e in creatinine_events)
            # For positives, keep labs up to ESRD date (inclusive by date)
            if esrd_date is not None:
                # Require at least one lab on the ESRD date to match original
                # pipeline
                labs_on_esrd_date = [
                    e
                    for e in creatinine_events
                    if e.timestamp.date() == esrd_date.date()
                ]
                if not labs_on_esrd_date:
                    return []
                considered_creatinine = [
                    e
                    for e in creatinine_events
                    if e.timestamp.date() <= esrd_date.date()
                ]
                has_esrd = 1
                duration_days = (esrd_date.date() - t0.date()).days
            else:
                considered_creatinine = creatinine_events
                has_esrd = 0
                last_lab_time = max(e.timestamp for e in considered_creatinine)
                duration_days = (last_lab_time.date() - t0.date()).days

            # Need at least two labs in the window
            if len(considered_creatinine) < 2 or duration_days <= 0:
                return []

            # Dispatch per setting
            if self.setting == "time_invariant":
                return self._process_time_invariant(
                    patient,
                    t0,
                    age,
                    gender,
                    duration_days,
                    has_esrd,
                    considered_creatinine,
                    esrd_date,
                )
            else:
                return self._process_time_variant(
                    patient,
                    t0,
                    age,
                    gender,
                    duration_days,
                    has_esrd,
                    considered_creatinine,
                    esrd_date,
                )

        else:  # heterogeneous
            # Consider creatinine, protein, albumin
            creatinine_events = [
                e
                for e in lab_events
                if e.itemid in self._CREATININE_ITEMIDS and _valid_numeric(e)
            ]
            protein_events = [
                e
                for e in lab_events
                if e.itemid in self._PROTEIN_ITEMIDS and _valid_numeric(e)
            ]
            albumin_events = [
                e
                for e in lab_events
                if e.itemid in self._ALBUMIN_ITEMIDS and _valid_numeric(e)
            ]

            # Need creatinine to derive egfr at minimum
            if not creatinine_events:
                return []

            # t0 is min across all available labs for this scenario
            timestamps = [
                e.timestamp
                for e in (creatinine_events + protein_events + albumin_events)
                if e.timestamp is not None
            ]
            if not timestamps:
                return []
            t0 = min(timestamps)

            if esrd_date is not None:
                # Require at least one lab on ESRD date
                any_on_esrd = any(
                    e.timestamp.date() == esrd_date.date()
                    for e in (creatinine_events + protein_events + albumin_events)
                )
                if not any_on_esrd:
                    return []
                considered_creatinine = [
                    e
                    for e in creatinine_events
                    if e.timestamp.date() <= esrd_date.date()
                ]
                considered_protein = [
                    e for e in protein_events if e.timestamp.date() <= esrd_date.date()
                ]
                considered_albumin = [
                    e for e in albumin_events if e.timestamp.date() <= esrd_date.date()
                ]
                has_esrd = 1
                duration_days = (esrd_date.date() - t0.date()).days
            else:
                considered_creatinine = creatinine_events
                considered_protein = protein_events
                considered_albumin = albumin_events
                has_esrd = 0
                last_time = max(
                    [
                        e.timestamp
                        for e in (
                            considered_creatinine
                            + considered_protein
                            + considered_albumin
                        )
                    ]
                )
                duration_days = (last_time.date() - t0.date()).days

            # Ensure at least two total timepoints across any lab
            total_events = len(
                {
                    e.timestamp
                    for e in (
                        considered_creatinine + considered_protein + considered_albumin
                    )
                }
            )
            if total_events < 2 or duration_days <= 0:
                return []

            return self._process_heterogeneous(
                patient,
                t0,
                age,
                gender,
                duration_days,
                has_esrd,
                considered_creatinine,
                considered_protein,
                considered_albumin,
                esrd_date,
            )

    def _process_time_invariant(
        self,
        patient,
        t0,
        age,
        gender,
        duration_days,
        has_esrd,
        considered_creatinine,
        esrd_date,
    ):
        """
        Process for time-invariant analysis aligned with original
        NON_TIME_VARIANT.

        - Positives: pick lab on ESRD date (last that day) and compute egfr
        - Negatives: pick last available lab
        """
        # Choose target creatinine event
        if has_esrd and esrd_date is not None:
            same_day_events = [
                e
                for e in considered_creatinine
                if e.timestamp.date() == esrd_date.date()
            ]
            if not same_day_events:
                return []
            target_event = max(same_day_events, key=lambda x: x.timestamp)
        else:
            target_event = max(considered_creatinine, key=lambda x: x.timestamp)

        try:
            creatinine_value = float(target_event.valuenum)
        except (ValueError, TypeError):
            return []
        if creatinine_value <= 0:
            return []

        egfr = self._calculate_egfr(creatinine_value, age, gender)

        # Comorbidities before first lab (t0)
        diagnoses = patient.get_events(event_type="diagnoses_icd")
        comorbidities = [
            e.icd_code
            for e in diagnoses
            if e.timestamp is not None and e.timestamp <= t0 and e.icd_code
        ]

        # Race from admissions (optional meta)
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
        self,
        patient,
        t0,
        age,
        gender,
        duration_days,
        has_esrd,
        considered_creatinine,
        esrd_date,
    ):
        """
        Process for time-varying analysis aligned with original
        TIME_VARIANT.

        Build series from first lab (t0) up to ESRD date (if positive) or last
        lab (negative).
        """
        considered_creatinine.sort(key=lambda x: x.timestamp)
        lab_measurements = []
        for e in considered_creatinine:
            try:
                creatinine_value = float(e.valuenum)
                if creatinine_value <= 0:
                    continue
            except (ValueError, TypeError):
                continue

            days_from_t0 = (e.timestamp.date() - t0.date()).days
            egfr_value = self._calculate_egfr(creatinine_value, age, gender)
            m = {
                "timestamp": days_from_t0,
                "egfr": egfr_value,
                "creatinine": creatinine_value,
            }
            if esrd_date is not None:
                m["has_esrd_step"] = int(e.timestamp.date() == esrd_date.date())
            lab_measurements.append(m)

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
        self,
        patient,
        t0,
        age,
        gender,
        duration_days,
        has_esrd,
        creatinine_events,
        protein_events,
        albumin_events,
        esrd_date,
    ):
        """Process for heterogeneous analysis with per-timestep missing flags.

        Missing flags use names: egfr_missing, protein_missing, albumin_missing
        (0/1).
        """
        measurements_by_time: Dict[int, Dict[str, Any]] = {}

        def _upsert(days: int, updates: Dict[str, Any]):
            if days not in measurements_by_time:
                measurements_by_time[days] = {
                    "timestamp": days,
                    "egfr_missing": 1,
                    "protein_missing": 1,
                    "albumin_missing": 1,
                    "egfr": 0.0,
                    "protein": 0.0,
                    "albumin": 0.0,
                    "creatinine": 0.0,
                }
            measurements_by_time[days].update(updates)

        # Within-window events already considered by caller
        for e in creatinine_events:
            days = (e.timestamp.date() - t0.date()).days
            try:
                cr = float(e.valuenum)
            except (ValueError, TypeError):
                continue
            if cr <= 0:
                continue
            egfr = self._calculate_egfr(cr, age, gender)
            _upsert(days, {"egfr": egfr, "creatinine": cr, "egfr_missing": 0})

        for e in protein_events:
            days = (e.timestamp.date() - t0.date()).days
            try:
                pv = float(e.valuenum)
            except (ValueError, TypeError):
                continue
            if pv <= 0:
                continue
            _upsert(days, {"protein": pv, "protein_missing": 0})

        for e in albumin_events:
            days = (e.timestamp.date() - t0.date()).days
            try:
                av = float(e.valuenum)
            except (ValueError, TypeError):
                continue
            if av <= 0:
                continue
            _upsert(days, {"albumin": av, "albumin_missing": 0})

        if len(measurements_by_time) < 2:
            return []

        lab_measurements: List[Dict[str, Any]] = []
        for days in sorted(measurements_by_time.keys()):
            m = measurements_by_time[days]
            if esrd_date is not None:
                # Set step-level ESRD flag when day matches ESRD date
                m["has_esrd_step"] = int(
                    (t0.date() + timedelta(days=days)) == esrd_date.date()
                )
            lab_measurements.append(m)

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
