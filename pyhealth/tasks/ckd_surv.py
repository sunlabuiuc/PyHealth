from typing import Any, Dict, List, Literal, Union, Type, Optional
from datetime import timedelta
import polars as pl

from .base_task import BaseTask
from pyhealth.processors import (
    SequenceProcessor,
    TensorProcessor,
    RawProcessor,
)


class MIMIC4CKDSurvAnalysis(BaseTask):
    """Survival analysis for CKD progression on MIMIC-IV (CKD -> ESRD).

    This task prepares patient-level samples for survival modeling using
    MIMIC-IV tables (patients, admissions, diagnoses_icd, labevents). It
    supports three settings that change the input form:

    - "time_invariant": single-row snapshot per patient
    - "time_variant": time series with a single modality stream
    - "heterogeneous": time series with multiple lab modalities

    The time origin (t0) is the first available lab in the window.
    Positive cases are censored at the ESRD date (inclusive-by-date) and
    negatives at the last available lab. Durations are computed in days
    from t0.

    Inputs and outputs by setting
    - Common output (all settings):
        - duration_days: Tensor (float), days between t0 and censoring
        - has_esrd: Tensor (int, 0/1), whether ESRD occurred in the window

    - time_invariant inputs:
        - demographics: Sequence ([age_group, gender_str, race])
        - age: Tensor (float)
        - gender: Sequence (["M"|"F"]) for modeling as categorical
        - baseline_egfr: Tensor (float), from a single target lab
        - comorbidities: Sequence (ICD codes prior to t0)

    - time_variant inputs:
        - demographics: Sequence ([age_group, gender_str])
        - age: Tensor (float)
        - gender: Sequence (["M"|"F"]) for modeling as categorical
        - lab_measurements: Raw list[dict], ordered by days since t0
            Each element includes:
            - timestamp: int, days since t0
            - creatinine: float (if present)
                        - egfr: float (if present)
                        - has_esrd_step: int (0/1), only when ESRD-day exists
                        - extras via extra_lab_itemids (e.g. bun)
                        - bun_missing flag (0 present, 1 missing)

    - heterogeneous inputs:
        - demographics, age, gender: same as time_variant
        - lab_measurements: Raw list[dict] with multimodal labs per day
            Each element includes (when present):
            - timestamp: int
            - creatinine: float
            - egfr: float, derived from creatinine, age, gender
            - protein: float, albumin: float
            - egfr_missing/protein_missing/albumin_missing: int (0/1)
            - has_esrd_step: int (0/1) on the ESRD day
            - Any configured extras plus {name}_missing flags

    Parameters
    - setting: one of ["time_invariant", "time_variant", "heterogeneous"]
    - min_age: minimum age (years) to include in cohort (default 18)
    - prediction_window_days: not used to truncate currently; reserved
    - extra_lab_itemids: optional dict mapping feature name -> list of
        labevents.itemid strings to include as extra modalities. For each
        name,
        two fields may appear in lab_measurements: {name} (float) and
        {name}_missing (int 0/1). Values are aligned to days since t0.

    Notes
    - eGFR uses the CKD-EPI 2021 formula with base coefficient 142. See
        https://pubmed.ncbi.nlm.nih.gov/34554658/
    - Positives require at least one lab event recorded on the ESRD date,
        matching the original pipeline semantics.

    Example
    -------
    >>> from pyhealth.datasets import MIMIC4Dataset
    >>> from pyhealth.tasks.ckd_surv import MIMIC4CKDSurvAnalysis
    >>> dataset = MIMIC4Dataset(
    ...     root="/path/to/mimiciv/demo",
    ...     tables=[
    ...         "patients", "admissions", "labevents", "diagnoses_icd"
    ...     ],
    ...     dev=True,
    ... )
    >>> task = MIMIC4CKDSurvAnalysis(
    ...     setting="time_variant",
    ...     extra_lab_itemids={"bun": ["51006"]},
    ... )
    >>> dataset.set_task(task)
    >>> samples = dataset.samples
    >>> sample = samples[0]
    >>> sorted(sample.keys())
    ['age', 'demographics', 'duration_days', 'gender', 'has_esrd',
     'lab_measurements', 'patient_id']
    >>> sample['lab_measurements'][0].keys()
    dict_keys(['timestamp', 'egfr_missing', 'protein_missing',
               'albumin_missing', 'egfr', 'protein', 'albumin',
               'creatinine', 'has_esrd_step', 'bun_missing', 'bun'])

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
        extra_lab_itemids: Optional[Dict[str, List[str]]] = None,
    ):

        if setting not in self._SURVIVAL_SETTINGS:
            raise ValueError(f"Setting must be one of {self._SURVIVAL_SETTINGS}")

        self.setting = setting
        self.min_age = min_age
        self.prediction_window_days = prediction_window_days
        self.task_name = f"MIMIC4CKDSurvAnalysis_{self.setting}"
        # Optional extensibility: additional lab item IDs to extract from
        # labevents. Dict maps feature name -> list of itemids. Values will
        # appear inside lab_measurements as the feature name and a
        # corresponding "{feature}_missing" flag (0 present, 1 missing).
        self.extra_lab_itemids = extra_lab_itemids or {}
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

        # Pre-compute extra lab events (validated numeric)
        extra_events_map: Dict[str, List[Any]] = {}
        for feat_name, itemids in self.extra_lab_itemids.items():
            if not itemids:
                continue
            extra_events_map[feat_name] = [
                e for e in lab_events if e.itemid in itemids and _valid_numeric(e)
            ]

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
                # Filter extras by ESRD date if positive
                if has_esrd and esrd_date is not None:
                    filtered_extras: Dict[str, List[Any]] = {}
                    for name, events in extra_events_map.items():
                        filtered_extras[name] = [
                            e for e in events if e.timestamp.date() <= esrd_date.date()
                        ]
                else:
                    filtered_extras = extra_events_map

                return self._process_time_variant(
                    patient,
                    t0,
                    age,
                    gender,
                    duration_days,
                    has_esrd,
                    considered_creatinine,
                    esrd_date,
                    filtered_extras,
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
                for e in (
                    creatinine_events
                    + protein_events
                    + albumin_events
                    + [ev for lst in extra_events_map.values() for ev in lst]
                )
                if e.timestamp is not None
            ]
            if not timestamps:
                return []
            t0 = min(timestamps)

            if esrd_date is not None:
                # Require at least one lab on ESRD date
                any_on_esrd = any(
                    e.timestamp.date() == esrd_date.date()
                    for e in (
                        creatinine_events
                        + protein_events
                        + albumin_events
                        + [ev for lst in extra_events_map.values() for ev in lst]
                    )
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
                considered_extras = {
                    name: [e for e in events if e.timestamp.date() <= esrd_date.date()]
                    for name, events in extra_events_map.items()
                }
                has_esrd = 1
                duration_days = (esrd_date.date() - t0.date()).days
            else:
                considered_creatinine = creatinine_events
                considered_protein = protein_events
                considered_albumin = albumin_events
                considered_extras = extra_events_map
                has_esrd = 0
                last_time = max(
                    [
                        e.timestamp
                        for e in (
                            considered_creatinine
                            + considered_protein
                            + considered_albumin
                            + [ev for lst in considered_extras.values() for ev in lst]
                        )
                    ]
                )
                duration_days = (last_time.date() - t0.date()).days

            # Ensure at least two total timepoints across any lab
            total_events = len(
                {
                    e.timestamp
                    for e in (
                        considered_creatinine
                        + considered_protein
                        + considered_albumin
                        + [ev for lst in considered_extras.values() for ev in lst]
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
                considered_extras,
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
        extra_events_map: Optional[Dict[str, List[Any]]] = None,
    ):
        """
        Process for time-varying analysis aligned with original
        TIME_VARIANT.

        Build series from first lab (t0) up to ESRD date (if positive) or last
        lab (negative).
        """
        # Build union of timepoints across creatinine and extras
        extra_events_map = extra_events_map or {}
        measurements_by_time: Dict[int, Dict[str, Any]] = {}

        def _ensure_day(day: int):
            if day not in measurements_by_time:
                measurements_by_time[day] = {"timestamp": day}
                for name in extra_events_map.keys():
                    measurements_by_time[day][f"{name}_missing"] = 1
                    measurements_by_time[day][name] = 0.0

        # Creatinine and egfr
        considered_creatinine.sort(key=lambda x: x.timestamp)
        for e in considered_creatinine:
            try:
                creatinine_value = float(e.valuenum)
                if creatinine_value <= 0:
                    continue
            except (ValueError, TypeError):
                continue

            days_from_t0 = (e.timestamp.date() - t0.date()).days
            egfr_value = self._calculate_egfr(creatinine_value, age, gender)
            _ensure_day(days_from_t0)
            m = measurements_by_time[days_from_t0]
            m["egfr"] = egfr_value
            m["creatinine"] = creatinine_value
            if esrd_date is not None:
                m["has_esrd_step"] = int(e.timestamp.date() == esrd_date.date())
        # Extras
        for name, events in extra_events_map.items():
            for e in events:
                day = (e.timestamp.date() - t0.date()).days
                try:
                    val = float(e.valuenum)
                except (ValueError, TypeError):
                    continue
                if val <= 0:
                    continue
                _ensure_day(day)
                measurements_by_time[day][name] = val
                measurements_by_time[day][f"{name}_missing"] = 0

        lab_measurements = [
            measurements_by_time[d] for d in sorted(measurements_by_time.keys())
        ]

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
        extra_events_map: Optional[Dict[str, List[Any]]] = None,
    ):
        """Process for heterogeneous analysis with per-timestep missing flags.

        Missing flags use names: egfr_missing, protein_missing, albumin_missing
        (0/1).
        """
        measurements_by_time: Dict[int, Dict[str, Any]] = {}
        extra_events_map = extra_events_map or {}

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

        # Extras
        for name, events in extra_events_map.items():
            for e in events:
                days = (e.timestamp.date() - t0.date()).days
                try:
                    val = float(e.valuenum)
                except (ValueError, TypeError):
                    continue
                if val <= 0:
                    continue
                _upsert(days, {name: val, f"{name}_missing": 0})

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
