# Description: Catheter-associated urinary infection prediction task for MIMIC-IV dataset

from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import polars as pl
from pyhealth.medcode import CrossMap

from .base_task import BaseTask


class _CatheterInfectionBase(BaseTask):
    """Shared helpers for catheter-associated infection prediction tasks."""

    _MAPPER_CACHE: ClassVar[Dict[Tuple[str, str], Optional[CrossMap]]] = {}
    MISSING_TOKEN: ClassVar[str] = "<missing>"

    @staticmethod
    def _task_name_with_same_visit(base_name: str, same_visit: bool) -> str:
        return f"{base_name}_samevisit_{str(same_visit).lower()}"

    @staticmethod
    def _task_name_with_map_ccscm(base_name: str, map_ccscm: bool) -> str:
        return f"{base_name}_mapccscm_{str(map_ccscm).lower()}"

    # ICD-10 diagnosis/procedure codes indicating catheter use
    CATHETER_CODES_ICD10: ClassVar[Set[str]] = {
        "Y846",  # Y84.6  — urinary catheterization as cause of abnormal reaction
        "Z466",  # Z46.6  — encounter for fitting/adjustment of urinary device
        "Z4682",  # Z46.82 — encounter for fitting/adjustment of non-vascular catheter
        "Z935",  # Z93.5  — cystostomy status
        "Z936",  # Z93.6  — other artificial urinary opening status
        "0T9B70Z",  # Foley catheter placement
        "0T2BX0Z",  # Foley removal
        "0T9C7ZZ",  # Routine Foley placement
    }

    # ICD-10 urinary-catheter complication families (prefix matching)
    CATHETER_PREFIXES_ICD10: ClassVar[Tuple[str, ...]] = (
        "T8301",  # T83.010-T83.018 breakdown
        "T8302",  # T83.020-T83.028 displacement
        "T8303",  # T83.030-T83.038 leakage
        "T8309",  # T83.090-T83.098 other mechanical complications
    )

    # ICD-9 diagnosis/procedure/external cause subset indicating catheter use
    CATHETER_CODES_ICD9: ClassVar[Set[str]] = {
        "99631",  # Mechanical complication of urethral catheter
        "99632",  # Mechanical complication of intrauterine contraceptive device
        "E8705",  # Misadventure in catheterization
        "E8796",  # Urinary catheterization causing abnormal reaction
    }

    # ICD-9 cardiac catheterization procedure range 37.21-37.23
    CATHETER_PREFIXES_ICD9: ClassVar[Tuple[str, ...]] = (
        "3721",
        "3722",
        "3723",
    )

    # -----------------------------------------------------------------------
    # Infection codes — Tier 1: Unconditional positive
    # Any admission containing these codes is a positive CAUTI event,
    # regardless of whether a catheter code is present in the same admission.
    # -----------------------------------------------------------------------
    INFECTION_CODES_UNCONDITIONAL_ICD10: ClassVar[Set[str]] = {
        "T83511A",  # T83.511A — CAUTI, initial encounter
        "T83518A",
        "T83518D",
        "T83518S",  # Other urinary catheter infection
        "T83519A",
        "T83519D",
        "T83519S",  # Unspecified urinary catheter infection
    }
    # Note: T83.511D / T83.511S (ongoing complication of a prior CAUTI) are
    # intentionally excluded from positive labels — they signal an existing
    # complication, not a new infection event, and including them as positive
    # labels could introduce temporal confusion.  They remain in the feature
    # vocabulary as regular diagnosis tokens.

    INFECTION_CODES_UNCONDITIONAL_ICD9: ClassVar[Set[str]] = {
        "99664",  # Infection due to indwelling urinary catheter
    }

    # -----------------------------------------------------------------------
    # Infection codes — Tier 2: Conditional positive
    # Positive ONLY when a catheter code also appears in the same admission.
    # Without catheter co-occurrence these are too non-specific to attribute
    # to CAUTI (e.g., community-acquired UTI).
    # -----------------------------------------------------------------------
    INFECTION_CODES_CONDITIONAL_ICD10: ClassVar[Set[str]] = {
        "N390",  # N39.0 — UTI, site unspecified
        "N10",  # N10   — acute pyelonephritis
        "R8271",  # R82.71 — bacteriuria
    }
    INFECTION_PREFIXES_CONDITIONAL_ICD10: ClassVar[Tuple[str, ...]] = (
        "N30",  # N30.x — cystitis
        "N34",  # N34.x — urethritis
    )
    # No conditional tier for ICD-9: the single remaining ICD-9 infection code
    # (996.64) is already catheter-specific and unconditional.

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
        "Sodium",
        "Potassium",
        "Chloride",
        "Bicarbonate",
        "Glucose",
        "Calcium",
        "Magnesium",
        "Anion Gap",
        "Osmolality",
        "Phosphate",
    ]

    LABITEMS: ClassVar[List[str]] = [
        item for items in LAB_CATEGORIES.values() for item in items
    ]

    def _zero_lab_vector(self) -> List[float]:
        return [0.0] * len(self.LAB_CATEGORY_ORDER)

    def _ensure_nonempty_sequence(self, values: List[str]) -> List[str]:
        cleaned = [v for v in values if v]
        if cleaned:
            return cleaned
        return [self.MISSING_TOKEN]

    @classmethod
    def _get_mapper(cls, source_vocab: str, target_vocab: str) -> Optional[CrossMap]:
        key = (source_vocab, target_vocab)
        if key in cls._MAPPER_CACHE:
            return cls._MAPPER_CACHE[key]

        try:
            mapper = CrossMap.load(source_vocab, target_vocab)
        except Exception:
            mapper = None
        cls._MAPPER_CACHE[key] = mapper
        return mapper

    @staticmethod
    def _normalize_code(code: str) -> str:
        return code.replace(".", "").strip().upper()

    def _map_condition_to_tokens(
        self, code: str, version: Any, map_ccscm: bool = True
    ) -> List[str]:
        normalized = self._normalize_code(code)
        if not map_ccscm:
            return [f"ICD_{normalized}"]

        version_str = str(version)

        if version_str == "9":
            mapper = self._get_mapper("ICD9CM", "CCSCM")
        elif version_str == "10":
            mapper = self._get_mapper("ICD10CM", "CCSCM")
        else:
            mapper = None

        if mapper is not None:
            try:
                mapped = [v.strip().upper() for v in mapper.map(code) if v]
            except Exception:
                mapped = []
            if mapped:
                return [f"CCSCM_{v}" for v in sorted(set(mapped))]

        return [f"ICD_{normalized}"]

    def _map_procedure_to_tokens(
        self, code: str, version: Any, map_ccscm: bool = True
    ) -> List[str]:
        normalized = self._normalize_code(code)
        if not map_ccscm:
            return [f"ICDPROC_{normalized}"]

        version_str = str(version)

        if version_str == "9":
            mapper = self._get_mapper("ICD9PROC", "CCSPROC")
        elif version_str == "10":
            mapper = self._get_mapper("ICD10PROC", "CCSPROC")
        else:
            mapper = None

        if mapper is not None:
            try:
                mapped = [v.strip().upper() for v in mapper.map(code) if v]
            except Exception:
                mapped = []
            if mapped:
                return [f"CCSPROC_{v}" for v in sorted(set(mapped))]

        return [f"ICDPROC_{normalized}"]

    def _map_ndc_to_atc3_tokens(self, ndc_code: str | None) -> List[str]:
        if not ndc_code:
            return []

        mapper = self._get_mapper("NDC", "ATC")
        if mapper is None:
            return []

        try:
            mapped = mapper.map(ndc_code, target_kwargs={"level": 3})
        except Exception:
            return []

        cleaned = [v.strip().upper() for v in mapped if v]
        return [f"ATC3_{v}" for v in sorted(set(cleaned))]

    def _is_catheter_code(self, code: str | None, version: Any) -> bool:
        """Check if an ICD code indicates catheter use."""
        if not code:
            return False

        normalized = self._normalize_code(code)
        version_str = str(version)

        if version_str == "10":
            if normalized in self.CATHETER_CODES_ICD10:
                return True
            return normalized.startswith(self.CATHETER_PREFIXES_ICD10)

        if version_str == "9":
            if normalized in self.CATHETER_CODES_ICD9:
                return True
            return normalized.startswith(self.CATHETER_PREFIXES_ICD9)

        return False

    def _is_unconditional_infection_code(self, code: str | None, version: Any) -> bool:
        """Return True if code marks a CAUTI event regardless of catheter co-occurrence.

        These codes are catheter-specific by definition (e.g., T83.511A explicitly
        names the catheter as the device) and require no additional context.
        """
        if not code:
            return False
        normalized = self._normalize_code(code)
        version_str = str(version)
        if version_str == "10":
            return normalized in self.INFECTION_CODES_UNCONDITIONAL_ICD10
        if version_str == "9":
            return normalized in self.INFECTION_CODES_UNCONDITIONAL_ICD9
        return False

    def _is_conditional_infection_code(self, code: str | None, version: Any) -> bool:
        """Return True if code is a CAUTI positive ONLY when a catheter code co-occurs
        in the same admission.

        Codes like N39.0 (UTI) are common and non-specific — they become attributable
        to CAUTI only when a catheter code appears in the same encounter.
        """
        if not code:
            return False
        normalized = self._normalize_code(code)
        version_str = str(version)
        if version_str == "10":
            if normalized in self.INFECTION_CODES_CONDITIONAL_ICD10:
                return True
            return normalized.startswith(self.INFECTION_PREFIXES_CONDITIONAL_ICD10)
        # No conditional tier for ICD-9
        return False

    def _is_infection_code(self, code: str | None, version: Any) -> bool:
        """Convenience wrapper: True if code is unconditional OR conditional infection."""
        return self._is_unconditional_infection_code(
            code, version
        ) or self._is_conditional_infection_code(code, version)

    def _build_lab_vector(self, lab_df: pl.DataFrame) -> List[float]:
        """Build a 10D lab feature vector from lab events DataFrame."""
        if lab_df.height == 0:
            return self._zero_lab_vector()

        filtered = (
            lab_df.with_columns(
                [
                    pl.col("labevents/itemid").cast(pl.Utf8),
                    pl.col("labevents/valuenum").cast(pl.Float64),
                ]
            )
            .filter(pl.col("labevents/itemid").is_in(self.LABITEMS))
            .filter(pl.col("labevents/valuenum").is_not_null())
        )

        if filtered.height == 0:
            return self._zero_lab_vector()

        vector: List[float] = []
        for category in self.LAB_CATEGORY_ORDER:
            itemids = self.LAB_CATEGORIES[category]
            cat_df = filtered.filter(pl.col("labevents/itemid").is_in(itemids))
            if cat_df.height > 0:
                values = cat_df["labevents/valuenum"].drop_nulls()
                mean_value = float(values.mean()) if len(values) > 0 else 0.0
                vector.append(mean_value)  # type: ignore
            else:
                vector.append(0.0)
        return vector

    def _determine_positive_label(
        self,
        diagnoses: List[Any],
        procedures: List[Any],
    ) -> Tuple[bool, bool, bool, bool]:
        """Scan diagnoses and procedures to determine CAUTI label flags.

        Returns:
            (is_positive, has_catheter, has_unconditional, has_conditional)
        """
        has_catheter = False
        has_unconditional = False
        has_conditional = False

        for diag in diagnoses:
            code = getattr(diag, "icd_code", None)
            version = getattr(diag, "icd_version", None)
            if not code:
                continue
            if self._is_catheter_code(code, version):
                has_catheter = True
            if self._is_unconditional_infection_code(code, version):
                has_unconditional = True
            if self._is_conditional_infection_code(code, version):
                has_conditional = True

        for proc in procedures:
            code = getattr(proc, "icd_code", None)
            version = getattr(proc, "icd_version", None)
            if not code:
                continue
            if self._is_catheter_code(code, version):
                has_catheter = True

        is_positive = has_unconditional or (has_conditional and has_catheter)
        return (is_positive, has_catheter, has_unconditional, has_conditional)


class CatheterAssociatedInfectionPredictionStageNetMIMIC4(_CatheterInfectionBase):
    """StageNet-style patient-level catheter infection prediction task.

    Predicts catheter-associated urinary tract infection (CAUTI) from longitudinal
    EHR data. Each qualifying infection admission generates a positive sample whose
    features are the admissions that preceded it (including full pre-catheter history).
    A patient with multiple CAUTI events produces multiple independent positive samples.
    Patients with catheter evidence but no infection produce one negative sample.

    Infection Code Tiers
    --------------------
    - **Unconditional**: T83.511A, T83.518A/D/S, T83.519A/D/S, ICD-9 996.64.
      Any admission containing one of these codes is a positive CAUTI event.
    - **Conditional**: N39.0, N10, R82.71, N30.x, N34.x.
      Positive only if a catheter code also appears in the same admission.

    same_visit Parameter
    --------------------
    When ``same_visit=True`` (default), the infection admission's non-infection
    codes (catheter codes, comorbidities, labs) are appended to the feature window
    with all infection codes masked out.  This is appropriate for CAUTI because
    foley catheters are typically placed and removed within a single admission, so
    the catheter code and infection code co-occur in the same encounter.
    When ``same_visit=False``, features are restricted to prior admissions only.

    Features (per admission in the feature window)
    -----------------------------------------------
    - icd_codes: StageNet tuple (time deltas in hours + ICD code token sequences)
    - labs: StageNet tensor tuple (time deltas + 10D mean lab vectors)
    """

    task_name: str = "CatheterAssociatedInfectionPredictionStageNetMIMIC4"

    def __init__(
        self,
        padding: int = 0,
        same_visit: bool = True,
        map_ccscm: bool = True,
    ):
        """Initialize task.

        Args:
            padding: StageNet sequence padding length.
            same_visit: If True (default), include same-admission features with
                infection codes masked.  If False, use prior admissions only.
            map_ccscm: If True (default), map ICD diagnosis/procedure codes to
                CCS categories; if False, keep normalized ICD tokens.
        """
        self.padding = padding
        self.same_visit = same_visit
        self.map_ccscm = map_ccscm
        self.task_name = self._task_name_with_map_ccscm(
            self._task_name_with_same_visit(type(self).task_name, self.same_visit),
            self.map_ccscm,
        )
        self.input_schema: Dict[str, Tuple[str, Dict[str, Any]]] = {  # type: ignore
            "icd_codes": ("stagenet", {"padding": padding}),
            "labs": ("stagenet_tensor", {}),
        }
        self.output_schema: Dict[str, str] = {"label": "binary"}  # type: ignore

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Create StageNet samples for one patient."""
        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []

        admissions = sorted(admissions, key=lambda x: x.timestamp)

        # Running feature lists — accumulate across all admissions (including masked
        # infection admissions so future events can see prior CAUTI history).
        all_icd_codes: List[List[str]] = []
        all_icd_times: List[float] = []
        all_lab_values: List[List[float]] = []
        all_lab_times: List[float] = []

        all_samples: List[Dict[str, Any]] = []
        has_any_catheter: bool = False
        previous_admission_time: Optional[datetime] = None
        infection_event_count: int = 0
        neg_count: int = 0

        for admission in admissions:
            admission_time = getattr(admission, "timestamp", None)
            if admission_time is None:
                continue

            dischtime_str = getattr(admission, "dischtime", None)
            try:
                admission_dischtime: Optional[datetime] = (
                    datetime.strptime(dischtime_str, "%Y-%m-%d %H:%M:%S")
                    if dischtime_str else None
                )
            except (ValueError, AttributeError):
                admission_dischtime = None

            time_delta = (
                (admission_time - previous_admission_time).total_seconds() / 3600.0
                if previous_admission_time is not None
                else 0.0
            )

            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )

            is_infection_event, has_catheter, has_unconditional, has_conditional = (
                self._determine_positive_label(diagnoses, procedures)
            )

            visit_codes: List[str] = []
            visit_codes_masked: List[str] = []  # infection codes stripped
            seen: Set[str] = set()
            seen_masked: Set[str] = set()

            for diag in diagnoses:
                code = getattr(diag, "icd_code", None)
                version = getattr(diag, "icd_version", None)
                if not code:
                    continue

                is_infect = self._is_infection_code(code, version)
                mapped_tokens = self._map_condition_to_tokens(
                    code,
                    version,
                    map_ccscm=self.map_ccscm,
                )
                for token in mapped_tokens:
                    prefixed = f"D_{token}"
                    if prefixed not in seen:
                        seen.add(prefixed)
                        visit_codes.append(prefixed)
                    if not is_infect and prefixed not in seen_masked:
                        seen_masked.add(prefixed)
                        visit_codes_masked.append(prefixed)

            for proc in procedures:
                code = getattr(proc, "icd_code", None)
                version = getattr(proc, "icd_version", None)
                if not code:
                    continue

                mapped_tokens = self._map_procedure_to_tokens(
                    code,
                    version,
                    map_ccscm=self.map_ccscm,
                )
                for token in mapped_tokens:
                    prefixed = f"P_{token}"
                    if prefixed not in seen:
                        seen.add(prefixed)
                        visit_codes.append(prefixed)
                    if prefixed not in seen_masked:
                        seen_masked.add(prefixed)
                        visit_codes_masked.append(prefixed)

            # Temporal lab cutoff: for the infection admission with same_visit=True,
            # use admittime as end to exclude all labs (charttime >= admittime).
            if is_infection_event and self.same_visit:
                lab_end = admission_time
            else:
                lab_end = admission_dischtime

            lab_df = patient.get_events(
                event_type="labevents",
                start=admission_time,
                end=lab_end,
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            lab_vector = self._build_lab_vector(lab_df)

            has_any_catheter |= has_catheter

            if is_infection_event:
                infection_event_count += 1

                if self.same_visit:
                    masked_codes = self._ensure_nonempty_sequence(visit_codes_masked)
                    feat_icd_codes = all_icd_codes + [masked_codes]
                    feat_icd_times = all_icd_times + [time_delta]
                    feat_lab_values = all_lab_values + [lab_vector]
                    feat_lab_times = all_lab_times + [time_delta]
                else:
                    feat_icd_codes = list(all_icd_codes)
                    feat_icd_times = list(all_icd_times)
                    feat_lab_values = list(all_lab_values)
                    feat_lab_times = list(all_lab_times)

                if not feat_icd_codes:
                    feat_icd_codes = [[f"D_{self.MISSING_TOKEN}"]]
                    feat_icd_times = [0.0]
                if not feat_lab_values:
                    feat_lab_values = [self._zero_lab_vector()]
                    feat_lab_times = [0.0]

                base_id = f"{patient.patient_id}_cauti{infection_event_count}"
                new_samples: List[Dict[str, Any]] = [
                    {
                        "patient_id": patient.patient_id,
                        "record_id": base_id,
                        "icd_codes": (list(feat_icd_times), list(feat_icd_codes)),
                        "labs": (list(feat_lab_times), list(feat_lab_values)),
                        "label": 1,
                    }
                ]

                # Suffix augmentation: drop progressively earlier admissions
                if len(feat_icd_codes) > 1:
                    for start in range(1, len(feat_icd_codes)):
                        new_samples.append(
                            {
                                "patient_id": patient.patient_id,
                                "record_id": f"{base_id}_aug{start}",
                                "icd_codes": (
                                    feat_icd_times[start:],
                                    feat_icd_codes[start:],
                                ),
                                "labs": (
                                    feat_lab_times[start:],
                                    feat_lab_values[start:],
                                ),
                                "label": 1,
                            }
                        )

                all_samples.extend(new_samples)

                # Add masked admission to running history for future events to see.
                masked_for_history = self._ensure_nonempty_sequence(visit_codes_masked)
                all_icd_codes.append(masked_for_history)
                all_icd_times.append(time_delta)
                all_lab_values.append(lab_vector)
                all_lab_times.append(time_delta)

            else:
                # Non-infection admission: accumulate full codes for feature history.
                all_icd_codes.append(self._ensure_nonempty_sequence(visit_codes))
                all_icd_times.append(time_delta)
                all_lab_values.append(lab_vector)
                all_lab_times.append(time_delta)

                # Emit a per-admission negative sample for catheter-evidence patients.
                if has_any_catheter:
                    neg_count += 1
                    all_samples.append(
                        {
                            "patient_id": patient.patient_id,
                            "record_id": f"{patient.patient_id}_neg{neg_count}",
                            "icd_codes": (list(all_icd_times), list(all_icd_codes)),
                            "labs": (list(all_lab_times), list(all_lab_values)),
                            "label": 0,
                        }
                    )

            previous_admission_time = admission_time

        return all_samples


class CatheterAssociatedInfectionPredictionMIMIC4(_CatheterInfectionBase):
    """Nested-sequence patient-level catheter infection prediction task.

    Predicts catheter-associated urinary tract infection (CAUTI) from longitudinal
    EHR data. Each qualifying infection admission generates a positive sample whose
    features are the admissions that preceded it (including full pre-catheter history).
    A patient with multiple CAUTI events produces multiple independent positive samples.
    Patients with catheter evidence but no infection produce one negative sample.

    Infection Code Tiers
    --------------------
    - **Unconditional**: T83.511A, T83.518A/D/S, T83.519A/D/S, ICD-9 996.64.
      Any admission containing one of these codes is a positive CAUTI event.
    - **Conditional**: N39.0, N10, R82.71, N30.x, N34.x.
      Positive only if a catheter code also appears in the same admission.

    same_visit Parameter
    --------------------
    When ``same_visit=True`` (default), the infection admission's non-infection
    codes (catheter codes, comorbidities, labs, drugs) are appended to the feature
    window with all infection codes masked out.  This is appropriate for CAUTI
    because foley catheters are typically placed and removed within a single
    admission, so the catheter code and infection code co-occur in the same encounter.
    When ``same_visit=False``, features are restricted to prior admissions only.

    Features (per admission in the feature window)
    -----------------------------------------------
    - conditions: nested_sequence of CCS-CM diagnosis tokens
    - procedures: nested_sequence of CCS-PCS procedure tokens
    - drugs: nested_sequence of ATC Level-3 drug tokens
    - labs: nested_sequence_floats of 10D mean lab vectors
    """

    task_name: str = "CatheterAssociatedInfectionPredictionMIMIC4"

    input_schema: Dict[str, str] = {
        "conditions": "nested_sequence",
        "procedures": "nested_sequence",
        "drugs": "nested_sequence",
        "labs": "nested_sequence_floats",
    }
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, same_visit: bool = True, map_ccscm: bool = True):
        """Initialize task.

        Args:
            same_visit: If True (default), include same-admission features with
                infection codes masked.  If False, use prior admissions only.
            map_ccscm: If True (default), map ICD diagnosis/procedure codes to
                CCS categories; if False, keep normalized ICD tokens.
        """
        self.same_visit = same_visit
        self.map_ccscm = map_ccscm
        self.task_name = self._task_name_with_map_ccscm(
            self._task_name_with_same_visit(type(self).task_name, self.same_visit),
            self.map_ccscm,
        )

    @staticmethod
    def _clean_sequence(values: List[Any]) -> List[str]:
        return [str(v).strip() for v in values if v is not None and str(v).strip()]

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Create nested-sequence samples for one patient."""
        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []

        admissions = sorted(admissions, key=lambda x: x.timestamp)

        # Running feature lists — accumulate across all admissions (including masked
        # infection admissions so future events can see prior CAUTI history).
        all_conditions: List[List[str]] = []
        all_procedures: List[List[str]] = []
        all_drugs: List[List[str]] = []
        all_labs: List[List[float]] = []

        all_samples: List[Dict[str, Any]] = []
        has_any_catheter: bool = False
        infection_event_count: int = 0
        neg_count: int = 0

        for admission in admissions:
            admission_time = getattr(admission, "timestamp", None)
            if admission_time is None:
                continue

            dischtime_str = getattr(admission, "dischtime", None)
            try:
                admission_dischtime: Optional[datetime] = (
                    datetime.strptime(dischtime_str, "%Y-%m-%d %H:%M:%S")
                    if dischtime_str else None
                )
            except (ValueError, AttributeError):
                admission_dischtime = None

            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                start=admission_time,
                end=admission_dischtime,
                filters=[("hadm_id", "==", admission.hadm_id)],
            )

            is_infection_event, has_catheter, has_unconditional, has_conditional = (
                self._determine_positive_label(diagnoses, procedures)
            )

            condition_codes: List[str] = []
            condition_codes_masked: List[str] = []  # infection codes stripped
            procedure_codes: List[str] = []

            for diag in diagnoses:
                code = getattr(diag, "icd_code", None)
                version = getattr(diag, "icd_version", None)
                if not code:
                    continue

                tokens = self._map_condition_to_tokens(
                    code,
                    version,
                    map_ccscm=self.map_ccscm,
                )
                condition_codes.extend(tokens)
                if not self._is_infection_code(code, version):
                    condition_codes_masked.extend(tokens)

            for proc in procedures:
                code = getattr(proc, "icd_code", None)
                version = getattr(proc, "icd_version", None)
                if not code:
                    continue

                procedure_codes.extend(
                    self._map_procedure_to_tokens(
                        code,
                        version,
                        map_ccscm=self.map_ccscm,
                    )
                )

            # Drug tokens (used for both infection and non-infection admissions)
            visit_drugs: List[str] = []
            for event in prescriptions:
                visit_drugs.extend(
                    self._map_ndc_to_atc3_tokens(getattr(event, "ndc", None))
                )
            visit_drugs = self._clean_sequence(list(dict.fromkeys(visit_drugs)))
            visit_drugs = self._ensure_nonempty_sequence(visit_drugs)

            # Temporal lab cutoff: for the infection admission with same_visit=True,
            # use admittime as end to exclude all labs (charttime >= admittime).
            if is_infection_event and self.same_visit:
                lab_end = admission_time
            else:
                lab_end = admission_dischtime

            lab_df = patient.get_events(
                event_type="labevents",
                start=admission_time,
                end=lab_end,
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            lab_vector = self._build_lab_vector(lab_df)

            has_any_catheter |= has_catheter

            if is_infection_event:
                infection_event_count += 1

                masked_cond = self._ensure_nonempty_sequence(
                    self._clean_sequence(condition_codes_masked)
                )
                masked_proc = self._ensure_nonempty_sequence(
                    self._clean_sequence(procedure_codes)
                )

                if self.same_visit:
                    feat_conditions = all_conditions + [masked_cond]
                    feat_procedures = all_procedures + [masked_proc]
                    feat_drugs = all_drugs + [visit_drugs]
                    feat_labs = all_labs + [lab_vector]
                else:
                    feat_conditions = list(all_conditions)
                    feat_procedures = list(all_procedures)
                    feat_drugs = list(all_drugs)
                    feat_labs = list(all_labs)

                if not feat_conditions:
                    feat_conditions = [[self.MISSING_TOKEN]]
                    feat_procedures = [[self.MISSING_TOKEN]]
                    feat_drugs = [[self.MISSING_TOKEN]]
                    feat_labs = [self._zero_lab_vector()]

                base_id = f"{patient.patient_id}_cauti{infection_event_count}"
                new_samples: List[Dict[str, Any]] = [
                    {
                        "patient_id": patient.patient_id,
                        "record_id": base_id,
                        "conditions": list(feat_conditions),
                        "procedures": list(feat_procedures),
                        "drugs": list(feat_drugs),
                        "labs": list(feat_labs),
                        "label": 1,
                    }
                ]

                # Suffix augmentation: drop progressively earlier admissions
                if len(feat_conditions) > 1:
                    for start in range(1, len(feat_conditions)):
                        new_samples.append(
                            {
                                "patient_id": patient.patient_id,
                                "record_id": f"{base_id}_aug{start}",
                                "conditions": feat_conditions[start:],
                                "procedures": feat_procedures[start:],
                                "drugs": feat_drugs[start:],
                                "labs": feat_labs[start:],
                                "label": 1,
                            }
                        )

                all_samples.extend(new_samples)

                # Add masked admission to running history for future events to see.
                all_conditions.append(masked_cond)
                all_procedures.append(masked_proc)
                all_drugs.append(visit_drugs)
                all_labs.append(lab_vector)

            else:
                # Non-infection admission: accumulate full codes for feature history.
                all_conditions.append(
                    self._ensure_nonempty_sequence(
                        self._clean_sequence(condition_codes)
                    )
                )
                all_procedures.append(
                    self._ensure_nonempty_sequence(
                        self._clean_sequence(procedure_codes)
                    )
                )
                all_drugs.append(visit_drugs)
                all_labs.append(lab_vector)

                # Emit a per-admission negative sample for catheter-evidence patients.
                if has_any_catheter:
                    neg_count += 1
                    all_samples.append(
                        {
                            "patient_id": patient.patient_id,
                            "record_id": f"{patient.patient_id}_neg{neg_count}",
                            "conditions": list(all_conditions),
                            "procedures": list(all_procedures),
                            "drugs": list(all_drugs),
                            "labs": list(all_labs),
                            "label": 0,
                        }
                    )

        return all_samples


class CatheterAssociatedInfectionPredictionStageNetMIMIC4DualContext(
    _CatheterInfectionBase
):
    """Dual-context StageNet CAUTI task.

    Emits both context modes for each sample:
    - visit_mode="current": same-visit features included
    - visit_mode="next": prior-admissions-only features
    """

    task_name: str = "CatheterAssociatedInfectionPredictionStageNetMIMIC4DualContext"

    def __init__(self, padding: int = 0, map_ccscm: bool = True):
        self.padding = padding
        self.map_ccscm = map_ccscm
        self.task_name = self._task_name_with_map_ccscm(
            type(self).task_name,
            self.map_ccscm,
        )
        self.input_schema: Dict[str, Tuple[str, Dict[str, Any]]] = {  # type: ignore
            "icd_codes": ("stagenet", {"padding": padding}),
            "labs": ("stagenet_tensor", {}),
        }
        self.output_schema: Dict[str, str] = {"label": "binary"}  # type: ignore

    @staticmethod
    def _tag_samples(
        samples: List[Dict[str, Any]], visit_mode: str
    ) -> List[Dict[str, Any]]:
        tagged: List[Dict[str, Any]] = []
        suffix = "current" if visit_mode == "current" else "next"
        for sample in samples:
            updated = dict(sample)
            updated["record_id"] = f"{sample['record_id']}_{suffix}"
            updated["visit_mode"] = visit_mode
            tagged.append(updated)
        return tagged

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        current_task = CatheterAssociatedInfectionPredictionStageNetMIMIC4(
            padding=self.padding,
            same_visit=True,
            map_ccscm=self.map_ccscm,
        )
        next_task = CatheterAssociatedInfectionPredictionStageNetMIMIC4(
            padding=self.padding,
            same_visit=False,
            map_ccscm=self.map_ccscm,
        )

        current_samples = self._tag_samples(current_task(patient), "current")
        next_samples = self._tag_samples(next_task(patient), "next")
        return current_samples + next_samples


class CatheterAssociatedInfectionPredictionMIMIC4DualContext(_CatheterInfectionBase):
    """Dual-context nested-sequence CAUTI task.

    Emits both context modes for each sample:
    - visit_mode="current": same-visit features included
    - visit_mode="next": prior-admissions-only features
    """

    task_name: str = "CatheterAssociatedInfectionPredictionMIMIC4DualContext"

    input_schema: Dict[str, str] = {
        "conditions": "nested_sequence",
        "procedures": "nested_sequence",
        "drugs": "nested_sequence",
        "labs": "nested_sequence_floats",
    }
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, map_ccscm: bool = True):
        self.map_ccscm = map_ccscm
        self.task_name = self._task_name_with_map_ccscm(
            type(self).task_name,
            self.map_ccscm,
        )

    @staticmethod
    def _tag_samples(
        samples: List[Dict[str, Any]], visit_mode: str
    ) -> List[Dict[str, Any]]:
        tagged: List[Dict[str, Any]] = []
        suffix = "current" if visit_mode == "current" else "next"
        for sample in samples:
            updated = dict(sample)
            updated["record_id"] = f"{sample['record_id']}_{suffix}"
            updated["visit_mode"] = visit_mode
            tagged.append(updated)
        return tagged

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        current_task = CatheterAssociatedInfectionPredictionMIMIC4(
            same_visit=True,
            map_ccscm=self.map_ccscm,
        )
        next_task = CatheterAssociatedInfectionPredictionMIMIC4(
            same_visit=False,
            map_ccscm=self.map_ccscm,
        )

        current_samples = self._tag_samples(current_task(patient), "current")
        next_samples = self._tag_samples(next_task(patient), "next")
        return current_samples + next_samples
