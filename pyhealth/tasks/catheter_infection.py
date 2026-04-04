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

    # ICD-10 diagnosis/procedure codes indicating catheter use
    CATHETER_CODES_ICD10: ClassVar[Set[str]] = {
        "Y846",  # Y84.6
        "Z466",  # Z46.6
        "Z4682",  # Z46.82
        "Z960",  # Z96.0
        "Z935",  # Z93.5 — cystostomy status
        "Z936",  # Z93.6 — other artificial urinary opening status
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
        "99639",  # Mechanical complication of other genitourinary device
        "99649",  # Mechanical complication of other vascular device
        "99662",  # Infection due to other vascular device (includes CVC)
        "99664",  # Infection due to indwelling urinary catheter
        "99668",  # Infection due to peritoneal dialysis catheter
        "V536",  # Fitting/adjustment of urinary devices
        "E8705",  # Misadventure in catheterization
        "E8796",  # Urinary catheterization causing abnormal reaction
    }

    # ICD-9 cardiac catheterization procedure range 37.21-37.23
    CATHETER_PREFIXES_ICD9: ClassVar[Tuple[str, ...]] = (
        "3721",
        "3722",
        "3723",
    )

    # ICD-10 diagnosis codes for catheter-associated urinary infection
    INFECTION_CODES_ICD10: ClassVar[Set[str]] = {
        "T83511A",
        "T83511D",
        "T83511S",
        "T83518A",
        "T83518D",
        "T83518S",
        "T83519A",
        "T83519D",
        "T83519S",
        "N390",   # N39.0 — UTI, site unspecified
        "N10",    # N10 — acute pyelonephritis
        "R8271",  # R82.71 — bacteriuria
    }

    # ICD-10 infection code families (prefix matching)
    INFECTION_PREFIXES_ICD10: ClassVar[Tuple[str, ...]] = (
        "N30",   # N30.x — cystitis
        "N34",   # N34.x — urethritis
        "B96",   # B96.x — bacterial agents (Klebsiella, Pseudomonas, etc.)
        "B37",   # B37.x — candidal infections
        "A41",   # A41.x — other sepsis (urosepsis)
        "R652",  # R65.2x — severe sepsis
    )

    # ICD-9 diagnosis codes indicating catheter/device-associated infection.
    INFECTION_CODES_ICD9: ClassVar[Set[str]] = {
        "99662",  # Infection due to other vascular device (includes CVC)
        "99664",  # Infection due to indwelling urinary catheter
        "99668",  # Infection due to peritoneal dialysis catheter
    }

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

    def _map_condition_to_tokens(self, code: str, version: Any) -> List[str]:
        normalized = self._normalize_code(code)
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

    def _map_procedure_to_tokens(self, code: str, version: Any) -> List[str]:
        normalized = self._normalize_code(code)
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

    def _is_infection_code(self, code: str | None, version: Any) -> bool:
        """Check if an ICD code indicates catheter-associated infection."""
        if not code:
            return False

        normalized = self._normalize_code(code)
        version_str = str(version)

        if version_str == "10":
            if normalized in self.INFECTION_CODES_ICD10:
                return True
            return normalized.startswith(self.INFECTION_PREFIXES_ICD10)

        if version_str == "9":
            return normalized in self.INFECTION_CODES_ICD9

        return False

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


class CatheterAssociatedInfectionPredictionStageNetMIMIC4(_CatheterInfectionBase):
    """StageNet-style patient-level catheter infection prediction task.

    This task creates PATIENT-LEVEL samples for patients with ICD-10 catheter
    evidence, then predicts whether they develop a catheter-associated urinary
    infection in a future admission. Features are collected only after the first
    catheter-coded admission and before the first infection-coded admission.

    Target Population:
        - Patients with at least one ICD-10 catheter code in diagnoses or procedures

    Label Definition:
        - Positive (1): Infection diagnosis occurs in a future admission after
          first catheter evidence
        - Negative (0): No qualifying infection after first catheter evidence

    Data Leakage Prevention:
        - Admissions are sorted chronologically
        - The first catheter-coded admission is used as temporal anchor
        - Only post-catheter, pre-infection admissions are used as features
        - Infection in or before the first catheter admission is excluded

    Features:
        - icd_codes: StageNet tuple (time deltas + ICD code sequences)
        - labs: StageNet tensor tuple (time deltas + 10D lab vectors)
    """

    task_name: str = "CatheterAssociatedInfectionPredictionStageNetMIMIC4"

    def __init__(self, padding: int = 0):
        """Initialize task with optional StageNet padding."""
        self.padding = padding
        self.input_schema: Dict[str, Tuple[str, Dict[str, Any]]] = {  # type: ignore
            "icd_codes": ("stagenet", {"padding": padding}),
            "labs": ("stagenet_tensor", {}),
        }
        self.output_schema: Dict[str, str] = {"label": "binary"}  # type: ignore

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Create one patient-level StageNet sample."""
        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []

        admissions = sorted(admissions, key=lambda x: x.timestamp)

        all_icd_codes: List[List[str]] = []
        all_icd_times: List[float] = []
        all_lab_values: List[List[float]] = []
        all_lab_times: List[float] = []

        first_catheter_time: Optional[datetime] = None
        previous_admission_time: Optional[datetime] = None
        has_infection_after_catheter = False

        for admission in admissions:
            admission_time = getattr(admission, "timestamp", None)
            if admission_time is None:
                continue

            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )

            has_catheter_in_admission = False
            has_infection_in_admission = False

            visit_codes: List[str] = []
            seen: Set[str] = set()

            for diag in diagnoses:
                code = getattr(diag, "icd_code", None)
                version = getattr(diag, "icd_version", None)
                if not code:
                    continue

                if self._is_catheter_code(code, version):
                    has_catheter_in_admission = True
                if self._is_infection_code(code, version):
                    has_infection_in_admission = True

                mapped_tokens = self._map_condition_to_tokens(code, version)
                for token in mapped_tokens:
                    normalized = f"D_{token}"
                    if normalized not in seen:
                        seen.add(normalized)
                        visit_codes.append(normalized)

            for proc in procedures:
                code = getattr(proc, "icd_code", None)
                version = getattr(proc, "icd_version", None)
                if not code:
                    continue

                if self._is_catheter_code(code, version):
                    has_catheter_in_admission = True

                mapped_tokens = self._map_procedure_to_tokens(code, version)
                for token in mapped_tokens:
                    normalized = f"P_{token}"
                    if normalized not in seen:
                        seen.add(normalized)
                        visit_codes.append(normalized)

            # Exclude patients where infection occurs before catheter exposure.
            if first_catheter_time is None and has_infection_in_admission:
                return []

            if first_catheter_time is None and has_catheter_in_admission:
                first_catheter_time = admission_time
                previous_admission_time = admission_time
                continue

            if first_catheter_time is None:
                continue

            # Stop at first future infection admission to avoid leakage.
            if has_infection_in_admission:
                has_infection_after_catheter = True
                break

            if previous_admission_time is None:
                time_from_previous = 0.0
            else:
                time_from_previous = (
                    admission_time - previous_admission_time
                ).total_seconds() / 3600.0
            previous_admission_time = admission_time

            visit_codes = self._ensure_nonempty_sequence(visit_codes)
            all_icd_codes.append(visit_codes)
            all_icd_times.append(time_from_previous)

            lab_df = patient.get_events(
                event_type="labevents",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            all_lab_values.append(self._build_lab_vector(lab_df))
            all_lab_times.append(time_from_previous)

        if first_catheter_time is None:
            return []

        if not all_icd_codes:
            all_icd_codes = [[f"D_{self.MISSING_TOKEN}"]]
            all_icd_times = [0.0]

        if not all_lab_values:
            all_lab_values = [self._zero_lab_vector()]
            all_lab_times = [0.0]

        label = int(has_infection_after_catheter)
        samples: List[Dict[str, Any]] = [
            {
                "patient_id": patient.patient_id,
                "record_id": patient.patient_id,
                "icd_codes": (all_icd_times, all_icd_codes),
                "labs": (all_lab_times, all_lab_values),
                "label": label,
            }
        ]

        # Positive-only augmentation with all valid temporal suffix cutoffs.
        if label == 1 and len(all_icd_codes) > 1:
            for start in range(1, len(all_icd_codes)):
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": f"{patient.patient_id}_aug_{start}",
                        "icd_codes": (
                            all_icd_times[start:],
                            all_icd_codes[start:],
                        ),
                        "labs": (
                            all_lab_times[start:],
                            all_lab_values[start:],
                        ),
                        "label": label,
                    }
                )

        return samples


class CatheterAssociatedInfectionPredictionMIMIC4(_CatheterInfectionBase):
    """Nested-sequence patient-level catheter infection prediction task.

    This variant aggregates post-catheter, pre-infection visits without StageNet
    time tuples. It uses nested sequence processors for visit-level code history
    and visit-level lab vectors.
    """

    task_name: str = "CatheterAssociatedInfectionPredictionMIMIC4"

    input_schema: Dict[str, str] = {
        "conditions": "nested_sequence",
        "procedures": "nested_sequence",
        "drugs": "nested_sequence",
        "labs": "nested_sequence_floats",
    }
    output_schema: Dict[str, str] = {"label": "binary"}

    @staticmethod
    def _clean_sequence(values: List[Any]) -> List[str]:
        return [str(v).strip() for v in values if v is not None and str(v).strip()]

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Create one patient-level nested-sequence sample."""
        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []

        admissions = sorted(admissions, key=lambda x: x.timestamp)

        all_conditions: List[List[str]] = []
        all_procedures: List[List[str]] = []
        all_drugs: List[List[str]] = []
        all_labs: List[List[float]] = []

        first_catheter_time: Optional[datetime] = None
        has_infection_after_catheter = False

        for admission in admissions:
            admission_time = getattr(admission, "timestamp", None)
            if admission_time is None:
                continue

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
                filters=[("hadm_id", "==", admission.hadm_id)],
            )

            has_catheter_in_admission = False
            has_infection_in_admission = False

            condition_codes: List[str] = []
            procedure_codes: List[str] = []

            for diag in diagnoses:
                code = getattr(diag, "icd_code", None)
                version = getattr(diag, "icd_version", None)
                if not code:
                    continue

                if self._is_catheter_code(code, version):
                    has_catheter_in_admission = True
                if self._is_infection_code(code, version):
                    has_infection_in_admission = True

                condition_codes.extend(self._map_condition_to_tokens(code, version))

            for proc in procedures:
                code = getattr(proc, "icd_code", None)
                version = getattr(proc, "icd_version", None)
                if not code:
                    continue

                if self._is_catheter_code(code, version):
                    has_catheter_in_admission = True

                procedure_codes.extend(self._map_procedure_to_tokens(code, version))

            if first_catheter_time is None and has_infection_in_admission:
                return []

            if first_catheter_time is None and has_catheter_in_admission:
                first_catheter_time = admission_time
                continue

            if first_catheter_time is None:
                continue

            if has_infection_in_admission:
                has_infection_after_catheter = True
                break

            visit_drugs: List[str] = []
            for event in prescriptions:
                visit_drugs.extend(
                    self._map_ndc_to_atc3_tokens(getattr(event, "ndc", None))
                )
            visit_drugs = self._clean_sequence(list(dict.fromkeys(visit_drugs)))
            visit_drugs = self._ensure_nonempty_sequence(visit_drugs)

            lab_df = patient.get_events(
                event_type="labevents",
                filters=[("hadm_id", "==", admission.hadm_id)],
                return_df=True,
            )
            lab_vector = self._build_lab_vector(lab_df)

            all_conditions.append(
                self._ensure_nonempty_sequence(self._clean_sequence(condition_codes))
            )
            all_procedures.append(
                self._ensure_nonempty_sequence(self._clean_sequence(procedure_codes))
            )
            all_drugs.append(visit_drugs)
            all_labs.append(lab_vector)

        if first_catheter_time is None:
            return []

        if len(all_conditions) == 0:
            all_conditions = [[self.MISSING_TOKEN]]
            all_procedures = [[self.MISSING_TOKEN]]
            all_drugs = [[self.MISSING_TOKEN]]
            all_labs = [self._zero_lab_vector()]

        label = int(has_infection_after_catheter)
        samples: List[Dict[str, Any]] = [
            {
                "patient_id": patient.patient_id,
                "record_id": patient.patient_id,
                "conditions": all_conditions,
                "procedures": all_procedures,
                "drugs": all_drugs,
                "labs": all_labs,
                "label": label,
            }
        ]

        # Positive-only augmentation with all valid temporal suffix cutoffs.
        if label == 1 and len(all_conditions) > 1:
            for start in range(1, len(all_conditions)):
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": f"{patient.patient_id}_aug_{start}",
                        "conditions": all_conditions[start:],
                        "procedures": all_procedures[start:],
                        "drugs": all_drugs[start:],
                        "labs": all_labs[start:],
                        "label": label,
                    }
                )

        return samples
