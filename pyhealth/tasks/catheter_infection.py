# Description: Catheter-associated urinary infection prediction task for MIMIC-IV dataset

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

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
    EHR data.

    Sample Construction
    -------------------
    Patient scope:
        Only patients with at least one admission containing catheter codes
        (ICD-10: Y84.6, Z46.6, Z46.82, Z93.5, Z93.6, T83.01x–T83.09x, 0T9B70Z,
        0T2BX0Z, 0T9C7ZZ; ICD-9: 99631, 99632, E8705, E8796) OR at least one
        qualifying CAUTI/infection event produce samples. Patients with no catheter
        or infection history anywhere in their record produce no samples.

    Positive samples:
        Each admission satisfying either criterion below generates one positive sample
        (plus suffix-augmented variants that drop progressively earlier admissions).
        A patient with N qualifying infection admissions produces N independent
        positive samples.

        - Unconditional: admission contains a catheter-specific infection code
          (T83.511A, T83.518A/D/S, T83.519A/D/S; ICD-9 99664).
        - Conditional: admission contains both a general UTI code (N39.0, N10,
          R82.71, N30.x, N34.x) AND a catheter code in the same encounter.

        Feature window (``same_visit`` controls target-admission inclusion):
          ``same_visit=True``  — prior admissions + current admission with infection
                                 codes masked (catheter codes retained).
          ``same_visit=False`` — prior admissions only; target admission excluded.

    Negative samples:
        Every non-infection admission of a catheter-relevant patient generates one
        negative sample, regardless of whether that specific admission itself contains
        catheter codes. This teaches the model that absence of catheter codes is
        causally sufficient to predict no CAUTI — even for patients with prior catheter
        or CAUTI history. The full set of non-infection admissions is only emitted once
        the patient is confirmed catheter-relevant (deferred single-pass approach).

        Feature window:
          ``same_visit=True``  — prior admissions + current admission (full codes).
          ``same_visit=False`` — prior admissions only.

    History accumulation:
        The running feature window grows admission-by-admission in chronological order.
        Infection admissions are appended to the history with infection codes masked,
        so subsequent admissions can see prior CAUTI history without leaking the label.

    Infection Code Tiers
    --------------------
    - **Unconditional**: T83.511A, T83.518A/D/S, T83.519A/D/S, ICD-9 99664.
      Any admission containing one of these codes is a positive CAUTI event.
    - **Conditional**: N39.0, N10, R82.71, N30.x, N34.x.
      Positive only if a catheter code also appears in the same admission.

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
        pending_negatives: List[Dict[str, Any]] = []
        previous_admission_time: Optional[datetime] = None
        infection_event_count: int = 0
        has_catheter_or_cauti: bool = False

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

            if is_infection_event:
                infection_event_count += 1
                has_catheter_or_cauti = True

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
                # Non-infection admission.
                # Snapshot BEFORE appending so same_visit=False can use prior-only window.
                prev_icd_codes = list(all_icd_codes)
                prev_icd_times = list(all_icd_times)
                prev_lab_values = list(all_lab_values)
                prev_lab_times = list(all_lab_times)

                all_icd_codes.append(self._ensure_nonempty_sequence(visit_codes))
                all_icd_times.append(time_delta)
                all_lab_values.append(lab_vector)
                all_lab_times.append(time_delta)

                if has_catheter:
                    has_catheter_or_cauti = True

                # Build a negative for every non-infection admission. Deferred: only
                # emitted if this patient is catheter-relevant (checked after the loop).
                # Feature window respects same_visit:
                #   True  → prior admissions + this one (current-visit style)
                #   False → prior admissions only (next-visit style)
                if self.same_visit:
                    feat_icd_codes = list(all_icd_codes)
                    feat_icd_times = list(all_icd_times)
                    feat_lab_values = list(all_lab_values)
                    feat_lab_times = list(all_lab_times)
                else:
                    feat_icd_codes = prev_icd_codes
                    feat_icd_times = prev_icd_times
                    feat_lab_values = prev_lab_values
                    feat_lab_times = prev_lab_times

                if not feat_icd_codes:
                    feat_icd_codes = [[f"D_{self.MISSING_TOKEN}"]]
                    feat_icd_times = [0.0]
                if not feat_lab_values:
                    feat_lab_values = [self._zero_lab_vector()]
                    feat_lab_times = [0.0]

                pending_negatives.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": None,  # assigned after the loop
                        "icd_codes": (feat_icd_times, feat_icd_codes),
                        "labs": (feat_lab_times, feat_lab_values),
                        "label": 0,
                    }
                )

            previous_admission_time = admission_time

        # Emit pending negatives only for catheter-relevant patients (those with at
        # least one catheter code or CAUTI event). Patients with no catheter or
        # infection history are outside the target population and produce no samples.
        if has_catheter_or_cauti:
            for i, neg in enumerate(pending_negatives, 1):
                neg["record_id"] = f"{patient.patient_id}_neg{i}"
            all_samples.extend(pending_negatives)

        return all_samples


class CatheterAssociatedInfectionPredictionMIMIC4(_CatheterInfectionBase):
    """Nested-sequence patient-level catheter infection prediction task.

    Predicts catheter-associated urinary tract infection (CAUTI) from longitudinal
    EHR data.

    Sample Construction
    -------------------
    Patient scope:
        Only patients with at least one admission containing catheter codes
        (ICD-10: Y84.6, Z46.6, Z46.82, Z93.5, Z93.6, T83.01x–T83.09x, 0T9B70Z,
        0T2BX0Z, 0T9C7ZZ; ICD-9: 99631, 99632, E8705, E8796) OR at least one
        qualifying CAUTI/infection event produce samples. Patients with no catheter
        or infection history anywhere in their record produce no samples.

    Positive samples:
        Each admission satisfying either criterion below generates one positive sample
        (plus suffix-augmented variants that drop progressively earlier admissions).
        A patient with N qualifying infection admissions produces N independent
        positive samples.

        - Unconditional: admission contains a catheter-specific infection code
          (T83.511A, T83.518A/D/S, T83.519A/D/S; ICD-9 99664).
        - Conditional: admission contains both a general UTI code (N39.0, N10,
          R82.71, N30.x, N34.x) AND a catheter code in the same encounter.

        Feature window (``same_visit`` controls target-admission inclusion):
          ``same_visit=True``  — prior admissions + current admission with infection
                                 codes masked (catheter codes retained).
          ``same_visit=False`` — prior admissions only; target admission excluded.

    Negative samples:
        Every non-infection admission of a catheter-relevant patient generates one
        negative sample, regardless of whether that specific admission itself contains
        catheter codes. This teaches the model that absence of catheter codes is
        causally sufficient to predict no CAUTI — even for patients with prior catheter
        or CAUTI history. The full set of non-infection admissions is only emitted once
        the patient is confirmed catheter-relevant (deferred single-pass approach).

        Feature window:
          ``same_visit=True``  — prior admissions + current admission (full codes).
          ``same_visit=False`` — prior admissions only.

    History accumulation:
        The running feature window grows admission-by-admission in chronological order.
        Infection admissions are appended to the history with infection codes masked,
        so subsequent admissions can see prior CAUTI history without leaking the label.

    Infection Code Tiers
    --------------------
    - **Unconditional**: T83.511A, T83.518A/D/S, T83.519A/D/S, ICD-9 99664.
      Any admission containing one of these codes is a positive CAUTI event.
    - **Conditional**: N39.0, N10, R82.71, N30.x, N34.x.
      Positive only if a catheter code also appears in the same admission.

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
        pending_negatives: List[Dict[str, Any]] = []
        infection_event_count: int = 0
        has_catheter_or_cauti: bool = False

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

            if is_infection_event:
                infection_event_count += 1
                has_catheter_or_cauti = True

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
                # Non-infection admission.
                # Snapshot BEFORE appending so same_visit=False can use prior-only window.
                prev_conditions = list(all_conditions)
                prev_procedures = list(all_procedures)
                prev_drugs = list(all_drugs)
                prev_labs = list(all_labs)

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

                if has_catheter:
                    has_catheter_or_cauti = True

                # Build a negative for every non-infection admission. Deferred: only
                # emitted if this patient is catheter-relevant (checked after the loop).
                # Feature window respects same_visit:
                #   True  → prior admissions + this one (current-visit style)
                #   False → prior admissions only (next-visit style)
                if self.same_visit:
                    feat_conditions = list(all_conditions)
                    feat_procedures = list(all_procedures)
                    feat_drugs = list(all_drugs)
                    feat_labs = list(all_labs)
                else:
                    feat_conditions = prev_conditions
                    feat_procedures = prev_procedures
                    feat_drugs = prev_drugs
                    feat_labs = prev_labs

                if not feat_conditions:
                    feat_conditions = [[self.MISSING_TOKEN]]
                    feat_procedures = [[self.MISSING_TOKEN]]
                    feat_drugs = [[self.MISSING_TOKEN]]
                    feat_labs = [self._zero_lab_vector()]

                pending_negatives.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": None,  # assigned after the loop
                        "conditions": feat_conditions,
                        "procedures": feat_procedures,
                        "drugs": feat_drugs,
                        "labs": feat_labs,
                        "label": 0,
                    }
                )

        # Emit pending negatives only for catheter-relevant patients (those with at
        # least one catheter code or CAUTI event). Patients with no catheter or
        # infection history are outside the target population and produce no samples.
        if has_catheter_or_cauti:
            for i, neg in enumerate(pending_negatives, 1):
                neg["record_id"] = f"{patient.patient_id}_neg{i}"
            all_samples.extend(pending_negatives)

        return all_samples


class CatheterAssociatedInfectionPredictionStageNetMIMIC4DualContext(
    _CatheterInfectionBase
):
    """Dual-context StageNet CAUTI task.

    Delegates to ``CatheterAssociatedInfectionPredictionStageNetMIMIC4`` twice —
    once with ``same_visit=True`` and once with ``same_visit=False`` — then tags each
    sample with a ``visit_mode`` field and a ``_current`` / ``_next`` record_id suffix.

    Emits both context modes for each sample:
    - ``visit_mode="current"`` (``same_visit=True``): feature window includes the
      target admission (infection codes masked for positives; full codes for negatives).
    - ``visit_mode="next"`` (``same_visit=False``): feature window uses prior
      admissions only; target admission excluded for both positives and negatives.

    Sample scope and negative-sampling logic are identical to the base class.
    See ``CatheterAssociatedInfectionPredictionStageNetMIMIC4`` for full details.
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

    Delegates to ``CatheterAssociatedInfectionPredictionMIMIC4`` twice —
    once with ``same_visit=True`` and once with ``same_visit=False`` — then tags each
    sample with a ``visit_mode`` field and a ``_current`` / ``_next`` record_id suffix.

    Emits both context modes for each sample:
    - ``visit_mode="current"`` (``same_visit=True``): feature window includes the
      target admission (infection codes masked for positives; full codes for negatives).
    - ``visit_mode="next"`` (``same_visit=False``): feature window uses prior
      admissions only; target admission excluded for both positives and negatives.

    Sample scope and negative-sampling logic are identical to the base class.
    See ``CatheterAssociatedInfectionPredictionMIMIC4`` for full details.
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


# ===========================================================================
# Temporal (NHSN SUTI 1a-aligned) CAUTI tasks
# ===========================================================================
#
# Source tables / columns (MIMIC-IV v2.2, keys in ``mimic4_ehr.yaml``):
#   hcpcsevents        hcpcs_cd ∈ {51702, 51703} indwelling catheter insertion
#                      (51701 = non-indwelling straight cath; recorded only),
#                      timestamp = chartdate (date only)
#   procedureevents    itemid 229351 "Foley Catheter", starttime → endtime
#   outputevents       itemid 226559 Foley / 226563 Suprapubic, charttime
#                      (226567 Straight Cath is intentionally excluded)
#   microbiologyevents spec_type_desc URINE*, test_itemid 90039 URINE CULTURE /
#                      90235 REFLEX URINE CULTURE, org_name, quantity/comments,
#                      timestamp = chartdate, attribute charttime
#
# Hard requirements (never relaxed for missing data): an indwelling catheter
# timed for > 2 consecutive calendar days (catheter day >= 3) and an eligible
# day on hospital day >= 3 while still admitted. Admissions failing either
# rule emit no sample. Among eligible admissions, the infection markers
# (M1/M2/M3) are combined by union (OR) to tolerate missing documentation.

CATHETER_CPT_INDWELLING: FrozenSet[str] = frozenset({"51702", "51703"})
CATHETER_CPT_NONINDWELLING: FrozenSet[str] = frozenset({"51701"})
FOLEY_PROCEDURE_ITEMIDS: FrozenSet[str] = frozenset({"229351"})
FOLEY_OUTPUT_ITEMIDS: FrozenSet[str] = frozenset({"226559", "226563"})
URINE_SPEC_TYPES: FrozenSet[str] = frozenset(
    {"URINE", "URINE,KIDNEY", "URINE,SUPRAPUBIC ASPIRATE"}
)
# Only culture tests count; urine NAATs (chlamydia, gonorrhea), Legionella
# antigen, fungal/viral/AFB cultures on the same specimen are ignored.
URINE_CULTURE_TEST_ITEMIDS: FrozenSet[str] = frozenset({"90039", "90235"})
URINE_CULTURE_TEST_NAMES: FrozenSet[str] = frozenset(
    {"URINE CULTURE", "REFLEX URINE CULTURE"}
)
BLOOD_SPEC_TYPES: FrozenSet[str] = frozenset({"BLOOD CULTURE"})

# NHSN excluded organisms (Candida/yeast, mold, dimorphic fungi, parasites) and
# mixed-flora results, matched as substrings of the upper-cased org_name.
EXCLUDED_ORGANISM_SUBSTRINGS: Tuple[str, ...] = (
    "YEAST",
    "CANDIDA",
    "FUNG",
    "MOLD",
    "ASPERGILLUS",
    "CRYPTOCOCC",
    "HISTOPLASMA",
    "BLASTOMYCES",
    "COCCIDIOIDES",
    "PARASIT",
    "TRICHOMONAS",
    "MIXED",
)
# Comments indicating a contaminated / polymicrobial (>2 species) specimen.
MIXED_FLORA_COMMENT_SUBSTRINGS: Tuple[str, ...] = (
    "MIXED BACTERIAL",
    "COLONY TYPES",
)

_COLONY_COUNT_RE = re.compile(
    r"(>=|≥|>|<=|≤|<)?\s*(\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\s*-\s*(\d{1,3}(?:,\d{3})+|\d+))?\s*(?:CFU|ORGANISMS|COL)",
    re.IGNORECASE,
)

_NULL_STRINGS = {"", "none", "nan", "nat", "<na>", "null"}


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in _NULL_STRINGS else text


def _attr(event: Any, key: str) -> str:
    """Cleaned string attribute of a PyHealth ``Event`` ("" if absent/null)."""
    return _clean_str(getattr(event, key, None))


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse a MIMIC timestamp (datetime, date, or string); None if missing."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    text = _clean_str(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _collect_catheter_days(
    admit_date: date,
    disch_date: Optional[date],
    hcpcs_events: Iterable[Any] = (),
    procedure_events: Iterable[Any] = (),
    output_events: Iterable[Any] = (),
) -> Tuple[Set[date], Set[str]]:
    """Return (indwelling catheter calendar days, evidence sources) for one admission.

    Takes the admission's ``hcpcsevents``, ``procedureevents`` and
    ``outputevents`` Events. Days come from CPT 51702/51703 ``chartdate``,
    Foley procedureevents (every day from ``starttime`` through ``endtime``),
    and Foley/suprapubic outputevents ``charttime``, clipped to the admission.
    CPT 51701 (non-indwelling) adds the ``cpt_nonindwelling`` source but no days.
    """
    days: Set[date] = set()
    sources: Set[str] = set()

    def in_stay(day: date) -> bool:
        if day < admit_date:
            return False
        return disch_date is None or day <= disch_date

    for event in hcpcs_events:
        code = _attr(event, "hcpcs_cd").upper()
        if code in CATHETER_CPT_NONINDWELLING:
            sources.add("cpt_nonindwelling")
        if code in CATHETER_CPT_INDWELLING and in_stay(event.timestamp.date()):
            days.add(event.timestamp.date())
            sources.add("cpt")

    for event in procedure_events:
        if _attr(event, "itemid") not in FOLEY_PROCEDURE_ITEMIDS:
            continue
        start = event.timestamp
        end = _parse_datetime(getattr(event, "endtime", None)) or start
        day = start.date()
        while day <= end.date():
            if in_stay(day):
                days.add(day)
                sources.add("icu_proc")
            day += timedelta(days=1)

    for event in output_events:
        if _attr(event, "itemid") not in FOLEY_OUTPUT_ITEMIDS:
            continue
        if in_stay(event.timestamp.date()):
            days.add(event.timestamp.date())
            sources.add("icu_output")

    return days, sources


def _catheter_episodes(days: Iterable[date]) -> List[Tuple[date, date]]:
    """Group catheter days into consecutive-day episodes.

    A full calendar day without a catheter splits an episode (NHSN interruption).
    """
    episodes: List[Tuple[date, date]] = []
    for day in sorted(set(days)):
        if episodes and day - episodes[-1][1] == timedelta(days=1):
            episodes[-1] = (episodes[-1][0], day)
        else:
            episodes.append((day, day))
    return episodes


def _episode_index_date(
    episode: Tuple[date, date],
    admit_date: date,
    min_catheter_days: int = 3,
    min_hospital_day: int = 3,
    disch_date: Optional[date] = None,
) -> Optional[date]:
    """First date the episode is CAUTI-eligible, or None if it never is.

    Eligible dates d satisfy: the catheter was in place for at least
    ``min_catheter_days`` consecutive days (placement = day 1), d is on or after
    catheter day ``min_catheter_days``, the catheter is present on d or was
    removed the day before, d is on hospital day >= ``min_hospital_day``, and
    the patient is still admitted on d (d <= ``disch_date`` when known).
    """
    start, end = episode
    if (end - start).days + 1 < min_catheter_days:
        return None
    index = max(
        start + timedelta(days=min_catheter_days - 1),
        admit_date + timedelta(days=min_hospital_day - 1),
    )
    if index > end + timedelta(days=1):
        return None
    if disch_date is not None and index > disch_date:
        return None
    return index


def _onset_eligible(
    onset_date: date,
    episodes: Sequence[Tuple[date, date]],
    admit_date: date,
    min_catheter_days: int = 3,
    min_hospital_day: int = 3,
) -> bool:
    """Check NHSN timing for an infection onset date (hard requirements).

    The onset must be on hospital day >= ``min_hospital_day`` and inside a
    timed catheter episode's eligible window (catheter day >=
    ``min_catheter_days`` through the day after removal). Without timed
    catheter episodes the onset is never eligible.
    """
    if (onset_date - admit_date).days + 1 < min_hospital_day:
        return False
    for start, end in episodes:
        if (end - start).days + 1 < min_catheter_days:
            continue
        if (
            start + timedelta(days=min_catheter_days - 1)
            <= onset_date
            <= end + timedelta(days=1)
        ):
            return True
    return False


def _parse_colony_count(text: Any) -> Optional[int]:
    """Best-effort CFU/mL estimate from a quantity/comment string.

    ``>100,000`` → 100001, ``<10,000`` → 9999, ranges ``a-b`` → b - 1.
    Returns None when no count is reported.
    """
    cleaned = _clean_str(text)
    if not cleaned:
        return None
    match = _COLONY_COUNT_RE.search(cleaned)
    if match is None:
        return None
    op, low, high = match.groups()
    low_value = int(low.replace(",", ""))
    if high:
        return int(high.replace(",", "")) - 1
    if op == ">":
        return low_value + 1
    if op == "<":
        return low_value - 1
    return low_value


def _is_excluded_organism(org_name: str) -> bool:
    upper = org_name.upper()
    return any(s in upper for s in EXCLUDED_ORGANISM_SUBSTRINGS)


def _is_urine_culture(event: Any) -> bool:
    """True for a urine specimen row from a urine culture test.

    Matches ``spec_type_desc`` against ``URINE_SPEC_TYPES`` and ``test_itemid``
    against ``URINE_CULTURE_TEST_ITEMIDS`` (falling back to ``test_name`` when
    ``test_itemid`` is not loaded).
    """
    if _attr(event, "spec_type_desc").upper() not in URINE_SPEC_TYPES:
        return False
    test_itemid = _attr(event, "test_itemid")
    if test_itemid:
        return test_itemid in URINE_CULTURE_TEST_ITEMIDS
    return _attr(event, "test_name").upper() in URINE_CULTURE_TEST_NAMES


def _micro_onset(event: Any) -> datetime:
    """Specimen collection time: ``charttime``, else the ``chartdate`` timestamp."""
    return _parse_datetime(getattr(event, "charttime", None)) or event.timestamp


@dataclass(frozen=True)
class _UrineSpecimen:
    specimen_id: str
    onset: datetime
    organisms: Tuple[str, ...]
    bacteria: Tuple[str, ...]
    qualifies: bool
    reason: str


def _summarize_urine_specimens(
    micro_events: Iterable[Any],
    cfu_threshold: int = 100_000,
) -> List[_UrineSpecimen]:
    """Group urine-culture Events by specimen and apply the NHSN culture criterion.

    Only urine culture tests count (see ``_is_urine_culture``). A specimen
    qualifies when it grew 1–2 distinct organisms, at least one of them a
    non-excluded bacterium, with no mixed-flora comment. A colony count is
    enforced (>= ``cfu_threshold``) only when one is reported; MIMIC rarely
    records counts on organism rows, so absence does not veto. Onset is the
    earliest collection time (``charttime``, falling back to ``chartdate``).
    """
    grouped: Dict[str, List[Any]] = {}
    for event in micro_events:
        if not _is_urine_culture(event):
            continue
        specimen_id = _attr(event, "micro_specimen_id") or (
            f"anon_{event.timestamp.isoformat()}"
        )
        grouped.setdefault(specimen_id, []).append(event)

    specimens: List[_UrineSpecimen] = []
    for specimen_id, events in grouped.items():
        organisms = tuple(
            sorted({_attr(e, "org_name").upper() for e in events if _attr(e, "org_name")})
        )
        bacteria = tuple(o for o in organisms if not _is_excluded_organism(o))
        comments = " ".join(_attr(e, "comments").upper() for e in events)
        counts = [
            c
            for e in events
            for c in (
                _parse_colony_count(getattr(e, "quantity", None)),
                _parse_colony_count(getattr(e, "comments", None)),
            )
            if c is not None
        ]

        if not organisms:
            qualifies, reason = False, "no_growth"
        elif any(s in comments for s in MIXED_FLORA_COMMENT_SUBSTRINGS):
            qualifies, reason = False, "mixed_flora"
        elif len(organisms) > 2:
            qualifies, reason = False, "gt_2_organisms"
        elif not bacteria:
            qualifies, reason = False, "excluded_organism_only"
        elif counts and max(counts) < cfu_threshold:
            qualifies, reason = False, "below_cfu_threshold"
        else:
            qualifies, reason = True, "qualifies"

        specimens.append(
            _UrineSpecimen(
                specimen_id=specimen_id,
                onset=min(_micro_onset(e) for e in events),
                organisms=organisms,
                bacteria=bacteria,
                qualifies=qualifies,
                reason=reason,
            )
        )
    return sorted(specimens, key=lambda s: s.onset)


def _has_secondary_abuti(
    specimen: _UrineSpecimen,
    micro_events: Iterable[Any],
    window_days: int = 3,
) -> bool:
    """True if a blood culture within ±window_days grows a matching organism."""
    targets = set(specimen.bacteria)
    if not targets:
        return False
    for event in micro_events:
        if _attr(event, "spec_type_desc").upper() not in BLOOD_SPEC_TYPES:
            continue
        if _attr(event, "org_name").upper() not in targets:
            continue
        delta = (_micro_onset(event).date() - specimen.onset.date()).days
        if abs(delta) <= window_days:
            return True
    return False


class _CatheterTemporalBase(_CatheterInfectionBase):
    """Shared per-admission cohort/label logic for the temporal CAUTI tasks.

    Inclusion (per admission, hard requirements — never relaxed for missing
    data): timed indwelling catheter days from CPT 51702/51703
    (``hcpcsevents.hcpcs_cd``) and ICU Foley procedureevents/outputevents form
    an episode of > 2 consecutive calendar days, giving an eligible day that is
    catheter day >= 3, hospital day >= 3, and on or before discharge.
    Admissions with only ICD catheter codes, short episodes, or discharge
    before hospital day 3 emit no sample.

    Positive markers (union; label = 1 if any enabled marker fires):
        M1: ICD unconditional CAUTI code (T83.511A, T83.518x, T83.519x, 996.64).
        M2: ICD conditional UTI code (N39.0, N10, R82.71, N30.x, N34.x).
        M3: qualifying urine culture (``test_itemid`` 90039/90235; see
            ``_summarize_urine_specimens``) collected on hospital day >= 3
            inside an eligible catheter window (catheter day >= 3 through the
            day after removal), outside the 14-day repeat-infection timeframe
            of a prior M3 event.

    Index time (feature cutoff): start of the first eligible day. Prior
    admissions contribute full features; the current admission contributes
    only timestamped events before the index time (prescriptions, labs, HCPCS
    tokens) and never its discharge-time ICD codes.

    Limitations: no symptom criterion (fever, suprapubic/CVA tenderness); the
    CFU threshold is enforced only when a count is reported; ward catheters
    are visible only via rare CPT rows, so timed episodes are mostly ICU; MIMIC
    is single-facility so transfer attribution reduces to the admission.
    M1/M2 have no in-admission timestamp, so the union label is broader than
    NHSN — stratify by ``positive_markers`` or ``nhsn_strict`` (M3 fired).
    """

    ALL_MARKERS: ClassVar[Tuple[str, ...]] = ("M1", "M2", "M3")
    # Bump when cohort or label logic changes: instance attributes form the
    # PyHealth task cache key, so a new version forces samples to be rebuilt.
    COHORT_VERSION: ClassVar[int] = 2

    def __init__(
        self,
        map_ccscm: bool = True,
        positive_markers: Sequence[str] = ("M1", "M2", "M3"),
        rit_days: int = 14,
        min_catheter_days: int = 3,
        min_hospital_day: int = 3,
        cfu_threshold: int = 100_000,
    ):
        """Initialize task.

        Args:
            map_ccscm: Map ICD codes to CCS categories (default True).
            positive_markers: Markers whose union defines label = 1.
            rit_days: NHSN repeat-infection timeframe for M3 events.
            min_catheter_days: Catheter day on which CAUTI eligibility begins.
            min_hospital_day: Hospital day on which HAI onset may begin.
            cfu_threshold: Minimum CFU/mL, enforced only when reported.
        """
        unknown = set(positive_markers) - set(self.ALL_MARKERS)
        if unknown:
            raise ValueError(f"Unknown positive markers: {sorted(unknown)}")
        self.map_ccscm = map_ccscm
        self.positive_markers = tuple(m for m in self.ALL_MARKERS if m in positive_markers)
        self.rit_days = rit_days
        self.min_catheter_days = min_catheter_days
        self.min_hospital_day = min_hospital_day
        self.cfu_threshold = cfu_threshold
        self.cohort_version = self.COHORT_VERSION
        name = self._task_name_with_map_ccscm(type(self).task_name, map_ccscm)
        if self.positive_markers != self.ALL_MARKERS:
            name = f"{name}_markers_{'-'.join(self.positive_markers)}"
        self.task_name = name

    @staticmethod
    def _events(
        patient: Any,
        table: str,
        hadm_id: Any = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        return_df: bool = False,
    ) -> Any:
        """``patient.get_events`` for one table, optionally one admission.

        Returns no events when the table was not loaded into the dataset, so
        optional tables (ICU, microbiology, HCPCS) can be omitted.
        """
        if f"{table}/hadm_id" not in patient.data_source.columns:
            return pl.DataFrame() if return_df else []
        filters = [("hadm_id", "==", hadm_id)] if hadm_id is not None else None
        return patient.get_events(
            event_type=table, start=start, end=end, filters=filters, return_df=return_df
        )

    @staticmethod
    def _hcpcs_token(code: str) -> str:
        return f"HCPCS_{code}"

    def _admission_records(self, patient: Any) -> List[Dict[str, Any]]:
        """Build per-admission feature blocks, inclusion flag, label and metadata."""
        admissions = sorted(
            patient.get_events(event_type="admissions"), key=lambda e: e.timestamp
        )
        records: List[Dict[str, Any]] = []
        last_m3_date: Optional[date] = None

        for admission in admissions:
            admission_time = admission.timestamp
            if admission_time is None:
                continue
            dischtime = _parse_datetime(getattr(admission, "dischtime", None))
            hadm_id = admission.hadm_id
            admit_date = admission_time.date()
            disch_date = dischtime.date() if dischtime else None

            diagnoses = self._events(patient, "diagnoses_icd", hadm_id)
            procedures = self._events(patient, "procedures_icd", hadm_id)
            prescriptions = self._events(
                patient, "prescriptions", hadm_id, admission_time, dischtime
            )
            hcpcs_events = self._events(patient, "hcpcsevents", hadm_id)
            procedure_events = self._events(patient, "procedureevents", hadm_id)
            output_events = self._events(patient, "outputevents", hadm_id)
            # Microbiology hadm_id is often null: take the admission's date window
            # (chartdate timestamps sit at 00:00) and keep matching/unlinked rows.
            micro_events = [
                e
                for e in self._events(
                    patient,
                    "microbiologyevents",
                    start=datetime.combine(admit_date, time.min),
                    end=dischtime,
                )
                if _attr(e, "hadm_id") in ("", str(hadm_id))
            ]

            _, has_icd_catheter, has_unconditional, has_conditional = (
                self._determine_positive_label(diagnoses, procedures)
            )

            # ---- catheter timeline (hard gate) -------------------------------
            catheter_days, catheter_sources = _collect_catheter_days(
                admit_date, disch_date, hcpcs_events, procedure_events, output_events
            )
            if has_icd_catheter:
                catheter_sources.add("icd")
            episodes = _catheter_episodes(catheter_days)
            index_dates = [
                d
                for d in (
                    _episode_index_date(
                        ep,
                        admit_date,
                        self.min_catheter_days,
                        self.min_hospital_day,
                        disch_date,
                    )
                    for ep in episodes
                )
                if d is not None
            ]
            # Hard gate: catheter > 2 consecutive days and hospital day >= 3.
            eligible = bool(index_dates)
            if eligible:
                index_time = max(
                    datetime.combine(min(index_dates), time.min), admission_time
                )
            else:
                index_time = admission_time

            # ---- positive markers --------------------------------------------
            fired: Set[str] = set()
            if has_unconditional:
                fired.add("M1")
            if has_conditional:  # catheter co-occurrence implied by the gate
                fired.add("M2")

            m3_specimens: List[_UrineSpecimen] = []
            if eligible:
                for specimen in _summarize_urine_specimens(
                    micro_events, self.cfu_threshold
                ):
                    if not specimen.qualifies:
                        continue
                    onset_date = specimen.onset.date()
                    if not _onset_eligible(
                        onset_date,
                        episodes,
                        admit_date,
                        self.min_catheter_days,
                        self.min_hospital_day,
                    ):
                        continue
                    if (
                        last_m3_date is not None
                        and (onset_date - last_m3_date).days < self.rit_days
                    ):
                        continue
                    last_m3_date = onset_date
                    m3_specimens.append(specimen)
            if m3_specimens:
                fired.add("M3")

            label = int(any(m in fired for m in self.positive_markers))

            # ---- features ----------------------------------------------------
            condition_codes: List[str] = []
            for diag in diagnoses:
                code = getattr(diag, "icd_code", None)
                if code:
                    condition_codes.extend(
                        self._map_condition_to_tokens(
                            code, getattr(diag, "icd_version", None), self.map_ccscm
                        )
                    )
            procedure_codes: List[str] = []
            for proc in procedures:
                code = getattr(proc, "icd_code", None)
                if code:
                    procedure_codes.extend(
                        self._map_procedure_to_tokens(
                            code, getattr(proc, "icd_version", None), self.map_ccscm
                        )
                    )
            hcpcs_all = [
                self._hcpcs_token(_attr(e, "hcpcs_cd").upper())
                for e in hcpcs_events
                if _attr(e, "hcpcs_cd")
            ]
            hcpcs_before_index = [
                self._hcpcs_token(_attr(e, "hcpcs_cd").upper())
                for e in hcpcs_events
                if _attr(e, "hcpcs_cd") and e.timestamp < index_time
            ]

            drugs_all: List[str] = []
            drugs_before_index: List[str] = []
            for event in prescriptions:
                tokens = self._map_ndc_to_atc3_tokens(getattr(event, "ndc", None))
                drugs_all.extend(tokens)
                if event.timestamp < index_time:
                    drugs_before_index.extend(tokens)

            # _build_lab_vector (shared with the ICD tasks) takes a DataFrame.
            labs_all = self._build_lab_vector(
                self._events(
                    patient,
                    "labevents",
                    hadm_id,
                    admission_time,
                    dischtime,
                    return_df=True,
                )
            )
            if index_time > admission_time:
                labs_before_index = self._build_lab_vector(
                    self._events(
                        patient,
                        "labevents",
                        hadm_id,
                        admission_time,
                        index_time - timedelta(milliseconds=1),
                        return_df=True,
                    )
                )
            else:
                labs_before_index = self._zero_lab_vector()

            def dedup(values: List[str]) -> List[str]:
                return self._ensure_nonempty_sequence(list(dict.fromkeys(values)))

            full_block = {
                "conditions": dedup(condition_codes),
                "procedures": dedup(procedure_codes + hcpcs_all),
                "drugs": dedup(drugs_all),
                "labs": labs_all,
            }
            partial_block = None
            if index_time > admission_time:
                partial_block = {
                    "conditions": [self.MISSING_TOKEN],
                    "procedures": dedup(hcpcs_before_index),
                    "drugs": dedup(drugs_before_index),
                    "labs": labs_before_index,
                }

            onset_time = m3_specimens[0].onset.isoformat() if m3_specimens else ""
            secondary_abuti = int(
                any(_has_secondary_abuti(s, micro_events) for s in m3_specimens)
            )
            records.append(
                {
                    "admission_time": admission_time,
                    "include": eligible,
                    "full": full_block,
                    "partial": partial_block,
                    "label": label,
                    "meta": {
                        "hadm_id": _clean_str(hadm_id),
                        "positive_markers": "|".join(sorted(fired)),
                        "catheter_sources": "|".join(sorted(catheter_sources)),
                        "nhsn_strict": int(bool(m3_specimens)),
                        "index_time": index_time.isoformat(),
                        "onset_time": onset_time,
                        "secondary_abuti": secondary_abuti,
                    },
                }
            )
        return records


class CatheterAssociatedInfectionPredictionMIMIC4Temporal(_CatheterTemporalBase):
    """Temporal, NHSN SUTI 1a-aligned CAUTI task (nested-sequence features).

    One sample per CAUTI-eligible admission (catheter > 2 days, hospital day
    >= 3); see ``_CatheterTemporalBase`` for inclusion, the M1/M2/M3 union
    label, index-time feature cutoff and limitations. Requires ``ehr_tables`` to include ``diagnoses_icd``,
    ``procedures_icd``, ``prescriptions``, ``labevents`` and, for timing,
    ``hcpcsevents``, ``procedureevents``, ``outputevents``, ``microbiologyevents``.

    Features (per admission in the feature window)
    -----------------------------------------------
    - conditions: nested_sequence of CCS-CM diagnosis tokens
    - procedures: nested_sequence of CCS-PCS procedure + ``HCPCS_<cpt>`` tokens
    - drugs: nested_sequence of ATC Level-3 drug tokens
    - labs: nested_sequence_floats of 10D mean lab vectors
    """

    task_name: str = "CatheterAssociatedInfectionPredictionMIMIC4Temporal"

    input_schema: Dict[str, str] = {
        "conditions": "nested_sequence",
        "procedures": "nested_sequence",
        "drugs": "nested_sequence",
        "labs": "nested_sequence_floats",
    }
    output_schema: Dict[str, str] = {"label": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Create temporal nested-sequence samples for one patient."""
        history: Dict[str, List[Any]] = {k: [] for k in self.input_schema}
        samples: List[Dict[str, Any]] = []

        for record in self._admission_records(patient):
            if record["include"]:
                window = {k: list(v) for k, v in history.items()}
                if record["partial"] is not None:
                    for key in window:
                        window[key].append(record["partial"][key])
                if not window["conditions"]:
                    window = {
                        "conditions": [[self.MISSING_TOKEN]],
                        "procedures": [[self.MISSING_TOKEN]],
                        "drugs": [[self.MISSING_TOKEN]],
                        "labs": [self._zero_lab_vector()],
                    }
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": f"{patient.patient_id}_{record['meta']['hadm_id']}",
                        **window,
                        "label": record["label"],
                        **record["meta"],
                    }
                )
            for key in history:
                history[key].append(record["full"][key])

        return samples


class CatheterAssociatedInfectionPredictionStageNetMIMIC4Temporal(_CatheterTemporalBase):
    """StageNet variant of ``CatheterAssociatedInfectionPredictionMIMIC4Temporal``.

    Features
    --------
    - icd_codes: StageNet tuple (hours since previous admission, ``D_``-prefixed
      diagnosis and ``P_``-prefixed procedure/HCPCS tokens)
    - labs: StageNet tensor tuple (same time deltas, 10D mean lab vectors)
    """

    task_name: str = "CatheterAssociatedInfectionPredictionStageNetMIMIC4Temporal"

    def __init__(self, padding: int = 0, **kwargs: Any):
        super().__init__(**kwargs)
        self.padding = padding
        self.input_schema: Dict[str, Tuple[str, Dict[str, Any]]] = {  # type: ignore
            "icd_codes": ("stagenet", {"padding": padding}),
            "labs": ("stagenet_tensor", {}),
        }
        self.output_schema: Dict[str, str] = {"label": "binary"}  # type: ignore

    def _stagenet_codes(self, block: Dict[str, Any]) -> List[str]:
        codes = [f"D_{c}" for c in block["conditions"] if c != self.MISSING_TOKEN]
        codes += [f"P_{c}" for c in block["procedures"] if c != self.MISSING_TOKEN]
        return codes or [f"D_{self.MISSING_TOKEN}"]

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Create temporal StageNet samples for one patient."""
        icd_codes: List[List[str]] = []
        icd_times: List[float] = []
        lab_values: List[List[float]] = []
        samples: List[Dict[str, Any]] = []
        previous_time: Optional[datetime] = None

        for record in self._admission_records(patient):
            admission_time = record["admission_time"]
            delta = (
                (admission_time - previous_time).total_seconds() / 3600.0
                if previous_time is not None
                else 0.0
            )
            if record["include"]:
                feat_codes = list(icd_codes)
                feat_times = list(icd_times)
                feat_labs = list(lab_values)
                if record["partial"] is not None:
                    feat_codes.append(self._stagenet_codes(record["partial"]))
                    feat_times.append(delta)
                    feat_labs.append(record["partial"]["labs"])
                if not feat_codes:
                    feat_codes = [[f"D_{self.MISSING_TOKEN}"]]
                    feat_times = [0.0]
                    feat_labs = [self._zero_lab_vector()]
                samples.append(
                    {
                        "patient_id": patient.patient_id,
                        "record_id": f"{patient.patient_id}_{record['meta']['hadm_id']}",
                        "icd_codes": (feat_times, feat_codes),
                        "labs": (list(feat_times), feat_labs),
                        "label": record["label"],
                        **record["meta"],
                    }
                )
            icd_codes.append(self._stagenet_codes(record["full"]))
            icd_times.append(delta)
            lab_values.append(record["full"]["labs"])
            previous_time = admission_time

        return samples
