"""Task definitions and target helpers for the EOL mistrust workflow."""

from __future__ import annotations

import re
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from .base_task import BaseTask

CODE_STATUS_ITEMIDS = {128, 223758}

EOL_MISTRUST_TASK_MAP = OrderedDict(
    [
        ("Left AMA", "left_ama"),
        ("Code Status", "code_status_dnr_dni_cmo"),
        ("In-hospital mortality", "in_hospital_mortality"),
    ]
)


def _require_columns(df: pd.DataFrame, required: Sequence[str], df_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{df_name} is missing required columns: {missing_str}")


def _normalize_token(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    value = str(value).strip().lower()
    if not value:
        return ""
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def _to_datetime(value) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def _calculate_age_years(admittime, dob) -> float:
    admit_time = _to_datetime(admittime)
    birth_time = _to_datetime(dob)
    if pd.isna(admit_time) or pd.isna(birth_time):
        return float("nan")

    seconds_per_year = 365.25 * 24 * 3600
    age = (admit_time.to_pydatetime() - birth_time.to_pydatetime()).total_seconds() / seconds_per_year
    return 90.0 if age > 200 else float(age)


def _calculate_los_days(admittime, dischtime) -> float:
    admit_time = _to_datetime(admittime)
    discharge_time = _to_datetime(dischtime)
    if pd.isna(admit_time) or pd.isna(discharge_time):
        return float("nan")
    return float((discharge_time - admit_time).total_seconds() / 86400.0)


def map_ethnicity_to_race(ethnicity) -> str:
    """Collapse raw MIMIC ethnicity strings into the study race groups."""

    text = str(ethnicity or "").upper()
    if "BLACK" in text or "AFRICAN" in text:
        return "BLACK"
    if "WHITE" in text or "EUROPEAN" in text or "PORTUGUESE" in text:
        return "WHITE"
    if "ASIAN" in text:
        return "ASIAN"
    if "HISPANIC" in text or "LATINO" in text or "SOUTH AMERICAN" in text:
        return "HISPANIC"
    if "NATIVE" in text or "AMERICAN INDIAN" in text or "ALASKA NATIVE" in text:
        return "NATIVE AMERICAN"
    return "OTHER"


def map_insurance_to_group(insurance) -> str:
    """Collapse raw insurance text into the three study groups."""

    text = str(insurance or "").strip().lower()
    normalized = re.sub(r"\s+", " ", text)
    if normalized in {"medicare", "medicaid", "government", "public"}:
        return "Public"
    if normalized in {"private"}:
        return "Private"
    if normalized in {"self pay", "self-pay", "self_pay"}:
        return "Self-Pay"
    return str(insurance or "")


def prepare_note_text(text) -> str:
    """Normalize note text by whitespace tokenization and rejoining only."""

    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return " ".join(str(text).split())


def build_left_ama_target(admissions: pd.DataFrame) -> pd.DataFrame:
    """Build the exact-match Left AMA target from admissions."""

    _require_columns(admissions, ["hadm_id", "discharge_location"], "admissions")
    targets = admissions[["hadm_id", "discharge_location"]].drop_duplicates("hadm_id").copy()
    targets["left_ama"] = (
        targets["discharge_location"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("LEFT AGAINST MEDICAL ADVICE")
        .astype(int)
    )
    return targets[["hadm_id", "left_ama"]].sort_values("hadm_id").reset_index(drop=True)


def build_in_hospital_mortality_target(admissions: pd.DataFrame) -> pd.DataFrame:
    """Build the in-hospital mortality target from admissions."""

    _require_columns(admissions, ["hadm_id", "hospital_expire_flag"], "admissions")
    targets = admissions[["hadm_id", "hospital_expire_flag"]].drop_duplicates("hadm_id").copy()
    targets["in_hospital_mortality"] = (
        pd.to_numeric(targets["hospital_expire_flag"], errors="coerce").fillna(0).astype(int)
    )
    return (
        targets[["hadm_id", "in_hospital_mortality"]]
        .sort_values("hadm_id")
        .reset_index(drop=True)
    )


def build_code_status_target(
    chartevents: pd.DataFrame,
    itemids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Build the code-status target using the required itemids only."""

    _require_columns(chartevents, ["hadm_id", "itemid", "value"], "chartevents")
    allowed_itemids = set(CODE_STATUS_ITEMIDS if itemids is None else itemids)

    if chartevents.empty:
        return pd.DataFrame(columns=["hadm_id", "code_status_dnr_dni_cmo"])

    codes = chartevents.loc[chartevents["itemid"].isin(allowed_itemids)].copy()
    if codes.empty:
        return pd.DataFrame(columns=["hadm_id", "code_status_dnr_dni_cmo"])

    normalized_value = codes["value"].map(_normalize_token)
    positive = normalized_value.apply(
        lambda value: int(
            ("dnr" in value)
            or ("dni" in value)
            or ("comfort" in value)
            or ("cmo" in value)
        )
    )
    target = (
        pd.DataFrame({"hadm_id": codes["hadm_id"], "code_status_dnr_dni_cmo": positive})
        .groupby("hadm_id", as_index=False)["code_status_dnr_dni_cmo"]
        .max()
        .sort_values("hadm_id")
    )
    return target.reset_index(drop=True)


def get_eol_mistrust_task_map() -> OrderedDict[str, str]:
    """Return the three downstream target names used by the study."""

    return OrderedDict(EOL_MISTRUST_TASK_MAP)


class EOLMistrustDownstreamMIMIC3(BaseTask):
    """Admission-level downstream prediction task for the EOL mistrust study."""

    task_name = "EOLMistrustDownstreamMIMIC3"

    def __init__(
        self,
        target: str = "in_hospital_mortality",
        include_notes: bool = False,
    ) -> None:
        if target not in set(EOL_MISTRUST_TASK_MAP.values()):
            raise ValueError(f"Unsupported EOL mistrust target: {target}")

        self.target = target
        self.include_notes = include_notes
        self.input_schema: Dict[str, str] = {
            "conditions": "sequence",
            "procedures": "sequence",
            "drugs": "sequence",
            "age": "float",
            "los_days": "float",
            "gender": "text",
            "insurance": "text",
            "race": "text",
        }
        if include_notes:
            self.input_schema["clinical_notes"] = "text"
        self.output_schema: Dict[str, str] = {target: "binary"}

    def _get_single_patient_event(self, patient: Any, event_type: str):
        events = patient.get_events(event_type=event_type)
        if not events:
            return None
        return events[0]

    def _get_codes_for_admission(self, patient: Any, event_type: str, hadm_id) -> List[str]:
        events = patient.get_events(
            event_type=event_type,
            filters=[("hadm_id", "==", hadm_id)],
        )
        values: List[str] = []
        for event in events:
            for attribute in ("icd9_code", "icd_code", "drug", "ndc"):
                value = getattr(event, attribute, None)
                if value is not None and str(value).strip():
                    values.append(str(value))
                    break
        return values

    def _get_note_text(self, patient: Any, hadm_id) -> str:
        notes = patient.get_events(
            event_type="noteevents",
            filters=[("hadm_id", "==", hadm_id)],
        )
        return prepare_note_text(" ".join(str(getattr(note, "text", "")) for note in notes))

    def _get_code_status_label(self, patient: Any, hadm_id) -> int:
        events = patient.get_events(
            event_type="chartevents",
            filters=[("hadm_id", "==", hadm_id)],
        )
        rows = []
        for event in events:
            rows.append(
                {
                    "hadm_id": getattr(event, "hadm_id", hadm_id),
                    "itemid": getattr(event, "itemid", None),
                    "value": getattr(event, "value", None),
                }
            )
        if not rows:
            return 0
        target = build_code_status_target(pd.DataFrame(rows))
        if target.empty:
            return 0
        return int(target["code_status_dnr_dni_cmo"].max())

    def _get_target_value(self, patient: Any, admission: Any) -> int:
        if self.target == "left_ama":
            discharge_location = str(getattr(admission, "discharge_location", "") or "")
            return int(discharge_location.strip().upper() == "LEFT AGAINST MEDICAL ADVICE")
        if self.target == "in_hospital_mortality":
            expire_flag = getattr(admission, "hospital_expire_flag", 0)
            try:
                return int(expire_flag)
            except (TypeError, ValueError):
                return 0
        if self.target == "code_status_dnr_dni_cmo":
            return self._get_code_status_label(patient, admission.hadm_id)
        raise ValueError(f"Unsupported EOL mistrust target: {self.target}")

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        admissions = patient.get_events(event_type="admissions")
        patient_event = self._get_single_patient_event(patient, "patients")
        if not admissions:
            return []

        samples: List[Dict[str, Any]] = []
        for admission in admissions:
            hadm_id = getattr(admission, "hadm_id", None)
            if hadm_id is None:
                continue

            conditions = self._get_codes_for_admission(patient, "diagnoses_icd", hadm_id)
            procedures = self._get_codes_for_admission(patient, "procedures_icd", hadm_id)
            drugs = self._get_codes_for_admission(patient, "prescriptions", hadm_id)

            sample: Dict[str, Any] = {
                "visit_id": hadm_id,
                "hadm_id": hadm_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "age": _calculate_age_years(
                    getattr(admission, "timestamp", None),
                    getattr(patient_event, "dob", None) if patient_event is not None else None,
                ),
                "los_days": _calculate_los_days(
                    getattr(admission, "timestamp", None),
                    getattr(admission, "dischtime", None),
                ),
                "gender": getattr(patient_event, "gender", None) if patient_event is not None else None,
                "insurance": map_insurance_to_group(getattr(admission, "insurance", None)),
                "race": map_ethnicity_to_race(getattr(admission, "ethnicity", None)),
                self.target: self._get_target_value(patient, admission),
            }
            if self.include_notes:
                sample["clinical_notes"] = self._get_note_text(patient, hadm_id)
            samples.append(sample)
        return samples


class EOLMistrustLeftAMAPredictionMIMIC3(EOLMistrustDownstreamMIMIC3):
    """Task wrapper for the Left AMA downstream target."""

    task_name = "EOLMistrustLeftAMAPredictionMIMIC3"

    def __init__(self, include_notes: bool = False) -> None:
        super().__init__(target="left_ama", include_notes=include_notes)


class EOLMistrustCodeStatusPredictionMIMIC3(EOLMistrustDownstreamMIMIC3):
    """Task wrapper for the code-status downstream target."""

    task_name = "EOLMistrustCodeStatusPredictionMIMIC3"

    def __init__(self, include_notes: bool = False) -> None:
        super().__init__(target="code_status_dnr_dni_cmo", include_notes=include_notes)


class EOLMistrustMortalityPredictionMIMIC3(EOLMistrustDownstreamMIMIC3):
    """Task wrapper for the in-hospital mortality downstream target."""

    task_name = "EOLMistrustMortalityPredictionMIMIC3"

    def __init__(self, include_notes: bool = False) -> None:
        super().__init__(target="in_hospital_mortality", include_notes=include_notes)


__all__ = [
    "CODE_STATUS_ITEMIDS",
    "EOL_MISTRUST_TASK_MAP",
    "EOLMistrustCodeStatusPredictionMIMIC3",
    "EOLMistrustDownstreamMIMIC3",
    "EOLMistrustLeftAMAPredictionMIMIC3",
    "EOLMistrustMortalityPredictionMIMIC3",
    "build_code_status_target",
    "build_in_hospital_mortality_target",
    "build_left_ama_target",
    "get_eol_mistrust_task_map",
    "map_ethnicity_to_race",
    "map_insurance_to_group",
    "prepare_note_text",
]
