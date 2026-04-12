"""Task definitions and target helpers for the EOL mistrust workflow.

Structure
---------
This module now keeps two logic families explicit:

1. Normal Path
   The corrected, cleaned task helpers used by the default research flow.
2. Paper-like Path
   The notebook-faithful special logic that only exists for paper compatibility.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .base_task import BaseTask

CODE_STATUS_ITEMIDS = {128, 223758}
CODE_STATUS_MODE_CORRECTED = "corrected"
CODE_STATUS_MODE_PAPER_LIKE = "paper_like"
DATASET_PREPARE_MODE_DEFAULT = "default"
DATASET_PREPARE_MODE_PAPER_LIKE = "paper_like"

CODE_STATUS_POSITIVE_SUBSTRINGS = (
    "dnr",
    "dni",
    "comfort",
    "cmo",
    "do_not_resusc",
    "do_not_intubat",
    "cpr_not_indicat",
)
CODE_STATUS_NOTEBOOK_POSITIVE_STRINGS = ("DNR", "DNI", "Comfort", "Do Not")
CODE_STATUS_NOTEBOOK_FULL_CODE_VALUES = {"Full Code", "Full code"}

EOL_MISTRUST_TASK_MAP = OrderedDict(
    [
        ("Left AMA", "left_ama"),
        ("Code Status", "code_status_dnr_dni_cmo"),
        ("In-hospital mortality", "in_hospital_mortality"),
    ]
)

_DATASET_PREPARE_ROUTE_SETTINGS = {
    DATASET_PREPARE_MODE_DEFAULT: {
        "paper_like_dataset_prepare": False,
        "code_status_mode": CODE_STATUS_MODE_CORRECTED,
    },
    DATASET_PREPARE_MODE_PAPER_LIKE: {
        "paper_like_dataset_prepare": True,
        "code_status_mode": CODE_STATUS_MODE_PAPER_LIKE,
    },
}


def _require_columns(df: pd.DataFrame, required: Sequence[str], df_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {', '.join(missing)}")


def _coerce_timestamp(value) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def _normalize_token(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _normalize_code_status_mode(mode: str | None) -> str:
    normalized = (
        CODE_STATUS_MODE_CORRECTED if mode is None else str(mode).strip().lower()
    )
    if normalized not in {CODE_STATUS_MODE_CORRECTED, CODE_STATUS_MODE_PAPER_LIKE}:
        raise ValueError(
            "code_status_mode must be one of "
            f"{CODE_STATUS_MODE_CORRECTED!r} or {CODE_STATUS_MODE_PAPER_LIKE!r}"
        )
    return normalized


def _normalize_dataset_prepare_mode(mode: str | None) -> str:
    normalized = (
        DATASET_PREPARE_MODE_DEFAULT if mode is None else str(mode).strip().lower()
    )
    if normalized not in _DATASET_PREPARE_ROUTE_SETTINGS:
        raise ValueError(
            "dataset_prepare_mode must be one of "
            f"{DATASET_PREPARE_MODE_DEFAULT!r} or "
            f"{DATASET_PREPARE_MODE_PAPER_LIKE!r}"
        )
    return normalized


def _calculate_age_years(admittime, dob) -> float:
    admit_time = _coerce_timestamp(admittime)
    birth_time = _coerce_timestamp(dob)
    if pd.isna(admit_time) or pd.isna(birth_time):
        return float("nan")

    seconds_per_year = 365.25 * 24 * 3600
    age_years = (
        admit_time.to_pydatetime() - birth_time.to_pydatetime()
    ).total_seconds() / seconds_per_year
    return 90.0 if age_years > 200 else float(age_years)


def _calculate_los_days(admittime, dischtime) -> float:
    admit_time = _coerce_timestamp(admittime)
    discharge_time = _coerce_timestamp(dischtime)
    if pd.isna(admit_time) or pd.isna(discharge_time):
        return float("nan")
    return float((discharge_time - admit_time).total_seconds() / 86400.0)


def _calculate_paper_like_los_days(admittime, dischtime) -> float:
    admit_time = _coerce_timestamp(admittime)
    discharge_time = _coerce_timestamp(dischtime)
    if pd.isna(admit_time) or pd.isna(discharge_time):
        return float("nan")
    return float((discharge_time - admit_time).seconds / 3600.0)


# ---------------------------------------------------------------------------
# Normal Path
# ---------------------------------------------------------------------------


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

    normalized = re.sub(r"\s+", " ", str(insurance or "").strip().lower())
    if normalized in {"medicare", "medicaid", "government", "public"}:
        return "Public"
    if normalized == "private":
        return "Private"
    return "Self-Pay"


def prepare_note_text(text) -> str:
    """Normalize note text by collapsing whitespace only."""

    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return " ".join(str(text).split())


def build_left_ama_target(admissions: pd.DataFrame) -> pd.DataFrame:
    """Build the Left AMA target from admissions discharge_location codes."""

    _require_columns(admissions, ["hadm_id", "discharge_location"], "admissions")
    targets = (
        admissions[["hadm_id", "discharge_location"]].drop_duplicates("hadm_id").copy()
    )
    discharge_location = (
        targets["discharge_location"].fillna("").astype(str).str.strip().str.upper()
    )
    targets["left_ama"] = discharge_location.isin(
        {"LEFT AGAINST MEDICAL ADVI", "LEFT AGAINST MEDICAL ADVICE"}
    ).astype(int)
    return (
        targets[["hadm_id", "left_ama"]].sort_values("hadm_id").reset_index(drop=True)
    )


def build_in_hospital_mortality_target(admissions: pd.DataFrame) -> pd.DataFrame:
    """Build the in-hospital mortality target from admissions."""

    _require_columns(admissions, ["hadm_id", "hospital_expire_flag"], "admissions")
    targets = (
        admissions[["hadm_id", "hospital_expire_flag"]]
        .drop_duplicates("hadm_id")
        .copy()
    )
    targets["in_hospital_mortality"] = (
        pd.to_numeric(
            targets["hospital_expire_flag"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    return (
        targets[["hadm_id", "in_hospital_mortality"]]
        .sort_values("hadm_id")
        .reset_index(drop=True)
    )


def is_positive_code_status_value(value) -> bool:
    """Return True when a raw code-status chart value indicates a positive label."""

    normalized = _normalize_token(value)
    return any(token in normalized for token in CODE_STATUS_POSITIVE_SUBSTRINGS)


def _build_code_status_target_normal(codes: pd.DataFrame) -> pd.DataFrame:
    labeled = codes.copy()
    labeled["code_status_dnr_dni_cmo"] = labeled["value"].map(
        lambda value: int(is_positive_code_status_value(value))
    )

    if "charttime" not in labeled.columns:
        return (
            labeled[["hadm_id", "code_status_dnr_dni_cmo"]]
            .groupby("hadm_id", as_index=False)["code_status_dnr_dni_cmo"]
            .max()
            .sort_values("hadm_id")
            .reset_index(drop=True)
        )

    labeled["_charttime"] = pd.to_datetime(labeled["charttime"], errors="coerce")
    labeled["_has_charttime"] = labeled["_charttime"].notna().astype(int)
    labeled["_event_order"] = range(len(labeled))
    latest = (
        labeled.sort_values(
            ["hadm_id", "_has_charttime", "_charttime", "_event_order"],
            kind="stable",
        )
        .groupby("hadm_id", as_index=False)
        .tail(1)[["hadm_id", "code_status_dnr_dni_cmo"]]
        .sort_values("hadm_id")
    )
    return latest.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Paper-like Path
# ---------------------------------------------------------------------------


def _advance_paper_like_code_status_label(
    current_label: int | None, value
) -> int | None:
    """Replicate the notebook's stateful overwrite behavior exactly."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return current_label

    text = str(value)
    if any(token in text for token in CODE_STATUS_NOTEBOOK_POSITIVE_STRINGS):
        return 1
    if text in CODE_STATUS_NOTEBOOK_FULL_CODE_VALUES:
        return 0
    return current_label


def _build_code_status_target_paper_like(codes: pd.DataFrame) -> pd.DataFrame:
    current_label: int | None = None
    notebook_targets: dict[int, int] = {}

    for row in codes.itertuples(index=False):
        current_label = _advance_paper_like_code_status_label(
            current_label,
            getattr(row, "value"),
        )
        if current_label is not None:
            notebook_targets[int(getattr(row, "hadm_id"))] = int(current_label)

    return (
        pd.DataFrame(
            {
                "hadm_id": sorted(notebook_targets),
                "code_status_dnr_dni_cmo": [
                    int(notebook_targets[hadm_id])
                    for hadm_id in sorted(notebook_targets)
                ],
            }
        )
        .sort_values("hadm_id")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Shared target entry points
# ---------------------------------------------------------------------------


def build_code_status_target(
    chartevents: pd.DataFrame,
    itemids: Iterable[int] | None = None,
    code_status_mode: str = CODE_STATUS_MODE_CORRECTED,
) -> pd.DataFrame:
    """Build the code-status target from the required itemids only."""

    _require_columns(chartevents, ["hadm_id", "itemid", "value"], "chartevents")
    if chartevents.empty:
        return pd.DataFrame(columns=["hadm_id", "code_status_dnr_dni_cmo"])

    allowed_itemids = set(CODE_STATUS_ITEMIDS if itemids is None else itemids)
    codes = chartevents.loc[chartevents["itemid"].isin(allowed_itemids)].copy()
    if codes.empty:
        return pd.DataFrame(columns=["hadm_id", "code_status_dnr_dni_cmo"])

    if _normalize_code_status_mode(code_status_mode) == CODE_STATUS_MODE_PAPER_LIKE:
        return _build_code_status_target_paper_like(codes)
    return _build_code_status_target_normal(codes)


def get_eol_mistrust_task_map() -> OrderedDict[str, str]:
    """Return the downstream target names used by the study."""

    return OrderedDict(EOL_MISTRUST_TASK_MAP)


class EOLMistrustDownstreamMIMIC3(BaseTask):
    """Admission-level downstream prediction task for the EOL mistrust study."""

    task_name = "EOLMistrustDownstreamMIMIC3"

    def __init__(
        self,
        target: str = "in_hospital_mortality",
        include_notes: bool = False,
        dataset_prepare_mode: str = DATASET_PREPARE_MODE_DEFAULT,
    ) -> None:
        if target not in set(EOL_MISTRUST_TASK_MAP.values()):
            raise ValueError(f"Unsupported EOL mistrust target: {target}")

        self.target = target
        self.include_notes = include_notes
        self.dataset_prepare_mode = _normalize_dataset_prepare_mode(
            dataset_prepare_mode
        )
        route_settings = _DATASET_PREPARE_ROUTE_SETTINGS[self.dataset_prepare_mode]
        self.paper_like_dataset_prepare = bool(
            route_settings["paper_like_dataset_prepare"]
        )
        self.code_status_mode = str(route_settings["code_status_mode"])
        self.input_schema: dict[str, str] = {
            "conditions": "sequence",
            "procedures": "sequence",
            "drugs": "sequence",
            "age": "tensor",
            "los_days": "tensor",
            "gender": "text",
            "insurance": "text",
            "race": "text",
        }
        if include_notes:
            self.input_schema["clinical_notes"] = "text"
        self.output_schema: dict[str, str] = {target: "binary"}

    def _get_codes_for_admission(
        self, patient: Any, event_type: str, hadm_id
    ) -> list[str]:
        events = self._get_events_for_admission(patient, event_type, hadm_id)
        values: list[str] = []
        for event in events:
            for attribute in ("icd9_code", "icd_code", "drug", "ndc"):
                value = getattr(event, attribute, None)
                if value is not None and str(value).strip():
                    values.append(str(value))
                    break
        return values

    def _get_events_for_admission(
        self, patient: Any, event_type: str, hadm_id
    ) -> list[Any]:
        events = patient.get_events(event_type=event_type)
        return [
            event
            for event in events
            if getattr(event, "hadm_id", None) == hadm_id
        ]

    def _get_note_text(self, patient: Any, hadm_id) -> str:
        notes = self._get_events_for_admission(patient, "noteevents", hadm_id)
        return prepare_note_text(
            " ".join(str(getattr(note, "text", "")) for note in notes)
        )

    def _get_code_status_label(self, patient: Any, hadm_id) -> int:
        events = self._get_events_for_admission(patient, "chartevents", hadm_id)
        rows = [
            {
                "hadm_id": getattr(event, "hadm_id", hadm_id),
                "itemid": getattr(event, "itemid", None),
                "value": getattr(event, "value", None),
                "charttime": getattr(event, "charttime", None),
            }
            for event in events
        ]
        if not rows:
            return 0
        target = build_code_status_target(
            pd.DataFrame(rows),
            code_status_mode=self.code_status_mode,
        )
        return 0 if target.empty else int(target["code_status_dnr_dni_cmo"].max())

    def _get_target_value(self, patient: Any, admission: Any) -> int:
        if self.target == "left_ama":
            discharge_location = str(getattr(admission, "discharge_location", "") or "")
            return int(
                discharge_location.strip().upper() == "LEFT AGAINST MEDICAL ADVICE"
            )
        if self.target == "in_hospital_mortality":
            expire_flag = getattr(admission, "hospital_expire_flag", 0)
            try:
                return int(expire_flag)
            except (TypeError, ValueError):
                return 0
        if self.target == "code_status_dnr_dni_cmo":
            return self._get_code_status_label(patient, admission.hadm_id)
        raise ValueError(f"Unsupported EOL mistrust target: {self.target}")

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []

        patient_events = patient.get_events(event_type="patients")
        patient_event = patient_events[0] if patient_events else None

        samples: list[dict[str, Any]] = []
        for admission in admissions:
            hadm_id = getattr(admission, "hadm_id", None)
            if hadm_id is None:
                continue

            admit_time = getattr(admission, "admittime", None) or getattr(
                admission, "timestamp", None
            )
            sample: dict[str, Any] = {
                "visit_id": hadm_id,
                "hadm_id": hadm_id,
                "patient_id": patient.patient_id,
                "conditions": self._get_codes_for_admission(
                    patient, "diagnoses_icd", hadm_id
                ),
                "procedures": self._get_codes_for_admission(
                    patient, "procedures_icd", hadm_id
                ),
                "drugs": self._get_codes_for_admission(
                    patient, "prescriptions", hadm_id
                ),
                "age": _calculate_age_years(
                    admit_time,
                    (
                        getattr(patient_event, "dob", None)
                        if patient_event is not None
                        else None
                    ),
                ),
                "los_days": (
                    _calculate_paper_like_los_days(
                        admit_time,
                        getattr(admission, "dischtime", None),
                    )
                    if self.paper_like_dataset_prepare
                    else _calculate_los_days(
                        admit_time,
                        getattr(admission, "dischtime", None),
                    )
                ),
                "gender": (
                    getattr(patient_event, "gender", None)
                    if patient_event is not None
                    else None
                ),
                "insurance": map_insurance_to_group(
                    getattr(admission, "insurance", None)
                ),
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

    def __init__(
        self,
        include_notes: bool = False,
        dataset_prepare_mode: str = DATASET_PREPARE_MODE_DEFAULT,
    ) -> None:
        super().__init__(
            target="left_ama",
            include_notes=include_notes,
            dataset_prepare_mode=dataset_prepare_mode,
        )


class EOLMistrustCodeStatusPredictionMIMIC3(EOLMistrustDownstreamMIMIC3):
    """Task wrapper for the code-status downstream target."""

    task_name = "EOLMistrustCodeStatusPredictionMIMIC3"

    def __init__(
        self,
        include_notes: bool = False,
        dataset_prepare_mode: str = DATASET_PREPARE_MODE_DEFAULT,
    ) -> None:
        super().__init__(
            target="code_status_dnr_dni_cmo",
            include_notes=include_notes,
            dataset_prepare_mode=dataset_prepare_mode,
        )


class EOLMistrustMortalityPredictionMIMIC3(EOLMistrustDownstreamMIMIC3):
    """Task wrapper for the in-hospital mortality downstream target."""

    task_name = "EOLMistrustMortalityPredictionMIMIC3"

    def __init__(
        self,
        include_notes: bool = False,
        dataset_prepare_mode: str = DATASET_PREPARE_MODE_DEFAULT,
    ) -> None:
        super().__init__(
            target="in_hospital_mortality",
            include_notes=include_notes,
            dataset_prepare_mode=dataset_prepare_mode,
        )


__all__ = [
    "CODE_STATUS_ITEMIDS",
    "CODE_STATUS_MODE_CORRECTED",
    "CODE_STATUS_MODE_PAPER_LIKE",
    "DATASET_PREPARE_MODE_DEFAULT",
    "DATASET_PREPARE_MODE_PAPER_LIKE",
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
