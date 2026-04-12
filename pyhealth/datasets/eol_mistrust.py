"""Utilities for reproducing the EOL mistrust preprocessing tables.

Notes
-----
This module owns dataset preparation only:
- cohort construction
- note/chartevent feature and label extraction
- treatment and acuity tables
- final admission-level modeling table assembly
"""

# pylint: disable=too-many-lines

import importlib.util
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd  # pylint: disable=import-error

from pyhealth.tasks.eol_mistrust import (
    CODE_STATUS_MODE_CORRECTED,
    CODE_STATUS_MODE_PAPER_LIKE,
    _advance_paper_like_code_status_label as _task_advance_paper_like_code_status_label,
    _normalize_code_status_mode as _task_normalize_code_status_mode,
    build_code_status_target as _build_task_code_status_target,
    build_in_hospital_mortality_target as _build_task_in_hospital_mortality_target,
    build_left_ama_target as _build_task_left_ama_target,
    is_positive_code_status_value as _task_is_positive_code_status_value,
    map_ethnicity_to_race as _task_map_ethnicity_to_race,
    map_insurance_to_group as _task_map_insurance_to_group,
    prepare_note_text as _task_prepare_note_text,
)


RACE_WHITE = "WHITE"
RACE_BLACK = "BLACK"
RACE_ASIAN = "ASIAN"
RACE_HISPANIC = "HISPANIC"
RACE_NATIVE_AMERICAN = "NATIVE AMERICAN"
RACE_OTHER = "OTHER"

INSURANCE_PUBLIC = "Public"
INSURANCE_PRIVATE = "Private"
INSURANCE_SELF_PAY = "Self-Pay"

RACE_CATEGORIES = [
    RACE_WHITE,
    RACE_BLACK,
    RACE_ASIAN,
    RACE_HISPANIC,
    RACE_NATIVE_AMERICAN,
    RACE_OTHER,
]

INSURANCE_CATEGORIES = [
    INSURANCE_PRIVATE,
    INSURANCE_PUBLIC,
    INSURANCE_SELF_PAY,
]

TABLE2_LABELS = {
    "1_1_sitter",
    "bath",
    "behavioral_interventions",
    "education_barrier",
    "education_learner",
    "education_method",
    "education_readiness",
    "education_topic",
    "family_communication_method",
    "family_meeting",
    "follows_commands",
    "gcs_verbal_response",
    "goal",
    "hair_washed",
    "harm_by_partner",
    "healthcare_proxy",
    "informed",
    "judgment",
    "non_violent_restraints",
    "orientation",
    "pain_assessment_method",
    "pain_level",
    "pain_management",
    "pain_present",
    "reason_for_restraint",
    "restraint_device",
    "restraint_type",
    "restraints_evaluated",
    "richmond_ras_scale",
    "riker_sas_scale",
    "safety_measures",
    "security",
    "side_rails",
    "sitter",
    "skin_care",
    "social_work_consult",
    "spokesperson_healthcare_proxy",
    "spiritual_support",
    "stress",
    "support_systems",
    "understand_agree_with_plan",
    "verbal_response",
    "violent_restraints",
    "wrist_restraints",
}

PAPER_LIKE_RELEVANT_LABELS = (
    "Family Communication",
    "Follows Commands",
    "Education Barrier",
    "Education Learner",
    "Education Method",
    "Education Readiness",
    "Education Topic #1",
    "Education Topic #2",
    "Pain",
    "Pain Level",
    "Pain Level (Rest)",
    "Pain Assess Method",
    "Restraint",
    "Restraint Type",
    "Restraint (Non-violent)",
    "Restraint Ordered (Non-violent)",
    "Restraint Location",
    "Reason For Restraint",
    "Spiritual Support",
    "Support Systems",
    "State",
    "Behavior",
    "Behavioral State",
    "Stress",
    "Safety",
    "Safety Measures_U_1",
    "Family",
    "Patient/Family Informed",
    "Pt./Family Informed",
    "Health Care Proxy",
    "BATH",
    "bath",
    "Bath",
    "Bed Bath",
    "Bedbath",
    "CHG Bath",
    "Skin Care",
    "Judgement",
    "Family Meeting held",
    "Emotional / physical / sexual harm by partner or close relation",
    "Verbal Response",
    "Side Rails",
    "Orientation",
    "RSBI Deferred",
    "Richmond-RAS Scale",
    "Riker-SAS Scale",
    "Status and Comfort",
    "Teaching directed toward",
    "Consults",
    "Social work consult",
    "Sitter",
    "security",
    "safety",
    "headache",
    "hairwashed",
    "observer",
)

CODE_STATUS_ITEMIDS = {128, 223758}

REQUIRED_RAW_TABLE_COLUMNS = {
    "admissions": [
        "hadm_id",
        "subject_id",
        "admittime",
        "dischtime",
        "ethnicity",
        "insurance",
        "discharge_location",
        "hospital_expire_flag",
        "has_chartevents_data",
    ],
    "patients": ["subject_id", "gender", "dob"],
    "icustays": ["hadm_id", "icustay_id", "intime", "outtime"],
    "noteevents": ["hadm_id", "category", "text", "iserror"],
    "chartevents": ["hadm_id", "itemid", "value", "icustay_id"],
    "d_items": ["itemid", "label", "dbsource"],
}

REQUIRED_MATERIALIZED_VIEW_COLUMNS = {
    "ventdurations": [
        "icustay_id",
        "ventnum",
        "starttime",
        "endtime",
        "duration_hours",
    ],
    "vasopressordurations": [
        "icustay_id",
        "vasonum",
        "starttime",
        "endtime",
        "duration_hours",
    ],
    "oasis": ["hadm_id", "icustay_id", "oasis"],
    "sapsii": ["hadm_id", "icustay_id", "sapsii"],
}

REQUIRED_JOIN_KEYS = {
    "subject_id",
    "hadm_id",
    "icustay_id",
    "itemid",
}

NONCOMPLIANCE_PATTERN = re.compile(r"\bnoncompliant\b", re.IGNORECASE)
_AUTOPSY_CONSENT_KEYWORDS = ("consent", "agree", "request")
_AUTOPSY_DECLINE_KEYWORDS = ("decline", "not consent", "refuse", "denied")
_AUTOPSY_CORRECTED_DECLINE_PHRASES = (
    "no autopsy",
    "not perform an autopsy",
    "not perform autopsy",
    "decision to not perform an autopsy",
    "decision made to not perform an autopsy",
    "do not want an autopsy",
    "did not want an autopsy",
    "not want an autopsy",
    "declining autopsy",
    "declining an autopsy",
)
_AUTOPSY_CORRECTED_CONSENT_PHRASES = (
    "autopsy permission was obtained",
    "permission for autopsy",
    "permission obtained for autopsy",
)
_AUTOPSY_SEGMENT_SPLIT_PATTERN = re.compile(r"[\n.;]+")
_AUTOPSY_CORRECTED_DECLINE_PATTERN = re.compile(
    r"(?:\b(?:declin\w*|refus\w*|deni\w*|not\s+consent(?:ed)?)\b(?:\W+\w+){0,5}\W+\bautopsy\b)"
    r"|(?:\bautopsy\b(?:\W+\w+){0,5}\W+\b(?:declin\w*|refus\w*|deni\w*|not\s+consent(?:ed)?)\b)",
    re.IGNORECASE,
)
_AUTOPSY_STUB_SEGMENT_PATTERN = re.compile(r"^(?:an?\s+)?autopsy\b", re.IGNORECASE)
AUTOPSY_LABEL_MODE_CORRECTED = "corrected"
AUTOPSY_LABEL_MODE_PAPER_LIKE = "paper_like"
_EOL_MISTRUST_MODEL_MODULE = None


def _require_columns(df: pd.DataFrame, required: Sequence[str], df_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{df_name} is missing required columns: {missing_str}")


def _load_eol_mistrust_model_module():
    global _EOL_MISTRUST_MODEL_MODULE
    if _EOL_MISTRUST_MODEL_MODULE is None:
        module_path = Path(__file__).resolve().parents[1] / "models" / "eol_mistrust.py"
        spec = importlib.util.spec_from_file_location(
            "pyhealth.models.eol_mistrust_dataset_compat",
            module_path,
        )
        module = importlib.util.module_from_spec(spec)
        if spec is None or spec.loader is None:
            raise ImportError(
                "Unable to load pyhealth.models.eol_mistrust compatibility module."
            )
        spec.loader.exec_module(module)
        _EOL_MISTRUST_MODEL_MODULE = module
    return _EOL_MISTRUST_MODEL_MODULE


def _filter_non_error_notes(noteevents: pd.DataFrame) -> pd.DataFrame:
    """Keep notes where iserror is NULL or not equal to 1."""

    iserror_numeric = pd.to_numeric(noteevents["iserror"], errors="coerce")
    keep_mask = noteevents["iserror"].isna() | iserror_numeric.ne(1)
    return noteevents.loc[keep_mask].copy()


def _normalize_note_categories(categories: Iterable[str] | None) -> set[str] | None:
    if categories is None:
        return None
    normalized = {
        str(category).strip().lower()
        for category in categories
        if str(category).strip()
    }
    return normalized or None


def _filter_note_categories(
    notes: pd.DataFrame,
    categories: Iterable[str] | None = None,
) -> pd.DataFrame:
    normalized_categories = _normalize_note_categories(categories)
    if normalized_categories is None:
        return notes.copy()

    _require_columns(notes, ["category"], "noteevents")
    category_series = notes["category"].fillna("").astype(str).str.strip().str.lower()
    return notes.loc[category_series.isin(normalized_categories)].copy()


def _classify_noncompliance(text: str) -> int:
    return int(bool(NONCOMPLIANCE_PATTERN.search(text)))


def _normalize_autopsy_label_mode(mode: str | None) -> str:
    normalized = (
        AUTOPSY_LABEL_MODE_CORRECTED if mode is None else str(mode).strip().lower()
    )
    if normalized not in {AUTOPSY_LABEL_MODE_CORRECTED, AUTOPSY_LABEL_MODE_PAPER_LIKE}:
        raise ValueError(
            "autopsy_label_mode must be one of "
            f"{AUTOPSY_LABEL_MODE_CORRECTED!r} or {AUTOPSY_LABEL_MODE_PAPER_LIKE!r}"
        )
    return normalized


def _classify_autopsy_lines_paper_like(lines: Iterable[str]) -> float:
    """Line-level autopsy classification matching the reference notebook.

    Each line is checked independently: if a line contains 'autopsy',
    look for consent keywords (consent, agree, request) and decline
    keywords (decline, not consent, refuse, denied) on that same line.

    Returns
    -------
    float
        1.0 = consent (proxy positive / mistrust)
        0.0 = decline (proxy negative / trust)
        NaN  = autopsy not mentioned, or ambiguous (both consent and decline)

    Parameters
    ----------
    lines : Iterable[str]
        Pre-lowered text lines (may come from one note or many).
    """
    consented = False
    declined = False
    for line in lines:
        if "autopsy" not in line:
            continue
        for kw in _AUTOPSY_DECLINE_KEYWORDS:
            if kw in line:
                declined = True
        for kw in _AUTOPSY_CONSENT_KEYWORDS:
            if kw in line:
                consented = True
    if not consented and not declined:
        return float("nan")
    if consented and declined:
        return float("nan")
    if consented:
        return 1.0
    return 0.0


def _classify_autopsy_lines_corrected(lines: Iterable[str]) -> float:
    """Line-level autopsy classification with a few explicit negative phrases.

    This keeps the notebook's overall structure but recognizes common
    negative phrasings such as ``no autopsy`` that the original notebook
    left unlabeled.
    """

    strong_consented = False
    weak_requested = False
    declined = False
    for line in lines:
        segments = [
            segment.strip()
            for segment in _AUTOPSY_SEGMENT_SPLIT_PATTERN.split(line)
            if segment.strip()
        ]
        for idx, segment in enumerate(segments):
            normalized_segment = segment
            if "autopsy" not in normalized_segment:
                continue

            candidate_segments = [normalized_segment]
            is_autopsy_stub = bool(
                _AUTOPSY_STUB_SEGMENT_PATTERN.fullmatch(normalized_segment)
            )
            if is_autopsy_stub and idx > 0 and "autopsy" not in segments[idx - 1]:
                candidate_segments.append(f"{segments[idx - 1]} {normalized_segment}")
            if (
                is_autopsy_stub
                and idx + 1 < len(segments)
                and "autopsy" not in segments[idx + 1]
                and (
                    "request" in segments[idx + 1]
                    or "consent" in segments[idx + 1]
                    or "agree" in segments[idx + 1]
                    or any(
                        phrase in segments[idx + 1]
                        for phrase in _AUTOPSY_CORRECTED_CONSENT_PHRASES
                    )
                )
            ):
                candidate_segments.append(f"{normalized_segment} {segments[idx + 1]}")

            segment_declined = False
            segment_has_request = False
            segment_has_strong_consent = False
            for candidate_segment in candidate_segments:
                segment_declined = (
                    segment_declined
                    or bool(
                        _AUTOPSY_CORRECTED_DECLINE_PATTERN.search(candidate_segment)
                    )
                    or any(
                        phrase in candidate_segment
                        for phrase in _AUTOPSY_CORRECTED_DECLINE_PHRASES
                    )
                )
                segment_has_request = segment_has_request or (
                    "request" in candidate_segment
                )
                segment_has_strong_consent = segment_has_strong_consent or (
                    ("consent" in candidate_segment)
                    or ("agree" in candidate_segment)
                    or any(
                        phrase in candidate_segment
                        for phrase in _AUTOPSY_CORRECTED_CONSENT_PHRASES
                    )
                )

            if segment_declined:
                declined = True
            # Treat explicit negative phrasing as stronger than generic request
            # wording. This keeps "request for autopsy was declined" negative
            # while still allowing clear consent/agreement phrases to remain
            # genuinely ambiguous if both positive and negative evidence appear.
            if segment_has_strong_consent and not segment_declined:
                strong_consented = True
            elif segment_has_request and not segment_declined:
                weak_requested = True

    if strong_consented and declined:
        return float("nan")
    if strong_consented:
        return 1.0
    if declined:
        return 0.0
    if weak_requested:
        return 1.0
    return float("nan")


def _classify_autopsy_lines(
    lines: Iterable[str],
    *,
    mode: str = AUTOPSY_LABEL_MODE_CORRECTED,
) -> float:
    normalized_mode = _normalize_autopsy_label_mode(mode)
    if normalized_mode == AUTOPSY_LABEL_MODE_PAPER_LIKE:
        return _classify_autopsy_lines_paper_like(lines)
    return _classify_autopsy_lines_corrected(lines)


def _to_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_localize(None)


def _normalize_hadm_ids(all_hadm_ids: Iterable[int] | None) -> list[int] | None:
    """Normalize an optional hadm_id iterable to a sorted unique integer list."""

    if all_hadm_ids is None:
        return None
    hadm_series = pd.Series(list(all_hadm_ids))
    hadm_numeric = pd.to_numeric(hadm_series, errors="coerce").dropna().astype(int)
    return sorted(set(hadm_numeric.tolist()))


def _read_csv_columns(
    csv_path: Path | str,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    """Read only the requested CSV columns and normalize headers to lowercase."""

    required = {column.lower() for column in required_columns}
    df = pd.read_csv(
        csv_path,
        usecols=lambda column: str(column).lower() in required,
        low_memory=False,
    )
    df.columns = [str(column).lower() for column in df.columns]
    return df


def _iter_csv_chunks(
    csv_path: Path | str,
    required_columns: Sequence[str],
    chunksize: int,
):
    """Yield CSV chunks with lowercase column names and only required columns."""

    required = {column.lower() for column in required_columns}
    reader = pd.read_csv(
        csv_path,
        usecols=lambda column: str(column).lower() in required,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        chunk.columns = [str(column).lower() for column in chunk.columns]
        yield chunk


def _calculate_age_years(admittime: pd.Series, dob: pd.Series) -> pd.Series:
    admittime = _to_datetime(admittime)
    dob = _to_datetime(dob)
    seconds_per_year = 365.25 * 24 * 3600

    ages: list[float] = []
    for admit, birth in zip(admittime, dob):
        if pd.isna(admit) or pd.isna(birth):
            ages.append(float("nan"))
            continue
        age = (
            admit.to_pydatetime() - birth.to_pydatetime()
        ).total_seconds() / seconds_per_year
        ages.append(90.0 if age > 200 else float(age))
    return pd.Series(ages, index=admittime.index, dtype=float)


def _normalize_token(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    value = str(value).strip().lower()
    if not value:
        return ""
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def _clean_feature_text(value) -> str:
    """Normalize display text for feature labels and values without lowercasing."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    cleaned = re.sub(r"\s+", " ", str(value).strip())
    cleaned = re.sub(r"^[^A-Za-z0-9]+", "", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9]+$", "", cleaned)
    return cleaned.strip()


def _normalize_label_match_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    cleaned = str(value).strip().lower().replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _normalize_paper_like_value(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "none"
    cleaned = re.sub(r"\s+", " ", str(value).strip().lower())
    return cleaned if cleaned else "none"


def _feature_text_display_score(value: str) -> tuple:
    cleaned = _clean_feature_text(value)
    if cleaned == "":
        return (-1, -1, -1, -1, -1, -1, "")

    has_alpha = any(char.isalpha() for char in cleaned)
    is_all_upper = has_alpha and cleaned.upper() == cleaned
    is_all_lower = has_alpha and cleaned.lower() == cleaned
    is_title_like = has_alpha and cleaned == cleaned.title()
    alpha_count = sum(char.isalpha() for char in cleaned)
    digit_count = sum(char.isdigit() for char in cleaned)
    punctuation_count = sum(
        (not char.isalnum()) and (not char.isspace()) for char in cleaned
    )
    return (
        int(is_title_like),
        int(not is_all_upper),
        int(not is_all_lower),
        alpha_count,
        -digit_count,
        -punctuation_count,
        -len(cleaned),
        cleaned.lower(),
    )


def _choose_preferred_feature_text(values: Iterable[str]) -> str:
    best = ""
    best_score = _feature_text_display_score("")
    for value in values:
        cleaned = _clean_feature_text(value)
        if cleaned == "":
            continue
        score = _feature_text_display_score(cleaned)
        if score > best_score:
            best = cleaned
            best_score = score
    return best


def _build_feature_label_metadata(
    items: pd.DataFrame,
) -> tuple[dict[int, str], dict[str, str]]:
    working = items.copy()
    working["normalized_label"] = working["label"].map(_normalize_token)
    working["display_label"] = working["label"].map(_clean_feature_text)

    label_display_lookup = (
        working.loc[
            working["display_label"] != "", ["normalized_label", "display_label"]
        ]
        .drop_duplicates()
        .groupby("normalized_label", sort=True)["display_label"]
        .agg(lambda series: _choose_preferred_feature_text(series.tolist()))
        .to_dict()
    )
    item_label_lookup = (
        working[["itemid", "normalized_label"]]
        .drop_duplicates("itemid")
        .set_index("itemid")["normalized_label"]
        .to_dict()
    )
    return item_label_lookup, label_display_lookup


_FEATURE_VALUE_MEASUREMENT_SUFFIXES = (
    "ppm",
    "mmhg",
    "kg",
    "kgs",
    "lb",
    "lbs",
    "cm",
    "mm",
    "ml",
    "cc",
    "mcg",
    "mg",
    "meq",
)


def _is_numeric_heavy_or_freeform_feature_value(
    normalized_value: str,
    display_value: str,
) -> bool:
    cleaned = _clean_feature_text(display_value)
    if cleaned == "":
        return True
    if len(cleaned) > 64 or len(cleaned.split()) > 10:
        return True

    normalized = str(normalized_value).strip("_")
    alpha_tokens = re.findall(r"[a-z]+", normalized)
    if not alpha_tokens:
        return True

    measurement_stripped = normalized
    for suffix in _FEATURE_VALUE_MEASUREMENT_SUFFIXES:
        if measurement_stripped.endswith(suffix):
            measurement_stripped = measurement_stripped[: -len(suffix)]
            break
    measurement_stripped = measurement_stripped.strip("_")
    if measurement_stripped and re.fullmatch(
        r"[-+]?\d+(?:_\d+)*", measurement_stripped
    ):
        return True

    digit_count = sum(char.isdigit() for char in cleaned)
    alpha_length = sum(len(token) for token in alpha_tokens)
    if digit_count >= 2 and alpha_length <= 3:
        return True
    return False


def _feature_display_name(
    normalized_label: str,
    normalized_value: str,
    label_display_lookup: Mapping[str, str],
    value_display_lookup: Mapping[tuple[str, str], str],
) -> str:
    display_label = label_display_lookup.get(
        normalized_label, _clean_feature_text(normalized_label)
    )
    display_value = value_display_lookup.get(
        (normalized_label, normalized_value),
        _clean_feature_text(normalized_value),
    )
    return f"{display_label}: {display_value}"


def _paper_like_feature_display_name(label: str, value: str) -> str:
    return f"{label}: {value}"


def _matches_paper_like_label(
    label: str,
    allowed_labels: Iterable[str] | None = None,
) -> bool:
    normalized_label = _normalize_label_match_text(label)
    if normalized_label == "":
        return False
    patterns = (
        [_normalize_label_match_text(item) for item in allowed_labels]
        if allowed_labels is not None
        else [_normalize_label_match_text(item) for item in PAPER_LIKE_RELEVANT_LABELS]
    )
    patterns = [pattern for pattern in patterns if pattern]
    return any(pattern in normalized_label for pattern in patterns)


def _paper_like_feature_pair(label: str, value) -> tuple[str, str] | None:
    normalized_label = _normalize_label_match_text(label)
    if normalized_label == "":
        return None
    normalized_value = _normalize_paper_like_value(value)

    if "reason for restraint" in normalized_label:
        if normalized_value in {"not applicable", "none"}:
            normalized_value = "none"
        elif ("threat" in normalized_value) or ("acute risk of" in normalized_value):
            normalized_value = "threat of harm"
        elif (
            ("confusion" in normalized_value)
            or ("delirium" in normalized_value)
            or (normalized_value == "impaired judgment")
            or (normalized_value == "sundowning")
        ):
            normalized_value = "confusion/delirium"
        elif (
            ("occurence" in normalized_value)
            or (normalized_value == "severe physical agitation")
            or (normalized_value == "violent/self des")
        ):
            normalized_value = "prescence of violence"
        elif normalized_value in {
            "ext/txinterfere",
            "protection of lines and tubes",
            "treatment interference",
        }:
            normalized_value = "treatment interference"
        elif "risk for fall" in normalized_value:
            normalized_value = "risk for falls"
        return ("reason for restraint", normalized_value)

    if "restraint location" in normalized_label:
        if normalized_value == "none":
            normalized_value = "none"
        elif "4 point rest" in normalized_value:
            normalized_value = "4 point restraint"
        else:
            normalized_value = "some restraint"
        return ("restraint location", normalized_value)

    if "restraint device" in normalized_label:
        if "sitter" in normalized_value:
            normalized_value = "sitter"
        elif "limb" in normalized_value:
            normalized_value = "limb"
        return ("restraint device", normalized_value)

    if "bath" in normalized_label:
        if "part" in normalized_label:
            normalized_value = "partial"
        elif "self" in normalized_value:
            normalized_value = "self"
        elif "refused" in normalized_value:
            normalized_value = "refused"
        elif "shave" in normalized_value:
            normalized_value = "shave"
        elif "hair" in normalized_value:
            normalized_value = "hair"
        elif "none" in normalized_value:
            normalized_value = "none"
        else:
            normalized_value = "done"
        return ("bath", normalized_value)

    if normalized_label in {"behavior", "behavioral state"}:
        return None

    if normalized_label.startswith("pain level"):
        return ("pain level", normalized_value)

    if normalized_label.startswith(
        ("pain management", "pain type", "pain cause", "pain location")
    ):
        return None

    if normalized_label.startswith("education topic"):
        return ("education topic", normalized_value)

    if normalized_label.startswith("safety measures"):
        return ("safety measures", normalized_value)

    if normalized_label.startswith("side rails"):
        return ("side rails", normalized_value)

    if normalized_label.startswith("status and comfort"):
        return ("status and comfort", normalized_value)

    if "informed" in normalized_label:
        return ("informed", normalized_value)

    return (normalized_label, normalized_value)


def _filter_chartevent_items(
    d_items: pd.DataFrame,
    allowed_labels: Iterable[str] | None = None,
    *,
    paper_like: bool = False,
) -> pd.DataFrame:
    _require_columns(d_items, ["itemid", "label", "dbsource"], "d_items")
    items = d_items.copy()
    items["normalized_label"] = items["label"].map(_normalize_token)
    if paper_like:
        mask = items["label"].map(
            lambda label: _matches_paper_like_label(
                label, allowed_labels=allowed_labels
            )
        )
        items = items.loc[mask].copy()
    elif allowed_labels is not None:
        allowed = {_normalize_token(label) for label in allowed_labels}
        items = items.loc[items["normalized_label"].isin(allowed)].copy()
    else:
        allowed_itemids = identify_table2_itemids(items)
        items = items.loc[items["itemid"].isin(allowed_itemids)].copy()

    items["itemid"] = pd.to_numeric(items["itemid"], errors="coerce")
    items = items.dropna(subset=["itemid"]).copy()
    items["itemid"] = items["itemid"].astype(int)
    return items


def _paper_like_feature_sets_from_rows(rows: pd.DataFrame) -> dict[str, set[int]]:
    feature_to_hadm: dict[str, set[int]] = defaultdict(set)
    if rows.empty:
        return feature_to_hadm
    unique_rows = rows[["hadm_id", "label", "value"]].drop_duplicates()
    for row in unique_rows.itertuples(index=False):
        feature_pair = _paper_like_feature_pair(
            str(getattr(row, "label")), getattr(row, "value")
        )
        if feature_pair is None:
            continue
        feature_name = _paper_like_feature_display_name(*feature_pair)
        feature_to_hadm[feature_name].add(int(getattr(row, "hadm_id")))
    return feature_to_hadm


def _binary_feature_matrix_from_feature_sets(
    feature_to_hadm: Mapping[str, set[int]],
    hadm_ids: Sequence[int],
) -> pd.DataFrame:
    feature_data: dict[str, object] = {"hadm_id": list(hadm_ids)}
    hadm_index = pd.Index(hadm_ids)
    for feature_name in sorted(feature_to_hadm, key=str.lower):
        feature_data[feature_name] = hadm_index.isin(
            feature_to_hadm[feature_name]
        ).astype(int)
    result = pd.DataFrame(feature_data)
    if "hadm_id" not in result.columns:
        result = pd.DataFrame(columns=["hadm_id"])
    feature_cols = [col for col in result.columns if col != "hadm_id"]
    if feature_cols:
        result[feature_cols] = result[feature_cols].fillna(0).astype(int)
    result = result.sort_values("hadm_id").drop_duplicates("hadm_id")
    return result.reset_index(drop=True)


def _matches_table2_concept(label: str) -> bool:
    """Return True when a d_items label matches a Table 2 concept by partial match."""

    normalized_label = _normalize_token(label)
    if normalized_label == "":
        return False
    return any(concept in normalized_label for concept in TABLE2_LABELS)


def _collect_required_join_keys(raw_tables: Mapping[str, pd.DataFrame]) -> set[str]:
    """Collect all join keys exposed by the core raw tables."""

    required_tables = ["admissions", "patients", "icustays", "chartevents", "d_items"]
    return set().union(*(set(raw_tables[name].columns) for name in required_tables))


def _validate_database_identity(
    schema_name: str | None,
    database_flavor: str | None,
) -> tuple[str, str]:
    """Validate and normalize the declared database flavor and schema."""

    resolved_schema = "mimiciii" if schema_name is None else str(schema_name).lower()
    resolved_flavor = (
        "postgresql" if database_flavor is None else str(database_flavor).lower()
    )
    if resolved_schema != "mimiciii":
        raise ValueError("Database schema must be mimiciii.")
    if resolved_flavor not in {"postgresql", "postgres"}:
        raise ValueError("Database flavor must be PostgreSQL.")
    return resolved_schema, resolved_flavor


def _validate_required_inputs(
    raw_tables: Mapping[str, pd.DataFrame],
    materialized_views: Mapping[str, pd.DataFrame],
) -> None:
    """Ensure all required raw tables, views, and columns are present."""

    missing_raw = sorted(set(REQUIRED_RAW_TABLE_COLUMNS) - set(raw_tables))
    if missing_raw:
        raise ValueError("Missing required raw tables: " + ", ".join(missing_raw))

    missing_views = sorted(
        set(REQUIRED_MATERIALIZED_VIEW_COLUMNS) - set(materialized_views)
    )
    if missing_views:
        raise ValueError(
            "Missing required materialized views: " + ", ".join(missing_views)
        )

    for table_name, required_columns in REQUIRED_RAW_TABLE_COLUMNS.items():
        _require_columns(raw_tables[table_name], required_columns, table_name)
    for view_name, required_columns in REQUIRED_MATERIALIZED_VIEW_COLUMNS.items():
        _require_columns(materialized_views[view_name], required_columns, view_name)


def _validate_text_access(noteevents: pd.DataFrame, chartevents: pd.DataFrame) -> None:
    """Ensure the text fields required for NLP and string matching are present."""

    if noteevents["text"].isna().all():
        raise ValueError("noteevents.text must be accessible for NLP steps.")
    if chartevents["value"].isna().all():
        raise ValueError(
            "chartevents.value must be accessible for string matching and feature extraction."
        )


def _validate_bridge_join(
    source_df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    join_column: str,
    error_message: str,
) -> None:
    """Ensure a required bridge join yields at least one row when source rows exist."""

    merged = source_df.merge(bridge_df, on=join_column, how="inner")
    if source_df.shape[0] > 0 and merged.empty:
        raise ValueError(error_message)


def map_ethnicity(ethnicity) -> str:
    """Dataset-facing alias of the task-owned race mapping helper."""

    return _task_map_ethnicity_to_race(ethnicity)


def map_insurance(insurance) -> str:
    """Dataset-facing alias of the task-owned insurance mapping helper."""

    return _task_map_insurance_to_group(insurance)


def prepare_note_text_for_sentiment(text) -> str:
    """Dataset-facing alias of the task-owned note normalization helper."""

    return _task_prepare_note_text(text)


def build_base_admissions(
    admissions: pd.DataFrame, patients: pd.DataFrame
) -> pd.DataFrame:
    """Join admissions to patients and keep only rows with chart events available."""

    _require_columns(
        admissions,
        [
            "hadm_id",
            "subject_id",
            "admittime",
            "dischtime",
            "ethnicity",
            "insurance",
            "discharge_location",
            "hospital_expire_flag",
            "has_chartevents_data",
        ],
        "admissions",
    )
    _require_columns(patients, ["subject_id", "gender", "dob"], "patients")

    admissions_df = admissions.copy()
    patients_df = patients.copy()
    admissions_df["admittime"] = _to_datetime(admissions_df["admittime"])
    admissions_df["dischtime"] = _to_datetime(admissions_df["dischtime"])
    patients_df["dob"] = _to_datetime(patients_df["dob"])

    merged = admissions_df.merge(
        patients_df[["subject_id", "gender", "dob"]],
        on="subject_id",
        how="left",
        validate="many_to_one",
    )
    merged = merged.loc[merged["has_chartevents_data"] == 1].copy()
    merged = merged.sort_values("hadm_id").drop_duplicates("hadm_id")
    return merged.reset_index(drop=True)


def build_demographics_table(
    base_admissions: pd.DataFrame,
    *,
    paper_like: bool = False,
) -> pd.DataFrame:
    """Derive race, age, LOS, and insurance-group fields for each admission.

    When ``paper_like=True``, ``los_days`` mirrors the reference notebook's
    modulo-24-hour representation (``timedelta.seconds / 3600``), while
    ``los_hours`` remains the true total LOS in hours so cohort filters keep
    using the cleaned duration semantics.
    """

    _require_columns(
        base_admissions,
        [
            "hadm_id",
            "subject_id",
            "admittime",
            "dischtime",
            "ethnicity",
            "insurance",
            "gender",
            "dob",
        ],
        "base_admissions",
    )

    df = base_admissions.copy()
    df["admittime"] = _to_datetime(df["admittime"])
    df["dischtime"] = _to_datetime(df["dischtime"])
    df["dob"] = _to_datetime(df["dob"])

    age_years = _calculate_age_years(df["admittime"], df["dob"])
    los_hours = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 3600.0
    if paper_like:
        los_days = (df["dischtime"] - df["admittime"]).dt.seconds / 3600.0
    else:
        los_days = los_hours / 24.0
    insurance_group = df["insurance"].map(map_insurance)

    demographics = pd.DataFrame(
        {
            "hadm_id": df["hadm_id"],
            "subject_id": df["subject_id"],
            "gender": df["gender"],
            "admittime": df["admittime"],
            "dischtime": df["dischtime"],
            "ethnicity": df["ethnicity"],
            "insurance_raw": df["insurance"],
            "race": df["ethnicity"].map(map_ethnicity),
            "age": age_years.astype(float),
            "los_hours": los_hours.astype(float),
            "los_days": los_days.astype(float),
            "insurance": insurance_group,
            "insurance_group": insurance_group,
        }
    )
    demographics = demographics.sort_values("hadm_id").drop_duplicates("hadm_id")
    return demographics.reset_index(drop=True)


def build_eol_cohort(
    base_admissions: pd.DataFrame, demographics: pd.DataFrame
) -> pd.DataFrame:
    """Build the end-of-life cohort used for treatment-disparity analysis."""

    _require_columns(
        base_admissions,
        ["hadm_id", "discharge_location", "hospital_expire_flag"],
        "base_admissions",
    )
    _require_columns(demographics, ["hadm_id", "los_hours"], "demographics")

    df = demographics.merge(
        base_admissions[["hadm_id", "discharge_location", "hospital_expire_flag"]],
        on="hadm_id",
        how="inner",
        validate="one_to_one",
    )
    discharge_location = df["discharge_location"].fillna("").str.upper()
    is_deceased = df["hospital_expire_flag"].fillna(0).astype(int) == 1
    is_hospice = discharge_location.str.contains("HOSPICE", na=False)
    is_snf = discharge_location.str.contains(
        r"SKILLED NURSING|\bSNF\b", na=False, regex=True
    )

    include = (df["los_hours"] >= 6) & (is_deceased | is_hospice | is_snf)
    df = df.loc[include].copy()
    df["discharge_category"] = "Skilled Nursing Facility"
    df.loc[is_hospice.loc[df.index], "discharge_category"] = "Hospice"
    df.loc[is_deceased.loc[df.index], "discharge_category"] = "Deceased"
    df = df.sort_values("hadm_id").drop_duplicates("hadm_id")
    return df.reset_index(drop=True)


def build_all_cohort(
    base_admissions: pd.DataFrame, icustays: pd.DataFrame
) -> pd.DataFrame:
    """Build the adult admission-level cohort with at least 12 cumulative ICU hours."""

    _require_columns(
        base_admissions, ["hadm_id", "admittime", "dob"], "base_admissions"
    )
    _require_columns(
        icustays, ["hadm_id", "icustay_id", "intime", "outtime"], "icustays"
    )

    base = base_admissions.copy()
    base["admittime"] = _to_datetime(base["admittime"])
    base["dob"] = _to_datetime(base["dob"])
    adult_hadm_ids = set(
        base.loc[_calculate_age_years(base["admittime"], base["dob"]) >= 18, "hadm_id"]
        .dropna()
        .tolist()
    )

    icu = icustays.copy()
    icu["intime"] = _to_datetime(icu["intime"])
    icu["outtime"] = _to_datetime(icu["outtime"])
    icu["icu_hours"] = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600.0
    icu["hadm_id"] = pd.to_numeric(icu["hadm_id"], errors="coerce")
    qualifying = set(
        icu.loc[icu["icu_hours"].ge(0)]
        .dropna(subset=["hadm_id"])
        .groupby("hadm_id", sort=True)["icu_hours"]
        .sum()
        .loc[lambda totals: totals >= 12]
        .index.astype(int)
        .tolist()
    )
    df = base.loc[base["hadm_id"].isin(adult_hadm_ids & qualifying)].copy()
    df = df.sort_values("hadm_id").drop_duplicates("hadm_id")
    return df.reset_index(drop=True)


def _merge_spans_for_hadm(spans: pd.DataFrame) -> float:
    if spans.empty:
        return 0.0

    spans = spans.sort_values("starttime")
    merged = []
    current_start = None
    current_end = None

    for row in spans.itertuples(index=False):
        start = row.starttime
        end = row.endtime
        if pd.isna(start) or pd.isna(end):
            continue
        if current_start is None:
            current_start = start
            current_end = end
            continue

        gap_minutes = (start - current_end).total_seconds() / 60.0
        if gap_minutes <= 600:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start = start
            current_end = end

    if current_start is not None:
        merged.append((current_start, current_end))

    total_minutes = 0.0
    for start, end in merged:
        total_minutes += (end - start).total_seconds() / 60.0
    return total_minutes


def _duration_totals_by_hadm(
    durations: pd.DataFrame,
    icustays: pd.DataFrame,
    number_col: str,
    output_col: str,
) -> pd.DataFrame:
    _require_columns(
        durations,
        ["icustay_id", number_col, "starttime", "endtime", "duration_hours"],
        output_col,
    )
    if durations.empty:
        return pd.DataFrame(columns=["hadm_id", output_col])

    bridge_columns = ["icustay_id", "hadm_id", "intime", "outtime"]
    bridge = icustays[bridge_columns].drop_duplicates()
    df = durations.copy()
    if "hadm_id" in df.columns:
        df = df.drop(columns=["hadm_id"])
    df["starttime"] = _to_datetime(df["starttime"])
    df["endtime"] = _to_datetime(df["endtime"])
    df = df.merge(bridge, on="icustay_id", how="inner", validate="many_to_one")
    df["intime"] = _to_datetime(df["intime"])
    df["outtime"] = _to_datetime(df["outtime"])
    df = df.loc[
        df["starttime"].notna()
        & df["endtime"].notna()
        & df["intime"].notna()
        & df["outtime"].notna()
        & df["starttime"].ge(df["intime"])
        & df["endtime"].le(df["outtime"])
    ].copy()
    if df.empty:
        return pd.DataFrame(columns=["hadm_id", output_col])

    totals = (
        df.groupby("hadm_id", sort=True)
        .apply(_merge_spans_for_hadm, include_groups=False)
        .rename(output_col)
        .reset_index()
    )
    return totals


def build_treatment_totals(
    icustays: pd.DataFrame,
    ventdurations: pd.DataFrame,
    vasopressordurations: pd.DataFrame,
    paper_like: bool = False,
) -> pd.DataFrame:
    """Compute admission-level ventilation and vasopressor totals in minutes.

    ``paper_like`` is retained for API compatibility, but ICU-window filtering
    now applies to both paths.
    """

    _require_columns(
        icustays, ["hadm_id", "icustay_id", "intime", "outtime"], "icustays"
    )
    del paper_like

    vent_totals = _duration_totals_by_hadm(
        ventdurations,
        icustays,
        number_col="ventnum",
        output_col="total_vent_min",
    )
    vaso_totals = _duration_totals_by_hadm(
        vasopressordurations,
        icustays,
        number_col="vasonum",
        output_col="total_vaso_min",
    )

    if vent_totals.empty and vaso_totals.empty:
        return pd.DataFrame(columns=["hadm_id", "total_vent_min", "total_vaso_min"])

    totals = pd.merge(vent_totals, vaso_totals, on="hadm_id", how="outer")
    totals = totals.sort_values("hadm_id").drop_duplicates("hadm_id")
    return totals.reset_index(drop=True)


def build_note_corpus(
    noteevents: pd.DataFrame,
    all_hadm_ids: Iterable[int] | None = None,
    categories: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Aggregate non-error notes into one concatenated note per admission."""

    _require_columns(noteevents, ["hadm_id", "text", "iserror"], "noteevents")

    notes = noteevents.copy()
    notes = _filter_non_error_notes(notes)
    notes = _filter_note_categories(notes, categories=categories)
    notes["text"] = notes["text"].map(prepare_note_text_for_sentiment)

    grouped = (
        notes.groupby("hadm_id", sort=True)["text"]
        .apply(
            lambda series: prepare_note_text_for_sentiment(
                " ".join(t for t in series if t)
            )
        )
        .reset_index(name="note_text")
    )

    if all_hadm_ids is not None:
        hadm_frame = pd.DataFrame({"hadm_id": list(all_hadm_ids)})
        grouped = hadm_frame.merge(grouped, on="hadm_id", how="left")

    grouped["note_text"] = grouped["note_text"].fillna("")
    grouped = grouped.sort_values("hadm_id").drop_duplicates("hadm_id")
    return grouped.reset_index(drop=True)


def _build_note_labels_from_corpus(note_corpus: pd.DataFrame) -> pd.DataFrame:
    """Create the two note-derived labels from an admission-level note corpus.

    Noncompliance uses the concatenated corpus text.  Autopsy labels are
    set to NaN here; the caller (``build_note_labels``) overwrites them with
    line-level results computed from raw noteevents before concatenation.
    NaN means "autopsy not mentioned" and those rows are excluded from
    proxy model training (but still scored).
    """

    _require_columns(note_corpus, ["hadm_id", "note_text"], "note_corpus")
    lowered = note_corpus["note_text"].fillna("").astype(str).str.lower()
    noncompliance = lowered.apply(_classify_noncompliance)

    labels = pd.DataFrame(
        {
            "hadm_id": note_corpus["hadm_id"],
            "noncompliance_label": noncompliance.astype(int),
            "autopsy_label": float("nan"),
        }
    )
    labels = labels.sort_values("hadm_id").drop_duplicates("hadm_id")
    return labels.reset_index(drop=True)


def _build_autopsy_labels_from_raw_notes(
    noteevents: pd.DataFrame,
    *,
    autopsy_label_mode: str = AUTOPSY_LABEL_MODE_CORRECTED,
) -> dict[int, int]:
    """Compute admission-level autopsy labels from raw (pre-concatenation) notes.

    Mirrors the reference notebook: iterate each note's lines individually so
    that consent/decline keywords are only matched on lines containing 'autopsy'.
    """
    admission_lines: dict[int, list[str]] = defaultdict(list)
    for hadm_id, text in zip(noteevents["hadm_id"], noteevents["text"]):
        raw = (
            str(text)
            if text is not None and not (isinstance(text, float) and pd.isna(text))
            else ""
        )
        for line in raw.lower().split("\n"):
            admission_lines[int(hadm_id)].append(line)

    return {
        hadm_id: _classify_autopsy_lines(lines, mode=autopsy_label_mode)
        for hadm_id, lines in admission_lines.items()
    }


def _build_note_labels_for_mode(
    noteevents: pd.DataFrame,
    *,
    all_hadm_ids: Iterable[int] | None,
    categories: Iterable[str] | None,
    autopsy_label_mode: str,
) -> pd.DataFrame:
    _require_columns(noteevents, ["hadm_id", "text", "iserror"], "noteevents")

    filtered = _filter_non_error_notes(noteevents)
    filtered = _filter_note_categories(filtered, categories=categories)
    corpus = build_note_corpus(
        noteevents,
        all_hadm_ids=all_hadm_ids,
        categories=categories,
    )
    labels = _build_note_labels_from_corpus(corpus)

    autopsy_map = _build_autopsy_labels_from_raw_notes(
        filtered,
        autopsy_label_mode=autopsy_label_mode,
    )
    labels["autopsy_label"] = labels["hadm_id"].map(autopsy_map)
    return labels


def build_note_labels(
    noteevents: pd.DataFrame,
    all_hadm_ids: Iterable[int] | None = None,
    categories: Iterable[str] | None = None,
    autopsy_label_mode: str = AUTOPSY_LABEL_MODE_CORRECTED,
) -> pd.DataFrame:
    """Create admission-level noncompliance and autopsy labels from notes.

    Normal Path
        corrected autopsy labeling
    Paper-like Path
        notebook-faithful autopsy labeling
    """

    normalized_mode = _normalize_autopsy_label_mode(autopsy_label_mode)
    return _build_note_labels_for_mode(
        noteevents,
        all_hadm_ids=all_hadm_ids,
        categories=categories,
        autopsy_label_mode=normalized_mode,
    )


def _resolve_note_artifact_category_filters(
    *,
    categories: Iterable[str] | None,
    corpus_categories: Iterable[str] | None,
    label_categories: Iterable[str] | None,
) -> tuple[set[str] | None, set[str] | None]:
    if corpus_categories is None:
        corpus_categories = categories
    if label_categories is None:
        label_categories = categories
    return (
        _normalize_note_categories(corpus_categories),
        _normalize_note_categories(label_categories),
    )


def _build_note_artifacts_from_csv_for_mode(
    noteevents_csv_path: Path | str,
    *,
    all_hadm_ids: Iterable[int] | None,
    categories: Iterable[str] | None,
    corpus_categories: Iterable[str] | None,
    label_categories: Iterable[str] | None,
    autopsy_label_mode: str,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    normalized_hadm_ids = _normalize_hadm_ids(all_hadm_ids)
    hadm_filter = set(normalized_hadm_ids) if normalized_hadm_ids is not None else None
    normalized_corpus_categories, normalized_label_categories = (
        _resolve_note_artifact_category_filters(
            categories=categories,
            corpus_categories=corpus_categories,
            label_categories=label_categories,
        )
    )

    corpus_fragments: dict[int, list[str]] = defaultdict(list)
    label_fragments: dict[int, list[str]] = defaultdict(list)
    autopsy_lines: dict[int, list[str]] = defaultdict(list)
    required_columns = ["hadm_id", "text", "iserror"]
    if (
        normalized_corpus_categories is not None
        or normalized_label_categories is not None
    ):
        required_columns.append("category")

    for chunk in _iter_csv_chunks(
        noteevents_csv_path,
        required_columns=required_columns,
        chunksize=chunksize,
    ):
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce")
        chunk = chunk.dropna(subset=["hadm_id"]).copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)

        if hadm_filter is not None:
            chunk = chunk.loc[chunk["hadm_id"].isin(hadm_filter)]
        if chunk.empty:
            continue

        chunk = _filter_non_error_notes(chunk)
        if chunk.empty:
            continue

        for hadm_id, raw_text in zip(chunk["hadm_id"], chunk["text"]):
            raw = (
                str(raw_text)
                if raw_text is not None
                and not (isinstance(raw_text, float) and pd.isna(raw_text))
                else ""
            )
            for line in raw.lower().split("\n"):
                stripped = line.strip()
                if stripped:
                    autopsy_lines[int(hadm_id)].append(stripped)

        chunk["text"] = chunk["text"].map(prepare_note_text_for_sentiment)
        chunk = chunk.loc[chunk["text"] != ""]
        if chunk.empty:
            continue

        corpus_chunk = _filter_note_categories(
            chunk, categories=normalized_corpus_categories
        )
        if not corpus_chunk.empty:
            grouped = corpus_chunk.groupby("hadm_id", sort=False)["text"].apply(
                lambda series: prepare_note_text_for_sentiment(" ".join(series))
            )
            for hadm_id, text in grouped.items():
                if text:
                    corpus_fragments[int(hadm_id)].append(text)

        label_chunk = _filter_note_categories(
            chunk, categories=normalized_label_categories
        )
        if label_chunk.empty:
            continue
        grouped = label_chunk.groupby("hadm_id", sort=False)["text"].apply(
            lambda series: prepare_note_text_for_sentiment(" ".join(series))
        )
        for hadm_id, text in grouped.items():
            if text:
                label_fragments[int(hadm_id)].append(text)

    if normalized_hadm_ids is not None:
        hadm_ids = normalized_hadm_ids
    else:
        hadm_ids = sorted(set(corpus_fragments) | set(label_fragments))

    corpus = pd.DataFrame(
        {
            "hadm_id": hadm_ids,
            "note_text": [
                prepare_note_text_for_sentiment(
                    " ".join(corpus_fragments.get(hadm_id, []))
                )
                for hadm_id in hadm_ids
            ],
        }
    )
    corpus = (
        corpus.sort_values("hadm_id").drop_duplicates("hadm_id").reset_index(drop=True)
    )
    label_corpus = pd.DataFrame(
        {
            "hadm_id": hadm_ids,
            "note_text": [
                prepare_note_text_for_sentiment(
                    " ".join(label_fragments.get(hadm_id, []))
                )
                for hadm_id in hadm_ids
            ],
        }
    )
    label_corpus = (
        label_corpus.sort_values("hadm_id")
        .drop_duplicates("hadm_id")
        .reset_index(drop=True)
    )
    labels = _build_note_labels_from_corpus(label_corpus)

    autopsy_map = {
        hadm_id: _classify_autopsy_lines(lines, mode=autopsy_label_mode)
        for hadm_id, lines in autopsy_lines.items()
    }
    labels["autopsy_label"] = labels["hadm_id"].map(autopsy_map)

    return corpus, labels


def build_note_artifacts_from_csv(
    noteevents_csv_path: Path | str,
    all_hadm_ids: Iterable[int] | None = None,
    categories: Iterable[str] | None = None,
    corpus_categories: Iterable[str] | None = None,
    label_categories: Iterable[str] | None = None,
    autopsy_label_mode: str = AUTOPSY_LABEL_MODE_CORRECTED,
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the note corpus and note-derived labels from a large CSV in chunks.

    Normal Path
        corrected autopsy labeling
    Paper-like Path
        notebook-faithful autopsy labeling
    """

    normalized_mode = _normalize_autopsy_label_mode(autopsy_label_mode)
    return _build_note_artifacts_from_csv_for_mode(
        noteevents_csv_path,
        all_hadm_ids=all_hadm_ids,
        categories=categories,
        corpus_categories=corpus_categories,
        label_categories=label_categories,
        autopsy_label_mode=normalized_mode,
        chunksize=chunksize,
    )


def build_note_corpus_from_csv(
    noteevents_csv_path: Path | str,
    all_hadm_ids: Iterable[int] | None = None,
    categories: Iterable[str] | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Build the admission-level note corpus from a large CSV in chunks."""

    corpus, _ = build_note_artifacts_from_csv(
        noteevents_csv_path=noteevents_csv_path,
        all_hadm_ids=all_hadm_ids,
        corpus_categories=categories,
        chunksize=chunksize,
    )
    return corpus


def build_note_labels_from_csv(
    noteevents_csv_path: Path | str,
    all_hadm_ids: Iterable[int] | None = None,
    categories: Iterable[str] | None = None,
    autopsy_label_mode: str = AUTOPSY_LABEL_MODE_CORRECTED,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Build note-derived labels from a large CSV in chunks.

    The study pipeline should normally leave ``categories`` unset so labels are
    derived from all non-error note types.
    """

    _, labels = build_note_artifacts_from_csv(
        noteevents_csv_path=noteevents_csv_path,
        all_hadm_ids=all_hadm_ids,
        label_categories=categories,
        autopsy_label_mode=autopsy_label_mode,
        chunksize=chunksize,
    )
    return labels


def identify_table2_itemids(d_items: pd.DataFrame) -> set[int]:
    """Identify chart itemids that match the paper's Table 2 concepts."""

    _require_columns(d_items, ["itemid", "label", "dbsource"], "d_items")
    matches = d_items["label"].map(_matches_table2_concept)
    return set(d_items.loc[matches, "itemid"].tolist())


def _resolve_chartevent_code_status_mode(
    *,
    paper_like: bool,
    code_status_mode: str | None,
) -> str:
    return _task_normalize_code_status_mode(
        code_status_mode
        if code_status_mode is not None
        else (CODE_STATUS_MODE_PAPER_LIKE if paper_like else CODE_STATUS_MODE_CORRECTED)
    )


def _iter_relevant_chartevent_csv_chunks(
    chartevents_csv_path: Path | str,
    *,
    relevant_itemids: set[int],
    hadm_filter: set[int] | None,
    chunksize: int,
):
    for chunk in _iter_csv_chunks(
        chartevents_csv_path,
        required_columns=["hadm_id", "itemid", "value", "icustay_id", "charttime"],
        chunksize=chunksize,
    ):
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce")
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
        chunk = chunk.dropna(subset=["hadm_id", "itemid"]).copy()
        if chunk.empty:
            continue

        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["itemid"] = chunk["itemid"].astype(int)

        if hadm_filter is not None:
            chunk = chunk.loc[chunk["hadm_id"].isin(hadm_filter)]
        if chunk.empty:
            continue

        chunk = chunk.loc[chunk["itemid"].isin(relevant_itemids)].copy()
        if not chunk.empty:
            yield chunk


def _accumulate_normal_feature_rows(
    feature_chunk: pd.DataFrame,
    *,
    item_label_lookup: Mapping[int, str],
    label_display_lookup: Mapping[str, str],
    feature_to_hadm: dict[tuple[str, str], set[int]],
    feature_value_display_lookup: dict[tuple[str, str], str],
) -> None:
    if feature_chunk.empty:
        return

    working = feature_chunk.copy()
    working["normalized_label"] = working["itemid"].map(item_label_lookup)
    working["normalized_value"] = working["value"].map(_normalize_token)
    working["display_label"] = working["normalized_label"].map(label_display_lookup)
    working["display_value"] = working["value"].map(_clean_feature_text)
    working = working.loc[
        (working["normalized_value"] != "") & (working["display_label"] != "")
    ].copy()
    if working.empty:
        return

    keep_mask = ~working.apply(
        lambda row: _is_numeric_heavy_or_freeform_feature_value(
            str(getattr(row, "normalized_value")),
            str(getattr(row, "display_value")),
        ),
        axis=1,
    )
    working = working.loc[keep_mask].copy()
    if working.empty:
        return

    unique_pairs = working[
        ["hadm_id", "normalized_label", "normalized_value", "display_value"]
    ].drop_duplicates()
    for row in unique_pairs.itertuples(index=False):
        key = (str(row.normalized_label), str(row.normalized_value))
        feature_value_display_lookup[key] = _choose_preferred_feature_text(
            [feature_value_display_lookup.get(key, ""), str(row.display_value)]
        )
        feature_to_hadm[key].add(int(row.hadm_id))


def _accumulate_paper_like_feature_rows(
    feature_chunk: pd.DataFrame,
    *,
    item_raw_label_lookup: Mapping[int, str],
    feature_to_hadm: dict[str, set[int]],
) -> None:
    if feature_chunk.empty:
        return

    working = feature_chunk.copy()
    working["label"] = working["itemid"].map(item_raw_label_lookup)
    paper_like_chunk = _paper_like_feature_sets_from_rows(
        working[["hadm_id", "label", "value"]]
    )
    for feature_name, hadm_ids in paper_like_chunk.items():
        feature_to_hadm[feature_name].update(hadm_ids)


def _finalize_normal_feature_matrix(
    *,
    feature_to_hadm: Mapping[tuple[str, str], set[int]],
    hadm_ids: Sequence[int],
    label_display_lookup: Mapping[str, str],
    feature_value_display_lookup: Mapping[tuple[str, str], str],
) -> pd.DataFrame:
    feature_keys = sorted(
        feature_to_hadm,
        key=lambda key: _feature_display_name(
            key[0],
            key[1],
            label_display_lookup,
            feature_value_display_lookup,
        ).lower(),
    )
    feature_data: dict[str, object] = {"hadm_id": list(hadm_ids)}
    hadm_index = pd.Index(hadm_ids)
    for feature_key in feature_keys:
        feature_name = _feature_display_name(
            feature_key[0],
            feature_key[1],
            label_display_lookup,
            feature_value_display_lookup,
        )
        feature_data[feature_name] = hadm_index.isin(
            feature_to_hadm[feature_key]
        ).astype(int)

    feature_matrix = pd.DataFrame(feature_data)
    if "hadm_id" not in feature_matrix.columns:
        feature_matrix = pd.DataFrame(columns=["hadm_id"])
    return (
        feature_matrix.sort_values("hadm_id")
        .drop_duplicates("hadm_id")
        .reset_index(drop=True)
    )


def _build_normal_feature_matrix_from_events(
    events: pd.DataFrame,
    *,
    items: pd.DataFrame,
    normalized_hadm_ids: list[int] | None,
) -> pd.DataFrame:
    item_label_lookup, label_display_lookup = _build_feature_label_metadata(items)
    merged = events.merge(
        items[["itemid", "normalized_label"]],
        on="itemid",
        how="inner",
        validate="many_to_one",
    )
    feature_to_hadm: dict[tuple[str, str], set[int]] = defaultdict(set)
    feature_value_display_lookup: dict[tuple[str, str], str] = {}
    _accumulate_normal_feature_rows(
        merged[["hadm_id", "itemid", "value", "normalized_label"]],
        item_label_lookup=item_label_lookup,
        label_display_lookup=label_display_lookup,
        feature_to_hadm=feature_to_hadm,
        feature_value_display_lookup=feature_value_display_lookup,
    )

    if normalized_hadm_ids is not None:
        hadm_ids = normalized_hadm_ids
    else:
        hadm_ids = (
            sorted(set().union(*feature_to_hadm.values())) if feature_to_hadm else []
        )
    return _finalize_normal_feature_matrix(
        feature_to_hadm=feature_to_hadm,
        hadm_ids=hadm_ids,
        label_display_lookup=label_display_lookup,
        feature_value_display_lookup=feature_value_display_lookup,
    )


def _build_paper_like_feature_matrix_from_events(
    events: pd.DataFrame,
    *,
    items: pd.DataFrame,
    normalized_hadm_ids: list[int] | None,
) -> pd.DataFrame:
    merged = events.merge(
        items[["itemid", "label"]],
        on="itemid",
        how="inner",
        validate="many_to_one",
    )
    feature_to_hadm = _paper_like_feature_sets_from_rows(
        merged[["hadm_id", "label", "value"]]
    )
    if normalized_hadm_ids is not None:
        hadm_ids = normalized_hadm_ids
    else:
        hadm_ids = (
            sorted(set().union(*feature_to_hadm.values())) if feature_to_hadm else []
        )
    return _binary_feature_matrix_from_feature_sets(feature_to_hadm, hadm_ids)


def _accumulate_corrected_code_status_rows(
    code_chunk: pd.DataFrame,
    *,
    code_status_positive: dict[int, int],
    code_status_latest: dict[int, tuple[tuple[int, int, int], int]],
    code_status_event_order_start: int,
) -> int:
    if code_chunk.empty:
        return code_status_event_order_start

    if "charttime" not in code_chunk.columns:
        positives = code_chunk["value"].map(
            lambda value: int(_task_is_positive_code_status_value(value))
        )
        for hadm_id, is_positive in zip(code_chunk["hadm_id"].astype(int), positives):
            code_status_positive[hadm_id] = max(
                code_status_positive.get(hadm_id, 0), int(is_positive)
            )
        return code_status_event_order_start

    event_order = code_status_event_order_start
    working = code_chunk.copy()
    working["charttime"] = pd.to_datetime(working["charttime"], errors="coerce")
    for row in working.itertuples(index=False):
        hadm_id = int(row.hadm_id)
        label = int(_task_is_positive_code_status_value(getattr(row, "value")))
        charttime = getattr(row, "charttime")
        has_charttime = int(not pd.isna(charttime))
        charttime_value = int(charttime.value) if has_charttime else -1
        sort_key = (has_charttime, charttime_value, event_order)
        event_order += 1
        previous = code_status_latest.get(hadm_id)
        if previous is None or sort_key > previous[0]:
            code_status_latest[hadm_id] = (sort_key, label)
    return event_order


def _accumulate_paper_like_code_status_rows(
    code_chunk: pd.DataFrame,
    *,
    current_label: int | None,
    targets: dict[int, int],
) -> int | None:
    for row in code_chunk.itertuples(index=False):
        hadm_id = int(getattr(row, "hadm_id"))
        current_label = _task_advance_paper_like_code_status_label(
            current_label,
            getattr(row, "value"),
        )
        if current_label is not None:
            targets[hadm_id] = int(current_label)
    return current_label


def _finalize_code_status_targets(
    *,
    normalized_code_status_mode: str,
    code_status_positive: Mapping[int, int],
    code_status_latest: Mapping[int, tuple[tuple[int, int, int], int]],
    code_status_paper_like: Mapping[int, int],
) -> pd.DataFrame:
    if normalized_code_status_mode == CODE_STATUS_MODE_PAPER_LIKE:
        target_map = code_status_paper_like
        values = [int(target_map[hadm_id]) for hadm_id in sorted(target_map)]
    elif code_status_latest:
        target_map = code_status_latest
        values = [int(target_map[hadm_id][1]) for hadm_id in sorted(target_map)]
    else:
        target_map = code_status_positive
        values = [int(target_map[hadm_id]) for hadm_id in sorted(target_map)]

    return (
        pd.DataFrame(
            {
                "hadm_id": sorted(target_map),
                "code_status_dnr_dni_cmo": values,
            }
        )
        .sort_values("hadm_id")
        .drop_duplicates("hadm_id")
        .reset_index(drop=True)
    )


def build_chartevent_artifacts_from_csv(
    chartevents_csv_path: Path | str,
    d_items: pd.DataFrame,
    allowed_labels: Iterable[str] | None = None,
    all_hadm_ids: Iterable[int] | None = None,
    chunksize: int = 500_000,
    paper_like: bool = False,
    code_status_mode: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the feature matrix and code-status targets from a large CSV in chunks."""

    items = _filter_chartevent_items(
        d_items,
        allowed_labels=allowed_labels,
        paper_like=paper_like,
    )
    item_label_lookup = _build_feature_label_metadata(items)[0]
    item_raw_label_lookup = (
        items.drop_duplicates("itemid").set_index("itemid")["label"].to_dict()
    )
    feature_itemids = set(item_label_lookup)
    relevant_itemids = feature_itemids | set(CODE_STATUS_ITEMIDS)
    normalized_hadm_ids = _normalize_hadm_ids(all_hadm_ids)
    hadm_filter = set(normalized_hadm_ids) if normalized_hadm_ids is not None else None
    normalized_code_status_mode = _resolve_chartevent_code_status_mode(
        paper_like=paper_like,
        code_status_mode=code_status_mode,
    )

    paper_like_feature_to_hadm: dict[str, set[int]] = defaultdict(set)
    feature_to_hadm: dict[tuple[str, str], set[int]] = defaultdict(set)
    feature_value_display_lookup: dict[tuple[str, str], str] = {}
    code_status_positive: dict[int, int] = {}
    code_status_latest: dict[int, tuple[tuple[int, int, int], int]] = {}
    code_status_paper_like: dict[int, int] = {}
    code_status_current_label: int | None = None
    code_status_event_order = 0

    item_label_lookup, label_display_lookup = _build_feature_label_metadata(items)
    for chunk in _iter_relevant_chartevent_csv_chunks(
        chartevents_csv_path,
        relevant_itemids=relevant_itemids,
        hadm_filter=hadm_filter,
        chunksize=chunksize,
    ):
        feature_chunk = chunk.loc[chunk["itemid"].isin(feature_itemids)].copy()
        if paper_like:
            _accumulate_paper_like_feature_rows(
                feature_chunk,
                item_raw_label_lookup=item_raw_label_lookup,
                feature_to_hadm=paper_like_feature_to_hadm,
            )
        else:
            _accumulate_normal_feature_rows(
                feature_chunk,
                item_label_lookup=item_label_lookup,
                label_display_lookup=label_display_lookup,
                feature_to_hadm=feature_to_hadm,
                feature_value_display_lookup=feature_value_display_lookup,
            )

        code_chunk = chunk.loc[chunk["itemid"].isin(CODE_STATUS_ITEMIDS)].copy()
        if normalized_code_status_mode == CODE_STATUS_MODE_PAPER_LIKE:
            code_status_current_label = _accumulate_paper_like_code_status_rows(
                code_chunk,
                current_label=code_status_current_label,
                targets=code_status_paper_like,
            )
        else:
            code_status_event_order = _accumulate_corrected_code_status_rows(
                code_chunk,
                code_status_positive=code_status_positive,
                code_status_latest=code_status_latest,
                code_status_event_order_start=code_status_event_order,
            )

    if normalized_hadm_ids is not None:
        hadm_ids = normalized_hadm_ids
    elif paper_like:
        hadm_ids = (
            sorted(set().union(*paper_like_feature_to_hadm.values()))
            if paper_like_feature_to_hadm
            else []
        )
    else:
        hadm_ids = (
            sorted(set().union(*feature_to_hadm.values())) if feature_to_hadm else []
        )

    if paper_like:
        feature_matrix = _binary_feature_matrix_from_feature_sets(
            paper_like_feature_to_hadm, hadm_ids
        )
    else:
        feature_matrix = _finalize_normal_feature_matrix(
            feature_to_hadm=feature_to_hadm,
            hadm_ids=hadm_ids,
            label_display_lookup=label_display_lookup,
            feature_value_display_lookup=feature_value_display_lookup,
        )

    code_status_targets = _finalize_code_status_targets(
        normalized_code_status_mode=normalized_code_status_mode,
        code_status_positive=code_status_positive,
        code_status_latest=code_status_latest,
        code_status_paper_like=code_status_paper_like,
    )
    return feature_matrix, code_status_targets


def build_chartevent_feature_matrix(
    chartevents: pd.DataFrame,
    d_items: pd.DataFrame,
    allowed_labels: Iterable[str] | None = None,
    all_hadm_ids: Iterable[int] | None = None,
    paper_like: bool = False,
) -> pd.DataFrame:
    """Build a binary admission-by-feature matrix from selected chart events."""

    _require_columns(
        chartevents, ["hadm_id", "itemid", "value", "icustay_id"], "chartevents"
    )

    events = chartevents.copy()
    items = _filter_chartevent_items(
        d_items,
        allowed_labels=allowed_labels,
        paper_like=paper_like,
    )

    normalized_hadm_ids = (
        _normalize_hadm_ids(all_hadm_ids) if all_hadm_ids is not None else None
    )
    if paper_like and normalized_hadm_ids is not None:
        events["hadm_id"] = pd.to_numeric(events["hadm_id"], errors="coerce")
        events = events.dropna(subset=["hadm_id"]).copy()
        events["hadm_id"] = events["hadm_id"].astype(int)
        events = events.loc[events["hadm_id"].isin(set(normalized_hadm_ids))].copy()

    if paper_like:
        return _build_paper_like_feature_matrix_from_events(
            events,
            items=items,
            normalized_hadm_ids=normalized_hadm_ids,
        )

    result = _build_normal_feature_matrix_from_events(
        events,
        items=items,
        normalized_hadm_ids=normalized_hadm_ids,
    )
    if all_hadm_ids is not None:
        result = pd.DataFrame({"hadm_id": list(all_hadm_ids)}).merge(
            result, on="hadm_id", how="left"
        )

    feature_cols = [col for col in result.columns if col != "hadm_id"]
    if feature_cols:
        result[feature_cols] = result[feature_cols].fillna(0).astype(int)
    return (
        result.sort_values("hadm_id").drop_duplicates("hadm_id").reset_index(drop=True)
    )


def build_chartevent_feature_matrix_from_csv(
    chartevents_csv_path: Path | str,
    d_items: pd.DataFrame,
    allowed_labels: Iterable[str] | None = None,
    all_hadm_ids: Iterable[int] | None = None,
    chunksize: int = 500_000,
    paper_like: bool = False,
) -> pd.DataFrame:
    """Build the binary feature matrix from a large chartevents CSV in chunks."""

    feature_matrix, _ = build_chartevent_artifacts_from_csv(
        chartevents_csv_path=chartevents_csv_path,
        d_items=d_items,
        allowed_labels=allowed_labels,
        all_hadm_ids=all_hadm_ids,
        chunksize=chunksize,
        paper_like=paper_like,
    )
    return feature_matrix


def build_acuity_scores(oasis: pd.DataFrame, sapsii: pd.DataFrame) -> pd.DataFrame:
    """Aggregate OASIS and SAPS II to one admission-level row per hadm_id."""

    _require_columns(oasis, ["hadm_id", "icustay_id", "oasis"], "oasis")
    _require_columns(sapsii, ["hadm_id", "icustay_id", "sapsii"], "sapsii")

    oasis_agg = oasis.groupby("hadm_id", as_index=False)["oasis"].max()
    sapsii_agg = sapsii.groupby("hadm_id", as_index=False)["sapsii"].max()
    acuity = oasis_agg.merge(sapsii_agg, on="hadm_id", how="outer")
    acuity = acuity.sort_values("hadm_id").drop_duplicates("hadm_id")
    return acuity.reset_index(drop=True)


def _build_gender_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({"hadm_id": df["hadm_id"]})
    gender = df["gender"].fillna("").str.upper()
    output["gender_f"] = (gender == "F").astype(int)
    output["gender_m"] = (gender == "M").astype(int)
    return output


def _build_insurance_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({"hadm_id": df["hadm_id"]})
    insurance_column = (
        "insurance_group" if "insurance_group" in df.columns else "insurance"
    )
    insurance = df[insurance_column].fillna("")
    output["insurance_private"] = (insurance == INSURANCE_PRIVATE).astype(int)
    output["insurance_public"] = (insurance == INSURANCE_PUBLIC).astype(int)
    output["insurance_self_pay"] = (insurance == INSURANCE_SELF_PAY).astype(int)
    return output


def _build_race_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({"hadm_id": df["hadm_id"]})
    race = df["race"].fillna("")
    output["race_white"] = (race == RACE_WHITE).astype(int)
    output["race_black"] = (race == RACE_BLACK).astype(int)
    output["race_asian"] = (race == RACE_ASIAN).astype(int)
    output["race_hispanic"] = (race == RACE_HISPANIC).astype(int)
    output["race_native_american"] = (race == RACE_NATIVE_AMERICAN).astype(int)
    output["race_other"] = (race == RACE_OTHER).astype(int)
    return output


def build_code_status_target_from_csv(
    chartevents_csv_path: Path | str,
    chunksize: int = 500_000,
    code_status_mode: str = CODE_STATUS_MODE_CORRECTED,
) -> pd.DataFrame:
    """Build the code-status target from a large chartevents CSV in chunks."""

    _, code_status_targets = build_chartevent_artifacts_from_csv(
        chartevents_csv_path=chartevents_csv_path,
        d_items=pd.DataFrame(columns=["itemid", "label", "dbsource"]),
        all_hadm_ids=None,
        chunksize=chunksize,
        code_status_mode=code_status_mode,
    )
    return code_status_targets


def _assemble_final_model_table(
    demographics: pd.DataFrame,
    all_cohort: pd.DataFrame,
    admissions: pd.DataFrame,
    code_status: pd.DataFrame,
    mistrust_scores: pd.DataFrame,
    include_race: bool = True,
    include_mistrust: bool = True,
) -> pd.DataFrame:
    """Shared implementation for final model table assembly."""

    _require_columns(
        demographics,
        ["hadm_id", "age", "los_days", "gender", "insurance", "race"],
        "demographics",
    )
    _require_columns(all_cohort, ["hadm_id"], "all_cohort")
    _require_columns(
        admissions,
        ["hadm_id", "subject_id", "discharge_location", "hospital_expire_flag"],
        "admissions",
    )
    _require_columns(
        mistrust_scores,
        [
            "hadm_id",
            "noncompliance_score_z",
            "autopsy_score_z",
            "negative_sentiment_score_z",
        ],
        "mistrust_scores",
    )
    _require_columns(code_status, ["hadm_id", "code_status_dnr_dni_cmo"], "code_status")

    cohort_hadm = pd.DataFrame(
        {
            "hadm_id": sorted(
                pd.to_numeric(all_cohort["hadm_id"], errors="coerce")
                .dropna()
                .astype(int)
                .unique()
            )
        }
    )
    demo = cohort_hadm.merge(demographics, on="hadm_id", how="left")

    final = cohort_hadm.copy()
    final = final.merge(
        admissions[["hadm_id", "subject_id"]].drop_duplicates("hadm_id"),
        on="hadm_id",
        how="left",
    )
    final = final.merge(
        demo[["hadm_id", "age", "los_days"]],
        on="hadm_id",
        how="left",
    )
    for col in ("age", "los_days"):
        std = final[col].std(ddof=0)
        if std > 0:
            final[col] = (final[col] - final[col].mean()) / std
    final = final.merge(_build_gender_one_hot(demo), on="hadm_id", how="left")
    final = final.merge(_build_insurance_one_hot(demo), on="hadm_id", how="left")

    if include_race:
        final = final.merge(_build_race_one_hot(demo), on="hadm_id", how="left")

    if include_mistrust:
        final = final.merge(mistrust_scores, on="hadm_id", how="left")

    admissions_targets = admissions[
        ["hadm_id", "discharge_location", "hospital_expire_flag"]
    ].drop_duplicates("hadm_id")
    left_ama = _build_task_left_ama_target(admissions_targets)
    mortality = _build_task_in_hospital_mortality_target(admissions_targets)
    final = final.merge(
        left_ama.merge(mortality, on="hadm_id", how="outer"),
        on="hadm_id",
        how="left",
    )
    final = final.merge(code_status, on="hadm_id", how="left")
    final["code_status_dnr_dni_cmo"] = pd.to_numeric(
        final["code_status_dnr_dni_cmo"],
        errors="coerce",
    ).astype("Int64")

    fill_zero_columns = [
        "gender_f",
        "gender_m",
        "insurance_private",
        "insurance_public",
        "insurance_self_pay",
        "left_ama",
        "in_hospital_mortality",
    ]
    if include_race:
        fill_zero_columns.extend(
            [
                "race_white",
                "race_black",
                "race_asian",
                "race_hispanic",
                "race_native_american",
                "race_other",
            ]
        )
    for column in fill_zero_columns:
        if column in final.columns:
            final[column] = final[column].fillna(0).astype(int)

    if final["subject_id"].isna().any():
        raise ValueError(
            "Final model table contains null subject_id values after admissions merge."
        )
    final["subject_id"] = pd.to_numeric(final["subject_id"], errors="raise").astype(int)
    final = final.drop(columns=["subject_id"])

    final = final.sort_values("hadm_id").drop_duplicates("hadm_id")
    return final.reset_index(drop=True)


def build_final_model_table(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    demographics: pd.DataFrame,
    all_cohort: pd.DataFrame,
    admissions: pd.DataFrame,
    chartevents: pd.DataFrame,
    d_items: pd.DataFrame,
    mistrust_scores: pd.DataFrame,
    include_race: bool = True,
    include_mistrust: bool = True,
) -> pd.DataFrame:
    """Assemble the final model table from raw chartevents.

    ``d_items`` is retained for API compatibility; the normal path uses the
    fixed code-status itemids defined by the task layer.
    """
    _require_columns(
        chartevents, ["hadm_id", "itemid", "value", "icustay_id"], "chartevents"
    )
    del d_items
    code_status = _build_task_code_status_target(
        chartevents,
        itemids=CODE_STATUS_ITEMIDS,
        code_status_mode=CODE_STATUS_MODE_CORRECTED,
    )
    return build_final_model_table_from_code_status_targets(
        demographics=demographics,
        all_cohort=all_cohort,
        admissions=admissions,
        code_status_targets=code_status,
        mistrust_scores=mistrust_scores,
        include_race=include_race,
        include_mistrust=include_mistrust,
    )


def build_final_model_table_from_code_status_targets(  # pylint: disable=too-many-arguments
    demographics: pd.DataFrame,
    all_cohort: pd.DataFrame,
    admissions: pd.DataFrame,
    code_status_targets: pd.DataFrame,
    mistrust_scores: pd.DataFrame,
    include_race: bool = True,
    include_mistrust: bool = True,
) -> pd.DataFrame:
    """Assemble the final model table using precomputed code-status targets."""

    return _assemble_final_model_table(
        demographics=demographics,
        all_cohort=all_cohort,
        admissions=admissions,
        code_status=code_status_targets,
        mistrust_scores=mistrust_scores,
        include_race=include_race,
        include_mistrust=include_mistrust,
    )


def write_minimal_deliverables(
    artifacts: dict[str, pd.DataFrame], output_dir: Path | str
) -> None:
    """Write the required CSV deliverables to disk without index columns."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filenames = {
        "base_admissions": "base_admissions.csv",
        "eol_cohort": "eol_cohort.csv",
        "all_cohort": "all_cohort.csv",
        "treatment_totals": "treatment_totals.csv",
        "chartevent_feature_matrix": "chartevent_feature_matrix.csv",
        "note_labels": "note_labels.csv",
        "mistrust_scores": "mistrust_scores.csv",
        "acuity_scores": "acuity_scores.csv",
        "final_model_table": "final_model_table.csv",
    }

    for key, filename in filenames.items():
        if key not in artifacts:
            continue
        df = artifacts[key].copy()
        if "hadm_id" in df.columns:
            df = df.sort_values("hadm_id")
        df.to_csv(output_path / filename, index=False)


def validate_database_environment(  # pylint: disable=too-many-locals
    raw_tables: Mapping[str, pd.DataFrame],
    materialized_views: Mapping[str, pd.DataFrame],
    schema_name: str | None = None,
    database_flavor: str | None = None,
) -> dict[str, object]:
    """Validate that the loaded MIMIC environment supports the full pipeline."""

    resolved_schema, resolved_flavor = _validate_database_identity(
        schema_name=schema_name,
        database_flavor=database_flavor,
    )
    _validate_required_inputs(raw_tables, materialized_views)

    admissions = raw_tables["admissions"]
    patients = raw_tables["patients"]
    icustays = raw_tables["icustays"]
    noteevents = raw_tables["noteevents"]
    chartevents = raw_tables["chartevents"]
    d_items = raw_tables["d_items"]
    ventdurations = materialized_views["ventdurations"]
    vasopressordurations = materialized_views["vasopressordurations"]

    available_join_keys = _collect_required_join_keys(raw_tables)
    missing_keys = sorted(REQUIRED_JOIN_KEYS - available_join_keys)
    if missing_keys:
        raise ValueError("Missing required join keys: " + ", ".join(missing_keys))

    base = build_base_admissions(admissions, patients)
    if len(base) <= 50000:
        raise ValueError(
            "Base admissions after admissions-patients join and "
            "has_chartevents_data filter must exceed 50,000 rows."
        )

    if base["subject_id"].isna().any():
        raise ValueError(
            "Base admissions contains null subject_id values after "
            "admissions-patients join."
        )
    if base["hadm_id"].isna().any():
        raise ValueError("Base admissions contains null hadm_id values.")
    if icustays["hadm_id"].isna().any() or icustays["icustay_id"].isna().any():
        raise ValueError(
            "icustays must provide non-null hadm_id and icustay_id for ICU bridging."
        )
    _validate_text_access(noteevents, chartevents)

    icu_bridge = icustays[["icustay_id", "hadm_id"]].drop_duplicates()
    _validate_bridge_join(
        ventdurations,
        icu_bridge,
        "icustay_id",
        "ventdurations must join to icustays through icustay_id.",
    )
    _validate_bridge_join(
        vasopressordurations,
        icu_bridge,
        "icustay_id",
        "vasopressordurations must join to icustays through icustay_id.",
    )
    _validate_bridge_join(
        chartevents,
        d_items[["itemid"]].drop_duplicates(),
        "itemid",
        "chartevents must join to d_items through itemid.",
    )

    acuity = build_acuity_scores(
        materialized_views["oasis"],
        materialized_views["sapsii"],
    )
    if acuity.empty:
        raise ValueError(
            "oasis and sapsii must join back to admissions on hadm_id "
            "and yield admission-level acuity rows."
        )

    supports_multiple_icustays = bool(
        icustays.groupby("hadm_id")["icustay_id"].nunique().gt(1).any()
    )

    return {
        "database_flavor": resolved_flavor,
        "schema_name": resolved_schema,
        "base_admissions_rows": int(len(base)),
        "raw_tables": sorted(raw_tables.keys()),
        "materialized_views": sorted(materialized_views.keys()),
        "supports_multiple_icustays_per_hadm": supports_multiple_icustays,
    }


# ---------------------------------------------------------------------------
# Compatibility Wrappers
# ---------------------------------------------------------------------------


def _call_model_compat(function_name: str, /, **kwargs):
    """Delegate deprecated dataset wrappers to the model-owned implementation."""

    model_module = _load_eol_mistrust_model_module()
    return getattr(model_module, function_name)(**kwargs)


def z_normalize_scores(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Deprecated wrapper around the model-owned score normalization helper."""

    return _call_model_compat("z_normalize_scores", score_table=df, columns=columns)


def build_proxy_probability_scores(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    label_column: str,
    estimator_factory: Callable[[], object] | None = None,
) -> pd.DataFrame:
    """Deprecated wrapper around the model-owned proxy score helper."""

    return _call_model_compat(
        "build_proxy_probability_scores",
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        label_column=label_column,
        estimator_factory=estimator_factory,
    )


def build_negative_sentiment_scores(
    note_corpus: pd.DataFrame,
    sentiment_fn: Callable[[str], tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Deprecated wrapper around the model-owned sentiment score helper."""

    return _call_model_compat(
        "build_negative_sentiment_mistrust_scores",
        note_corpus=note_corpus,
        sentiment_fn=sentiment_fn,
    )


def build_mistrust_score_table(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    note_corpus: pd.DataFrame,
    estimator_factory: Callable[[], object] | None = None,
    sentiment_fn: Callable[[str], tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Deprecated wrapper around the model-owned mistrust table builder."""

    return _call_model_compat(
        "build_mistrust_score_table",
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        note_corpus=note_corpus,
        estimator_factory=estimator_factory,
        sentiment_fn=sentiment_fn,
    )
