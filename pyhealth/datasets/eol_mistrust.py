"""Utilities for reproducing the EOL mistrust preprocessing and modeling tables."""
# pylint: disable=too-many-lines

import importlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd  # pylint: disable=import-error

from pyhealth.tasks.eol_mistrust import (
    build_code_status_target as _build_task_code_status_target,
    build_in_hospital_mortality_target as _build_task_in_hospital_mortality_target,
    build_left_ama_target as _build_task_left_ama_target,
)

_SENTIMENT_BACKEND: Callable[[str], tuple[float, float]] | None = None


def _load_transformers_sentiment() -> Callable[[str], tuple[float, float]]:
    """Load a transformers sentiment pipeline, preferring GPU when available."""

    transformers_module = importlib.import_module("transformers")
    torch_module = importlib.import_module("torch")

    pipeline_factory = getattr(transformers_module, "pipeline", None)
    if not callable(pipeline_factory):
        raise ModuleNotFoundError("transformers.pipeline is unavailable in the current environment.")

    try:  # pragma: no cover - logging surface depends on transformers version
        transformers_logging = importlib.import_module("transformers.utils.logging")
        set_verbosity_error = getattr(transformers_logging, "set_verbosity_error", None)
        if callable(set_verbosity_error):
            set_verbosity_error()
    except Exception:
        pass

    use_cuda = bool(getattr(torch_module, "cuda", None) and torch_module.cuda.is_available())
    device = 0 if use_cuda else -1
    classifier = pipeline_factory(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )

    def _transformers_sentiment(text: str) -> tuple[float, float]:
        cleaned = " ".join(str(text).split())
        if not cleaned:
            return (0.0, 0.0)
        result = classifier(cleaned[:2048], truncation=True)[0]
        label = str(result.get("label", "")).upper()
        score = float(result.get("score", 0.0))
        polarity = score if "POS" in label else -score
        return (polarity, 0.0)

    return _transformers_sentiment


def _default_sentiment_backend(text: str) -> tuple[float, float]:
    """Resolve and cache the default transformers sentiment backend lazily."""

    global _SENTIMENT_BACKEND
    if _SENTIMENT_BACKEND is None:
        _SENTIMENT_BACKEND = _load_transformers_sentiment()
    return _SENTIMENT_BACKEND(text)


pattern_sentiment = _default_sentiment_backend

try:
    from sklearn.linear_model import LogisticRegression  # pylint: disable=import-error
except ModuleNotFoundError:  # pragma: no cover - lightweight test env fallback
    class LogisticRegression:  # type: ignore[no-redef]
        """Fallback estimator that preserves the expected interface in test envs."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit(self, features, labels):
            """Raise when scikit-learn is unavailable for model fitting."""

            del features, labels
            raise ModuleNotFoundError(
                "scikit-learn is required for the default logistic regression estimator."
            )

        def predict_proba(self, features):
            """Raise when scikit-learn is unavailable for probability scoring."""

            del features
            raise ModuleNotFoundError(
                "scikit-learn is required for the default logistic regression estimator."
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
    "reason_for_restraint",
    "restraint_device",
    "richmond_ras_scale",
    "riker_sas_scale",
    "safety_measures",
    "security",
    "side_rails",
    "sitter",
    "skin_care",
    "social_work_consult",
    "spiritual_support",
    "stress",
    "support_systems",
    "understand_agree_with_plan",
    "verbal_response",
    "violent_restraints",
    "wrist_restraints",
}

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
    "ventdurations": ["icustay_id", "ventnum", "starttime", "endtime", "duration_hours"],
    "vasopressordurations": ["icustay_id", "vasonum", "starttime", "endtime", "duration_hours"],
    "oasis": ["hadm_id", "icustay_id", "oasis"],
    "sapsii": ["hadm_id", "icustay_id", "sapsii"],
}

REQUIRED_JOIN_KEYS = {
    "subject_id",
    "hadm_id",
    "icustay_id",
    "itemid",
}

NONCOMPLIANCE_PATTERNS = [
    "noncomplian",
    "non-complian",
    "nonadher",
    "non-adher",
    "noncompliance",
    "noncompliant",
    "refuses treatment",
    "refused treatment",
    "refused medication",
    "refuses medication",
]


def _require_columns(df: pd.DataFrame, required: Sequence[str], df_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{df_name} is missing required columns: {missing_str}")


def _filter_non_error_notes(noteevents: pd.DataFrame) -> pd.DataFrame:
    """Keep notes where iserror is NULL or not equal to 1."""

    iserror_numeric = pd.to_numeric(noteevents["iserror"], errors="coerce")
    keep_mask = noteevents["iserror"].isna() | iserror_numeric.ne(1)
    return noteevents.loc[keep_mask].copy()


def _extract_positive_class_probabilities(probabilities) -> list[float]:
    """Validate predict_proba output and return the positive-class column."""

    probability_frame = pd.DataFrame(probabilities)
    if probability_frame.shape[1] < 2:
        raise ValueError(
            "Estimator `predict_proba` output must have shape (n_samples, n_classes>=2)."
        )
    return probability_frame.iloc[:, 1].astype(float).tolist()


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


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
        age = (admit.to_pydatetime() - birth.to_pydatetime()).total_seconds() / seconds_per_year
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
    return re.sub(r"\s+", " ", str(value).strip())


def _matches_table2_concept(label: str) -> bool:
    """Return True when a d_items label matches a Table 2 concept by partial match."""

    normalized_label = _normalize_token(label)
    if normalized_label == "":
        return False
    return any(
        (concept in normalized_label) or (normalized_label in concept)
        for concept in TABLE2_LABELS
    )


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
    """Map raw MIMIC ethnicity strings to the paper's coarse race groups."""

    text = str(ethnicity or "").upper()
    if "BLACK" in text or "AFRICAN" in text:
        return RACE_BLACK
    if "WHITE" in text or "EUROPEAN" in text or "PORTUGUESE" in text:
        return RACE_WHITE
    if "ASIAN" in text:
        return RACE_ASIAN
    if "HISPANIC" in text or "LATINO" in text or "SOUTH AMERICAN" in text:
        return RACE_HISPANIC
    if (
        "NATIVE" in text
        or "AMERICAN INDIAN" in text
        or "ALASKA NATIVE" in text
    ):
        return RACE_NATIVE_AMERICAN
    return RACE_OTHER


def map_insurance(insurance) -> str:
    """Collapse raw MIMIC insurance values into the required three groups."""

    text = str(insurance or "").strip().lower()
    normalized = re.sub(r"\s+", " ", text)
    if normalized in {"medicare", "medicaid", "government", "public"}:
        return INSURANCE_PUBLIC
    if normalized in {"private"}:
        return INSURANCE_PRIVATE
    if normalized in {"self pay", "self-pay", "self_pay"}:
        return INSURANCE_SELF_PAY
    raise ValueError(f"Unexpected insurance value: {insurance}")


def prepare_note_text_for_sentiment(text) -> str:
    """Normalize note text using whitespace tokenization and rejoining only."""

    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    tokens = str(text).split()
    return " ".join(tokens)


def build_base_admissions(admissions: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
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


def build_demographics_table(base_admissions: pd.DataFrame) -> pd.DataFrame:
    """Derive race, age, LOS, and insurance-group fields for each admission."""

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


def build_eol_cohort(base_admissions: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
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
    is_snf = discharge_location.str.contains(r"SKILLED NURSING|\bSNF\b", na=False, regex=True)

    include = (df["los_hours"] >= 6) & (is_deceased | is_hospice | is_snf)
    df = df.loc[include].copy()
    df["discharge_category"] = "Skilled Nursing Facility"
    df.loc[is_hospice.loc[df.index], "discharge_category"] = "Hospice"
    df.loc[is_deceased.loc[df.index], "discharge_category"] = "Deceased"
    df = df.sort_values("hadm_id").drop_duplicates("hadm_id")
    return df.reset_index(drop=True)


def build_all_cohort(base_admissions: pd.DataFrame, icustays: pd.DataFrame) -> pd.DataFrame:
    """Build the admission-level cohort with at least one ICU stay of 12 hours."""

    _require_columns(base_admissions, ["hadm_id"], "base_admissions")
    _require_columns(icustays, ["hadm_id", "icustay_id", "intime", "outtime"], "icustays")

    icu = icustays.copy()
    icu["intime"] = _to_datetime(icu["intime"])
    icu["outtime"] = _to_datetime(icu["outtime"])
    icu["icu_los_hours"] = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600.0

    qualifying = icu.loc[icu["icu_los_hours"] >= 12, "hadm_id"].drop_duplicates()
    df = base_admissions.loc[base_admissions["hadm_id"].isin(set(qualifying))].copy()
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

    bridge = icustays[["icustay_id", "hadm_id"]].drop_duplicates()
    df = durations.copy()
    if "hadm_id" in df.columns:
        df = df.drop(columns=["hadm_id"])
    df["starttime"] = _to_datetime(df["starttime"])
    df["endtime"] = _to_datetime(df["endtime"])
    df = df.merge(bridge, on="icustay_id", how="inner", validate="many_to_one")

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
) -> pd.DataFrame:
    """Compute admission-level ventilation and vasopressor totals in minutes."""

    _require_columns(icustays, ["hadm_id", "icustay_id", "intime", "outtime"], "icustays")

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
) -> pd.DataFrame:
    """Aggregate non-error notes into one concatenated note per admission."""

    _require_columns(noteevents, ["hadm_id", "text", "iserror"], "noteevents")

    notes = noteevents.copy()
    notes = _filter_non_error_notes(notes)
    notes["text"] = notes["text"].map(prepare_note_text_for_sentiment)

    grouped = (
        notes.groupby("hadm_id", sort=True)["text"]
        .apply(lambda series: prepare_note_text_for_sentiment(" ".join(t for t in series if t)))
        .reset_index(name="note_text")
    )

    if all_hadm_ids is not None:
        hadm_frame = pd.DataFrame({"hadm_id": list(all_hadm_ids)})
        grouped = hadm_frame.merge(grouped, on="hadm_id", how="left")

    grouped["note_text"] = grouped["note_text"].fillna("")
    grouped = grouped.sort_values("hadm_id").drop_duplicates("hadm_id")
    return grouped.reset_index(drop=True)


def _build_note_labels_from_corpus(note_corpus: pd.DataFrame) -> pd.DataFrame:
    """Create the two note-derived labels from an admission-level note corpus."""

    _require_columns(note_corpus, ["hadm_id", "note_text"], "note_corpus")
    lowered = note_corpus["note_text"].fillna("").astype(str).str.lower()
    noncompliance = lowered.apply(
        lambda text: int(any(pattern in text for pattern in NONCOMPLIANCE_PATTERNS))
    )
    autopsy = lowered.apply(lambda text: int("autopsy" in text))

    labels = pd.DataFrame(
        {
            "hadm_id": note_corpus["hadm_id"],
            "noncompliance_label": noncompliance.astype(int),
            "autopsy_label": autopsy.astype(int),
        }
    )
    labels = labels.sort_values("hadm_id").drop_duplicates("hadm_id")
    return labels.reset_index(drop=True)


def build_note_labels(
    noteevents: pd.DataFrame,
    all_hadm_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Create admission-level noncompliance and autopsy labels from notes."""

    _require_columns(noteevents, ["hadm_id", "text", "iserror"], "noteevents")
    corpus = build_note_corpus(noteevents, all_hadm_ids=all_hadm_ids)
    return _build_note_labels_from_corpus(corpus)


def build_note_artifacts_from_csv(
    noteevents_csv_path: Path | str,
    all_hadm_ids: Iterable[int] | None = None,
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the note corpus and note-derived labels from a large CSV in chunks."""

    normalized_hadm_ids = _normalize_hadm_ids(all_hadm_ids)
    hadm_filter = set(normalized_hadm_ids) if normalized_hadm_ids is not None else None
    note_fragments: dict[int, list[str]] = defaultdict(list)

    for chunk in _iter_csv_chunks(
        noteevents_csv_path,
        required_columns=["hadm_id", "text", "iserror"],
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

        chunk["text"] = chunk["text"].map(prepare_note_text_for_sentiment)
        chunk = chunk.loc[chunk["text"] != ""]
        if chunk.empty:
            continue

        grouped = (
            chunk.groupby("hadm_id", sort=False)["text"]
            .apply(lambda series: prepare_note_text_for_sentiment(" ".join(series)))
        )
        for hadm_id, text in grouped.items():
            if text:
                note_fragments[int(hadm_id)].append(text)

    if normalized_hadm_ids is not None:
        hadm_ids = normalized_hadm_ids
    else:
        hadm_ids = sorted(note_fragments)

    corpus = pd.DataFrame(
        {
            "hadm_id": hadm_ids,
            "note_text": [
                prepare_note_text_for_sentiment(" ".join(note_fragments.get(hadm_id, [])))
                for hadm_id in hadm_ids
            ],
        }
    )
    corpus = corpus.sort_values("hadm_id").drop_duplicates("hadm_id").reset_index(drop=True)
    labels = _build_note_labels_from_corpus(corpus)
    return corpus, labels


def build_note_corpus_from_csv(
    noteevents_csv_path: Path | str,
    all_hadm_ids: Iterable[int] | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Build the admission-level note corpus from a large CSV in chunks."""

    corpus, _ = build_note_artifacts_from_csv(
        noteevents_csv_path=noteevents_csv_path,
        all_hadm_ids=all_hadm_ids,
        chunksize=chunksize,
    )
    return corpus


def build_note_labels_from_csv(
    noteevents_csv_path: Path | str,
    all_hadm_ids: Iterable[int] | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Build note-derived labels from a large CSV in chunks."""

    _, labels = build_note_artifacts_from_csv(
        noteevents_csv_path=noteevents_csv_path,
        all_hadm_ids=all_hadm_ids,
        chunksize=chunksize,
    )
    return labels


def identify_table2_itemids(d_items: pd.DataFrame) -> set[int]:
    """Identify chart itemids that match the paper's Table 2 concepts."""

    _require_columns(d_items, ["itemid", "label", "dbsource"], "d_items")
    matches = d_items["label"].map(_matches_table2_concept)
    return set(d_items.loc[matches, "itemid"].tolist())


def build_chartevent_artifacts_from_csv(
    chartevents_csv_path: Path | str,
    d_items: pd.DataFrame,
    allowed_labels: Iterable[str] | None = None,
    all_hadm_ids: Iterable[int] | None = None,
    chunksize: int = 500_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the feature matrix and code-status targets from a large CSV in chunks."""

    _require_columns(d_items, ["itemid", "label", "dbsource"], "d_items")

    items = d_items.copy()
    items["normalized_label"] = items["label"].map(_normalize_token)
    if allowed_labels is not None:
        allowed = {_normalize_token(label) for label in allowed_labels}
        items = items.loc[items["normalized_label"].isin(allowed)].copy()
    else:
        allowed_itemids = identify_table2_itemids(items)
        items = items.loc[items["itemid"].isin(allowed_itemids)].copy()

    items["itemid"] = pd.to_numeric(items["itemid"], errors="coerce")
    items = items.dropna(subset=["itemid"]).copy()
    items["itemid"] = items["itemid"].astype(int)

    feature_lookup = (
        items[["itemid", "label"]]
        .drop_duplicates("itemid")
        .set_index("itemid")["label"]
        .to_dict()
    )
    feature_itemids = set(feature_lookup)
    relevant_itemids = feature_itemids | set(CODE_STATUS_ITEMIDS)
    normalized_hadm_ids = _normalize_hadm_ids(all_hadm_ids)
    hadm_filter = set(normalized_hadm_ids) if normalized_hadm_ids is not None else None

    feature_to_hadm: dict[str, set[int]] = defaultdict(set)
    code_status_positive: dict[int, int] = {}

    for chunk in _iter_csv_chunks(
        chartevents_csv_path,
        required_columns=["hadm_id", "itemid", "value", "icustay_id"],
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
        if chunk.empty:
            continue

        feature_chunk = chunk.loc[chunk["itemid"].isin(feature_itemids)].copy()
        if not feature_chunk.empty:
            feature_chunk["label"] = feature_chunk["itemid"].map(feature_lookup)
            feature_chunk["normalized_value"] = feature_chunk["value"].map(_normalize_token)
            feature_chunk["display_label"] = feature_chunk["label"].map(_clean_feature_text)
            feature_chunk["display_value"] = feature_chunk["value"].map(_clean_feature_text)
            feature_chunk = feature_chunk.loc[
                (feature_chunk["normalized_value"] != "")
                & (feature_chunk["display_label"] != "")
            ].copy()
            if not feature_chunk.empty:
                feature_chunk["feature_name"] = (
                    feature_chunk["display_label"] + ": " + feature_chunk["display_value"]
                )
                unique_pairs = feature_chunk[["hadm_id", "feature_name"]].drop_duplicates()
                for feature_name, group in unique_pairs.groupby("feature_name", sort=False):
                    feature_to_hadm[str(feature_name)].update(group["hadm_id"].astype(int).tolist())

        code_chunk = chunk.loc[chunk["itemid"].isin(CODE_STATUS_ITEMIDS)].copy()
        if not code_chunk.empty:
            normalized_value = code_chunk["value"].map(_normalize_token)
            positives = normalized_value.apply(
                lambda value: int(
                    ("dnr" in value)
                    or ("dni" in value)
                    or ("comfort" in value)
                    or ("cmo" in value)
                )
            )
            for hadm_id, is_positive in zip(code_chunk["hadm_id"].astype(int), positives):
                code_status_positive[hadm_id] = max(
                    code_status_positive.get(hadm_id, 0),
                    int(is_positive),
                )

    if normalized_hadm_ids is not None:
        hadm_ids = normalized_hadm_ids
    else:
        hadm_ids = sorted(set().union(*feature_to_hadm.values())) if feature_to_hadm else []

    feature_names = sorted(feature_to_hadm)
    feature_data: dict[str, object] = {"hadm_id": hadm_ids}
    hadm_index = pd.Index(hadm_ids)
    for feature_name in feature_names:
        feature_data[feature_name] = hadm_index.isin(feature_to_hadm[feature_name]).astype(int)
    feature_matrix = pd.DataFrame(feature_data)
    if "hadm_id" not in feature_matrix.columns:
        feature_matrix = pd.DataFrame(columns=["hadm_id"])
    feature_matrix = (
        feature_matrix.sort_values("hadm_id")
        .drop_duplicates("hadm_id")
        .reset_index(drop=True)
    )

    code_status_targets = pd.DataFrame(
        {
            "hadm_id": sorted(code_status_positive),
            "code_status_dnr_dni_cmo": [
                int(code_status_positive[hadm_id]) for hadm_id in sorted(code_status_positive)
            ],
        }
    )
    code_status_targets = (
        code_status_targets.sort_values("hadm_id")
        .drop_duplicates("hadm_id")
        .reset_index(drop=True)
    )
    return feature_matrix, code_status_targets


def build_chartevent_feature_matrix(
    chartevents: pd.DataFrame,
    d_items: pd.DataFrame,
    allowed_labels: Iterable[str] | None = None,
    all_hadm_ids: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Build a binary admission-by-feature matrix from selected chart events."""

    _require_columns(chartevents, ["hadm_id", "itemid", "value", "icustay_id"], "chartevents")
    _require_columns(d_items, ["itemid", "label", "dbsource"], "d_items")

    events = chartevents.copy()
    items = d_items.copy()
    items["normalized_label"] = items["label"].map(_normalize_token)

    if allowed_labels is not None:
        allowed = {_normalize_token(label) for label in allowed_labels}
        items = items.loc[items["normalized_label"].isin(allowed)].copy()
    else:
        allowed_itemids = identify_table2_itemids(items)
        items = items.loc[items["itemid"].isin(allowed_itemids)].copy()

    merged = events.merge(
        items[["itemid", "label", "normalized_label"]],
        on="itemid",
        how="inner",
        validate="many_to_one",
    )
    merged["normalized_value"] = merged["value"].map(_normalize_token)
    merged["display_label"] = merged["label"].map(_clean_feature_text)
    merged["display_value"] = merged["value"].map(_clean_feature_text)
    merged = merged.loc[
        (merged["normalized_value"] != "") & (merged["display_label"] != "")
    ].copy()

    if merged.empty:
        result = pd.DataFrame(columns=["hadm_id"])
    else:
        merged["feature_name"] = merged["display_label"] + ": " + merged["display_value"]
        pivot = (
            merged.assign(feature_value=1)
            .pivot_table(
                index="hadm_id",
                columns="feature_name",
                values="feature_value",
                aggfunc="max",
                fill_value=0,
            )
            .reset_index()
        )
        pivot.columns.name = None
        result = pivot

    if all_hadm_ids is not None:
        hadm_frame = pd.DataFrame({"hadm_id": list(all_hadm_ids)})
        result = hadm_frame.merge(result, on="hadm_id", how="left")

    if "hadm_id" not in result.columns:
        result = pd.DataFrame(columns=["hadm_id"])

    feature_cols = [col for col in result.columns if col != "hadm_id"]
    if feature_cols:
        result[feature_cols] = result[feature_cols].fillna(0).astype(int)
    result = result.sort_values("hadm_id").drop_duplicates("hadm_id")
    return result.reset_index(drop=True)


def build_chartevent_feature_matrix_from_csv(
    chartevents_csv_path: Path | str,
    d_items: pd.DataFrame,
    allowed_labels: Iterable[str] | None = None,
    all_hadm_ids: Iterable[int] | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Build the binary feature matrix from a large chartevents CSV in chunks."""

    feature_matrix, _ = build_chartevent_artifacts_from_csv(
        chartevents_csv_path=chartevents_csv_path,
        d_items=d_items,
        allowed_labels=allowed_labels,
        all_hadm_ids=all_hadm_ids,
        chunksize=chunksize,
    )
    return feature_matrix


def z_normalize_scores(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Apply independent z-score normalization to the requested score columns."""

    normalized = df.copy()
    for column in columns:
        _require_columns(normalized, [column], "score_table")
        values = normalized[column].astype(float)
        mean = values.mean()
        std = values.std(ddof=0)
        if pd.isna(std) or std == 0:
            normalized[column] = 0.0
        else:
            normalized[column] = (values - mean) / std
    return normalized


def build_acuity_scores(oasis: pd.DataFrame, sapsii: pd.DataFrame) -> pd.DataFrame:
    """Aggregate OASIS and SAPS II to one admission-level row per hadm_id."""

    _require_columns(oasis, ["hadm_id", "icustay_id", "oasis"], "oasis")
    _require_columns(sapsii, ["hadm_id", "icustay_id", "sapsii"], "sapsii")

    oasis_agg = oasis.groupby("hadm_id", as_index=False)["oasis"].max()
    sapsii_agg = sapsii.groupby("hadm_id", as_index=False)["sapsii"].max()
    acuity = oasis_agg.merge(sapsii_agg, on="hadm_id", how="outer")
    acuity = acuity.sort_values("hadm_id").drop_duplicates("hadm_id")
    return acuity.reset_index(drop=True)


def build_proxy_probability_scores(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    label_column: str,
    estimator_factory: Callable[[], object] | None = None,
) -> pd.DataFrame:
    """Fit the proxy label model and return positive-class probabilities."""

    _require_columns(feature_matrix, ["hadm_id"], "feature_matrix")
    _require_columns(note_labels, ["hadm_id", label_column], "note_labels")

    feature_columns = [column for column in feature_matrix.columns if column != "hadm_id"]
    merged = feature_matrix.merge(
        note_labels[["hadm_id", label_column]],
        on="hadm_id",
        how="inner",
        validate="one_to_one",
    ).sort_values("hadm_id")

    feature_values = merged[feature_columns]
    y = merged[label_column].astype(int)

    if estimator_factory is None:
        estimator = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
    else:
        estimator = estimator_factory()

    estimator.fit(feature_values, y)
    probabilities = estimator.predict_proba(feature_values)
    score_column = (
        f"{label_column[:-6]}_score" if label_column.endswith("_label") else f"{label_column}_score"
    )

    scores = pd.DataFrame(
        {
            "hadm_id": merged["hadm_id"].tolist(),
            score_column: _extract_positive_class_probabilities(probabilities),
        }
    )
    scores = scores.sort_values("hadm_id").drop_duplicates("hadm_id")
    return scores.reset_index(drop=True)


def build_negative_sentiment_scores(
    note_corpus: pd.DataFrame,
    sentiment_fn: Callable[[str], tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Convert note sentiment polarity into an admission-level mistrust score."""

    _require_columns(note_corpus, ["hadm_id", "note_text"], "note_corpus")

    if sentiment_fn is None:
        sentiment_fn = pattern_sentiment

    rows = []
    for row in note_corpus.sort_values("hadm_id").itertuples(index=False):
        text = prepare_note_text_for_sentiment(row.note_text)
        if text == "":
            score = 0.0
        else:
            polarity, _ = sentiment_fn(text)
            score = -1.0 * float(polarity)
        rows.append({"hadm_id": row.hadm_id, "negative_sentiment_score": score})

    scores = pd.DataFrame(rows).sort_values("hadm_id").drop_duplicates("hadm_id")
    return scores.reset_index(drop=True)


def build_mistrust_score_table(
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    note_corpus: pd.DataFrame,
    estimator_factory: Callable[[], object] | None = None,
    sentiment_fn: Callable[[str], tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Build and normalize the three admission-level mistrust score vectors."""

    _require_columns(feature_matrix, ["hadm_id"], "feature_matrix")
    _require_columns(
        note_labels,
        ["hadm_id", "noncompliance_label", "autopsy_label"],
        "note_labels",
    )
    _require_columns(note_corpus, ["hadm_id", "note_text"], "note_corpus")

    noncompliance_scores = build_proxy_probability_scores(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        label_column="noncompliance_label",
        estimator_factory=estimator_factory,
    )
    autopsy_scores = build_proxy_probability_scores(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        label_column="autopsy_label",
        estimator_factory=estimator_factory,
    )
    negative_sentiment_scores = build_negative_sentiment_scores(
        note_corpus,
        sentiment_fn=sentiment_fn,
    )

    merged = (
        noncompliance_scores.merge(autopsy_scores, on="hadm_id", how="inner")
        .merge(negative_sentiment_scores, on="hadm_id", how="inner")
        .sort_values("hadm_id")
    )
    normalized = z_normalize_scores(
        merged,
        columns=[
            "noncompliance_score",
            "autopsy_score",
            "negative_sentiment_score",
        ],
    )
    normalized = normalized.rename(
        columns={
            "noncompliance_score": "noncompliance_score_z",
            "autopsy_score": "autopsy_score_z",
            "negative_sentiment_score": "negative_sentiment_score_z",
        }
    )
    normalized = normalized.sort_values("hadm_id").drop_duplicates("hadm_id")
    return normalized.reset_index(drop=True)


def _build_gender_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({"hadm_id": df["hadm_id"]})
    gender = df["gender"].fillna("").str.upper()
    output["gender_f"] = (gender == "F").astype(int)
    output["gender_m"] = (gender == "M").astype(int)
    return output


def _build_insurance_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({"hadm_id": df["hadm_id"]})
    insurance_column = "insurance_group" if "insurance_group" in df.columns else "insurance"
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


def _build_code_status_target(chartevents: pd.DataFrame, d_items: pd.DataFrame) -> pd.DataFrame:
    _require_columns(chartevents, ["hadm_id", "itemid", "value", "icustay_id"], "chartevents")
    _require_columns(d_items, ["itemid", "label", "dbsource"], "d_items")
    return _build_task_code_status_target(chartevents, itemids=CODE_STATUS_ITEMIDS)


def build_code_status_target_from_csv(
    chartevents_csv_path: Path | str,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Build the code-status target from a large chartevents CSV in chunks."""

    _, code_status_targets = build_chartevent_artifacts_from_csv(
        chartevents_csv_path=chartevents_csv_path,
        d_items=pd.DataFrame(columns=["itemid", "label", "dbsource"]),
        all_hadm_ids=None,
        chunksize=chunksize,
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
        ["hadm_id", "discharge_location", "hospital_expire_flag"],
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
        {"hadm_id": sorted(pd.to_numeric(all_cohort["hadm_id"], errors="coerce").dropna().astype(int).unique())}
    )
    demo = cohort_hadm.merge(demographics, on="hadm_id", how="left")

    final = cohort_hadm.copy()
    final = final.merge(
        demo[["hadm_id", "age", "los_days"]],
        on="hadm_id",
        how="left",
    )
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
    ).fillna(0).astype(int)

    fill_zero_columns = [
        "gender_f",
        "gender_m",
        "insurance_private",
        "insurance_public",
        "insurance_self_pay",
        "left_ama",
        "code_status_dnr_dni_cmo",
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
    """Assemble baseline, optional race, mistrust, and target columns."""
    code_status = _build_code_status_target(chartevents, d_items)
    return _assemble_final_model_table(
        demographics=demographics,
        all_cohort=all_cohort,
        admissions=admissions,
        code_status=code_status,
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


def write_minimal_deliverables(artifacts: dict[str, pd.DataFrame], output_dir: Path | str) -> None:
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
