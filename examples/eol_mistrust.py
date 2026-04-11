r"""Example workflow for the EOL mistrust study pipeline.

This script assumes you have already exported and combined the required MIMIC-III
tables into a local directory such as:

    EOL_Workspace/eol_mistrust_required_combined/
        mimiciii_clinical/
        mimiciii_notes/
        mimiciii_derived/

It demonstrates two related flows:

1. the study-style preprocessing + modeling pipeline built on pandas tables
2. an optional PyHealth task demo using the custom EOL mistrust YAML config

Implementation note: the sentiment metric in this repo uses the existing
transformers+torch stack rather than the original Pattern backend from the
reference notebooks. The example still follows the paper-style note scope by
building both the sentiment corpus and note-derived labels from all non-error
notes.

Recommended commands
--------------------
Formal managed runs (recommended)

The script now creates a managed run archive under
``EOL_Workspace/EOL_Result/EOL_(normal|Paperlike)_<timestamp>/``.
When ``--output-dir`` and ``--stream-cache-dir`` are omitted, deliverables,
runtime files, and stage cache directories are created automatically inside
that managed run folder.

Default / corrected pipeline

Formal cold-start run:
.\.venv\Scripts\python.exe examples\eol_mistrust.py --root EOL_Workspace\eol_mistrust_required_combined --compare-to-paper --repetitions 10

Formal smoke run:
.\.venv\Scripts\python.exe examples\eol_mistrust.py --root EOL_Workspace\eol_mistrust_required_combined --compare-to-paper --repetitions 1

Paper-like dataset preparation

Formal cold-start run:
.\.venv\Scripts\python.exe examples\eol_mistrust.py --root EOL_Workspace\eol_mistrust_required_combined --compare-to-paper --paper-like-dataset-prepare --repetitions 10

Formal smoke run:
.\.venv\Scripts\python.exe examples\eol_mistrust.py --root EOL_Workspace\eol_mistrust_required_combined --compare-to-paper --paper-like-dataset-prepare --repetitions 1

Optional fast reruns with shared cache

Default / corrected pipeline:
.\.venv\Scripts\python.exe examples\eol_mistrust.py --root EOL_Workspace\eol_mistrust_required_combined --stream-cache-dir EOL_Workspace --reuse-intermediates EOL_Workspace --compare-to-paper --repetitions 10
.\.venv\Scripts\python.exe examples\eol_mistrust.py --root EOL_Workspace\eol_mistrust_required_combined --stream-cache-dir EOL_Workspace --reuse-intermediates EOL_Workspace --compare-to-paper --repetitions 1

Paper-like dataset preparation:
.\.venv\Scripts\python.exe examples\eol_mistrust.py --root EOL_Workspace\eol_mistrust_required_combined --stream-cache-dir EOL_Workspace --reuse-intermediates EOL_Workspace --compare-to-paper --paper-like-dataset-prepare --repetitions 10
.\.venv\Scripts\python.exe examples\eol_mistrust.py --root EOL_Workspace\eol_mistrust_required_combined --stream-cache-dir EOL_Workspace --reuse-intermediates EOL_Workspace --compare-to-paper --paper-like-dataset-prepare --repetitions 1

Optional custom managed-run archive root:
.\.venv\Scripts\python.exe examples\eol_mistrust.py --root EOL_Workspace\eol_mistrust_required_combined --result-root EOL_Workspace\EOL_Result --compare-to-paper --repetitions 10



"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "EOL_Workspace" / "eol_mistrust_required_combined"
DEFAULT_CONFIG_PATH = REPO_ROOT / "pyhealth" / "datasets" / "configs" / "eol_mistrust.yaml"
DEFAULT_RESULT_ROOT = REPO_ROOT / "EOL_Workspace" / "EOL_Result"


def _load_local_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DATASET_MODULE = _load_local_module(
    "pyhealth_datasets_eol_mistrust_example_local",
    "pyhealth/datasets/eol_mistrust.py",
)
_MODEL_MODULE = _load_local_module(
    "pyhealth_models_eol_mistrust_example_local",
    "pyhealth/models/eol_mistrust.py",
)

build_acuity_scores = _DATASET_MODULE.build_acuity_scores
build_all_cohort = _DATASET_MODULE.build_all_cohort
build_base_admissions = _DATASET_MODULE.build_base_admissions
build_chartevent_artifacts_from_csv = _DATASET_MODULE.build_chartevent_artifacts_from_csv
build_demographics_table = _DATASET_MODULE.build_demographics_table
build_eol_cohort = _DATASET_MODULE.build_eol_cohort
build_final_model_table_from_code_status_targets = _DATASET_MODULE.build_final_model_table_from_code_status_targets
build_note_corpus_from_csv = _DATASET_MODULE.build_note_corpus_from_csv
build_note_labels_from_csv = _DATASET_MODULE.build_note_labels_from_csv
build_treatment_totals = _DATASET_MODULE.build_treatment_totals
validate_database_environment = _DATASET_MODULE.validate_database_environment
write_minimal_deliverables = _DATASET_MODULE.write_minimal_deliverables

EOLMistrustModel = _MODEL_MODULE.EOLMistrustModel
evaluate_downstream_average_weights = _MODEL_MODULE.evaluate_downstream_average_weights
build_autopsy_mistrust_scores = _MODEL_MODULE.build_autopsy_mistrust_scores
build_logistic_cv_estimator_factory = _MODEL_MODULE.build_logistic_cv_estimator_factory
build_negative_sentiment_mistrust_scores = _MODEL_MODULE.build_negative_sentiment_mistrust_scores
build_noncompliance_mistrust_scores = _MODEL_MODULE.build_noncompliance_mistrust_scores
get_downstream_feature_configurations = _MODEL_MODULE.get_downstream_feature_configurations
z_normalize_scores = _MODEL_MODULE.z_normalize_scores

MIMIC3Dataset = None
EOLMistrustMortalityPredictionMIMIC3 = None

RAW_TABLE_PATHS = {
    "admissions": "mimiciii_clinical/admissions.csv",
    "patients": "mimiciii_clinical/patients.csv",
    "icustays": "mimiciii_clinical/icustays.csv",
    "d_items": "mimiciii_clinical/d_items.csv",
}

EVENT_TABLE_PATHS = {
    "noteevents": "mimiciii_notes/noteevents.csv",
    "chartevents": "mimiciii_clinical/chartevents.csv",
}

MATERIALIZED_VIEW_PATHS = {
    "ventdurations": "mimiciii_derived/ventdurations.csv",
    "vasopressordurations": "mimiciii_derived/vasopressordurations.csv",
    "oasis": "mimiciii_derived/oasis.csv",
    "sapsii": "mimiciii_derived/sapsii.csv",
}

VALIDATION_EVENT_PROBE_ROWS = 50_000

PAPER_URL = "https://proceedings.mlr.press/v85/boag18a.html"
PAPER_PDF_URL = "https://proceedings.mlr.press/v85/boag18a/boag18a.pdf"

PAPER_TABLE1_COUNTS = {
    "Population Size": {"BLACK": 1214, "WHITE": 9987},
    "Insurance Private": {"BLACK": 141, "WHITE": 1594},
    "Insurance Public": {"BLACK": 1062, "WHITE": 8356},
    "Insurance Self-Pay": {"BLACK": 11, "WHITE": 37},
    "Discharge Deceased": {"BLACK": 401, "WHITE": 3869},
    "Discharge Hospice": {"BLACK": 40, "WHITE": 421},
    "Discharge Skilled Nursing Facility": {"BLACK": 773, "WHITE": 5697},
    "Gender F": {"BLACK": 733, "WHITE": 5012},
    "Gender M": {"BLACK": 481, "WHITE": 4975},
}

PAPER_TABLE1_CONTINUOUS = {
    "Length of stay (median days)": {
        "BLACK": {"center": 13.90, "lower": 5.55, "upper": 19.56},
        "WHITE": {"center": 14.08, "lower": 6.45, "upper": 19.45},
    },
    "Age (median years)": {
        "BLACK": {"center": 71.31, "lower": 60.21, "upper": 80.36},
        "WHITE": {"center": 77.87, "lower": 66.61, "upper": 84.93},
    },
}

PAPER_TABLE2_TREATMENT = {
    "total_vent_min": {
        "n_black": 510,
        "n_white": 4810,
        "median_black": 3180.0,
        "median_white": 2520.0,
        "pvalue": 0.005,
    },
    "total_vaso_min": {
        "n_black": 453,
        "n_white": 4456,
        "median_black": 2046.0,
        "median_white": 1770.0,
        "pvalue": 0.12,
    },
}

PAPER_TABLE3_WEIGHTS = {
    "noncompliance": {
        "positive": [
            ("riker-sas scale: agitated", 0.7013),
            ("education readiness: no", 0.2540),
            ("pain level: 7-mod to severe", 0.2168),
        ],
        "negative": [
            ("state: alert", -1.0156),
            ("pain: none", -0.5427),
            ("richmond-ras scale: 0 alert and calm", -0.3598),
        ],
    },
    "autopsy": {
        "positive": [
            ("reapplied restraints", 0.1153),
            ("restraint type: soft limb", 0.0980),
            ("orientation: oriented 3x", 0.0363),
        ],
        "negative": [
            ("pain present: no", -0.2689),
            ("spokesperson is healthcare proxy", -0.2271),
            ("family communication: talked to m.d.", -0.1184),
        ],
    },
}

PAPER_TABLE3_FEATURE_ALIASES = {
    "autopsy": {
        "reapplied restraints": (
            "restraints evaluated: restraintreapply",
            "restraints evaluated: reapplied",
            "restraints evaluated v1: restraint reapplied",
            "restraints evaluated v2: reapplied",
        ),
        "orientation: oriented 3x": (
            "orientation: oriented x 3",
            "orientation: oriented x3",
        ),
        "spokesperson is healthcare proxy": (
            "is the spokesperson the health care proxy: 1",
        ),
        "family communication: talked to m.d.": (
            "family communication: family talked to md",
            "family communication: fam talked to md",
        ),
    },
}

PAPER_TABLE4_CORRELATIONS = {
    tuple(sorted(("oasis", "sapsii"))): 0.679,
    tuple(sorted(("oasis", "noncompliance_score_z"))): 0.050,
    tuple(sorted(("oasis", "autopsy_score_z"))): -0.012,
    tuple(sorted(("oasis", "negative_sentiment_score_z"))): 0.075,
    tuple(sorted(("sapsii", "noncompliance_score_z"))): 0.013,
    tuple(sorted(("sapsii", "autopsy_score_z"))): -0.013,
    tuple(sorted(("sapsii", "negative_sentiment_score_z"))): 0.086,
    tuple(sorted(("noncompliance_score_z", "autopsy_score_z"))): 0.262,
    tuple(sorted(("noncompliance_score_z", "negative_sentiment_score_z"))): 0.058,
    tuple(sorted(("autopsy_score_z", "negative_sentiment_score_z"))): 0.044,
}

PAPER_TABLE5_AUC = {
    ("Left AMA", "Baseline"): {"n_rows": 48071, "auc_mean": 0.859, "auc_std": 0.014},
    ("Left AMA", "Baseline + Race"): {"n_rows": 48071, "auc_mean": 0.861, "auc_std": 0.014},
    ("Left AMA", "Baseline + Noncompliant"): {"n_rows": 48071, "auc_mean": 0.869, "auc_std": 0.012},
    ("Left AMA", "Baseline + Autopsy"): {"n_rows": 48071, "auc_mean": 0.861, "auc_std": 0.012},
    ("Left AMA", "Baseline + Neg-Sentiment"): {"n_rows": 48071, "auc_mean": 0.859, "auc_std": 0.013},
    ("Left AMA", "Baseline + ALL"): {"n_rows": 48071, "auc_mean": 0.873, "auc_std": 0.012},
    ("Code Status", "Baseline"): {"n_rows": 39815, "auc_mean": 0.763, "auc_std": 0.013},
    ("Code Status", "Baseline + Race"): {"n_rows": 39815, "auc_mean": 0.766, "auc_std": 0.014},
    ("Code Status", "Baseline + Noncompliant"): {"n_rows": 39815, "auc_mean": 0.767, "auc_std": 0.013},
    ("Code Status", "Baseline + Autopsy"): {"n_rows": 39815, "auc_mean": 0.773, "auc_std": 0.011},
    ("Code Status", "Baseline + Neg-Sentiment"): {"n_rows": 39815, "auc_mean": 0.765, "auc_std": 0.014},
    ("Code Status", "Baseline + ALL"): {"n_rows": 39815, "auc_mean": 0.782, "auc_std": 0.012},
    ("In-hospital mortality", "Baseline"): {"n_rows": 48071, "auc_mean": 0.600, "auc_std": 0.011},
    ("In-hospital mortality", "Baseline + Race"): {"n_rows": 48071, "auc_mean": 0.614, "auc_std": 0.011},
    ("In-hospital mortality", "Baseline + Noncompliant"): {"n_rows": 48071, "auc_mean": 0.614, "auc_std": 0.010},
    ("In-hospital mortality", "Baseline + Autopsy"): {"n_rows": 48071, "auc_mean": 0.603, "auc_std": 0.012},
    ("In-hospital mortality", "Baseline + Neg-Sentiment"): {"n_rows": 48071, "auc_mean": 0.615, "auc_std": 0.010},
    ("In-hospital mortality", "Baseline + ALL"): {"n_rows": 48071, "auc_mean": 0.635, "auc_std": 0.010},
}

PAPER_TABLE6_WEIGHTS = {
    "Left AMA": {
        "noncompliant": (0.52, 0.09),
        "autopsy": (0.01, 0.03),
        "negative sentiment": (0.00, 0.02),
        "race: asian": (0.00, 0.00),
        "race: black": (0.03, 0.12),
        "race: hispanic": (0.00, 0.00),
        "race: other": (-0.15, 0.19),
        "race: white": (-0.02, 0.06),
        "race: native american": (0.00, 0.00),
        "gender: male": (0.00, 0.00),
        "gender: female": (-0.40, 0.20),
        "insurance: private": (-1.01, 0.21),
        "insurance: public": (0.00, 0.00),
        "insurance: self-pay": (0.00, 0.00),
        "length-of-stay": (-1.44, 0.37),
        "age": (-2.10, 0.21),
    },
    "Code Status": {
        "noncompliant": (0.27, 0.04),
        "autopsy": (-0.44, 0.05),
        "negative sentiment": (0.09, 0.03),
        "race: asian": (0.00, 0.00),
        "race: black": (-0.22, 0.19),
        "race: hispanic": (-0.17, 0.21),
        "race: other": (-0.12, 0.17),
        "race: white": (0.06, 0.15),
        "race: native american": (0.00, 0.00),
        "gender: male": (-0.85, 1.40),
        "gender: female": (-0.49, 1.39),
        "insurance: private": (-0.94, 0.29),
        "insurance: public": (-0.02, 0.28),
        "insurance: self-pay": (-0.02, 0.24),
        "length-of-stay": (-0.70, 0.10),
        "age": (0.42, 0.02),
    },
    "In-hospital mortality": {
        "noncompliant": (0.16, 0.03),
        "autopsy": (0.02, 0.02),
        "negative sentiment": (0.16, 0.03),
        "race: asian": (-0.05, 0.03),
        "race: black": (-0.53, 0.31),
        "race: hispanic": (-0.58, 0.34),
        "race: other": (0.15, 0.30),
        "race: white": (-0.26, 0.30),
        "race: native american": (0.00, 0.00),
        "gender: male": (-0.67, 0.99),
        "gender: female": (-0.59, 0.99),
        "insurance: private": (-0.96, 0.95),
        "insurance: public": (-0.50, 0.95),
        "insurance: self-pay": (-0.21, 0.68),
        "length-of-stay": (0.08, 0.03),
        "age": (0.20, 0.02),
    },
}

TABLE6_FEATURE_NAME_MAP = {
    "noncompliance_score_z": "noncompliant",
    "autopsy_score_z": "autopsy",
    "negative_sentiment_score_z": "negative sentiment",
    "race_asian": "race: asian",
    "race_black": "race: black",
    "race_hispanic": "race: hispanic",
    "race_other": "race: other",
    "race_white": "race: white",
    "race_native_american": "race: native american",
    "gender_m": "gender: male",
    "gender_f": "gender: female",
    "insurance_private": "insurance: private",
    "insurance_public": "insurance: public",
    "insurance_self_pay": "insurance: self-pay",
    "los_days": "length-of-stay",
    "age": "age",
}


def _read_csvs(root: Path, path_map: dict[str, str]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, relative_path in path_map.items():
        csv_path = root / relative_path
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required table for EOL example: {csv_path}")
        table = pd.read_csv(csv_path, low_memory=False)
        table.columns = [str(column).lower() for column in table.columns]
        tables[name] = table
    return tables


def load_eol_mistrust_tables(
    root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load the raw tables and materialized views required by the pipeline."""

    raw_tables = _read_csvs(root, RAW_TABLE_PATHS)
    materialized_views = _read_csvs(root, MATERIALIZED_VIEW_PATHS)
    return raw_tables, materialized_views


def _read_csv_probe(
    root: Path,
    relative_path: str,
    *,
    nrows: int = VALIDATION_EVENT_PROBE_ROWS,
) -> pd.DataFrame:
    """Load a lightweight probe frame for validation of large event CSVs."""

    csv_path = root / relative_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required table for EOL example: {csv_path}")
    table = pd.read_csv(csv_path, low_memory=False, nrows=nrows)
    table.columns = [str(column).lower() for column in table.columns]
    return table


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def _has_columns(frame: object, required_columns: set[str]) -> bool:
    """Return True when *frame* is a DataFrame containing all required columns."""

    return isinstance(frame, pd.DataFrame) and required_columns.issubset(frame.columns)


def _format_count_percent(count: int, total: int) -> str:
    if total <= 0:
        return str(int(count))
    return f"{int(count)} ({100.0 * float(count) / float(total):.2f}%)"


def _format_continuous_summary(center: float, lower: float, upper: float) -> str:
    return f"{center:.2f} [{lower:.2f}, {upper:.2f}]"


def _note_present_hadm_ids(note_corpus: pd.DataFrame) -> list[int]:
    """Return sorted admission ids with at least one non-empty aggregated note."""

    hadm_ids = pd.to_numeric(
        note_corpus.loc[note_corpus["note_text"].fillna("").astype(str).str.strip() != "", "hadm_id"],
        errors="coerce",
    )
    return sorted(hadm_ids.dropna().astype(int).unique().tolist())


def build_paper_table1_comparison(eol_cohort: pd.DataFrame) -> pd.DataFrame:
    """Compare the run EOL cohort demographics against Table 1 from the paper."""

    cohort = eol_cohort[eol_cohort["race"].isin(["BLACK", "WHITE"])].copy()
    totals = {race: int((cohort["race"] == race).sum()) for race in ("BLACK", "WHITE")}
    rows: list[dict[str, object]] = []

    metric_specs = [
        ("Population Size", None, None),
        ("Insurance Private", "insurance_group", "Private"),
        ("Insurance Public", "insurance_group", "Public"),
        ("Insurance Self-Pay", "insurance_group", "Self-Pay"),
        ("Discharge Deceased", "discharge_category", "Deceased"),
        ("Discharge Hospice", "discharge_category", "Hospice"),
        ("Discharge Skilled Nursing Facility", "discharge_category", "Skilled Nursing Facility"),
        ("Gender F", "gender", "F"),
        ("Gender M", "gender", "M"),
    ]
    for metric, column, target_value in metric_specs:
        for race in ("BLACK", "WHITE"):
            race_frame = cohort[cohort["race"] == race]
            if column is None:
                run_numeric = int(len(race_frame))
                run_display = str(run_numeric)
            else:
                run_numeric = int((race_frame[column] == target_value).sum())
                run_display = _format_count_percent(run_numeric, totals[race])
            paper_numeric = int(PAPER_TABLE1_COUNTS[metric][race])
            if column is None:
                paper_display = str(paper_numeric)
            else:
                paper_display = _format_count_percent(
                    paper_numeric,
                    PAPER_TABLE1_COUNTS["Population Size"][race],
                )
            rows.append(
                {
                    "metric": metric,
                    "race": race,
                    "paper_value": paper_display,
                    "run_value": run_display,
                    "paper_numeric": paper_numeric,
                    "run_numeric": run_numeric,
                    "delta_numeric": int(run_numeric - paper_numeric),
                }
            )

    for metric, paper_values in PAPER_TABLE1_CONTINUOUS.items():
        for race in ("BLACK", "WHITE"):
            race_frame = cohort[cohort["race"] == race]
            series_name = "los_days" if metric == "Length of stay (median days)" else "age"
            series = pd.to_numeric(race_frame[series_name], errors="coerce").dropna()
            if series.empty:
                run_numeric = float("nan")
                run_lower = float("nan")
                run_upper = float("nan")
            else:
                run_numeric = float(series.median())
                run_lower = float(series.quantile(0.25))
                run_upper = float(series.quantile(0.75))
            paper_numeric = float(paper_values[race]["center"])
            paper_lower = float(paper_values[race]["lower"])
            paper_upper = float(paper_values[race]["upper"])
            rows.append(
                {
                    "metric": metric,
                    "race": race,
                    "summary_stat": "median_iqr",
                    "paper_value": _format_continuous_summary(paper_numeric, paper_lower, paper_upper),
                    "run_value": _format_continuous_summary(run_numeric, run_lower, run_upper),
                    "paper_numeric": paper_numeric,
                    "run_numeric": run_numeric,
                    "paper_interval_lower": paper_lower,
                    "paper_interval_upper": paper_upper,
                    "run_interval_lower": run_lower,
                    "run_interval_upper": run_upper,
                    "delta_numeric": float(run_numeric - paper_numeric),
                }
            )

    return pd.DataFrame(rows)


def build_paper_table2_comparison(race_treatment_results: pd.DataFrame) -> pd.DataFrame:
    """Compare run race-based treatment durations against Table 2 / Figure 2 from the paper."""

    if race_treatment_results.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, row in race_treatment_results.iterrows():
        treatment = row["treatment"]
        if treatment not in PAPER_TABLE2_TREATMENT:
            continue
        paper = PAPER_TABLE2_TREATMENT[treatment]
        run_median_black = float(row["median_black"])
        run_median_white = float(row["median_white"])
        run_pvalue = float(row["pvalue"])
        rows.append(
            {
                "treatment": treatment,
                "paper_n_black": int(paper["n_black"]),
                "run_n_black": int(row["n_black"]),
                "paper_n_white": int(paper["n_white"]),
                "run_n_white": int(row["n_white"]),
                "paper_median_black": float(paper["median_black"]),
                "run_median_black": run_median_black,
                "delta_median_black": run_median_black - float(paper["median_black"]),
                "paper_median_white": float(paper["median_white"]),
                "run_median_white": run_median_white,
                "delta_median_white": run_median_white - float(paper["median_white"]),
                "paper_pvalue": float(paper["pvalue"]),
                "run_pvalue": run_pvalue,
            }
        )
    return pd.DataFrame(rows)


def build_paper_table3_comparison(feature_weight_summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare run proxy model top-3 feature weights against Table 3 from the paper."""

    rows: list[dict[str, object]] = []
    for model_name, weights_dict in feature_weight_summaries.items():
        if model_name not in PAPER_TABLE3_WEIGHTS:
            continue
        paper_model = PAPER_TABLE3_WEIGHTS[model_name]

        # weights_dict may be a dict with "all"/"positive"/"negative" keys
        if isinstance(weights_dict, dict):
            all_weights = weights_dict.get("all")
            if not isinstance(all_weights, pd.DataFrame) or all_weights.empty:
                continue
        elif isinstance(weights_dict, pd.DataFrame):
            all_weights = weights_dict
        else:
            continue

        if "weight" not in all_weights.columns or "feature" not in all_weights.columns:
            continue

        # Build a lookup from lowercase feature name to weight
        run_lookup = {
            str(f).lower().strip(): float(w)
            for f, w in zip(all_weights["feature"], all_weights["weight"])
        }
        alias_lookup = {
            str(f).lower().strip(): str(f)
            for f in all_weights["feature"]
        }
        model_aliases = PAPER_TABLE3_FEATURE_ALIASES.get(model_name, {})

        for direction in ("positive", "negative"):
            for rank, (paper_feature, paper_weight) in enumerate(
                paper_model[direction], start=1
            ):
                normalized_paper_feature = paper_feature.lower().strip()
                matched_feature = alias_lookup.get(normalized_paper_feature)
                run_weight = run_lookup.get(normalized_paper_feature, float("nan"))
                if pd.isna(run_weight):
                    for alias in model_aliases.get(paper_feature, ()):
                        normalized_alias = alias.lower().strip()
                        alias_weight = run_lookup.get(normalized_alias, float("nan"))
                        if not pd.isna(alias_weight):
                            run_weight = alias_weight
                            matched_feature = alias_lookup.get(normalized_alias, alias)
                            break
                rows.append(
                    {
                        "proxy_model": model_name,
                        "direction": direction,
                        "rank": int(rank),
                        "paper_feature": paper_feature,
                        "paper_weight": float(paper_weight),
                        "run_feature": matched_feature,
                        "run_weight": run_weight,
                        "delta_weight": run_weight - float(paper_weight)
                        if not pd.isna(run_weight)
                        else float("nan"),
                        "run_feature_found": not pd.isna(run_weight),
                    }
                )
    return pd.DataFrame(rows)


def build_paper_table3_snapshot(feature_weight_summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Capture the run's top positive/negative proxy weights for qualitative review."""

    rows: list[dict[str, object]] = []
    for model_name, weights_dict in feature_weight_summaries.items():
        # Handle both dict-of-DataFrames and plain DataFrame inputs
        if isinstance(weights_dict, dict):
            working = weights_dict.get("all")
            if not isinstance(working, pd.DataFrame) or working.empty:
                continue
        elif isinstance(weights_dict, pd.DataFrame):
            working = weights_dict
        else:
            continue
        if "weight" not in working.columns or "feature" not in working.columns:
            continue
        positive = working[working["weight"] > 0].sort_values("weight", ascending=False).head(3)
        negative = working[working["weight"] < 0].sort_values("weight", ascending=True).head(3)
        for direction, frame in (("positive", positive), ("negative", negative)):
            for rank, row in enumerate(frame.itertuples(index=False), start=1):
                rows.append(
                    {
                        "proxy_model": model_name,
                        "direction": direction,
                        "rank": int(rank),
                        "feature": getattr(row, "feature"),
                        "weight": float(getattr(row, "weight")),
                    }
                )
    return pd.DataFrame(rows)


def build_paper_table4_comparison(acuity_correlations: pd.DataFrame) -> pd.DataFrame:
    """Compare run acuity/mistrust correlations against Table 4 from the paper."""

    rows: list[dict[str, object]] = []
    for row in acuity_correlations.itertuples(index=False):
        key = _canonical_pair(getattr(row, "feature_a"), getattr(row, "feature_b"))
        if key not in PAPER_TABLE4_CORRELATIONS:
            continue
        paper_corr = float(PAPER_TABLE4_CORRELATIONS[key])
        run_corr = float(getattr(row, "correlation"))
        rows.append(
            {
                "feature_a": key[0],
                "feature_b": key[1],
                "paper_correlation": paper_corr,
                "run_correlation": run_corr,
                "delta_correlation": float(run_corr - paper_corr),
            }
        )
    return pd.DataFrame(rows)


def build_paper_table5_comparison(downstream_auc_results: pd.DataFrame) -> pd.DataFrame:
    """Compare downstream AUCs against Table 5 from the paper."""

    rows: list[dict[str, object]] = []
    for row in downstream_auc_results.itertuples(index=False):
        key = (getattr(row, "task"), getattr(row, "configuration"))
        if key not in PAPER_TABLE5_AUC:
            continue
        paper = PAPER_TABLE5_AUC[key]
        paper_mean = float(paper["auc_mean"])
        paper_std = float(paper["auc_std"])
        run_mean = float(getattr(row, "auc_mean"))
        run_std = float(getattr(row, "auc_std"))
        rows.append(
            {
                "task": key[0],
                "configuration": key[1],
                "paper_n_rows": int(paper["n_rows"]),
                "run_n_rows": int(getattr(row, "n_rows")),
                "paper_auc_mean": paper_mean,
                "run_auc_mean": run_mean,
                "delta_auc_mean": float(run_mean - paper_mean),
                "paper_auc_std": paper_std,
                "run_auc_std": run_std,
                "delta_auc_std": float(run_std - paper_std),
                "n_valid_auc": int(getattr(row, "n_valid_auc")),
            }
        )
    return pd.DataFrame(rows)


def build_paper_table6_comparison(downstream_weight_results: pd.DataFrame) -> pd.DataFrame:
    """Compare Baseline + ALL downstream average weights against Table 6 from the paper."""

    if downstream_weight_results.empty:
        return pd.DataFrame()

    working = downstream_weight_results.copy()
    working = working[working["configuration"] == "Baseline + ALL"].copy()
    if working.empty:
        return pd.DataFrame()
    working["paper_feature"] = working["feature"].map(TABLE6_FEATURE_NAME_MAP)
    working = working[working["paper_feature"].notna()].copy()

    rows: list[dict[str, object]] = []
    for row in working.itertuples(index=False):
        task_name = getattr(row, "task")
        feature_name = getattr(row, "paper_feature")
        if task_name not in PAPER_TABLE6_WEIGHTS:
            continue
        if feature_name not in PAPER_TABLE6_WEIGHTS[task_name]:
            continue
        paper_mean, paper_std = PAPER_TABLE6_WEIGHTS[task_name][feature_name]
        run_mean = float(getattr(row, "weight_mean"))
        run_std = float(getattr(row, "weight_std"))
        # Paper Table 6 reports 1.96*std (95% CI half-width), not raw std
        run_std_ci = run_std * 1.96
        rows.append(
            {
                "task": task_name,
                "feature": feature_name,
                "paper_weight_mean": float(paper_mean),
                "run_weight_mean": run_mean,
                "delta_weight_mean": float(run_mean - float(paper_mean)),
                "paper_weight_std": float(paper_std),
                "run_weight_std": run_std_ci,
                "n_valid_weights": int(getattr(row, "n_valid_weights")),
            }
        )
    return pd.DataFrame(rows)


def _ensure_downstream_weight_results(
    artifacts: dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]],
    *,
    repetitions: int,
) -> pd.DataFrame:
    existing = artifacts.get("downstream_weight_results")
    if isinstance(existing, pd.DataFrame) and not existing.empty:
        return existing
    final_model_table = artifacts.get("final_model_table")
    if not isinstance(final_model_table, pd.DataFrame) or final_model_table.empty:
        return pd.DataFrame()
    computed = evaluate_downstream_average_weights(
        final_model_table=final_model_table,
        repetitions=repetitions,
    )
    artifacts["downstream_weight_results"] = computed
    return computed


def _render_run_table_summary(
    artifacts: dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]],
    *,
    repetitions: int,
) -> str:
    """Render a run-only Table 1-6 summary without direct paper comparisons."""

    validation_summary = artifacts.get("validation_summary", {})
    autopsy_proxy_enabled = True
    dataset_prepare_mode = "unknown"
    if isinstance(validation_summary, dict):
        autopsy_proxy_enabled = bool(validation_summary.get("autopsy_proxy_enabled", True))
        dataset_prepare_mode = str(validation_summary.get("dataset_prepare_mode", "unknown"))

    feature_weight_summaries = artifacts.get("feature_weight_summaries", {})
    if not isinstance(feature_weight_summaries, dict):
        feature_weight_summaries = {}

    eol_cohort = artifacts.get("eol_cohort")
    table1 = (
        build_paper_table1_comparison(eol_cohort)
        if _has_columns(
            eol_cohort,
            {"race", "insurance_group", "discharge_category", "gender", "los_days", "age"},
        )
        else pd.DataFrame()
    )
    race_treatment = artifacts.get("race_treatment_results")
    table2 = (
        build_paper_table2_comparison(race_treatment)
        if _has_columns(
            race_treatment,
            {"treatment", "n_black", "n_white", "median_black", "median_white", "pvalue"},
        )
        and not race_treatment.empty
        else pd.DataFrame()
    )
    table3 = build_paper_table3_snapshot(feature_weight_summaries)
    if not autopsy_proxy_enabled and not table3.empty and "proxy_model" in table3.columns:
        table3 = table3.loc[table3["proxy_model"] != "autopsy"].reset_index(drop=True)
    acuity_correlations = artifacts.get("acuity_correlations")
    table4 = (
        build_paper_table4_comparison(acuity_correlations)
        if _has_columns(acuity_correlations, {"feature_a", "feature_b", "correlation"})
        else pd.DataFrame()
    )
    downstream_auc_results = artifacts.get("downstream_auc_results")
    table5 = (
        build_paper_table5_comparison(downstream_auc_results)
        if _has_columns(
            downstream_auc_results,
            {"task", "configuration", "n_rows", "auc_mean", "auc_std"},
        )
        else pd.DataFrame()
    )
    table6_source = _ensure_downstream_weight_results(artifacts, repetitions=repetitions)
    table6 = build_paper_table6_comparison(table6_source)
    if not autopsy_proxy_enabled and not table6.empty and "feature" in table6.columns:
        table6 = table6.loc[table6["feature"] != "autopsy"].reset_index(drop=True)

    lines = [
        "Run Table Results",
        f"dataset_prepare_mode: {dataset_prepare_mode}",
        f"autopsy_proxy_enabled: {autopsy_proxy_enabled}",
        "",
    ]

    if not table1.empty:
        lines.append("Table 1")
        for row in table1.itertuples(index=False):
            lines.append(f"- {row.metric}")
            lines.append(f"  {row.race}: {row.run_value}")
        lines.append("")

    if not table2.empty:
        lines.append("Table 2")
        for row in table2.itertuples(index=False):
            lines.append(f"- {row.treatment}")
            lines.append(
                f"  BLACK: n={int(row.run_n_black)}, median={float(row.run_median_black):.1f}"
            )
            lines.append(
                f"  WHITE: n={int(row.run_n_white)}, median={float(row.run_median_white):.1f}"
            )
            if not pd.isna(row.run_pvalue):
                lines.append(f"  pvalue: {float(row.run_pvalue)}")
        lines.append("")

    if not table3.empty:
        lines.append("Table 3")
        for proxy_model in table3["proxy_model"].drop_duplicates().tolist():
            lines.append(f"- {proxy_model}")
            proxy_rows = table3.loc[table3["proxy_model"] == proxy_model]
            for direction in ("positive", "negative"):
                direction_rows = proxy_rows.loc[proxy_rows["direction"] == direction]
                if direction_rows.empty:
                    continue
                lines.append(f"  {direction}:")
                for row in direction_rows.itertuples(index=False):
                    lines.append(
                        f"    #{int(row.rank)}: {row.feature} = {float(row.weight):.4f}"
                    )
        lines.append("")

    if not table4.empty:
        lines.append("Table 4")
        for row in table4.itertuples(index=False):
            lines.append(
                f"- {row.feature_a} vs {row.feature_b}: {float(row.run_correlation):.3f}"
            )
        lines.append("")

    if not table5.empty:
        lines.append("Table 5")
        for row in table5.itertuples(index=False):
            lines.append(f"- {row.task} | {row.configuration}")
            lines.append(f"  n_rows: {int(row.run_n_rows)}")
            lines.append(f"  auc_mean: {float(row.run_auc_mean):.3f}")
            lines.append(f"  auc_std: {float(row.run_auc_std):.3f}")
        lines.append("")

    if not table6.empty:
        lines.append("Table 6")
        for task_name in table6["task"].drop_duplicates().tolist():
            lines.append(f"- {task_name}")
            task_rows = table6.loc[table6["task"] == task_name]
            for row in task_rows.itertuples(index=False):
                lines.append(
                    f"  {row.feature}: mean={float(row.run_weight_mean):.3f}, std={float(row.run_weight_std):.3f}"
                )
        lines.append("")

    return "\n".join(lines)


def write_run_table_summary_artifacts(
    artifacts: dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]],
    *,
    output_dir: Path,
    repetitions: int,
) -> None:
    """Write a run-only Table 1-6 summary without paper-vs-run formatting."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_table_summary.txt").write_text(
        _render_run_table_summary(artifacts, repetitions=repetitions) + "\n",
        encoding="utf-8",
    )


def build_paper_comparison_outputs(
    artifacts: dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]],
    *,
    repetitions: int,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    """Build paper-aligned comparison tables from an example run."""

    validation_summary = artifacts.get("validation_summary", {})
    autopsy_proxy_enabled = True
    if isinstance(validation_summary, dict):
        autopsy_proxy_enabled = bool(validation_summary.get("autopsy_proxy_enabled", True))

    feature_weight_summaries = artifacts.get("feature_weight_summaries", {})
    if not isinstance(feature_weight_summaries, dict):
        feature_weight_summaries = {}

    table1 = build_paper_table1_comparison(artifacts["eol_cohort"]) if isinstance(artifacts.get("eol_cohort"), pd.DataFrame) else pd.DataFrame()
    race_treatment = artifacts.get("race_treatment_results")
    table2 = build_paper_table2_comparison(race_treatment) if isinstance(race_treatment, pd.DataFrame) and not race_treatment.empty else pd.DataFrame()
    table3_comparison = build_paper_table3_comparison(feature_weight_summaries)
    table3_snapshot = build_paper_table3_snapshot(feature_weight_summaries)
    table4 = build_paper_table4_comparison(artifacts["acuity_correlations"]) if isinstance(artifacts.get("acuity_correlations"), pd.DataFrame) else pd.DataFrame()
    table5 = build_paper_table5_comparison(artifacts["downstream_auc_results"]) if isinstance(artifacts.get("downstream_auc_results"), pd.DataFrame) else pd.DataFrame()
    table6_source = _ensure_downstream_weight_results(artifacts, repetitions=repetitions)
    table6 = build_paper_table6_comparison(table6_source)
    if not autopsy_proxy_enabled and not table6.empty and "feature" in table6.columns:
        table6 = table6.loc[table6["feature"] != "autopsy"].reset_index(drop=True)

    summary = {
        "paper_url": PAPER_URL,
        "paper_pdf_url": PAPER_PDF_URL,
        "table1_rows": int(len(table1)),
        "table2_rows": int(len(table2)),
        "table3_comparison_rows": int(len(table3_comparison)),
        "table3_snapshot_rows": int(len(table3_snapshot)),
        "table4_rows": int(len(table4)),
        "table5_rows": int(len(table5)),
        "table6_rows": int(len(table6)),
        "table2_max_abs_delta_median": (
            float(
                max(
                    table2["delta_median_black"].abs().max(),
                    table2["delta_median_white"].abs().max(),
                )
            )
            if not table2.empty
            else None
        ),
        "table3_comparison_features_found": (
            int(table3_comparison["run_feature_found"].sum())
            if not table3_comparison.empty
            else 0
        ),
        "table3_comparison_features_total": int(len(table3_comparison)),
        "table3_comparison_max_abs_delta": (
            float(table3_comparison["delta_weight"].dropna().abs().max())
            if not table3_comparison.empty and table3_comparison["delta_weight"].notna().any()
            else None
        ),
        "table4_max_abs_delta": (
            float(table4["delta_correlation"].abs().max()) if not table4.empty else None
        ),
        "table5_max_abs_delta": (
            float(table5["delta_auc_mean"].abs().max()) if not table5.empty else None
        ),
        "table6_max_abs_delta": (
            float(table6["delta_weight_mean"].abs().max()) if not table6.empty else None
        ),
    }

    return {
        "summary": summary,
        "table1_comparison": table1,
        "table2_comparison": table2,
        "table3_comparison": table3_comparison,
        "table3_snapshot": table3_snapshot,
        "table4_comparison": table4,
        "table5_comparison": table5,
        "table6_comparison": table6,
    }


def write_paper_comparison_artifacts(
    comparison_outputs: dict[str, pd.DataFrame | dict[str, object]],
    output_dir: Path,
    *,
    include_summary: bool = True,
) -> None:
    """Write paper comparison tables and summary next to the example deliverables."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, artifact in comparison_outputs.items():
        if isinstance(artifact, pd.DataFrame):
            artifact.to_csv(output_dir / f"{name}.csv", index=False)
        elif isinstance(artifact, dict):
            (output_dir / f"{name}.json").write_text(json.dumps(artifact, indent=2))
    if include_summary:
        (output_dir / "paper_comparison_summary.txt").write_text(
            _render_paper_comparison_summary(comparison_outputs) + "\n",
            encoding="utf-8",
        )


def _render_paper_comparison_summary(
    comparison_outputs: dict[str, pd.DataFrame | dict[str, object]],
) -> str:
    lines: list[str] = []

    summary = comparison_outputs.get("summary", {})
    if isinstance(summary, dict):
        lines.append("Paper comparison summary:")
        for key in (
            "table1_rows",
            "table2_rows",
            "table3_snapshot_rows",
            "table4_rows",
            "table5_rows",
            "table6_rows",
            "table4_max_abs_delta",
            "table5_max_abs_delta",
            "table6_max_abs_delta",
        ):
            value = summary.get(key)
            if value is not None:
                lines.append(f"  {key}: {value}")

    table1 = comparison_outputs.get("table1_comparison")
    if isinstance(table1, pd.DataFrame) and not table1.empty:
        lines.append("")
        lines.append("Table 1 vs Paper:")
        for row in table1.itertuples(index=False):
            lines.append(f"  {row.metric} | {row.race} | paper={row.paper_value} | run={row.run_value}")

    table2 = comparison_outputs.get("table2_comparison")
    if isinstance(table2, pd.DataFrame) and not table2.empty:
        lines.append("")
        lines.append("Table 2 vs Paper:")
        for row in table2.itertuples(index=False):
            lines.append(
                "  "
                f"{row.treatment} | "
                f"black n {int(row.paper_n_black)}->{int(row.run_n_black)}, median {row.paper_median_black:.1f}->{row.run_median_black:.1f} | "
                f"white n {int(row.paper_n_white)}->{int(row.run_n_white)}, median {row.paper_median_white:.1f}->{row.run_median_white:.1f}"
            )

    table3 = comparison_outputs.get("table3_comparison")
    if isinstance(table3, pd.DataFrame) and not table3.empty:
        lines.append("")
        lines.append("Table 3 vs Paper:")
        for row in table3.itertuples(index=False):
            run_weight = "missing" if pd.isna(row.run_weight) else f"{float(row.run_weight):.4f}"
            lines.append(
                "  "
                f"{row.proxy_model} | {row.direction} #{int(row.rank)} | {row.paper_feature} | "
                f"paper={float(row.paper_weight):.4f} | run={run_weight} | found={bool(row.run_feature_found)}"
            )

    table4 = comparison_outputs.get("table4_comparison")
    if isinstance(table4, pd.DataFrame) and not table4.empty:
        lines.append("")
        lines.append("Table 4 vs Paper:")
        for row in table4.itertuples(index=False):
            lines.append(
                "  "
                f"{row.feature_a} vs {row.feature_b} | "
                f"paper={float(row.paper_correlation):.3f} | run={float(row.run_correlation):.3f}"
            )

    table5 = comparison_outputs.get("table5_comparison")
    if isinstance(table5, pd.DataFrame) and not table5.empty:
        lines.append("")
        lines.append("Table 5 vs Paper:")
        for row in table5.itertuples(index=False):
            lines.append(
                "  "
                f"{row.task} | {row.configuration} | "
                f"n {int(row.paper_n_rows)}->{int(row.run_n_rows)} | "
                f"auc {float(row.paper_auc_mean):.3f}->{float(row.run_auc_mean):.3f}"
            )

    table6 = comparison_outputs.get("table6_comparison")
    if isinstance(table6, pd.DataFrame) and not table6.empty:
        lines.append("")
        lines.append("Table 6 vs Paper:")
        for row in table6.itertuples(index=False):
            lines.append(
                "  "
                f"{row.task} | {row.feature} | "
                f"paper={float(row.paper_weight_mean):.3f} | run={float(row.run_weight_mean):.3f}"
            )

    return "\n".join(lines)


def _print_paper_comparison_summary(
    comparison_outputs: dict[str, pd.DataFrame | dict[str, object]],
) -> None:
    rendered = _render_paper_comparison_summary(comparison_outputs)
    if rendered:
        print()
        print(rendered)


def _log_stage(stage_start: float, pipeline_start: float, message: str) -> None:
    """Print a timing log line for a pipeline stage."""
    elapsed_stage = time.time() - stage_start
    elapsed_total = time.time() - pipeline_start
    print(f"[{elapsed_total:7.1f}s total | {elapsed_stage:6.1f}s] {message}", flush=True)


class _RouteSettings:
    def __init__(
        self,
        *,
        mode_name: str,
        autopsy_enabled: bool,
        autopsy_label_mode: str,
        code_status_mode: str,
        score_columns: list[str] | None,
        feature_configurations: dict[str, list[str]] | None,
        downstream_estimator_mode: str,
        downstream_estimator_factory_resolver: object | None,
    ) -> None:
        self.mode_name = mode_name
        self.autopsy_enabled = autopsy_enabled
        self.autopsy_label_mode = autopsy_label_mode
        self.code_status_mode = code_status_mode
        self.score_columns = score_columns
        self.feature_configurations = feature_configurations
        self.downstream_estimator_mode = downstream_estimator_mode
        self.downstream_estimator_factory_resolver = downstream_estimator_factory_resolver


def _current_run_timestamp() -> str:
    """Return the timestamp suffix used for managed run archive directories."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _managed_run_route_label(route_settings: _RouteSettings) -> str:
    """Return the user-facing route label used in managed run directory names."""

    return "Paperlike" if route_settings.mode_name == "paper_like" else "normal"


def _build_managed_run_name(route_settings: _RouteSettings, timestamp: str) -> str:
    """Return the managed run directory name for the given route and timestamp."""

    return f"EOL_{_managed_run_route_label(route_settings)}_{timestamp}"


def _prepare_managed_run_directories(
    *,
    result_root: Path,
    route_settings: _RouteSettings,
    output_dir: Path | None,
    stream_cache_dir: Path | None,
) -> dict[str, Path | str]:
    """Create a managed run archive directory and resolve default output/cache paths."""

    timestamp = _current_run_timestamp()
    base_name = _build_managed_run_name(route_settings, timestamp)
    run_name = base_name
    run_dir = result_root / run_name
    suffix = 1
    while run_dir.exists():
        run_name = f"{base_name}_{suffix:02d}"
        run_dir = result_root / run_name
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    resolved_output_dir = output_dir if output_dir is not None else run_dir / "result"
    resolved_stream_cache_dir = (
        stream_cache_dir if stream_cache_dir is not None else run_dir / "cache"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_stream_cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "output_dir": resolved_output_dir,
        "stream_cache_dir": resolved_stream_cache_dir,
    }


def _collect_core_artifact_shapes(
    artifacts: dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]],
) -> dict[str, list[int]]:
    """Collect the core DataFrame shapes used in run summaries and manifests."""

    shapes: dict[str, list[int]] = {}
    for key in (
        "base_admissions",
        "all_cohort",
        "eol_cohort",
        "chartevent_feature_matrix",
        "note_labels",
        "mistrust_scores",
        "final_model_table",
    ):
        df = artifacts.get(key)
        if isinstance(df, pd.DataFrame):
            shapes[key] = [int(df.shape[0]), int(df.shape[1])]
    return shapes


def _render_managed_run_summary(
    *,
    run_name: str,
    run_dir: Path,
    route_settings: _RouteSettings,
    args: argparse.Namespace,
    resolved_output_dir: Path,
    resolved_stream_cache_dir: Path,
    started_at: datetime,
    finished_at: datetime,
    total_runtime_seconds: float,
    artifacts: dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]],
) -> str:
    """Render a human-readable managed run summary."""

    validation_summary = artifacts.get("validation_summary", {})
    lines = [
        "EOL managed run summary:",
        f"managed_run_name: {run_name}",
        f"managed_run_dir: {run_dir}",
        f"route_mode: {route_settings.mode_name}",
        f"autopsy_proxy_enabled: {route_settings.autopsy_enabled}",
        f"started_at: {started_at.isoformat(timespec='seconds')}",
        f"finished_at: {finished_at.isoformat(timespec='seconds')}",
        f"total_runtime_seconds: {total_runtime_seconds:.3f}",
        f"result_dir: {resolved_output_dir}",
        f"stream_cache_base_dir: {resolved_stream_cache_dir}",
        (
            f"paper_comparison_summary_file: {run_dir / 'paper_comparison_summary.txt'}"
            if args.compare_to_paper
            else "paper_comparison_summary_file: disabled"
        ),
        f"run_table_summary_file: {run_dir / 'run_table_summary.txt'}",
        f"reuse_intermediates: {args.reuse_intermediates}",
        f"compare_to_paper: {args.compare_to_paper}",
        f"paper_like_dataset_prepare: {args.paper_like_dataset_prepare}",
        f"repetitions: {args.repetitions}",
        f"note_chunksize: {args.note_chunksize}",
        f"chartevent_chunksize: {args.chartevent_chunksize}",
        f"command: {' '.join(sys.argv)}",
        "",
        "Validation summary:",
    ]

    if isinstance(validation_summary, dict):
        for key, value in validation_summary.items():
            lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("Core artifact shapes:")
    for key, shape in _collect_core_artifact_shapes(artifacts).items():
        lines.append(f"  {key}: ({shape[0]}, {shape[1]})")

    return "\n".join(lines) + "\n"


def _write_managed_run_artifacts(
    *,
    run_name: str,
    run_dir: Path,
    route_settings: _RouteSettings,
    args: argparse.Namespace,
    resolved_output_dir: Path,
    resolved_stream_cache_dir: Path,
    started_at: datetime,
    finished_at: datetime,
    total_runtime_seconds: float,
    artifacts: dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]],
) -> None:
    """Write managed run archive files under EOL_Result/EOL_<route>_<timestamp>."""

    summary_text = _render_managed_run_summary(
        run_name=run_name,
        run_dir=run_dir,
        route_settings=route_settings,
        args=args,
        resolved_output_dir=resolved_output_dir,
        resolved_stream_cache_dir=resolved_stream_cache_dir,
        started_at=started_at,
        finished_at=finished_at,
        total_runtime_seconds=total_runtime_seconds,
        artifacts=artifacts,
    )
    (run_dir / "RUN_SUMMARY.txt").write_text(summary_text, encoding="utf-8")
    (run_dir / "RUN_TIME.txt").write_text(
        "\n".join(
            [
                "EOL run timing:",
                f"managed_run_name: {run_name}",
                f"started_at: {started_at.isoformat(timespec='seconds')}",
                f"finished_at: {finished_at.isoformat(timespec='seconds')}",
                f"total_runtime_seconds: {total_runtime_seconds:.3f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "managed_run_name": run_name,
        "managed_run_dir": str(run_dir),
        "route_mode": route_settings.mode_name,
        "autopsy_proxy_enabled": route_settings.autopsy_enabled,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "total_runtime_seconds": round(float(total_runtime_seconds), 6),
        "result_dir": str(resolved_output_dir),
        "stream_cache_base_dir": str(resolved_stream_cache_dir),
        "reuse_intermediates": (
            str(args.reuse_intermediates) if args.reuse_intermediates is not None else None
        ),
        "compare_to_paper": bool(args.compare_to_paper),
        "paper_like_dataset_prepare": bool(args.paper_like_dataset_prepare),
        "repetitions": int(args.repetitions),
        "note_chunksize": int(args.note_chunksize),
        "chartevent_chunksize": int(args.chartevent_chunksize),
        "command": sys.argv,
        "validation_summary": artifacts.get("validation_summary", {}),
        "core_artifact_shapes": _collect_core_artifact_shapes(artifacts),
    }
    (run_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    write_run_table_summary_artifacts(
        artifacts,
        output_dir=run_dir,
        repetitions=int(args.repetitions),
    )

    comparison_outputs = artifacts.get("paper_comparison")
    if args.compare_to_paper and isinstance(comparison_outputs, dict):
        (run_dir / "paper_comparison_summary.txt").write_text(
            _render_paper_comparison_summary(comparison_outputs) + "\n",
            encoding="utf-8",
        )


def _build_route_settings(paper_like_dataset_prepare: bool) -> _RouteSettings:
    if paper_like_dataset_prepare:
        return _RouteSettings(
            mode_name="paper_like",
            autopsy_enabled=True,
            autopsy_label_mode="paper_like",
            code_status_mode="paper_like",
            score_columns=None,
            feature_configurations=None,
            downstream_estimator_mode="default",
            downstream_estimator_factory_resolver=None,
        )

    return _RouteSettings(
        mode_name="default",
        autopsy_enabled=False,
        autopsy_label_mode="corrected",
        code_status_mode="corrected",
        score_columns=_normal_route_score_columns(),
        feature_configurations=_normal_route_feature_configurations(),
        downstream_estimator_mode="task_balanced_logistic_cv",
        downstream_estimator_factory_resolver=_normal_route_downstream_estimator_factory_resolver(),
    )


def _resolve_stage_cache_dir(
    *,
    output_dir: Path | None,
    stream_cache_dir: Path | None,
    route_settings: _RouteSettings,
) -> Path | None:
    """Return the directory used for streamed-stage checkpoint CSVs."""

    if stream_cache_dir is not None:
        return Path(stream_cache_dir) / route_settings.mode_name
    return output_dir


def _has_reuse_cache_files(directory: Path) -> bool:
    required = (
        "note_corpus.csv",
        "note_labels.csv",
        "chartevent_feature_matrix.csv",
        "code_status_targets.csv",
    )
    return all((directory / filename).exists() for filename in required)


def _resolve_reuse_dir(
    reuse_intermediates: Path | None,
    *,
    route_settings: _RouteSettings,
) -> Path | None:
    """Resolve the reuse directory, allowing a base cache dir with mode subfolders."""

    if reuse_intermediates is None:
        return None
    direct = Path(reuse_intermediates)
    if _has_reuse_cache_files(direct):
        return direct
    mode_dir = direct / route_settings.mode_name
    if _has_reuse_cache_files(mode_dir):
        return mode_dir
    return direct


def _write_stage_cache_frame(
    output_dir: Path | None,
    filename: str,
    frame: pd.DataFrame,
) -> None:
    """Persist a reusable CSV artifact as soon as its stage completes."""

    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / filename, index=False)


def _disable_autopsy_scores(mistrust_scores: pd.DataFrame) -> pd.DataFrame:
    """Return a schema-stable score table with the autopsy proxy disabled."""

    if "autopsy_score_z" not in mistrust_scores.columns:
        return mistrust_scores
    adjusted = mistrust_scores.copy()
    adjusted["autopsy_score_z"] = 0.0
    return adjusted


def _normal_route_score_columns() -> list[str]:
    return ["noncompliance_score_z", "negative_sentiment_score_z"]


def _normal_route_feature_configurations() -> dict[str, list[str]]:
    configs = get_downstream_feature_configurations()
    adjusted: dict[str, list[str]] = {}
    for name, columns in configs.items():
        if name == "Baseline + Autopsy":
            continue
        adjusted[name] = [column for column in columns if column != "autopsy_score_z"]
    return adjusted


def _normal_route_downstream_estimator_factory_resolver():
    """Return task-specific balanced LogisticRegressionCV factories for the corrected route."""

    task_specs = {
        "Left AMA": {
            "Cs": [0.01, 0.03, 0.1, 0.3],
            "class_weight": "balanced",
            "scoring": "roc_auc",
        },
        "Code Status": {
            "Cs": [0.01, 0.03, 0.1, 0.3],
            "class_weight": "balanced",
            "scoring": "roc_auc",
        },
        "In-hospital mortality": {
            "Cs": [0.03, 0.1, 0.3, 1.0],
            "class_weight": "balanced",
            "scoring": "roc_auc",
        },
    }
    cached_factories: dict[str, object] = {}

    def _resolver(task_name: str, _config_name: str):
        spec = task_specs.get(task_name)
        if spec is None:
            return None
        if task_name not in cached_factories:
            cached_factories[task_name] = build_logistic_cv_estimator_factory(**spec)
        return cached_factories[task_name]

    return _resolver


def _filter_metric_frame(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if "metric" not in frame.columns:
        return frame
    return frame.loc[frame["metric"] != metric].reset_index(drop=True)


def _disable_autopsy_outputs(
    model_outputs: dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]],
) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]]:
    """Strip autopsy-specific analysis outputs from the default route."""

    adjusted = dict(model_outputs)

    feature_weight_summaries = adjusted.get("feature_weight_summaries")
    if isinstance(feature_weight_summaries, dict):
        adjusted["feature_weight_summaries"] = {
            name: table
            for name, table in feature_weight_summaries.items()
            if name != "autopsy"
        }

    for key in ("race_gap_results", "trust_treatment_results", "trust_treatment_by_acuity_results"):
        frame = adjusted.get(key)
        if isinstance(frame, pd.DataFrame):
            adjusted[key] = _filter_metric_frame(frame, "autopsy_score_z")

    cdf_plot_data = adjusted.get("trust_treatment_cdf_plot_data")
    if isinstance(cdf_plot_data, dict):
        adjusted["trust_treatment_cdf_plot_data"] = {
            name: _filter_metric_frame(frame, "autopsy_score_z")
            if isinstance(frame, pd.DataFrame)
            else frame
            for name, frame in cdf_plot_data.items()
        }

    acuity_correlations = adjusted.get("acuity_correlations")
    if isinstance(acuity_correlations, pd.DataFrame):
        filtered = acuity_correlations.copy()
        if {"feature_a", "feature_b"}.issubset(filtered.columns):
            filtered = filtered.loc[
                (filtered["feature_a"] != "autopsy_score_z")
                & (filtered["feature_b"] != "autopsy_score_z")
            ]
        adjusted["acuity_correlations"] = filtered.reset_index(drop=True)

    downstream_auc_results = adjusted.get("downstream_auc_results")
    if (
        isinstance(downstream_auc_results, pd.DataFrame)
        and "configuration" in downstream_auc_results.columns
    ):
        adjusted["downstream_auc_results"] = downstream_auc_results.loc[
            downstream_auc_results["configuration"] != "Baseline + Autopsy"
        ].reset_index(drop=True)

    downstream_weight_results = adjusted.get("downstream_weight_results")
    if (
        isinstance(downstream_weight_results, pd.DataFrame)
        and "feature" in downstream_weight_results.columns
    ):
        adjusted["downstream_weight_results"] = downstream_weight_results.loc[
            downstream_weight_results["feature"] != "autopsy_score_z"
        ].reset_index(drop=True)

    return adjusted


def _build_or_reuse_mistrust_scores(
    *,
    model: object,
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    note_corpus: pd.DataFrame,
    reuse_dir: Path | None,
    stage_cache_dir: Path | None,
    pipeline_start: float,
    autopsy_enabled: bool,
) -> pd.DataFrame:
    cached_path = None if reuse_dir is None else reuse_dir / "mistrust_scores.csv"
    if cached_path is not None and cached_path.exists():
        t0 = time.time()
        print(f"[REUSE] Loading mistrust_scores from {reuse_dir}", flush=True)
        mistrust_scores = pd.read_csv(cached_path, low_memory=False)
        _log_stage(t0, pipeline_start, f"Reused mistrust scores ({len(mistrust_scores)} rows)")
        return mistrust_scores

    if hasattr(model, "estimator_factory") and hasattr(model, "sentiment_fn"):
        estimator_factory = getattr(model, "estimator_factory")
        sentiment_fn = getattr(model, "sentiment_fn")
        t_total = time.time()

        t0 = time.time()
        noncompliance = build_noncompliance_mistrust_scores(
            feature_matrix=feature_matrix,
            note_labels=note_labels,
            estimator_factory=estimator_factory,
        )
        _log_stage(t0, pipeline_start, f"Built noncompliance proxy scores ({len(noncompliance)} rows)")

        if autopsy_enabled:
            t0 = time.time()
            autopsy = build_autopsy_mistrust_scores(
                feature_matrix=feature_matrix,
                note_labels=note_labels,
                estimator_factory=estimator_factory,
            )
            _log_stage(t0, pipeline_start, f"Built autopsy proxy scores ({len(autopsy)} rows)")
        else:
            autopsy = pd.DataFrame(
                {
                    "hadm_id": noncompliance["hadm_id"].astype(int),
                    "autopsy_score": 0.0,
                }
            )

        t0 = time.time()
        sentiment = build_negative_sentiment_mistrust_scores(
            note_corpus=note_corpus,
            sentiment_fn=sentiment_fn,
        )
        _log_stage(t0, pipeline_start, f"Built negative sentiment scores ({len(sentiment)} rows)")

        merged = (
            noncompliance.merge(autopsy, on="hadm_id", how="inner", validate="one_to_one")
            .merge(sentiment, on="hadm_id", how="inner", validate="one_to_one")
            .sort_values("hadm_id")
        )
        mistrust_scores = z_normalize_scores(
            merged,
            columns=["noncompliance_score", "autopsy_score", "negative_sentiment_score"],
        ).rename(
            columns={
                "noncompliance_score": "noncompliance_score_z",
                "autopsy_score": "autopsy_score_z",
                "negative_sentiment_score": "negative_sentiment_score_z",
            }
        ).reset_index(drop=True)
        _write_stage_cache_frame(stage_cache_dir, "mistrust_scores.csv", mistrust_scores)
        _log_stage(t_total, pipeline_start, "Built mistrust scores (proxy models + sentiment)")
        return mistrust_scores

    t0 = time.time()
    mistrust_scores = model.build_mistrust_scores(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        note_corpus=note_corpus,
    )
    _write_stage_cache_frame(stage_cache_dir, "mistrust_scores.csv", mistrust_scores)
    _log_stage(t0, pipeline_start, "Built mistrust scores (proxy models + sentiment)")
    return mistrust_scores


def _build_or_reuse_note_artifacts(
    *,
    noteevents_csv_path: Path,
    all_cohort: pd.DataFrame,
    reuse_dir: Path | None,
    stage_cache_dir: Path | None,
    route_settings: _RouteSettings,
    note_chunksize: int,
    pipeline_start: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], pd.DataFrame]:
    can_reuse = (
        reuse_dir is not None
        and (reuse_dir / "note_corpus.csv").exists()
        and (reuse_dir / "note_labels.csv").exists()
    )

    t0 = time.time()
    if can_reuse:
        print(f"[REUSE] Loading note_corpus & note_labels from {reuse_dir}", flush=True)
        note_corpus = pd.read_csv(reuse_dir / "note_corpus.csv", low_memory=False)
        note_labels = pd.read_csv(reuse_dir / "note_labels.csv", low_memory=False)
        note_present_hadm_ids = _note_present_hadm_ids(note_corpus)
        filtered_all_cohort = all_cohort.loc[all_cohort["hadm_id"].isin(note_present_hadm_ids)].copy()
        _log_stage(
            t0,
            pipeline_start,
            f"Reused note artifacts ({len(note_corpus)} corpus rows, {len(note_labels)} label rows)",
        )
        return note_corpus, note_labels, note_present_hadm_ids, filtered_all_cohort

    note_corpus = build_note_corpus_from_csv(
        noteevents_csv_path=noteevents_csv_path,
        all_hadm_ids=all_cohort["hadm_id"],
        categories=None,
        chunksize=note_chunksize,
    )
    note_present_hadm_ids = _note_present_hadm_ids(note_corpus)
    filtered_all_cohort = all_cohort.loc[all_cohort["hadm_id"].isin(note_present_hadm_ids)].copy()
    note_corpus = note_corpus.loc[note_corpus["hadm_id"].isin(note_present_hadm_ids)].copy()
    _write_stage_cache_frame(stage_cache_dir, "note_corpus.csv", note_corpus)
    _log_stage(t0, pipeline_start, f"Streamed note corpus ({len(note_corpus)} rows)")

    t0 = time.time()
    note_labels = build_note_labels_from_csv(
        noteevents_csv_path=noteevents_csv_path,
        all_hadm_ids=note_present_hadm_ids,
        autopsy_label_mode=route_settings.autopsy_label_mode,
        chunksize=note_chunksize,
    )
    _write_stage_cache_frame(stage_cache_dir, "note_labels.csv", note_labels)
    _log_stage(t0, pipeline_start, f"Streamed note labels ({len(note_labels)} rows)")
    return note_corpus, note_labels, note_present_hadm_ids, filtered_all_cohort


def _build_or_reuse_chartevent_artifacts(
    *,
    chartevents_csv_path: Path,
    d_items: pd.DataFrame,
    note_present_hadm_ids: list[int],
    reuse_dir: Path | None,
    stage_cache_dir: Path | None,
    route_settings: _RouteSettings,
    chartevent_chunksize: int,
    pipeline_start: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    can_reuse = (
        reuse_dir is not None
        and (reuse_dir / "chartevent_feature_matrix.csv").exists()
        and (reuse_dir / "code_status_targets.csv").exists()
    )

    t0 = time.time()
    if can_reuse:
        print(f"[REUSE] Loading feature_matrix & code_status_targets from {reuse_dir}", flush=True)
        feature_matrix = pd.read_csv(reuse_dir / "chartevent_feature_matrix.csv", low_memory=False)
        code_status_targets = pd.read_csv(reuse_dir / "code_status_targets.csv", low_memory=False)
        _log_stage(
            t0,
            pipeline_start,
            f"Reused chartevent artifacts ({len(feature_matrix)} feature rows, {len(code_status_targets)} target rows)",
        )
        return feature_matrix, code_status_targets

    feature_matrix, code_status_targets = build_chartevent_artifacts_from_csv(
        chartevents_csv_path=chartevents_csv_path,
        d_items=d_items,
        all_hadm_ids=note_present_hadm_ids,
        chunksize=chartevent_chunksize,
        paper_like=route_settings.autopsy_enabled,
        code_status_mode=route_settings.code_status_mode,
    )
    _write_stage_cache_frame(stage_cache_dir, "chartevent_feature_matrix.csv", feature_matrix)
    _write_stage_cache_frame(stage_cache_dir, "code_status_targets.csv", code_status_targets)
    _log_stage(t0, pipeline_start, f"Streamed chartevents ({len(feature_matrix)} feature rows)")
    return feature_matrix, code_status_targets


def build_eol_mistrust_outputs(
    root: Path,
    repetitions: int = 100,
    include_downstream_weight_summary: bool = False,
    include_cdf_plot_data: bool = False,
    compare_to_paper: bool = False,
    output_dir: Path | None = None,
    stream_cache_dir: Path | None = None,
    note_chunksize: int = 100_000,
    chartevent_chunksize: int = 500_000,
    reuse_intermediates: Path | None = None,
    paper_like_dataset_prepare: bool = False,
) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]]:
    """Run the local end-to-end EOL mistrust workflow over downloaded CSV files.

    When *reuse_intermediates* points to a previous output directory that
    contains cached CSV artifacts (note_corpus, note_labels,
    chartevent_feature_matrix, code_status_targets, optionally
    mistrust_scores), the expensive CSV streaming stages are skipped and
    those frames are loaded from disk instead. Everything downstream is
    recomputed unless a reusable ``mistrust_scores.csv`` is also present.
    """

    t_pipeline = time.time()
    route_settings = _build_route_settings(paper_like_dataset_prepare)

    # ------------------------------------------------------------------
    # Stage 1: load small raw tables & materialized views (fast)
    # ------------------------------------------------------------------
    t0 = time.time()
    raw_tables, materialized_views = load_eol_mistrust_tables(root)
    _log_stage(t0, t_pipeline, "Loaded raw tables & materialized views")

    validation = {
        "schema_name": "mimiciii",
        "database_flavor": "postgresql",
        "raw_tables": sorted(raw_tables.keys()),
        "materialized_views": sorted(materialized_views.keys()),
        "dataset_prepare_mode": route_settings.mode_name,
        "autopsy_proxy_enabled": route_settings.autopsy_enabled,
    }
    _stage_cache_dir = _resolve_stage_cache_dir(
        output_dir=output_dir,
        stream_cache_dir=stream_cache_dir,
        route_settings=route_settings,
    )
    _reuse_dir = _resolve_reuse_dir(
        reuse_intermediates,
        route_settings=route_settings,
    )
    if _stage_cache_dir is not None:
        validation["stream_cache_dir"] = str(_stage_cache_dir)

    admissions = raw_tables["admissions"]
    patients = raw_tables["patients"]
    icustays = raw_tables["icustays"]
    d_items = raw_tables["d_items"]
    noteevents_csv_path = root / EVENT_TABLE_PATHS["noteevents"]
    chartevents_csv_path = root / EVENT_TABLE_PATHS["chartevents"]

    t0 = time.time()
    validation_raw_tables = dict(raw_tables)
    if "noteevents" not in validation_raw_tables and noteevents_csv_path.exists():
        validation_raw_tables["noteevents"] = _read_csv_probe(
            root,
            EVENT_TABLE_PATHS["noteevents"],
        )
    if "chartevents" not in validation_raw_tables and chartevents_csv_path.exists():
        validation_raw_tables["chartevents"] = _read_csv_probe(
            root,
            EVENT_TABLE_PATHS["chartevents"],
        )
    if {"noteevents", "chartevents"}.issubset(validation_raw_tables):
        validation = validate_database_environment(
            validation_raw_tables,
            materialized_views,
            schema_name="mimiciii",
            database_flavor="postgresql",
        )
    validation["dataset_prepare_mode"] = route_settings.mode_name
    validation["autopsy_proxy_enabled"] = route_settings.autopsy_enabled
    if _stage_cache_dir is not None:
        validation["stream_cache_dir"] = str(_stage_cache_dir)
    _log_stage(t0, t_pipeline, "Validated database environment")

    # ------------------------------------------------------------------
    # Stage 2: build cohorts & demographics (fast)
    # ------------------------------------------------------------------
    t0 = time.time()
    base_admissions = build_base_admissions(admissions, patients)
    demographics = build_demographics_table(
        base_admissions,
        paper_like=route_settings.autopsy_enabled,
    )
    all_cohort = build_all_cohort(base_admissions, icustays)
    eol_cohort = build_eol_cohort(base_admissions, demographics)
    treatment_totals = build_treatment_totals(
        icustays=icustays,
        ventdurations=materialized_views["ventdurations"],
        vasopressordurations=materialized_views["vasopressordurations"],
        paper_like=route_settings.autopsy_enabled,
    )
    _log_stage(t0, t_pipeline, "Built cohorts & demographics")

    # ------------------------------------------------------------------
    # Stage 3: note corpus + note labels (SLOW — stream noteevents.csv)
    # ------------------------------------------------------------------
    note_corpus, note_labels, note_present_hadm_ids, all_cohort = _build_or_reuse_note_artifacts(
        noteevents_csv_path=noteevents_csv_path,
        all_cohort=all_cohort,
        reuse_dir=_reuse_dir,
        stage_cache_dir=_stage_cache_dir,
        route_settings=route_settings,
        note_chunksize=note_chunksize,
        pipeline_start=t_pipeline,
    )

    # ------------------------------------------------------------------
    # Stage 4: chartevents feature matrix + code status (SLOW — stream chartevents.csv)
    # ------------------------------------------------------------------
    feature_matrix, code_status_targets = _build_or_reuse_chartevent_artifacts(
        chartevents_csv_path=chartevents_csv_path,
        d_items=d_items,
        note_present_hadm_ids=note_present_hadm_ids,
        reuse_dir=_reuse_dir,
        stage_cache_dir=_stage_cache_dir,
        route_settings=route_settings,
        chartevent_chunksize=chartevent_chunksize,
        pipeline_start=t_pipeline,
    )

    t0 = time.time()
    acuity_scores = build_acuity_scores(
        materialized_views["oasis"],
        materialized_views["sapsii"],
    )
    _log_stage(t0, t_pipeline, "Built acuity scores")

    # ------------------------------------------------------------------
    # Stage 5: mistrust model + downstream evaluation (recomputed always)
    # ------------------------------------------------------------------
    model = EOLMistrustModel(repetitions=repetitions)
    mistrust_scores = _build_or_reuse_mistrust_scores(
        model=model,
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        note_corpus=note_corpus,
        reuse_dir=_reuse_dir,
        stage_cache_dir=_stage_cache_dir,
        pipeline_start=t_pipeline,
        autopsy_enabled=route_settings.autopsy_enabled,
    )
    if not route_settings.autopsy_enabled:
        mistrust_scores = _disable_autopsy_scores(mistrust_scores)
        _write_stage_cache_frame(_stage_cache_dir, "mistrust_scores.csv", mistrust_scores)

    t0 = time.time()
    final_model_table = build_final_model_table_from_code_status_targets(
        demographics=demographics,
        all_cohort=all_cohort,
        admissions=admissions,
        code_status_targets=code_status_targets,
        mistrust_scores=mistrust_scores,
    )
    _log_stage(t0, t_pipeline, f"Built final model table ({len(final_model_table)} rows)")

    validation["base_admissions_rows"] = int(len(base_admissions))
    validation["all_cohort_rows"] = int(len(all_cohort))
    validation["eol_cohort_rows"] = int(len(eol_cohort))
    validation["downstream_estimator_mode"] = route_settings.downstream_estimator_mode
    t0 = time.time()
    model_outputs = model.run(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        note_corpus=note_corpus,
        demographics=demographics,
        eol_cohort=eol_cohort,
        treatment_totals=treatment_totals,
        acuity_scores=acuity_scores,
        final_model_table=final_model_table,
        include_downstream_weight_summary=include_downstream_weight_summary,
        include_cdf_plot_data=include_cdf_plot_data,
        precomputed_mistrust_scores=mistrust_scores,
        score_columns=route_settings.score_columns,
        feature_configurations=route_settings.feature_configurations,
        downstream_estimator_factory_resolver=route_settings.downstream_estimator_factory_resolver,
    )
    if not route_settings.autopsy_enabled:
        model_outputs = _disable_autopsy_outputs(model_outputs)
    _log_stage(t0, t_pipeline, "Finished model.run() (downstream evaluation)")

    artifacts: dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]] = {
        "validation_summary": validation,
        "base_admissions": base_admissions,
        "demographics": demographics,
        "all_cohort": all_cohort,
        "eol_cohort": eol_cohort,
        "treatment_totals": treatment_totals,
        "note_corpus": note_corpus,
        "note_labels": note_labels,
        "chartevent_feature_matrix": feature_matrix,
        "acuity_scores": acuity_scores,
        "mistrust_scores": mistrust_scores,
        "final_model_table": final_model_table,
    }
    artifacts.update(model_outputs)

    if output_dir is not None:
        t0 = time.time()
        write_minimal_deliverables(
            {
                "base_admissions": base_admissions,
                "eol_cohort": eol_cohort,
                "all_cohort": all_cohort,
                "treatment_totals": treatment_totals,
                "chartevent_feature_matrix": feature_matrix,
                "note_labels": note_labels,
                "mistrust_scores": mistrust_scores,
                "acuity_scores": acuity_scores,
                "final_model_table": final_model_table,
            },
            output_dir=output_dir,
        )
        _log_stage(t0, t_pipeline, "Wrote deliverables + reuse cache to disk")

    t0 = time.time()
    comparison_outputs = build_paper_comparison_outputs(
        artifacts,
        repetitions=repetitions,
    )
    artifacts["paper_comparison"] = comparison_outputs
    if output_dir is not None:
        write_paper_comparison_artifacts(
            comparison_outputs,
            output_dir=output_dir / "paper_comparison",
            include_summary=compare_to_paper,
        )
    _log_stage(t0, t_pipeline, "Built & wrote paper table artifacts")

    _log_stage(t_pipeline, t_pipeline, "=== Pipeline complete ===")
    return artifacts


def run_task_demo(root: Path, config_path: Path) -> None:
    """Build a PyHealth sample dataset with the custom EOL mistrust YAML config."""

    global MIMIC3Dataset
    global EOLMistrustMortalityPredictionMIMIC3

    if MIMIC3Dataset is None:
        from pyhealth.datasets import MIMIC3Dataset as _MIMIC3Dataset

        MIMIC3Dataset = _MIMIC3Dataset
    if EOLMistrustMortalityPredictionMIMIC3 is None:
        from pyhealth.tasks.eol_mistrust import (
            EOLMistrustMortalityPredictionMIMIC3 as _EOLMistrustMortalityPredictionMIMIC3,
        )

        EOLMistrustMortalityPredictionMIMIC3 = _EOLMistrustMortalityPredictionMIMIC3

    base_dataset = MIMIC3Dataset(
        root=str(root),
        tables=["chartevents", "noteevents", "d_items"],
        dataset_name="eol_mistrust_mimic3",
        config_path=str(config_path),
        cache_dir=tempfile.mkdtemp(),
        dev=True,
    )
    base_dataset.stats()

    task = EOLMistrustMortalityPredictionMIMIC3(include_notes=True)
    sample_dataset = base_dataset.set_task(task, num_workers=1)
    sample_dataset.stats()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EOL mistrust example workflow.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing the combined EOL mistrust CSV exports.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the EOL mistrust dataset YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for writing the required CSV deliverables. "
            "When omitted, the script writes them under "
            "result_root/EOL_<route>_<timestamp>/result."
        ),
    )
    parser.add_argument(
        "--stream-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional base directory for streamed-stage reuse CSVs. "
            "When set, note/chartevent checkpoints are written under "
            "stream_cache_dir/{default|paper_like} as soon as each stage finishes. "
            "When omitted, the script writes them under "
            "result_root/EOL_<route>_<timestamp>/cache/{default|paper_like}."
        ),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help=(
            "Managed run archive root. Each invocation creates "
            "result_root/EOL_(normal|Paperlike)_<timestamp> with run summaries, "
            "runtime metadata, and default result/cache directories when explicit "
            "paths are not provided."
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=100,
        help="Number of downstream 60/40 evaluation repetitions.",
    )
    parser.add_argument(
        "--include-downstream-weight-summary",
        action="store_true",
        help="Also compute average downstream regularized weights across repetitions.",
    )
    parser.add_argument(
        "--include-cdf-plot-data",
        action="store_true",
        help="Also build empirical CDF data for race-based and trust-based treatment plots.",
    )
    parser.add_argument(
        "--compare-to-paper",
        action="store_true",
        help=(
            "Also write the human-readable paper comparison summary and print it. "
            "Structured paper comparison CSV/JSON artifacts under output_dir/paper_comparison "
            "are always generated."
        ),
    )
    parser.add_argument(
        "--task-demo",
        action="store_true",
        help="Also build a PyHealth sample dataset with the custom EOL mistrust task.",
    )
    parser.add_argument(
        "--note-chunksize",
        type=int,
        default=100_000,
        help="Chunk size for streamed noteevents processing.",
    )
    parser.add_argument(
        "--chartevent-chunksize",
        type=int,
        default=500_000,
        help="Chunk size for streamed chartevents processing.",
    )
    parser.add_argument(
        "--reuse-intermediates",
        type=Path,
        default=None,
        help=(
            "Path to a previous output directory containing cached CSV artifacts "
            "(note_corpus.csv, note_labels.csv, chartevent_feature_matrix.csv, "
            "code_status_targets.csv).  This may point either directly to the cache "
            "directory or to a base stream-cache dir containing mode subfolders. "
            "When set, the expensive CSV streaming stages are skipped and those "
            "frames are loaded from disk instead."
        ),
    )
    parser.add_argument(
        "--paper-like-dataset-prepare",
        action="store_true",
        help=(
            "Use notebook-style data preparation for treatment totals and chartevent "
            "feature extraction while keeping the default corrected pipeline available."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    route_settings = _build_route_settings(args.paper_like_dataset_prepare)
    result_root = getattr(args, "result_root", DEFAULT_RESULT_ROOT)
    managed_run = _prepare_managed_run_directories(
        result_root=result_root,
        route_settings=route_settings,
        output_dir=args.output_dir,
        stream_cache_dir=args.stream_cache_dir,
    )
    resolved_output_dir = managed_run["output_dir"]
    resolved_stream_cache_dir = managed_run["stream_cache_dir"]
    run_dir = managed_run["run_dir"]
    run_name = managed_run["run_name"]
    started_at = datetime.now()
    total_start = time.time()

    artifacts = build_eol_mistrust_outputs(
        root=args.root,
        repetitions=args.repetitions,
        include_downstream_weight_summary=args.include_downstream_weight_summary,
        include_cdf_plot_data=args.include_cdf_plot_data,
        compare_to_paper=args.compare_to_paper,
        output_dir=resolved_output_dir,
        stream_cache_dir=resolved_stream_cache_dir,
        note_chunksize=args.note_chunksize,
        chartevent_chunksize=args.chartevent_chunksize,
        reuse_intermediates=args.reuse_intermediates,
        paper_like_dataset_prepare=args.paper_like_dataset_prepare,
    )
    finished_at = datetime.now()
    total_runtime_seconds = time.time() - total_start
    _write_managed_run_artifacts(
        run_name=str(run_name),
        run_dir=Path(run_dir),
        route_settings=route_settings,
        args=args,
        resolved_output_dir=Path(resolved_output_dir),
        resolved_stream_cache_dir=Path(resolved_stream_cache_dir),
        started_at=started_at,
        finished_at=finished_at,
        total_runtime_seconds=total_runtime_seconds,
        artifacts=artifacts,
    )

    print("Validation summary:")
    print(artifacts["validation_summary"])
    print()
    print("Core artifact shapes:")
    for key in (
        "base_admissions",
        "all_cohort",
        "eol_cohort",
        "chartevent_feature_matrix",
        "note_labels",
        "mistrust_scores",
        "final_model_table",
    ):
        df = artifacts[key]
        if isinstance(df, pd.DataFrame):
            print(f"  {key}: {df.shape}")

    print()
    print(f"Managed run archive: {run_dir}")
    print(f"Wrote required deliverables to: {resolved_output_dir}")
    print(f"Wrote paper comparison artifacts to: {resolved_output_dir / 'paper_comparison'}")
    stream_cache_path = artifacts["validation_summary"].get("stream_cache_dir")
    if stream_cache_path is not None:
        print(f"Streamed-stage cache directory: {stream_cache_path}")

    if args.compare_to_paper:
        comparison_outputs = artifacts.get("paper_comparison")
        if isinstance(comparison_outputs, dict):
            _print_paper_comparison_summary(comparison_outputs)

    if args.task_demo:
        print()
        print("Running PyHealth task demo...")
        run_task_demo(root=args.root, config_path=args.config_path)


if __name__ == "__main__":
    main()
