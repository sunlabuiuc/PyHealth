r"""Run the EOL mistrust workflow.

Expected data root::

    EOL_Workspace/eol_mistrust_required_combined/
        mimiciii_clinical/
        mimiciii_notes/
        mimiciii_derived/

This example supports two uses:

1. Full research pipeline
   pandas-based preprocessing -> proxy construction -> downstream evaluation
   -> result writing
2. PyHealth-native proof/demo
   ``BaseDataset -> BaseTask -> BaseModel`` with optional normal-path
   ``Trainer.train() -> Trainer.evaluate()``

Managed runs are written under::

    EOL_Workspace/EOL_Result/EOL_(normal|Paperlike)_<timestamp>/


Recommended commands
--------------------

Full pipeline, normal::

    .\.venv\Scripts\python.exe examples\eol_mistrust_mortality_classifier.py --root EOL_Workspace\eol_mistrust_required_combined --repetitions 10

Full pipeline, paper-like::

    .\.venv\Scripts\python.exe examples\eol_mistrust_mortality_classifier.py --root EOL_Workspace\eol_mistrust_required_combined --paper-like-dataset-prepare --repetitions 10
    
Full pipeline . Route ablation, normal vs paper-like::

    .\.venv\Scripts\python.exe examples\eol_mistrust_mortality_classifier.py --root EOL_Workspace\eol_mistrust_required_combined --ablation-study --repetitions 1


Native proof, normal::

    .\.venv\Scripts\python.exe -m unittest tests.core.test_eol_mistrust_model.TestEOLMistrustClassifier.test_classifier_runs_end_to_end_for_normal_full_feature_path

Native proof, paper-like::

    .\.venv\Scripts\python.exe -m unittest tests.core.test_eol_mistrust_model.TestEOLMistrustClassifier.test_classifier_runs_end_to_end_for_paper_like_full_feature_path

Native train/eval demo, normal only::

    .\.venv\Scripts\python.exe examples\eol_mistrust_mortality_classifier.py --root EOL_Workspace\eol_mistrust_required_combined --task-demo --task-demo-train-eval
    
    
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA_ROOT = REPO_ROOT / "EOL_Workspace" / "eol_mistrust_required_combined"
DEFAULT_CONFIG_PATH = REPO_ROOT / "pyhealth" / "datasets" / "configs" / "eol_mistrust.yaml"
DEFAULT_RESULT_ROOT = REPO_ROOT / "EOL_Workspace" / "EOL_Result"
DEFAULT_NOTE_CHUNKSIZE = 100_000
DEFAULT_CHARTEVENT_CHUNKSIZE = 500_000


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

EOLMistrustClassifier = None
EOLMistrustDataset = None
EOLMistrustMortalityPredictionMIMIC3 = None
get_dataloader = None
split_by_patient = None
Trainer = None

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

RUN_TABLE1_RACES = ("BLACK", "WHITE")

RUN_TABLE1_COUNT_SPECS = [
    ("Population Size", None, None),
    ("Insurance Private", "insurance_group", "Private"),
    ("Insurance Public", "insurance_group", "Public"),
    ("Insurance Self-Pay", "insurance_group", "Self-Pay"),
    ("Discharge Deceased", "discharge_category", "Deceased"),
    ("Discharge Hospice", "discharge_category", "Hospice"),
    (
        "Discharge Skilled Nursing Facility",
        "discharge_category",
        "Skilled Nursing Facility",
    ),
    ("Gender F", "gender", "F"),
    ("Gender M", "gender", "M"),
]

RUN_TABLE1_CONTINUOUS_SPECS = {
    "Length of stay (median days)": "los_days",
    "Age (median years)": "age",
}

RUN_TABLE2_TREATMENT_ORDER = ["total_vent_min", "total_vaso_min"]

RUN_TABLE3_PROXY_ORDER = ["noncompliance", "autopsy"]

RUN_TABLE4_FEATURE_ORDER = [
    "autopsy_score_z",
    "negative_sentiment_score_z",
    "noncompliance_score_z",
    "oasis",
    "sapsii",
]

RUN_TABLE5_TASK_ORDER = ["Left AMA", "Code Status", "In-hospital mortality"]

RUN_TABLE5_CONFIGURATION_ORDER = [
    "Baseline",
    "Baseline + Race",
    "Baseline + Noncompliant",
    "Baseline + Autopsy",
    "Baseline + Neg-Sentiment",
    "Baseline + ALL",
]

RUN_TABLE6_TASK_ORDER = ["Left AMA", "Code Status", "In-hospital mortality"]

RUN_TABLE6_FEATURE_NAME_MAP = {
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

RUN_TABLE6_FEATURE_ORDER = [
    "age",
    "length-of-stay",
    "gender: female",
    "gender: male",
    "insurance: private",
    "insurance: public",
    "insurance: self-pay",
    "race: white",
    "race: black",
    "race: asian",
    "race: hispanic",
    "race: native american",
    "race: other",
    "noncompliant",
    "autopsy",
    "negative sentiment",
]


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


def _ordered_present_values(
    preferred_values: list[str] | tuple[str, ...],
    present_values: list[str],
) -> list[str]:
    """Return present values in a stable preferred-first order."""

    present_lookup = {str(value) for value in present_values}
    ordered = [value for value in preferred_values if value in present_lookup]
    remaining = sorted(present_lookup.difference(ordered))
    return ordered + remaining


def _build_run_table1_summary(eol_cohort: pd.DataFrame) -> pd.DataFrame:
    """Build run-only Table 1 demographics summaries."""

    cohort = eol_cohort[eol_cohort["race"].isin(RUN_TABLE1_RACES)].copy()
    totals = {race: int((cohort["race"] == race).sum()) for race in RUN_TABLE1_RACES}
    rows: list[dict[str, object]] = []

    for metric, column, target_value in RUN_TABLE1_COUNT_SPECS:
        for race in RUN_TABLE1_RACES:
            race_frame = cohort[cohort["race"] == race]
            if column is None:
                run_numeric = int(len(race_frame))
                run_display = str(run_numeric)
            else:
                run_numeric = int((race_frame[column] == target_value).sum())
                run_display = _format_count_percent(run_numeric, totals[race])
            rows.append(
                {
                    "metric": metric,
                    "race": race,
                    "run_value": run_display,
                    "run_numeric": run_numeric,
                }
            )

    for metric, series_name in RUN_TABLE1_CONTINUOUS_SPECS.items():
        for race in RUN_TABLE1_RACES:
            race_frame = cohort[cohort["race"] == race]
            series = pd.to_numeric(race_frame[series_name], errors="coerce").dropna()
            if series.empty:
                run_numeric = float("nan")
                run_lower = float("nan")
                run_upper = float("nan")
            else:
                run_numeric = float(series.median())
                run_lower = float(series.quantile(0.25))
                run_upper = float(series.quantile(0.75))
            rows.append(
                {
                    "metric": metric,
                    "race": race,
                    "summary_stat": "median_iqr",
                    "run_value": _format_continuous_summary(run_numeric, run_lower, run_upper),
                    "run_numeric": run_numeric,
                    "run_interval_lower": run_lower,
                    "run_interval_upper": run_upper,
                }
            )

    return pd.DataFrame(rows)


def _build_run_table2_summary(race_treatment_results: pd.DataFrame) -> pd.DataFrame:
    """Build run-only Table 2 treatment summaries."""

    if race_treatment_results.empty:
        return pd.DataFrame()

    return race_treatment_results[
        ["treatment", "n_black", "n_white", "median_black", "median_white", "pvalue"]
    ].copy()


def _build_run_table3_summary(
    feature_weight_summaries: dict[str, pd.DataFrame | dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Build run-only Table 3 top positive/negative proxy features."""

    rows: list[dict[str, object]] = []
    for model_name, weights_dict in feature_weight_summaries.items():
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

        positive = (
            all_weights[all_weights["weight"] > 0]
            .sort_values("weight", ascending=False)
            .head(3)
        )
        negative = (
            all_weights[all_weights["weight"] < 0]
            .sort_values("weight", ascending=True)
            .head(3)
        )
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


def _build_run_table4_summary(acuity_correlations: pd.DataFrame) -> pd.DataFrame:
    """Build run-only Table 4 correlation summaries."""

    keyed_rows: dict[tuple[str, str], dict[str, object]] = {}
    for row in acuity_correlations.itertuples(index=False):
        feature_a = str(getattr(row, "feature_a"))
        feature_b = str(getattr(row, "feature_b"))
        key = _canonical_pair(feature_a, feature_b)
        keyed_rows[key] = {
            "feature_a": key[0],
            "feature_b": key[1],
            "correlation": float(getattr(row, "correlation")),
        }
    return pd.DataFrame(keyed_rows.values())


def _build_run_table5_summary(downstream_auc_results: pd.DataFrame) -> pd.DataFrame:
    """Build run-only Table 5 downstream AUC summaries."""

    return downstream_auc_results[
        ["task", "configuration", "n_rows", "auc_mean", "auc_std", "n_valid_auc"]
    ].copy()


def _ensure_downstream_weight_results(
    artifacts: dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]],
    *,
    repetitions: int,
) -> pd.DataFrame:
    validation_summary = artifacts.get("validation_summary", {})
    autopsy_proxy_enabled = True
    if isinstance(validation_summary, dict):
        autopsy_proxy_enabled = bool(
            validation_summary.get("autopsy_proxy_enabled", True)
        )
    existing = artifacts.get("downstream_weight_results")
    if isinstance(existing, pd.DataFrame) and not existing.empty:
        return existing
    if isinstance(existing, pd.DataFrame) and not autopsy_proxy_enabled:
        return existing
    final_model_table = artifacts.get("final_model_table")
    if not isinstance(final_model_table, pd.DataFrame) or final_model_table.empty:
        return pd.DataFrame()
    computed = evaluate_downstream_average_weights(
        final_model_table=final_model_table,
        repetitions=repetitions,
    )
    if not autopsy_proxy_enabled and "feature" in computed.columns:
        computed = computed.loc[
            computed["feature"] != "autopsy_score_z"
        ].reset_index(drop=True)
    artifacts["downstream_weight_results"] = computed
    return computed


def _build_run_table6_summary(
    artifacts: dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]],
    *,
    repetitions: int,
    autopsy_proxy_enabled: bool,
) -> pd.DataFrame:
    """Build run-only Table 6 downstream weight summaries."""

    table6_source = _ensure_downstream_weight_results(artifacts, repetitions=repetitions)
    if not isinstance(table6_source, pd.DataFrame) or table6_source.empty:
        return pd.DataFrame()
    required = {"task", "configuration", "feature", "weight_mean", "weight_std"}
    if not required.issubset(table6_source.columns):
        return pd.DataFrame()
    work = table6_source.loc[
        table6_source["configuration"] == "Baseline + ALL"
    ].copy()
    work["feature"] = work["feature"].map(RUN_TABLE6_FEATURE_NAME_MAP)
    work = work.loc[work["feature"].notna()].copy()
    if not autopsy_proxy_enabled:
        work = work.loc[work["feature"] != "autopsy"].copy()
    if work.empty:
        return pd.DataFrame()
    return work.rename(
        columns={
            "weight_mean": "run_weight_mean",
            "weight_std": "run_weight_std",
        }
    )[
        ["task", "feature", "run_weight_mean", "run_weight_std"]
    ].reset_index(drop=True)


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
        _build_run_table1_summary(eol_cohort)
        if _has_columns(
            eol_cohort,
            {"race", "insurance_group", "discharge_category", "gender", "los_days", "age"},
        )
        else pd.DataFrame()
    )
    race_treatment = artifacts.get("race_treatment_results")
    table2 = (
        _build_run_table2_summary(race_treatment)
        if _has_columns(
            race_treatment,
            {"treatment", "n_black", "n_white", "median_black", "median_white", "pvalue"},
        )
        and not race_treatment.empty
        else pd.DataFrame()
    )
    table3 = _build_run_table3_summary(feature_weight_summaries)
    if not autopsy_proxy_enabled and not table3.empty and "proxy_model" in table3.columns:
        table3 = table3.loc[table3["proxy_model"] != "autopsy"].reset_index(drop=True)
    acuity_correlations = artifacts.get("acuity_correlations")
    table4 = (
        _build_run_table4_summary(acuity_correlations)
        if _has_columns(acuity_correlations, {"feature_a", "feature_b", "correlation"})
        else pd.DataFrame()
    )
    if not autopsy_proxy_enabled and not table4.empty:
        table4 = table4.loc[
            (table4["feature_a"] != "autopsy_score_z")
            & (table4["feature_b"] != "autopsy_score_z")
        ].reset_index(drop=True)
    downstream_auc_results = artifacts.get("downstream_auc_results")
    table5 = (
        _build_run_table5_summary(downstream_auc_results)
        if _has_columns(
            downstream_auc_results,
            {"task", "configuration", "n_rows", "auc_mean", "auc_std", "n_valid_auc"},
        )
        else pd.DataFrame()
    )
    if not autopsy_proxy_enabled and not table5.empty:
        table5 = table5.loc[
            table5["configuration"] != "Baseline + Autopsy"
        ].reset_index(drop=True)
    table6 = _build_run_table6_summary(
        artifacts,
        repetitions=repetitions,
        autopsy_proxy_enabled=autopsy_proxy_enabled,
    )

    lines = [
        "Run Table Results",
        f"Route: {'Paper-like' if dataset_prepare_mode == 'paper_like' else 'Normal' if dataset_prepare_mode == 'default' else dataset_prepare_mode}",
        f"dataset_prepare_mode: {dataset_prepare_mode}",
        f"autopsy_proxy_enabled: {autopsy_proxy_enabled}",
        f"repetitions: {repetitions}",
        "",
    ]

    if not table1.empty:
        lines.append("Table 1")
        table1_metric_order = _ordered_present_values(
            [metric for metric, _, _ in RUN_TABLE1_COUNT_SPECS]
            + list(RUN_TABLE1_CONTINUOUS_SPECS.keys()),
            table1["metric"].drop_duplicates().astype(str).tolist(),
        )
        for metric in table1_metric_order:
            metric_rows = table1.loc[table1["metric"] == metric]
            if metric_rows.empty:
                continue
            lines.append(f"- {metric}")
            for race in RUN_TABLE1_RACES:
                race_rows = metric_rows.loc[metric_rows["race"] == race]
                if race_rows.empty:
                    continue
                lines.append(f"  {race}: {race_rows.iloc[0]['run_value']}")
        lines.append("")

    if not table2.empty:
        lines.append("Table 2")
        table2_by_treatment = {
            str(row["treatment"]): row for _, row in table2.iterrows()
        }
        treatment_order = _ordered_present_values(
            RUN_TABLE2_TREATMENT_ORDER,
            list(table2_by_treatment.keys()),
        )
        for treatment in treatment_order:
            row = table2_by_treatment.get(treatment)
            if row is None:
                continue
            lines.append(f"- {row.treatment}")
            lines.append(
                f"  BLACK: n={int(row.n_black)}, median={float(row.median_black):.1f}"
            )
            lines.append(
                f"  WHITE: n={int(row.n_white)}, median={float(row.median_white):.1f}"
            )
            if not pd.isna(row.pvalue):
                lines.append(f"  pvalue: {float(row.pvalue)}")
        lines.append("")

    if not table3.empty:
        lines.append("Table 3")
        proxy_order = _ordered_present_values(
            RUN_TABLE3_PROXY_ORDER,
            table3["proxy_model"].drop_duplicates().astype(str).tolist(),
        )
        for proxy_model in proxy_order:
            lines.append(f"- {proxy_model}")
            proxy_rows = table3.loc[table3["proxy_model"] == proxy_model]
            for direction in ("positive", "negative"):
                direction_rows = proxy_rows.loc[
                    proxy_rows["direction"] == direction
                ].sort_values("rank")
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
        table4_keyed = {
            _canonical_pair(row["feature_a"], row["feature_b"]): row
            for _, row in table4.iterrows()
        }
        feature_rank = {
            feature_name: index
            for index, feature_name in enumerate(RUN_TABLE4_FEATURE_ORDER)
        }
        for key in sorted(
            table4_keyed.keys(),
            key=lambda pair: (
                feature_rank.get(pair[0], len(feature_rank)),
                feature_rank.get(pair[1], len(feature_rank)),
                pair[0],
                pair[1],
            ),
        ):
            row = table4_keyed.get(key)
            if row is None:
                continue
            lines.append(
                f"- {row.feature_a} vs {row.feature_b}: {float(row.correlation):.3f}"
            )
        lines.append("")

    if not table5.empty:
        lines.append("Table 5")
        table5_keyed = {
            (str(row["task"]), str(row["configuration"])): row
            for _, row in table5.iterrows()
        }
        task_order = _ordered_present_values(
            RUN_TABLE5_TASK_ORDER,
            table5["task"].drop_duplicates().astype(str).tolist(),
        )
        for task_name in task_order:
            present_configs = table5.loc[
                table5["task"] == task_name, "configuration"
            ].astype(str).tolist()
            config_order = _ordered_present_values(
                RUN_TABLE5_CONFIGURATION_ORDER,
                present_configs,
            )
            for configuration in config_order:
                row = table5_keyed.get((task_name, configuration))
                if row is None:
                    continue
                lines.append(f"- {row.task} | {row.configuration}")
                lines.append(f"  n_rows: {int(row.n_rows)}")
                lines.append(f"  auc_mean: {float(row.auc_mean):.3f}")
                lines.append(f"  auc_std: {float(row.auc_std):.3f}")
        lines.append("")

    if not table6.empty:
        lines.append("Table 6")
        table6_task_order = _ordered_present_values(
            RUN_TABLE6_TASK_ORDER,
            table6["task"].drop_duplicates().astype(str).tolist(),
        )
        for task_name in table6_task_order:
            lines.append(f"- {task_name}")
            task_rows = table6.loc[table6["task"] == task_name]
            task_row_lookup = {
                str(row["feature"]): row for _, row in task_rows.iterrows()
            }
            feature_order = _ordered_present_values(
                RUN_TABLE6_FEATURE_ORDER,
                list(task_row_lookup.keys()),
            )
            for feature_name in feature_order:
                row = task_row_lookup.get(feature_name)
                if row is None:
                    continue
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


def _build_ablation_run_name(timestamp: str) -> str:
    """Return the managed run directory name for the route ablation study."""

    return f"EOL_ablation_normal_vs_paperlike_{timestamp}"


def _prepare_managed_run_directories(
    *,
    result_root: Path,
    route_settings: _RouteSettings,
    output_dir: Path | None,
) -> dict[str, Path | str]:
    """Create a managed run archive directory and resolve the output path."""

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
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "output_dir": resolved_output_dir,
    }


def _prepare_ablation_run_directories(
    *,
    result_root: Path,
    output_dir: Path | None,
) -> dict[str, Path | str]:
    """Create a managed run archive directory for the normal-vs-paper-like study."""

    timestamp = _current_run_timestamp()
    base_name = _build_ablation_run_name(timestamp)
    run_name = base_name
    run_dir = output_dir if output_dir is not None else result_root / run_name
    suffix = 1
    while run_dir.exists():
        run_name = f"{base_name}_{suffix:02d}"
        run_dir = (
            output_dir.parent / run_name if output_dir is not None else result_root / run_name
        )
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return {
        "run_name": run_name,
        "run_dir": run_dir,
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
        f"run_table_summary_file: {run_dir / 'run_table_summary.txt'}",
        f"paper_like_dataset_prepare: {args.paper_like_dataset_prepare}",
        f"repetitions: {args.repetitions}",
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
        started_at=started_at,
        finished_at=finished_at,
        total_runtime_seconds=total_runtime_seconds,
        artifacts=artifacts,
    )
    (run_dir / "RUN_SUMMARY.txt").write_text(summary_text, encoding="utf-8")

    manifest = {
        "managed_run_name": run_name,
        "managed_run_dir": str(run_dir),
        "route_mode": route_settings.mode_name,
        "autopsy_proxy_enabled": route_settings.autopsy_enabled,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "total_runtime_seconds": round(float(total_runtime_seconds), 6),
        "result_dir": str(resolved_output_dir),
        "paper_like_dataset_prepare": bool(args.paper_like_dataset_prepare),
        "repetitions": int(args.repetitions),
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


def _route_display_label(route_settings: _RouteSettings) -> str:
    """Return the user-facing display name for a route."""

    return "Paper-like" if route_settings.mode_name == "paper_like" else "Normal"


def _namespace_from_args_like(args: argparse.Namespace | object) -> argparse.Namespace:
    """Return an argparse.Namespace built from an argparse-like object."""

    if isinstance(args, argparse.Namespace):
        return argparse.Namespace(**vars(args))

    attributes = {
        name: getattr(args, name)
        for name in dir(args)
        if not name.startswith("_") and not callable(getattr(args, name))
    }
    return argparse.Namespace(**attributes)


def _route_auc_summary_lines(
    artifacts: dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]],
) -> list[str]:
    """Return a compact AUC summary for the route ablation appendix output."""

    auc_results = artifacts.get("downstream_auc_results")
    if not isinstance(auc_results, pd.DataFrame) or auc_results.empty:
        return []
    required = {"task", "configuration", "auc_mean", "auc_std"}
    if not required.issubset(auc_results.columns):
        return []

    lines: list[str] = []
    for task_name in RUN_TABLE5_TASK_ORDER:
        row = auc_results.loc[
            (auc_results["task"] == task_name)
            & (auc_results["configuration"] == "Baseline + ALL")
        ]
        if row.empty:
            continue
        selected = row.iloc[0]
        lines.append(
            f"  {task_name} | Baseline + ALL: "
            f"auc_mean: {float(selected['auc_mean']):.3f}, "
            f"auc_std: {float(selected['auc_std']):.3f}"
        )
    return lines


def _route_has_autopsy_weight(
    artifacts: dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]],
) -> bool:
    """Return whether the run includes an autopsy weight in Table 6 output."""

    weights = artifacts.get("downstream_weight_results")
    if not isinstance(weights, pd.DataFrame) or weights.empty:
        return False
    if "feature" not in weights.columns:
        return False
    return bool((weights["feature"] == "autopsy_score_z").any())


def _render_ablation_summary(
    *,
    run_name: str,
    run_dir: Path,
    repetitions: int,
    started_at: datetime,
    finished_at: datetime,
    total_runtime_seconds: float,
    route_results: list[dict[str, object]],
) -> str:
    """Render a compact route-ablation summary for Normal vs Paper-like."""

    lines = [
        "Route Ablation Study",
        f"managed_run_name: {run_name}",
        f"managed_run_dir: {run_dir}",
        "ablation_variable: route (Normal vs Paper-like)",
        "ablation_focus: corrected default path without autopsy vs paper-like path with autopsy",
        f"started_at: {started_at.isoformat(timespec='seconds')}",
        f"finished_at: {finished_at.isoformat(timespec='seconds')}",
        f"total_runtime_seconds: {total_runtime_seconds:.3f}",
        f"repetitions: {repetitions}",
        f"command: {' '.join(sys.argv)}",
        "",
    ]

    for route_result in route_results:
        route_settings = route_result["route_settings"]
        artifacts = route_result["artifacts"]
        route_run_dir = route_result["run_dir"]
        route_output_dir = route_result["output_dir"]
        validation_summary = artifacts.get("validation_summary", {})
        final_model_table = artifacts.get("final_model_table")
        final_model_rows = (
            int(final_model_table.shape[0])
            if isinstance(final_model_table, pd.DataFrame)
            else 0
        )
        autopsy_proxy_enabled = False
        if isinstance(validation_summary, dict):
            autopsy_proxy_enabled = bool(
                validation_summary.get("autopsy_proxy_enabled", False)
            )

        lines.extend(
            [
                f"{_route_display_label(route_settings)}:",
                f"  route_mode: {route_settings.mode_name}",
                f"  autopsy_proxy_enabled: {autopsy_proxy_enabled}",
                f"  final_model_table_rows: {final_model_rows}",
                f"  run_dir: {route_run_dir}",
                f"  result_dir: {route_output_dir}",
                f"  run_summary_file: {route_run_dir / 'RUN_SUMMARY.txt'}",
                f"  run_table_summary_file: {route_run_dir / 'run_table_summary.txt'}",
                f"  has_autopsy_weight: {_route_has_autopsy_weight(artifacts)}",
            ]
        )
        lines.extend(_route_auc_summary_lines(artifacts))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _run_single_managed_route(
    *,
    root: Path,
    repetitions: int,
    route_settings: _RouteSettings,
    route_output_dir: Path,
    route_run_dir: Path,
    route_run_name: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Run one route and write the standard managed-run artifacts."""

    route_run_dir.mkdir(parents=True, exist_ok=True)
    route_output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()
    total_start = time.time()
    artifacts = build_eol_mistrust_outputs(
        root=root,
        repetitions=repetitions,
        output_dir=route_output_dir,
        paper_like_dataset_prepare=(route_settings.mode_name == "paper_like"),
    )
    finished_at = datetime.now()
    total_runtime_seconds = time.time() - total_start

    route_args = _namespace_from_args_like(args)
    route_args.paper_like_dataset_prepare = route_settings.mode_name == "paper_like"
    _write_managed_run_artifacts(
        run_name=route_run_name,
        run_dir=route_run_dir,
        route_settings=route_settings,
        args=route_args,
        resolved_output_dir=route_output_dir,
        started_at=started_at,
        finished_at=finished_at,
        total_runtime_seconds=total_runtime_seconds,
        artifacts=artifacts,
    )
    return {
        "route_settings": route_settings,
        "run_dir": route_run_dir,
        "output_dir": route_output_dir,
        "artifacts": artifacts,
        "started_at": started_at,
        "finished_at": finished_at,
        "total_runtime_seconds": total_runtime_seconds,
    }


def _run_route_ablation_study(args: argparse.Namespace) -> None:
    """Run the explicit Normal vs Paper-like route ablation study."""

    if getattr(args, "task_demo", False) or getattr(args, "task_demo_train_eval", False):
        raise ValueError(
            "--ablation-study cannot be combined with --task-demo or "
            "--task-demo-train-eval."
        )

    result_root = getattr(args, "result_root", DEFAULT_RESULT_ROOT)
    ablation_run = _prepare_ablation_run_directories(
        result_root=result_root,
        output_dir=args.output_dir,
    )
    run_name = str(ablation_run["run_name"])
    run_dir = Path(ablation_run["run_dir"])
    started_at = datetime.now()
    total_start = time.time()

    normal_settings = _build_route_settings(False)
    paperlike_settings = _build_route_settings(True)
    route_results = [
        _run_single_managed_route(
            root=args.root,
            repetitions=args.repetitions,
            route_settings=normal_settings,
            route_output_dir=run_dir / "normal" / "result",
            route_run_dir=run_dir / "normal",
            route_run_name=f"{run_name}_normal",
            args=args,
        ),
        _run_single_managed_route(
            root=args.root,
            repetitions=args.repetitions,
            route_settings=paperlike_settings,
            route_output_dir=run_dir / "paper_like" / "result",
            route_run_dir=run_dir / "paper_like",
            route_run_name=f"{run_name}_paper_like",
            args=args,
        ),
    ]
    finished_at = datetime.now()
    total_runtime_seconds = time.time() - total_start

    summary_text = _render_ablation_summary(
        run_name=run_name,
        run_dir=run_dir,
        repetitions=int(args.repetitions),
        started_at=started_at,
        finished_at=finished_at,
        total_runtime_seconds=total_runtime_seconds,
        route_results=route_results,
    )
    (run_dir / "ABLATION_SUMMARY.txt").write_text(summary_text, encoding="utf-8")

    print(f"Managed route ablation archive: {run_dir}")
    print(f"Wrote ablation summary to: {run_dir / 'ABLATION_SUMMARY.txt'}")


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


def _build_mistrust_scores(
    *,
    model: object,
    feature_matrix: pd.DataFrame,
    note_labels: pd.DataFrame,
    note_corpus: pd.DataFrame,
    pipeline_start: float,
    autopsy_enabled: bool,
) -> pd.DataFrame:
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
        _log_stage(t_total, pipeline_start, "Built mistrust scores (proxy models + sentiment)")
        return mistrust_scores

    t0 = time.time()
    mistrust_scores = model.build_mistrust_scores(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        note_corpus=note_corpus,
    )
    _log_stage(t0, pipeline_start, "Built mistrust scores (proxy models + sentiment)")
    return mistrust_scores


def _build_note_artifacts(
    *,
    noteevents_csv_path: Path,
    all_cohort: pd.DataFrame,
    route_settings: _RouteSettings,
    note_chunksize: int,
    pipeline_start: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], pd.DataFrame]:
    t0 = time.time()
    note_corpus = build_note_corpus_from_csv(
        noteevents_csv_path=noteevents_csv_path,
        all_hadm_ids=all_cohort["hadm_id"],
        categories=None,
        chunksize=note_chunksize,
    )
    note_present_hadm_ids = _note_present_hadm_ids(note_corpus)
    filtered_all_cohort = all_cohort.loc[all_cohort["hadm_id"].isin(note_present_hadm_ids)].copy()
    note_corpus = note_corpus.loc[note_corpus["hadm_id"].isin(note_present_hadm_ids)].copy()
    _log_stage(t0, pipeline_start, f"Streamed note corpus ({len(note_corpus)} rows)")

    t0 = time.time()
    note_labels = build_note_labels_from_csv(
        noteevents_csv_path=noteevents_csv_path,
        all_hadm_ids=note_present_hadm_ids,
        autopsy_label_mode=route_settings.autopsy_label_mode,
        chunksize=note_chunksize,
    )
    _log_stage(t0, pipeline_start, f"Streamed note labels ({len(note_labels)} rows)")
    return note_corpus, note_labels, note_present_hadm_ids, filtered_all_cohort


def _build_chartevent_artifacts(
    *,
    chartevents_csv_path: Path,
    d_items: pd.DataFrame,
    note_present_hadm_ids: list[int],
    route_settings: _RouteSettings,
    chartevent_chunksize: int,
    pipeline_start: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    feature_matrix, code_status_targets = build_chartevent_artifacts_from_csv(
        chartevents_csv_path=chartevents_csv_path,
        d_items=d_items,
        all_hadm_ids=note_present_hadm_ids,
        chunksize=chartevent_chunksize,
        paper_like=route_settings.autopsy_enabled,
        code_status_mode=route_settings.code_status_mode,
    )
    _log_stage(t0, pipeline_start, f"Streamed chartevents ({len(feature_matrix)} feature rows)")
    return feature_matrix, code_status_targets


def build_eol_mistrust_outputs(
    root: Path,
    repetitions: int = 100,
    output_dir: Path | None = None,
    paper_like_dataset_prepare: bool = False,
) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]]:
    """Run the local end-to-end EOL mistrust workflow over downloaded CSV files."""

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
    note_corpus, note_labels, note_present_hadm_ids, all_cohort = _build_note_artifacts(
        noteevents_csv_path=noteevents_csv_path,
        all_cohort=all_cohort,
        route_settings=route_settings,
        note_chunksize=DEFAULT_NOTE_CHUNKSIZE,
        pipeline_start=t_pipeline,
    )

    # ------------------------------------------------------------------
    # Stage 4: chartevents feature matrix + code status (SLOW — stream chartevents.csv)
    # ------------------------------------------------------------------
    feature_matrix, code_status_targets = _build_chartevent_artifacts(
        chartevents_csv_path=chartevents_csv_path,
        d_items=d_items,
        note_present_hadm_ids=note_present_hadm_ids,
        route_settings=route_settings,
        chartevent_chunksize=DEFAULT_CHARTEVENT_CHUNKSIZE,
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
    mistrust_scores = _build_mistrust_scores(
        model=model,
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        note_corpus=note_corpus,
        pipeline_start=t_pipeline,
        autopsy_enabled=route_settings.autopsy_enabled,
    )
    if not route_settings.autopsy_enabled:
        mistrust_scores = _disable_autopsy_scores(mistrust_scores)

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
        include_downstream_weight_summary=False,
        include_cdf_plot_data=False,
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
        _log_stage(t0, t_pipeline, "Wrote deliverables to disk")

    _log_stage(t_pipeline, t_pipeline, "=== Pipeline complete ===")
    return artifacts


def run_task_demo(
    root: Path,
    config_path: Path,
    dataset_prepare_mode: str = "default",
    train_and_evaluate: bool = False,
) -> None:
    """Build a PyHealth sample dataset with the custom EOL mistrust YAML config."""

    global EOLMistrustClassifier
    global EOLMistrustDataset
    global EOLMistrustMortalityPredictionMIMIC3
    global get_dataloader
    global split_by_patient
    global Trainer

    if train_and_evaluate and dataset_prepare_mode != "default":
        raise ValueError(
            "Native train/eval demo is only supported for the default normal path."
        )

    if EOLMistrustDataset is None or get_dataloader is None or split_by_patient is None:
        from pyhealth.datasets import (
            EOLMistrustDataset as _EOLMistrustDataset,
            get_dataloader as _get_dataloader,
            split_by_patient as _split_by_patient,
        )

        EOLMistrustDataset = _EOLMistrustDataset
        get_dataloader = _get_dataloader
        split_by_patient = _split_by_patient
    if EOLMistrustClassifier is None:
        from pyhealth.models import EOLMistrustClassifier as _EOLMistrustClassifier

        EOLMistrustClassifier = _EOLMistrustClassifier
    if EOLMistrustMortalityPredictionMIMIC3 is None:
        from pyhealth.tasks.eol_mistrust import (
            EOLMistrustMortalityPredictionMIMIC3 as _EOLMistrustMortalityPredictionMIMIC3,
        )

        EOLMistrustMortalityPredictionMIMIC3 = _EOLMistrustMortalityPredictionMIMIC3
    if train_and_evaluate and Trainer is None:
        from pyhealth.trainer import Trainer as _Trainer

        Trainer = _Trainer

    def _close_unique_datasets(*datasets: object) -> None:
        seen: set[int] = set()
        for dataset in datasets:
            if dataset is None:
                continue
            dataset_id = id(dataset)
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            close_fn = getattr(dataset, "close", None)
            if callable(close_fn):
                close_fn()

    with tempfile.TemporaryDirectory() as cache_dir:
        base_dataset = EOLMistrustDataset(
            root=str(root),
            tables=None,
            dataset_name="eol_mistrust",
            config_path=str(config_path),
            cache_dir=cache_dir,
            dev=True,
            dataset_prepare_mode=dataset_prepare_mode,
        )
        base_dataset.stats()

        task = EOLMistrustMortalityPredictionMIMIC3(
            include_notes=True,
            dataset_prepare_mode=dataset_prepare_mode,
        )
        sample_dataset = base_dataset.set_task(task, num_workers=1)
        train_dataset = None
        val_dataset = None
        test_dataset = None
        try:
            model = EOLMistrustClassifier(dataset=sample_dataset)
            if train_and_evaluate:
                train_dataset, val_dataset, test_dataset = split_by_patient(
                    sample_dataset,
                    [0.6, 0.2, 0.2],
                )
                train_dataloader = get_dataloader(
                    train_dataset, batch_size=32, shuffle=True
                )
                val_dataloader = get_dataloader(
                    val_dataset, batch_size=32, shuffle=False
                )
                test_dataloader = get_dataloader(
                    test_dataset, batch_size=32, shuffle=False
                )
                trainer = Trainer(
                    model=model,
                    metrics=["accuracy"],
                    enable_logging=False,
                )
                trainer.train(
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    test_dataloader=test_dataloader,
                    epochs=1,
                    monitor="accuracy",
                    load_best_model_at_last=False,
                )
                scores = trainer.evaluate(test_dataloader)
                print(f"Task demo evaluation scores: {scores}")
            else:
                batch = next(
                    iter(get_dataloader(sample_dataset, batch_size=2, shuffle=False))
                )
                outputs = model(**batch)
                print(f"Task demo forward keys: {sorted(outputs.keys())}")
        finally:
            _close_unique_datasets(sample_dataset, train_dataset, val_dataset, test_dataset)


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
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help=(
            "Managed run archive root. Each invocation creates "
            "result_root/EOL_(normal|Paperlike)_<timestamp> with run summaries, "
            "runtime metadata, and a default result directory when an explicit "
            "output path is not provided."
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=100,
        help="Number of downstream 60/40 evaluation repetitions.",
    )
    parser.add_argument(
        "--ablation-study",
        action="store_true",
        help=(
            "Run the explicit Normal vs Paper-like route ablation study and "
            "write an ABLATION_SUMMARY.txt under a managed run directory."
        ),
    )
    parser.add_argument(
        "--task-demo",
        action="store_true",
        help="Also build a PyHealth sample dataset with the custom EOL mistrust task.",
    )
    parser.add_argument(
        "--task-demo-train-eval",
        action="store_true",
        help=(
            "When used with --task-demo, run a one-epoch native PyHealth "
            "train/evaluate demo on the default normal path."
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
    if getattr(args, "ablation_study", False):
        _run_route_ablation_study(args)
        return

    route_settings = _build_route_settings(args.paper_like_dataset_prepare)
    result_root = getattr(args, "result_root", DEFAULT_RESULT_ROOT)
    managed_run = _prepare_managed_run_directories(
        result_root=result_root,
        route_settings=route_settings,
        output_dir=args.output_dir,
    )
    resolved_output_dir = managed_run["output_dir"]
    run_dir = managed_run["run_dir"]
    run_name = managed_run["run_name"]
    started_at = datetime.now()
    total_start = time.time()

    artifacts = build_eol_mistrust_outputs(
        root=args.root,
        repetitions=args.repetitions,
        output_dir=resolved_output_dir,
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

    task_demo = getattr(args, "task_demo", False)
    task_demo_train_eval = getattr(args, "task_demo_train_eval", False)
    if task_demo or task_demo_train_eval:
        print()
        print("Running PyHealth task demo...")
        run_task_demo(
            root=args.root,
            config_path=args.config_path,
            dataset_prepare_mode=route_settings.mode_name,
            train_and_evaluate=task_demo_train_eval,
        )


if __name__ == "__main__":
    main()
