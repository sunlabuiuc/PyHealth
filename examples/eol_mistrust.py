"""Example workflow for the EOL mistrust study pipeline.

This script assumes you have already exported and combined the required MIMIC-III
tables into a local directory such as:

    downloads/eol_mistrust_required_combined/
        mimiciii_clinical/
        mimiciii_notes/
        mimiciii_derived/

It demonstrates two related flows:

1. the study-style preprocessing + modeling pipeline built on pandas tables
2. an optional PyHealth task demo using the custom EOL mistrust YAML config

Implementation note: the sentiment metric in this repo uses the existing
transformers+torch stack rather than the original Pattern backend from the
reference notebooks. The example therefore builds the sentiment corpus from
`Discharge summary` notes only, while label extraction still uses all non-error
notes.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets.eol_mistrust import (
    build_acuity_scores,
    build_all_cohort,
    build_base_admissions,
    build_chartevent_artifacts_from_csv,
    build_demographics_table,
    build_eol_cohort,
    build_final_model_table_from_code_status_targets,
    build_note_corpus_from_csv,
    build_note_labels_from_csv,
    build_treatment_totals,
    write_minimal_deliverables,
)
from pyhealth.models.eol_mistrust import EOLMistrustModel
from pyhealth.tasks.eol_mistrust import EOLMistrustMortalityPredictionMIMIC3


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "downloads" / "eol_mistrust_required_combined"
DEFAULT_CONFIG_PATH = REPO_ROOT / "pyhealth" / "datasets" / "configs" / "eol_mistrust.yaml"

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


def build_eol_mistrust_outputs(
    root: Path,
    repetitions: int = 100,
    include_downstream_weight_summary: bool = False,
    include_cdf_plot_data: bool = False,
    output_dir: Path | None = None,
    note_chunksize: int = 100_000,
    chartevent_chunksize: int = 500_000,
) -> dict[str, pd.DataFrame | dict[str, pd.DataFrame] | dict[str, object]]:
    """Run the local end-to-end EOL mistrust workflow over downloaded CSV files."""

    raw_tables, materialized_views = load_eol_mistrust_tables(root)
    validation = {
        "schema_name": "mimiciii",
        "database_flavor": "postgresql",
        "raw_tables": sorted(raw_tables.keys()),
        "materialized_views": sorted(materialized_views.keys()),
    }

    admissions = raw_tables["admissions"]
    patients = raw_tables["patients"]
    icustays = raw_tables["icustays"]
    d_items = raw_tables["d_items"]
    noteevents_csv_path = root / EVENT_TABLE_PATHS["noteevents"]
    chartevents_csv_path = root / EVENT_TABLE_PATHS["chartevents"]

    base_admissions = build_base_admissions(admissions, patients)
    demographics = build_demographics_table(base_admissions)
    all_cohort = build_all_cohort(base_admissions, icustays)
    eol_cohort = build_eol_cohort(base_admissions, demographics)
    treatment_totals = build_treatment_totals(
        icustays=icustays,
        ventdurations=materialized_views["ventdurations"],
        vasopressordurations=materialized_views["vasopressordurations"],
    )
    note_corpus = build_note_corpus_from_csv(
        noteevents_csv_path=noteevents_csv_path,
        all_hadm_ids=all_cohort["hadm_id"],
        categories=["Discharge summary"],
        chunksize=note_chunksize,
    )
    note_labels = build_note_labels_from_csv(
        noteevents_csv_path=noteevents_csv_path,
        all_hadm_ids=all_cohort["hadm_id"],
        chunksize=note_chunksize,
    )
    feature_matrix, code_status_targets = build_chartevent_artifacts_from_csv(
        chartevents_csv_path=chartevents_csv_path,
        d_items=d_items,
        all_hadm_ids=all_cohort["hadm_id"],
        chunksize=chartevent_chunksize,
    )
    acuity_scores = build_acuity_scores(
        materialized_views["oasis"],
        materialized_views["sapsii"],
    )

    model = EOLMistrustModel(repetitions=repetitions)
    mistrust_scores = model.build_mistrust_scores(
        feature_matrix=feature_matrix,
        note_labels=note_labels,
        note_corpus=note_corpus,
    )
    final_model_table = build_final_model_table_from_code_status_targets(
        demographics=demographics,
        all_cohort=all_cohort,
        admissions=admissions,
        code_status_targets=code_status_targets,
        mistrust_scores=mistrust_scores,
    )
    validation["base_admissions_rows"] = int(len(base_admissions))
    validation["all_cohort_rows"] = int(len(all_cohort))
    validation["eol_cohort_rows"] = int(len(eol_cohort))
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
    )

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

    return artifacts


def run_task_demo(root: Path, config_path: Path) -> None:
    """Build a PyHealth sample dataset with the custom EOL mistrust YAML config."""

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
        help="Optional directory for writing the required CSV deliverables.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifacts = build_eol_mistrust_outputs(
        root=args.root,
        repetitions=args.repetitions,
        include_downstream_weight_summary=args.include_downstream_weight_summary,
        include_cdf_plot_data=args.include_cdf_plot_data,
        output_dir=args.output_dir,
        note_chunksize=args.note_chunksize,
        chartevent_chunksize=args.chartevent_chunksize,
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

    if args.output_dir is not None:
        print()
        print(f"Wrote required deliverables to: {args.output_dir}")

    if args.task_demo:
        print()
        print("Running PyHealth task demo...")
        run_task_demo(root=args.root, config_path=args.config_path)


if __name__ == "__main__":
    main()
