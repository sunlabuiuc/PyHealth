import argparse
import json
import os
from datetime import datetime
from typing import List, Set

import numpy as np
from pyhealth.datasets import MIMIC3NoteDataset
from pyhealth.metrics import multilabel_metrics_fn
from pyhealth.models.sdoh_icd9_llm import SDOHICD9LLM
from pyhealth.tasks.sdoh_icd9_detection import TARGET_CODES
from pyhealth.tasks.sdoh_utils import codes_to_multihot, load_sdoh_icd9_labels


def parse_args():
    """Parse CLI arguments for SDOH ICD-9 note evaluation."""
    parser = argparse.ArgumentParser(
        description="Admission-level SDOH ICD-9 evaluation with per-note LLM calls."
    )
    parser.add_argument("--mimic-root", required=True, help="Root folder for MIMIC-III CSVs")
    parser.add_argument("--label-csv-path", required=True, help="Path to sdoh_icd9_dataset.csv")
    parser.add_argument(
        "--label-source",
        default="manual",
        choices=["manual", "true"],
        help="Which labels to use as primary ground truth.",
    )
    parser.add_argument(
        "--max-notes",
        default="all",
        help="Limit notes per admission (e.g., 1, 2, 5, or 'all').",
    )
    parser.add_argument(
        "--max-admissions",
        default="all",
        help="Limit admissions to process (e.g., 5 or 'all').",
    )
    parser.add_argument(
        "--note-categories",
        help="Comma-separated NOTE_CATEGORY values to include (optional).",
    )
    parser.add_argument("--output-dir", default=".", help="Directory to save outputs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    """Run admission-level evaluation with per-note LLM calls."""
    args = parse_args()
    target_codes = list(TARGET_CODES)
    label_map = load_sdoh_icd9_labels(args.label_csv_path, target_codes)

    include_categories = (
        [cat.strip() for cat in args.note_categories.split(",")]
        if args.note_categories
        else None
    )
    if str(args.max_notes).lower() == "all":
        max_notes = None
    else:
        try:
            max_notes = int(args.max_notes)
        except ValueError as exc:
            raise ValueError("--max-notes must be an integer or 'all'") from exc
        if max_notes <= 0:
            raise ValueError("--max-notes must be a positive integer or 'all'")
    if str(args.max_admissions).lower() == "all":
        max_admissions = None
    else:
        try:
            max_admissions = int(args.max_admissions)
        except ValueError as exc:
            raise ValueError("--max-admissions must be an integer or 'all'") from exc
    if max_admissions <= 0:
        raise ValueError("--max-admissions must be a positive integer or 'all'")

    hadm_ids = list(label_map.keys())
    if max_admissions is not None:
        hadm_ids = hadm_ids[:max_admissions]
        label_map = {hadm_id: label_map[hadm_id] for hadm_id in hadm_ids}

    note_dataset = MIMIC3NoteDataset(
        root=args.mimic_root,
        target_codes=target_codes,
        hadm_ids=hadm_ids,
        include_categories=include_categories,
    )
    sample_dataset = note_dataset.set_task(
        label_source=args.label_source,
        label_map=label_map,
    )

    dry_run = args.dry_run or not os.environ.get("OPENAI_API_KEY")
    model = SDOHICD9LLM(
        target_codes=target_codes,
        dry_run=dry_run,
        max_notes=max_notes,
    )

    results = []
    predicted_codes_all: List[Set[str]] = []
    manual_codes_all: List[Set[str]] = []
    true_codes_all: List[Set[str]] = []

    for sample in sample_dataset:
        predicted_codes, note_results = model.predict_admission_with_notes(
            sample["notes"],
            sample.get("note_categories"),
            sample.get("chartdates"),
        )
        predicted_codes_all.append(predicted_codes)
        visit_id = str(sample.get("visit_id", ""))
        label_entry = label_map.get(visit_id, {"manual": set(), "true": set()})
        manual_codes = set(label_entry["manual"])
        true_codes = set(label_entry["true"])
        manual_codes_all.append(manual_codes)
        true_codes_all.append(true_codes)

        results.append(
            {
                "visit_id": sample.get("visit_id"),
                "patient_id": sample.get("patient_id"),
                "num_notes": sample.get("num_notes"),
                "text_length": sample.get("text_length"),
                "is_gap_case": sample.get("is_gap_case"),
                "manual_codes": ",".join(sorted(manual_codes)),
                "true_codes": ",".join(sorted(true_codes)),
                "predicted_codes": ",".join(sorted(predicted_codes)),
                "note_results": json.dumps(note_results),
            }
        )

    y_pred = np.stack(
        [codes_to_multihot(codes, target_codes).numpy() for codes in predicted_codes_all],
        axis=0,
    )
    y_manual = np.stack(
        [codes_to_multihot(codes, target_codes).numpy() for codes in manual_codes_all],
        axis=0,
    )
    y_true = np.stack(
        [codes_to_multihot(codes, target_codes).numpy() for codes in true_codes_all],
        axis=0,
    )

    metrics_list = [
        "accuracy",
        "hamming_loss",
        "f1_micro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
    ]
    metrics_manual = multilabel_metrics_fn(y_manual, y_pred, metrics=metrics_list)
    metrics_true = multilabel_metrics_fn(y_true, y_pred, metrics=metrics_list)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(
        args.output_dir, f"admission_level_results_per_note_{timestamp}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    metrics_path = os.path.join(
        args.output_dir, f"admission_level_metrics_per_note_{timestamp}.json"
    )
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "evaluation_timestamp": timestamp,
                "processing_method": "per_note",
                "total_admissions": len(results),
                "dry_run": dry_run,
                "manual_labels_metrics": metrics_manual,
                "true_codes_metrics": metrics_true,
            },
            f,
            indent=2,
        )

    print("Saved results to:", results_path)
    print("Saved metrics to:", metrics_path)
    print("Manual labels micro F1:", metrics_manual.get("f1_micro"))


if __name__ == "__main__":
    main()
