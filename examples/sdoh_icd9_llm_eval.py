import argparse
import json
import os
from datetime import datetime
from typing import List, Set

import numpy as np
from pyhealth.datasets import MIMIC3NotesDataset
from pyhealth.metrics import multilabel_metrics_fn
from pyhealth.models.sdoh_icd9_llm import SDOHICD9LLM
from pyhealth.tasks.sdoh_icd9_detection import TARGET_CODES
from pyhealth.tasks.sdoh_utils import codes_to_multihot


def parse_args():
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
    parser.add_argument("--output-dir", default=".", help="Directory to save outputs.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    target_codes = list(TARGET_CODES)

    noteevents_path = f"{args.mimic_root}/NOTEEVENTS.csv.gz"
    note_dataset = MIMIC3NotesDataset(
        noteevents_path=noteevents_path,
        label_csv_path=args.label_csv_path,
        target_codes=target_codes,
    )
    sample_dataset = note_dataset.set_task(label_source=args.label_source)

    dry_run = args.dry_run or not os.environ.get("OPENAI_API_KEY")
    model = SDOHICD9LLM(target_codes=target_codes, dry_run=dry_run)

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
        manual_codes_all.append(set(sample.get("manual_codes", [])))
        true_codes_all.append(set(sample.get("true_codes", [])))

        results.append(
            {
                "visit_id": sample.get("visit_id"),
                "patient_id": sample.get("patient_id"),
                "num_notes": sample.get("num_notes"),
                "text_length": sample.get("text_length"),
                "is_gap_case": sample.get("is_gap_case"),
                "manual_codes": ",".join(sorted(sample.get("manual_codes", []))),
                "true_codes": ",".join(sorted(sample.get("true_codes", []))),
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
