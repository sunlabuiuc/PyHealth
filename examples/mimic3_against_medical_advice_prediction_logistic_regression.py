"""Ablation example for MIMIC-III left-AMA prediction on CPU.

This script demonstrates Option 3 (standalone task) usage with multiple
feature-group configurations and a lightweight PyHealth model.

Notes:
    Configs prefixed with ``baseline`` mirror the paper-style reproduction
    feature groups.

    Configs prefixed with ``novel_`` are the intended extension/ablation
    study for coursework credit because they are not reported in the original
    paper table:
    - restricting note-derived features to nursing notes
    - tightening the ICU cohort to 24 hours

    The MIMIC-III demo subset can be extremely label-sparse. When a requested
    configuration produces a split with only one label class, the script will
    skip that configuration instead of failing the whole run.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import get_dataloader, split_by_patient
from pyhealth.models import LogisticRegression
from pyhealth.tasks import AgainstMedicalAdvicePredictionMIMIC3
from pyhealth.trainer import Trainer


CONFIG_DEFAULTS: Dict[str, Dict[str, object]] = {
    "baseline": {
        "include_baseline": True,
        "include_race": False,
        "include_mistrust": False,
        "include_codes": False,
    },
    "baseline_race": {
        "include_baseline": True,
        "include_race": True,
        "include_mistrust": False,
        "include_codes": False,
    },
    "baseline_noncompliance": {
        "include_baseline": True,
        "include_race": False,
        "include_mistrust": True,
        "mistrust_feature_set": "noncompliance",
        "include_codes": False,
    },
    "baseline_all": {
        "include_baseline": True,
        "include_race": True,
        "include_mistrust": True,
        "mistrust_feature_set": "all",
        "include_codes": False,
    },
    "novel_nursing_all": {
        "include_baseline": True,
        "include_race": True,
        "include_mistrust": True,
        "mistrust_feature_set": "all",
        "include_codes": False,
        "note_categories": ["nursing"],
    },
    "novel_icu24_all": {
        "include_baseline": True,
        "include_race": True,
        "include_mistrust": True,
        "mistrust_feature_set": "all",
        "include_codes": False,
        "min_icu_hours": 24.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MIMIC-III left-AMA ablation with LogisticRegression."
    )
    parser.add_argument("--root", required=True, help="Path to MIMIC-III root.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory (default: temp directory).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=1, help="Task workers.")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable PyHealth dev mode (first 1000 patients).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "baseline",
            "baseline_all",
            "novel_nursing_all",
            "novel_icu24_all",
        ],
        choices=sorted(CONFIG_DEFAULTS.keys()),
        help="Configuration names to run.",
    )
    parser.add_argument(
        "--include-codes",
        action="store_true",
        help="Force include code features in all selected configs.",
    )
    parser.add_argument(
        "--min-icu-hours",
        type=float,
        default=None,
        help="Optional minimum ICU-hours filter.",
    )
    parser.add_argument(
        "--note-categories",
        nargs="+",
        default=None,
        help="Optional note category filter (case-insensitive).",
    )
    return parser.parse_args()


def build_tables(config: Dict[str, object]) -> List[str]:
    tables = ["noteevents"]
    if config.get("include_codes", False):
        tables.extend(["diagnoses_icd", "procedures_icd", "prescriptions"])
    return tables


def _extract_binary_label(sample) -> int:
    label = sample["left_ama"]
    if hasattr(label, "detach"):
        return int(label.detach().cpu().numpy().reshape(-1)[0])
    return int(np.asarray(label).reshape(-1)[0])


def _count_classes(dataset) -> Dict[int, int]:
    counts = {0: 0, 1: 0}
    for idx in range(len(dataset)):
        counts[_extract_binary_label(dataset[idx])] += 1
    return counts


def run_one_config(args: argparse.Namespace, config_name: str) -> Dict[str, object]:
    config = dict(CONFIG_DEFAULTS[config_name])
    if args.include_codes:
        config["include_codes"] = True
    task_note_categories = config.pop("note_categories", None)
    task_min_icu_hours = config.pop("min_icu_hours", None)
    if args.note_categories is not None:
        task_note_categories = args.note_categories
    if args.min_icu_hours is not None and task_min_icu_hours is None:
        task_min_icu_hours = args.min_icu_hours

    print(f"\n=== Running config: {config_name} ===")
    print(f"Config args: {config}")

    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="mimic3_ama_cache_")
    sample_dataset = None
    dataset = MIMIC3Dataset(
        root=args.root,
        tables=build_tables(config),
        cache_dir=cache_dir,
        dev=args.dev,
    )
    task = AgainstMedicalAdvicePredictionMIMIC3(
        exclude_minors=True,
        min_icu_hours=task_min_icu_hours,
        note_categories=task_note_categories,
        **config,
    )
    try:
        try:
            sample_dataset = dataset.set_task(task, num_workers=args.num_workers)
        except Exception as exc:
            if "Expected 2 unique labels, got 1" in str(exc):
                return {
                    "status": "skipped",
                    "reason": (
                        "Dataset/task combination only contains one label class "
                        "for this run."
                    ),
                }
            raise
        if len(sample_dataset) < 10:
            return {
                "status": "skipped",
                "reason": (
                    f"Too few samples ({len(sample_dataset)}) for config "
                    f"'{config_name}'."
                ),
            }

        train_ds, val_ds, test_ds = split_by_patient(
            sample_dataset,
            ratios=[0.8, 0.1, 0.1],
            seed=args.seed,
        )
        split_counts = {
            "train": _count_classes(train_ds),
            "val": _count_classes(val_ds),
            "test": _count_classes(test_ds),
        }
        if any(min(counts.values()) == 0 for counts in split_counts.values()):
            return {
                "status": "skipped",
                "reason": (
                    "One or more splits have only a single label class. "
                    f"Split counts: {split_counts}"
                ),
            }

        train_loader = get_dataloader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = LogisticRegression(dataset=sample_dataset, embedding_dim=64)
        trainer = Trainer(
            model=model,
            metrics=[
                "roc_auc",
                "pr_auc",
                "balanced_accuracy",
                "f1",
                "precision",
                "recall",
            ],
            enable_logging=False,
        )
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            monitor="roc_auc",
        )
        test_scores = trainer.evaluate(test_loader)
        test_scores["status"] = "ok"
        test_scores["split_counts"] = split_counts
        return test_scores
    finally:
        if sample_dataset is not None:
            sample_dataset.close()


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    results: Dict[str, Dict[str, object]] = {}
    for config_name in args.configs:
        try:
            results[config_name] = run_one_config(args, config_name)
            print(f"Result ({config_name}): {results[config_name]}")
        except Exception as exc:
            print(f"Config '{config_name}' failed: {exc}")

    print("\n=== Summary ===")
    for name, scores in results.items():
        print(f"{name}: {scores}")


if __name__ == "__main__":
    main()
