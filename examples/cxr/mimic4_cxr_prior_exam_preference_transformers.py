"""Run the PriorExamPreferenceTask context ablation with a PyHealth model.

Author: Rishi Gopalan
Paper: Direct Preference Optimization for Suppressing Hallucinated Prior
    Exams in Radiology Report Generation (Banerjee et al., 2024)
Paper link: https://arxiv.org/abs/2406.06496

The three runs use the same ReXPref-Prior release splits, model, seed and
training configuration.  Only ``PriorExamPreferenceTask.context_mode``
changes.  The target is whether ReXPref-Prior rewrote or deleted a sentence;
this experiment does not independently label hallucinations.

The output contains aggregate counts and test metrics only.  It never writes
report text or identifiers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pyhealth.datasets import MIMIC4CXRDataset, get_dataloader
from pyhealth.models import TransformersModel
from pyhealth.tasks import PriorExamPreferenceTask
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed

CONTEXT_MODES = ("sentence_only", "neighbor_context", "section_context")
METRICS = ["precision", "recall", "f1", "pr_auc", "roc_auc"]


def parse_args() -> argparse.Namespace:
    """Parses data paths and the shared experiment configuration."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cxr-root", required=True)
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--val-annotations", required=True)
    parser.add_argument("--test-annotations", required=True)
    parser.add_argument("--model-name", default="prajjwal1/bert-tiny")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prior_exam_preference_ablation.json"),
    )
    return parser.parse_args()


def release_subsets(samples):
    """Returns release subsets and rejects empty or patient-leaking splits."""
    indices = {split: [] for split in ("train", "val", "test")}
    patients = {split: set() for split in indices}
    labels = {split: Counter() for split in indices}
    for index in range(len(samples)):
        sample = samples[index]
        split = sample["split"]
        if split not in indices:
            raise ValueError(f"unexpected release split: {split!r}")
        indices[split].append(index)
        patients[split].add(sample["patient_id"])
        labels[split][int(sample["needs_edit"].item())] += 1

    for split, split_indices in indices.items():
        if not split_indices:
            raise ValueError(f"the {split} split has no matched samples")
        if set(labels[split]) != {0, 1}:
            raise ValueError(f"the {split} split must contain both binary labels")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if patients[left] & patients[right]:
            raise ValueError(f"patient leakage between {left} and {right}")

    subsets = {
        split: samples.subset(split_indices) for split, split_indices in indices.items()
    }
    counts = {
        split: {
            "patients": len(patients[split]),
            "samples": len(indices[split]),
            "unchanged": labels[split][0],
            "needs_edit": labels[split][1],
        }
        for split in indices
    }
    return subsets, counts


def run_mode(dataset, annotation_files, args, context_mode):
    """Trains and evaluates one context configuration."""
    set_seed(args.seed)
    samples = dataset.set_task(
        PriorExamPreferenceTask(annotation_files, context_mode=context_mode),
        num_workers=args.num_workers,
    )
    subsets, counts = release_subsets(samples)
    train, val, test = (subsets[split] for split in ("train", "val", "test"))

    set_seed(args.seed)
    model = TransformersModel(
        dataset=train,
        model_name=args.model_name,
        max_length=args.max_length,
    )
    trainer = Trainer(
        model=model,
        metrics=METRICS,
        device=args.device,
        enable_logging=False,
    )
    trainer.train(
        get_dataloader(train, args.batch_size, shuffle=True, seed=args.seed),
        val_dataloader=get_dataloader(val, args.batch_size),
        epochs=args.epochs,
        optimizer_params={"lr": args.learning_rate},
        monitor="f1",
        monitor_criterion="max",
    )
    scores = trainer.evaluate(get_dataloader(test, args.batch_size))
    return {
        "counts": counts,
        "test": {name: float(value) for name, value in scores.items()},
    }


def main() -> None:
    """Runs all context modes and records comparable aggregate results."""
    args = parse_args()
    annotation_files = {
        "train": args.train_annotations,
        "val": args.val_annotations,
        "test": args.test_annotations,
    }
    dataset = MIMIC4CXRDataset(
        root=args.cxr_root,
        tables=["metadata"],
        num_workers=args.num_workers,
    )
    results = {
        "target": "needs_edit",
        "model": args.model_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "contexts": {
            mode: run_mode(dataset, annotation_files, args, mode)
            for mode in CONTEXT_MODES
        },
    }
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
