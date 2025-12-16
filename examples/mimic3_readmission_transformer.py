"""
Example: MIMIC-III 30-day readmission prediction with Transformer.

This example demonstrates a full PyHealth pipeline using:

- pyhealth.datasets.MIMIC3Dataset
- pyhealth.tasks.readmission_prediction_mimic3_fn
- pyhealth.models.Transformer
- pyhealth.trainer.Trainer
- pyhealth.metrics.binary.binary_metrics_fn

The pipeline follows the standard five-stage PyHealth workflow:

    1. Load EHR data with MIMIC3Dataset
    2. Define a readmission prediction task via readmission_prediction_mimic3_fn
    3. Build a Transformer model for binary outcome prediction
    4. Train the model with Trainer
    5. Evaluate the model with binary metrics (PR-AUC, ROC-AUC, loss)

Typical usage with the Synthetic MIMIC-III demo:

    python examples/mimic3_readmission_transformer.py \
        --root https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/ \
        --epochs 3 \
        --batch-size 32 \
        --dev

Typical usage with a local MIMIC-III installation:

    python examples/mimic3_readmission_transformer.py \
        --root /path/to/mimiciii/root \
        --epochs 5 \
        --batch-size 32

Notes
-----
- You must have PyHealth installed (e.g. `pip install -e .` from the repo root).
- Access to real MIMIC-III requires PhysioNet credentialing; the Synthetic
  MIMIC-III demo hosted by PyHealth can be used for non-sensitive experiments.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.metrics.binary import binary_metrics_fn
from pyhealth.models import Transformer
from pyhealth.tasks import readmission_prediction_mimic3_fn
from pyhealth.trainer import Trainer

# Defaults consistent with PyHealth documentation
DEFAULT_MIMIC3_SYNTHETIC_ROOT = (
    "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/"
)
DEFAULT_TABLES = ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"]


def run_mimic3_readmission_transformer(
    root: str,
    tables: Optional[List[str]] = None,
    batch_size: int = 32,
    epochs: int = 5,
    dev: bool = False,
    num_workers: int = 0,
) -> Dict[str, float]:
    """
    Run an end-to-end MIMIC-III 30-day readmission pipeline with a Transformer model.

    This function is factored out so it can be imported and tested programmatically,
    in addition to being invoked via the CLI entry point defined below.

    Parameters
    ----------
    root:
        Root directory or URL where MIMIC-III (or Synthetic MIMIC-III) data is stored.
        Examples:
            - "/srv/local/data/physionet.org/files/mimiciii/1.4"
            - "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/"
    tables:
        List of MIMIC-III tables to load. If None, defaults to
        ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"].
    batch_size:
        Batch size used for all dataloaders.
    epochs:
        Number of training epochs.
    dev:
        If True, enables MIMIC3Dataset dev mode (small subset for quick tests).
    num_workers:
        Number of workers for DataLoader(s). 0 is usually safest.

    Returns
    -------
    metrics:
        A dictionary with evaluation metrics on the test split, including:
            - "loss": average test loss from Trainer.inference
            - "pr_auc": area under the precision–recall curve
            - "roc_auc": area under the ROC curve
            - "num_samples": number of evaluated samples
    """
    if tables is None:
        tables = list(DEFAULT_TABLES)

    # 1. Load and process MIMIC-III dataset
    dataset = MIMIC3Dataset(
        root=root,
        tables=tables,
        # Map NDC codes to ATC level 3 (as in PyHealth docs)
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        dev=dev,
    )

    # 2. Define readmission prediction task and generate samples
    #    Each sample includes:
    #      - "visit_id", "patient_id"
    #      - "conditions", "procedures", "drugs"
    #      - "label" (0/1, readmission indicator)
    sample_dataset = dataset.set_task(task_fn=readmission_prediction_mimic3_fn)

    # 3. Split by patient to avoid leakage across sets
    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])

    train_loader = get_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = get_dataloader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = get_dataloader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # 4. Instantiate Transformer for binary readmission prediction
    #
    # Feature keys follow the schema produced by readmission_prediction_mimic3_fn:
    #   - "conditions": ICD codes per visit (sequence)
    #   - "procedures": procedure codes per visit (sequence)
    #   - "drugs": ATC codes derived from NDC (sequence)
    model = Transformer(
        dataset=sample_dataset,
        feature_keys=["conditions", "procedures", "drugs"],
        label_key="label",
        mode="binary",
    )

    # 5. Train with the standard Trainer API
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="pr_auc_samples",
    )

    # 6. Evaluate on the held-out test set
    y_true, y_prob, loss = trainer.inference(test_loader)
    metrics = binary_metrics_fn(
        y_true,
        y_prob,
        metrics=["pr_auc", "roc_auc"],
    )

    # Attach loss and a simple count for sanity checking
    metrics["loss"] = float(loss)
    metrics["num_samples"] = int(len(y_true))

    return metrics


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the example script."""
    parser = argparse.ArgumentParser(
        description="MIMIC-III 30-day readmission prediction using PyHealth + Transformer."
    )

    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_MIMIC3_SYNTHETIC_ROOT,
        help=(
            "Root path or URL for MIMIC-III data. "
            "Defaults to the Synthetic MIMIC-III demo hosted by PyHealth."
        ),
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=DEFAULT_TABLES,
        help=(
            "MIMIC-III tables to load. "
            'Default: "DIAGNOSES_ICD PROCEDURES_ICD PRESCRIPTIONS".'
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training/evaluation (default: 32).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable MIMIC3Dataset dev mode (small subset for quick runs/tests).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save metrics as a JSON file.",
    )

    return parser.parse_args()


def main() -> None:
    """Command-line entry point for the example."""
    args = _parse_args()

    metrics = run_mimic3_readmission_transformer(
        root=args.root,
        tables=args.tables,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dev=args.dev,
        num_workers=args.num_workers,
    )

    print("\n=== MIMIC-III Readmission Prediction (Transformer) ===")
    print(f"Root: {args.root}")
    print(f"Tables: {args.tables}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics written to: {args.output_json}")


if __name__ == "__main__":
    main()
