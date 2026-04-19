"""TPC on MIMIC-IV length-of-stay prediction with simple ablations.

Contributor: Hasham Ul Haq (huhaq2)
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
    in the Intensive Care Unit
Paper link: https://arxiv.org/abs/2007.09483
Description: Example script for evaluating the PyHealth-adapted TPC model on
    the existing MIMIC-IV temporal length-of-stay task with small ablations.

This example is designed for reproducible course-project style experiments:
it runs on the bundled MIMIC-IV demo by default and can be pointed at full
MIMIC-IV via ``EHR_ROOT``.

Suggested usage:
    python examples/length_of_stay/length_of_stay_mimic4_tpc.py
    python examples/length_of_stay/length_of_stay_mimic4_tpc.py --quick-test
    EHR_ROOT=/path/to/mimiciv/2.2 python examples/length_of_stay/length_of_stay_mimic4_tpc.py

Ablations included:
    - kernel size
    - number of TPC layers
    - hidden dimension
    - dropout

Expected MIMIC-IV files for the temporal LOS task:
    - hosp/patients.csv.gz
    - hosp/admissions.csv.gz
    - hosp/diagnoses_icd.csv.gz
    - hosp/procedures_icd.csv.gz
    - hosp/labevents.csv.gz
    - hosp/d_labitems.csv.gz
    - icu/icustays.csv.gz

Note:
    ``MIMIC4EHRDataset`` automatically includes core EHR tables such as
    patients, admissions, and ICU stays. Keep those files in the EHR root even
    though the example only passes diagnoses, procedures, and labevents in
    ``ehr_tables``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEMO_EHR_ROOT = REPO_ROOT / "test-resources" / "core" / "mimic4demo"

# Prefer the local repository checkout when this script is run directly.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
    split_by_sample,
)
from pyhealth.models import TPC
from pyhealth.tasks import LengthOfStayStageNetMIMIC4
from pyhealth.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use dev mode and only the first ablation for a smoke test.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally cap the SampleDataset size after task construction.",
    )
    parser.add_argument(
        "--single-ablation",
        action="store_true",
        help="Run only the first ablation config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the default number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size for all ablations.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["auto", "patient", "sample"],
        default="auto",
        help="Choose how to split the dataset for evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataset/task worker count.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    return parser.parse_args()


def build_dataset(
    ehr_root: str,
    quick_test: bool,
    max_samples: int | None = None,
    num_workers: int | None = None,
):
    worker_count = num_workers if num_workers is not None else (1 if quick_test else 4)
    base_dataset = MIMIC4Dataset(
        ehr_root=ehr_root,
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
        dev=quick_test,
        num_workers=worker_count,
    )
    sample_dataset = base_dataset.set_task(
        LengthOfStayStageNetMIMIC4(padding=20),
        num_workers=worker_count,
    )
    if max_samples is not None:
        capped_size = min(max_samples, len(sample_dataset))
        sample_dataset = sample_dataset.subset(range(capped_size))
        print(f"Capped dataset to {len(sample_dataset)} samples for local testing.")
    return sample_dataset


def run_ablation(
    sample_dataset,
    device: str,
    config: dict,
    epochs: int,
    split_mode: str = "auto",
) -> dict:
    if len(sample_dataset) < 3 or len(sample_dataset.patient_to_index) < 3:
        print(
            "Dataset too small for patient-level train/val/test splitting; "
            "reusing the same dataset for all three loaders."
        )
        train_dataset = sample_dataset
        val_dataset = sample_dataset
        test_dataset = sample_dataset
    elif split_mode == "sample":
        print("Using sample-level train/val/test split.")
        train_dataset, val_dataset, test_dataset = split_by_sample(
            sample_dataset,
            ratios=[0.7, 0.1, 0.2],
        )
    elif split_mode == "patient":
        print("Using patient-level train/val/test split.")
        train_dataset, val_dataset, test_dataset = split_by_patient(
            sample_dataset,
            ratios=[0.7, 0.1, 0.2],
        )
    else:
        print("Using patient-level train/val/test split.")
        train_dataset, val_dataset, test_dataset = split_by_patient(
            sample_dataset,
            ratios=[0.7, 0.1, 0.2],
        )

    train_loader = get_dataloader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = TPC(
        dataset=sample_dataset,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
    )
    trainer = Trainer(
        model=model,
        device=device,
        metrics=["accuracy", "f1_weighted", "f1_macro"],
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="accuracy",
        optimizer_params={"lr": config["lr"]},
    )
    results = trainer.evaluate(test_loader)
    return results


if __name__ == "__main__":
    args = parse_args()
    ehr_root = os.environ.get("EHR_ROOT", str(DEMO_EHR_ROOT))
    sample_dataset = build_dataset(
        ehr_root=ehr_root,
        quick_test=args.quick_test,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
    )

    print(f"Loaded sample dataset with {len(sample_dataset)} samples from: {ehr_root}")
    print(f"Input schema: {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")

    ablations = [
        {
            "name": "base",
            "embedding_dim": 128,
            "hidden_dim": 128,
            "num_layers": 2,
            "kernel_size": 3,
            "dropout": 0.1,
            "lr": 1e-3,
            "batch_size": 16 if args.quick_test else 32,
        },
        {
            "name": "wider_hidden",
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_layers": 2,
            "kernel_size": 3,
            "dropout": 0.1,
            "lr": 1e-3,
            "batch_size": 32,
        },
        {
            "name": "larger_kernel",
            "embedding_dim": 128,
            "hidden_dim": 128,
            "num_layers": 2,
            "kernel_size": 5,
            "dropout": 0.1,
            "lr": 1e-3,
            "batch_size": 32,
        },
        {
            "name": "deeper_stack",
            "embedding_dim": 128,
            "hidden_dim": 128,
            "num_layers": 4,
            "kernel_size": 3,
            "dropout": 0.2,
            "lr": 1e-3,
            "batch_size": 32,
        },
    ]

    if args.quick_test or args.single_ablation:
        ablations = ablations[:1]

    if args.batch_size is not None:
        for config in ablations:
            config["batch_size"] = args.batch_size

    epochs = args.epochs if args.epochs is not None else (1 if args.quick_test else 10)
    summary = []
    for config in ablations:
        print("\n" + "=" * 80)
        print(f"Running ablation: {config['name']}")
        print(config)
        results = run_ablation(
            sample_dataset=sample_dataset,
            device=args.device,
            config=config,
            epochs=epochs,
            split_mode=args.split_mode,
        )
        print(f"Results: {results}")
        summary.append((config["name"], results))

    print("\nAblation summary")
    for name, results in summary:
        print(
            f"- {name}: accuracy={results.get('accuracy', float('nan')):.4f}, "
            f"f1_weighted={results.get('f1_weighted', float('nan')):.4f}, "
            f"f1_macro={results.get('f1_macro', float('nan')):.4f}"
        )
