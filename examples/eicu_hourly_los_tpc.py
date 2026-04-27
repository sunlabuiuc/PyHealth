"""
eICU hourly remaining length-of-stay (LoS) training and ablation script
for the Temporal Pointwise Convolution (TPC) model.

This script runs the TPC model through the true PyHealth ``BaseModel``
pipeline:

    1. Load an eICU base dataset with the custom YAML config.
    2. Convert it to a task-specific ``SampleDataset`` using ``HourlyLOSEICU``.
    3. Split the task dataset by patient.
    4. Create dataloaders with PyHealth's ``get_dataloader``.
    5. Instantiate ``TPC(dataset=task_dataset, ...)``.
    6. Train using ``pyhealth.trainer.Trainer``.
    7. Evaluate scalar regression loss, MAE, and RMSE.

This file is intentionally designed to work with synthetic, dev, or real
eICU roots. For project verification and fast iteration, it should be run
against synthetic data using small sample caps.

Example:
    >>> EICU_ROOT=/path/to/synthetic/eicu_demo PYTHONPATH=. \\
    ... python3 examples/eicu_hourly_los_tpc.py \\
    ...     --epochs 1 \\
    ...     --batch_size 2 \\
    ...     --max_samples 24
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict

import numpy as np
import torch
from pyhealth.datasets import SampleDataset

DEFAULT_EICU_ROOT = os.environ.get("EICU_ROOT", "YOUR_EICU_DATASET_PATH")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pyhealth.datasets import eICUDataset, get_dataloader, split_by_patient
from pyhealth.models.tpc import TPC
from pyhealth.tasks.hourly_los import HourlyLOSEICU
from pyhealth.trainer import Trainer


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Seed value applied to Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)





def infer_model_dims(task_dataset) -> tuple[int, int]:
    """Infer ``input_dim`` and ``static_dim`` from the processed task dataset.

    Args:
        task_dataset: PyHealth ``SampleDataset`` returned by ``set_task()``.

    Returns:
        A tuple ``(input_dim, static_dim)``.

    Raises:
        ValueError: If the first sample does not contain a valid
            ``time_series`` or ``static`` field.
    """
    first_sample = task_dataset[0]

    if "time_series" not in first_sample:
        raise ValueError("Task sample is missing required field 'time_series'")
    if "static" not in first_sample:
        raise ValueError("Task sample is missing required field 'static'")

    time_series = first_sample["time_series"]
    static = first_sample["static"]

    if not isinstance(time_series, torch.Tensor):
        time_series = torch.tensor(time_series, dtype=torch.float32)
    if not isinstance(static, torch.Tensor):
        static = torch.tensor(static, dtype=torch.float32)

    if time_series.ndim != 2:
        raise ValueError(
            "Expected task sample 'time_series' to have shape [T, 3F], got "
            f"{tuple(time_series.shape)}"
        )

    feature_dim = time_series.shape[1]
    if feature_dim % 3 != 0:
        raise ValueError(
            "Expected 'time_series' last dimension divisible by 3 for "
            f"[value, mask, decay], got {feature_dim}"
        )

    input_dim = feature_dim // 3
    static_dim = int(static.shape[0])

    return input_dim, static_dim


def evaluate_regression(model: TPC, dataloader) -> Dict[str, float]:
    """Evaluate scalar regression loss, MAE, and RMSE.

    Args:
        model: Trained TPC model.
        dataloader: Evaluation dataloader.

    Returns:
        Dictionary containing ``loss``, ``mae``, and ``rmse``.
    """
    model.eval()

    total_loss = 0.0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)

            batch_loss = outputs["loss"].item()
            y_pred = outputs["y_prob"].detach().cpu().view(-1)
            y_true = outputs["y_true"].detach().cpu().view(-1)

            total_loss += batch_loss
            total_abs_error += torch.sum(torch.abs(y_pred - y_true)).item()
            total_sq_error += torch.sum((y_pred - y_true) ** 2).item()
            total_count += int(y_true.numel())

    mean_loss = total_loss / max(len(dataloader), 1)
    mae = total_abs_error / max(total_count, 1)
    rmse = (total_sq_error / max(total_count, 1)) ** 0.5

    return {
        "loss": mean_loss,
        "mae": mae,
        "rmse": rmse,
    }


def parse_args():
    """Parse command-line arguments for the eICU TPC training script.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run eICU hourly remaining LoS prediction with TPC."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_EICU_ROOT,
        help=(
            "Path to eICU dataset root. Defaults to the EICU_ROOT "
            "environment variable when set."
        ),
    )

    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=None, 
        help="Path to the PyHealth cache directory."
    )
    parser.add_argument("--dev", action="store_true", help="Use dev mode dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=128,
        help=(
            "Approximate total sample cap across train/val/test splits for "
            "fast synthetic or smoke-style runs."
        ),
    )
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--temporal_channels", type=int, default=4)
    parser.add_argument("--pointwise_channels", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--fc_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--loss",
        type=str,
        choices=["msle", "mse"],
        default="msle",
        help="Loss function for H3 ablation (default: msle).",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=["full", "temporal_only", "pointwise_only"],
        default="full",
        help="Architectural ablation: full TPC, temporal only, or pointwise only.",
    )
    parser.add_argument(
        "--shared_temporal",
        action="store_true",
        help="Use shared temporal convolution weights across features.",
    )
    parser.add_argument(
        "--no_skip_connections",
        action="store_true",
        help="Disable concatenative skip connections.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of CPU workers for task transformations"
    )
    return parser.parse_args()


def main():
    """Run the full eICU hourly LoS training and evaluation pipeline."""
    args = parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("TPC eICU Hourly LoS Run Configuration")
    print("=" * 80)
    print(f"root: {args.root}")
    print(f"dev: {args.dev}")
    print(f"epochs: {args.epochs}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"max_samples: {args.max_samples}")
    print(f"num_layers: {args.num_layers}")
    print(f"temporal_channels: {args.temporal_channels}")
    print(f"pointwise_channels: {args.pointwise_channels}")
    print(f"kernel_size: {args.kernel_size}")
    print(f"fc_dim: {args.fc_dim}")
    print(f"dropout: {args.dropout}")
    print(f"lr: {args.lr}")
    print(f"loss: {args.loss}")
    print(f"model_variant: {args.model_variant}")
    print(f"shared_temporal: {args.shared_temporal}")
    print(f"no_skip_connections: {args.no_skip_connections}")
    print(f"seed: {args.seed}")
    print("=" * 80)

    base_dataset = eICUDataset(
        root=args.root,
        tables=[
            "patient",
            "lab",
            "respiratorycharting",
            "nursecharting",
            "vitalperiodic",
            "vitalaperiodic",
            "pasthistory",
            "admissiondx",
            "diagnosis",
        ],
        dev=args.dev,
        cache_dir=args.cache_dir,
        config_path=os.path.join(
            REPO_ROOT,
            "pyhealth/datasets/configs/eicu_tpc.yaml",
        ),
    )

    task_dataset = base_dataset.set_task(
        HourlyLOSEICU(
            time_series_tables=[
                "lab",
                "respiratorycharting",
                "nursecharting",
                "vitalperiodic",
                "vitalaperiodic",
            ],
            time_series_features={
                "lab": [
                    "-basos",
                    "-eos",
                    "-lymphs",
                    "-monos",
                    "-polys",
                    "ALT (SGPT)",
                    "AST (SGOT)",
                    "BUN",
                    "Base Excess",
                    "FiO2",
                    "HCO3",
                    "Hct",
                    "Hgb",
                    "MCH",
                    "MCHC",
                    "MCV",
                    "MPV",
                    "O2 Sat (%)",
                    "PT",
                    "PT - INR",
                    "PTT",
                    "RBC",
                    "RDW",
                    "WBC x 1000",
                    "albumin",
                    "alkaline phos.",
                    "anion gap",
                    "bedside glucose",
                    "bicarbonate",
                    "calcium",
                    "chloride",
                    "creatinine",
                    "glucose",
                    "lactate",
                    "magnesium",
                    "pH",
                    "paCO2",
                    "paO2",
                    "phosphate",
                    "platelets x 1000",
                    "potassium",
                    "sodium",
                    "total bilirubin",
                    "total protein",
                    "troponin - I",
                    "urinary specific gravity",
                ],
                "respiratorycharting": [
                    "Exhaled MV",
                    "Exhaled TV (patient)",
                    "LPM O2",
                    "Mean Airway Pressure",
                    "Peak Insp. Pressure",
                    "PEEP",
                    "Plateau Pressure",
                    "Pressure Support",
                    "RR (patient)",
                    "SaO2",
                    "TV/kg IBW",
                    "Tidal Volume (set)",
                    "Total RR",
                    "Vent Rate",
                ],
                "nursecharting": [
                    "Bedside Glucose",
                    "Delirium Scale/Score",
                    "Glasgow coma score",
                    "Heart Rate",
                    "Invasive BP",
                    "Non-Invasive BP",
                    "O2 Admin Device",
                    "O2 L/%",
                    "O2 Saturation",
                    "Pain Score/Goal",
                    "Respiratory Rate",
                    "Sedation Score/Goal",
                    "Temperature",
                ],
                "vitalperiodic": [
                    "cvp",
                    "heartrate",
                    "respiration",
                    "sao2",
                    "st1",
                    "st2",
                    "st3",
                    "systemicdiastolic",
                    "systemicmean",
                    "systemicsystolic",
                    "temperature",
                ],
                "vitalaperiodic": [
                    "noninvasivediastolic",
                    "noninvasivemean",
                    "noninvasivesystolic",
                ],
            },
            numeric_static_features=[
                "age",
                "admissionheight",
                "admissionweight",
            ],
            # Keep the task contract stable for true BaseModel training.
            categorical_static_features=[],
            diagnosis_tables=[],
            include_diagnoses=False,
            diagnosis_time_limit_hours=5,
            min_history_hours=5,
            max_hours=48,
        ),
        num_workers=args.num_workers,
    )

    if len(task_dataset) == 0:
        raise RuntimeError(
            "No samples were generated by HourlyLOSEICU. "
            "Check synthetic data, feature mappings, joins, cache, or dataset mode."
        )

    input_dim, static_dim = infer_model_dims(task_dataset)

    print(f"task samples: {len(task_dataset)}")
    print(f"model input_dim: {input_dim}")
    print(f"model static_dim: {static_dim}")

    # For very small synthetic datasets, use sample-level split instead of patient-level

    # Use patient-level split when possible. For very small synthetic datasets,
    # reuse one non-empty split for validation so Trainer can run.
    num_samples = len(task_dataset)

    if num_samples < 20:
        train_ds, _, test_ds = split_by_patient(task_dataset, [0.5, 0.0, 0.5])
        val_ds = test_ds
    else:
        train_ds, val_ds, test_ds = split_by_patient(task_dataset, [0.70, 0.15, 0.15])

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(
            "Dataset split produced an empty train, validation, or test split."
        )

    print(f"train samples: {len(train_ds)}")
    print(f"val samples: {len(val_ds)}")
    print(f"test samples: {len(test_ds)}")

    train_loader = get_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        #num_workers=args.num_workers,
    )
    val_loader = get_dataloader(val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        #num_workers=args.num_workers,
    )
    test_loader = get_dataloader(test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        #num_workers=args.num_workers,
    )
    use_temporal = args.model_variant in {"full", "temporal_only"}
    use_pointwise = args.model_variant in {"full", "pointwise_only"}
    use_skip_connections = not args.no_skip_connections

    model = TPC(
        dataset=task_dataset,
        input_dim=input_dim,
        static_dim=static_dim,
        temporal_channels=args.temporal_channels,
        pointwise_channels=args.pointwise_channels,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        fc_dim=args.fc_dim,
        dropout=args.dropout,
        shared_temporal=args.shared_temporal,
        use_temporal=use_temporal,
        use_pointwise=use_pointwise,
        use_skip_connections=use_skip_connections,
        loss_name=args.loss,
    )

    trainer = Trainer(
        model=model,
        device="cpu",
        enable_logging=False,
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
    )

    val_results = evaluate_regression(model, val_loader)
    test_results = evaluate_regression(model, test_loader)

    print("=" * 80)
    print("Run complete")
    print(f"model_variant: {args.model_variant}")
    print(f"shared_temporal: {args.shared_temporal}")
    print(f"no_skip_connections: {args.no_skip_connections}")
    print(f"loss: {args.loss}")
    print(f"val_loss: {val_results['loss']:.4f}")
    print(f"val_mae: {val_results['mae']:.4f}")
    print(f"val_rmse: {val_results['rmse']:.4f}")
    print(f"test_loss: {test_results['loss']:.4f}")
    print(f"test_mae: {test_results['mae']:.4f}")
    print(f"test_rmse: {test_results['rmse']:.4f}")
    print(
        "ABLATION_SUMMARY "
        f"model_variant={args.model_variant} "
        f"shared_temporal={args.shared_temporal} "
        f"no_skip_connections={args.no_skip_connections} "
        f"loss={args.loss} "
        f"val_loss={val_results['loss']:.4f} "
        f"val_mae={val_results['mae']:.4f} "
        f"val_rmse={val_results['rmse']:.4f} "
        f"test_loss={test_results['loss']:.4f} "
        f"test_mae={test_results['mae']:.4f} "
        f"test_rmse={test_results['rmse']:.4f}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()