"""
MIMIC-IV hourly remaining length-of-stay (LoS) training and evaluation script
for the Temporal Pointwise Convolution (TPC) model.

This script runs the TPC model through the true PyHealth ``BaseModel``
pipeline:

    1. Load a MIMIC-IV base dataset with the custom YAML config.
    2. Convert it to a task-specific ``SampleDataset`` using ``HourlyLOSEICU``.
    3. Split the task dataset by patient.
    4. Create dataloaders with PyHealth's ``get_dataloader``.
    5. Instantiate ``TPC(dataset=task_dataset, ...)``.
    6. Train using ``pyhealth.trainer.Trainer``.
    7. Evaluate scalar regression loss, MAE, and RMSE.

This file is intended to work with synthetic, dev, or real MIMIC-IV roots.
For project verification and fast iteration, it should be run against
synthetic data using a small sample cap.

Example:
    >>> MIMIC4_ROOT=/path/to/synthetic/mimic4_demo PYTHONPATH=. \\
    ... python3 examples/mimic4_hourly_los_tpc.py \\
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

DEFAULT_MIMIC4_ROOT = os.environ.get("MIMIC4_ROOT", "YOUR_MIMIC4_DATASET_PATH")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
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
    """Parse command-line arguments for the MIMIC-IV TPC training script.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run MIMIC-IV hourly remaining LoS prediction with TPC."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=DEFAULT_MIMIC4_ROOT,
        help=(
            "Path to the MIMIC-IV dataset root (directory containing hosp/ and "
            "icu/). Defaults to the MIMIC4_ROOT environment variable when set."
        ),
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
        help="Loss function for H3 comparison (default: msle).",
    )
    parser.add_argument(
        "--max_hours",
        type=int,
        default=336,
        help="Maximum ICU hours to model (14 days = 336 hours).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
    "--num_workers",
    type=int,
    default=1,
    help="Number of CPU workers for task transformations"
)
    parser.add_argument(
    "--cache_dir",
    type=str,
    default=None,
    help="Path to the PyHealth cache directory.",
)
    return parser.parse_args()


def main():
    """Run the full MIMIC-IV hourly LoS training and evaluation pipeline."""
    args = parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("TPC MIMIC-IV Hourly LoS Run Configuration")
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
    print(f"max_hours: {args.max_hours}")
    print(f"seed: {args.seed}")
    print("=" * 80)

    base_dataset = MIMIC4Dataset(
        ehr_root=args.root,
        ehr_tables=[
            "patients",
            "admissions",
            "icustays",
            "chartevents",
            "labevents",
        ],
        ehr_config_path=os.path.join(
            REPO_ROOT,
            "pyhealth/datasets/configs/mimic4_ehr_tpc.yaml",
        ),
        dev=args.dev,
        cache_dir=args.cache_dir, # Added proper cache directory support
    )

    task_dataset = base_dataset.set_task(
        HourlyLOSEICU(
            time_series_tables=["chartevents", "labevents"],
            time_series_features={
                "chartevents": [
                    "Activity / Mobility (JH-HLM)",
                    "Apnea Interval",
                    "Arterial Blood Pressure Alarm - High",
                    "Arterial Blood Pressure Alarm - Low",
                    "Arterial Blood Pressure diastolic",
                    "Arterial Blood Pressure mean",
                    "Arterial Blood Pressure systolic",
                    "Braden Score",
                    "Current Dyspnea Assessment",
                    "Daily Weight",
                    "Expiratory Ratio",
                    "Fspn High",
                    "GCS - Eye Opening",
                    "GCS - Motor Response",
                    "GCS - Verbal Response",
                    "Glucose finger stick (range 70-100)",
                    "Heart Rate",
                    "Heart Rate Alarm - Low",
                    "Heart rate Alarm - High",
                    "Inspired O2 Fraction",
                    "Mean Airway Pressure",
                    "Minute Volume",
                    "Minute Volume Alarm - High",
                    "Minute Volume Alarm - Low",
                    "Non Invasive Blood Pressure diastolic",
                    "Non Invasive Blood Pressure mean",
                    "Non Invasive Blood Pressure systolic",
                    "Non-Invasive Blood Pressure Alarm - High",
                    "Non-Invasive Blood Pressure Alarm - Low",
                    "O2 Flow",
                    "O2 Saturation Pulseoxymetry Alarm - Low",
                    "O2 saturation pulseoxymetry",
                    "PEEP set",
                    "PSV Level",
                    "Pain Level",
                    "Pain Level Response",
                    "Paw High",
                    "Peak Insp. Pressure",
                    "Phosphorous",
                    "Plateau Pressure",
                    "Resp Alarm - High",
                    "Resp Alarm - Low",
                    "Respiratory Rate",
                    "Respiratory Rate (Set)",
                    "Respiratory Rate (Total)",
                    "Respiratory Rate (spontaneous)",
                    "Richmond-RAS Scale",
                    "Strength L Arm",
                    "Strength L Leg",
                    "Strength R Arm",
                    "Strength R Leg",
                    "Temperature Fahrenheit",
                    "Tidal Volume (observed)",
                    "Tidal Volume (set)",
                    "Tidal Volume (spontaneous)",
                    "Total PEEP Level",
                    "Ventilator Mode",
                    "Vti High",
                ],
                "labevents": [
                    "Alanine Aminotransferase (ALT)",
                    "Alkaline Phosphatase",
                    "Anion Gap",
                    "Asparate Aminotransferase (AST)",
                    "Base Excess",
                    "Bicarbonate",
                    "Bilirubin, Total",
                    "Calcium, Total",
                    "Calculated Total CO2",
                    "Chloride",
                    "Creatinine",
                    "Free Calcium",
                    "Glucose",
                    "Hematocrit",
                    "Hematocrit, Calculated",
                    "Hemoglobin",
                    "INR(PT)",
                    "Lactate",
                    "MCH",
                    "MCHC",
                    "MCV",
                    "Magnesium",
                    "Oxygen Saturation",
                    "PT",
                    "PTT",
                    "Phosphate",
                    "Platelet Count",
                    "Potassium",
                    "Potassium, Whole Blood",
                    "RDW",
                    "RDW-SD",
                    "Red Blood Cells",
                    "Sodium",
                    "Sodium, Whole Blood",
                    "Temperature",
                    "Urea Nitrogen",
                    "White Blood Cells",
                    "pCO2",
                    "pH",
                    "pO2",
                ],
            },
            numeric_static_features=[],
            categorical_static_features=[],
            min_history_hours=5,
            max_hours=args.max_hours,
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
        #num_workers=dl_workers,
    )
    val_loader = get_dataloader(val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        #num_workers=dl_workers,
    )
    test_loader = get_dataloader(test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        #num_workers=dl_workers,
    )

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
        loss_name=args.loss,
    )

    
    trainer = Trainer(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
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
    print(f"loss: {args.loss}")
    print(f"val_loss: {val_results['loss']:.4f}")
    print(f"val_mae: {val_results['mae']:.4f}")
    print(f"val_rmse: {val_results['rmse']:.4f}")
    print(f"test_loss: {test_results['loss']:.4f}")
    print(f"test_mae: {test_results['mae']:.4f}")
    print(f"test_rmse: {test_results['rmse']:.4f}")
    print(
        "MIMIC_SUMMARY "
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