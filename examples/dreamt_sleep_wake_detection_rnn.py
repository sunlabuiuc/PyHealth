"""Ablation study: SleepWakeDetectionDREAMT with varying feature configurations.

This script reproduces the core experimental comparison from the DREAMT paper
(Wang et al., CHIL 2024) within PyHealth. It tests how different feature subsets
affect binary sleep-wake classification performance.

Experimental Setup:
We compare three feature configurations using PyHealth's RNN model:

  1. all_features: Full feature set from Wang et al. (ACC, TEMP, BVP, EDA,
     HR, ACC_INDEX, HRV_HFD). This replicates the paper's complete input space.
  2. acc_only: Accelerometry-derived features only. This tests whether motion
     signals alone are sufficient for sleep-wake detection.
  3. top2_paper: ACC_INDEX + HRV_HFD. These are the two strongest predictors
     identified in the paper through feature importance analysis.

Hypothesis:
Based on the paper's findings, top2_paper should perform comparably
to all_features, since ACC_INDEX and HRV_HFD were the dominant predictors.
acc_only should underperform, and this would confirm that cardiac signals (HRV_HFD)
add meaningful information.

Usage:
To test with the real DREAMT dataset, first download and prepare the data as per
the instructions in the DREAMT dataset documentation. Then run:
python examples/dreamt_sleep_wake_detection_rnn.py --root /path/to/dreamt/2.1.0/

To test with synthetic data (no PhysioNet access required), simply run:
python examples/dreamt_sleep_wake_detection_rnn.py --synthetic

References:
Wang et al. "Addressing wearable sleep tracking inequity: a new
dataset and novel methods for a population with sleep disorders."
CHIL 2024, PMLR 248:380-396.
"""

import argparse
import os
import tempfile

import pandas as pd
import numpy as np

from pyhealth.datasets import DREAMTDataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.datasets.sample_dataset import SampleDataset
from pyhealth.models import RNN
from pyhealth.tasks.sleep_wake_detection_dreamt import (
    SleepWakeDetectionDREAMT,
    FEATURE_COLUMNS,
)
from pyhealth.trainer import Trainer

# Feature configurations to compare
ACC_FEATURES = [c for c in FEATURE_COLUMNS if c.startswith("ACC")]
TOP2_FEATURES = ["ACC_INDEX", "HRV_HFD"]

CONFIGURATIONS = {
    "all_features": FEATURE_COLUMNS,
    "acc_only": ACC_FEATURES,
    "top2_paper": TOP2_FEATURES,
}

# Synthetic data helpers (for testing without access to real dataset)

def _make_synthetic_dreamt_root(tmp_dir: str, n_patients: int = 5) -> str:
    """Create a minimal synthetic DREAMT directory for demo purposes.

    Generates participant_info.csv and one feature CSV per patient with
    random signal features and alternating sleep/wake labels.

    Args:
        tmp_dir: Directory to write synthetic files into.
        n_patients: Number of synthetic patients to generate.

    Returns:
        Path to the synthetic DREAMT root directory.
    """
    os.makedirs(os.path.join(tmp_dir, "data_64Hz"), exist_ok=True)

    # participant_info.csv
    participant_info = pd.DataFrame({
        "SID": [f"S{i:03d}" for i in range(1, n_patients + 1)],
        "AGE": np.random.randint(30, 75, n_patients),
        "GENDER": ["M" if i % 2 == 0 else "F" for i in range(n_patients)],
        "BMI": np.random.uniform(25, 40, n_patients).round(1),
        "OAHI": np.random.uniform(0, 30, n_patients).round(1),
        "AHI": np.random.uniform(0, 40, n_patients).round(1),
        "Mean_SaO2": [f"{v:.1f}%" for v in np.random.uniform(90, 98, n_patients)],
        "Arousal Index": np.random.uniform(10, 50, n_patients).round(1),
        "MEDICAL_HISTORY": ["None"] * n_patients,
        "Sleep_Disorders": ["Sleep Apnea"] * n_patients,
    })
    participant_info.to_csv(
        os.path.join(tmp_dir, "participant_info.csv"), index=False
    )

    # One feature CSV per patient (50 epochs each)
    stages = ["W", "N1", "N2", "N3", "R"]
    n_epochs = 50
    for sid in participant_info["SID"]:
        data = {col: np.random.randn(n_epochs).tolist() for col in FEATURE_COLUMNS}
        data["Sleep_Stage"] = [stages[i % len(stages)] for i in range(n_epochs)]
        df = pd.DataFrame(data)
        df.to_csv(
            os.path.join(tmp_dir, "data_64Hz", f"{sid}_whole_df.csv"), index=False
        )

    return tmp_dir

# Core experiment runner

def run_configuration(
    dataset_root: str,
    config_name: str,
    feature_columns: list,
) -> dict:
    """Run a single feature configuration and return evaluation metrics.

    Args:
        dataset_root: Path to DREAMT dataset root directory.
        config_name: Human-readable name for this configuration.
        feature_columns: List of feature column names to use.

    Returns:
        Dict of evaluation metrics from PyHealth Trainer.
    """
    print(f"\n  Running: {config_name} ({len(feature_columns)} features)")

    dataset = DREAMTDataset(root=dataset_root)
    task = SleepWakeDetectionDREAMT(feature_columns=feature_columns)
    samples = dataset.set_task(task)

    train_ds, val_ds, test_ds = split_by_patient(samples, [0.7, 0.1, 0.2])
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

    model = RNN(
        dataset=samples,
        feature_keys=["features"],
        label_key="label",
        mode="binary",
    )
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=10,
        monitor="roc_auc",
    )
    metrics = trainer.evaluate(test_loader)
    return metrics


def main(dataset_root: str) -> None:
    """Run all feature configurations and print a comparison table.

    Args:
        dataset_root: Path to DREAMT dataset root directory.
    """
    print("=" * 65)
    print("Ablation Study: SleepWakeDetectionDREAMT Feature Configurations")
    print("=" * 65)
    print(
        "\nComparing three feature sets using RNN on binary sleep-wake "
        "classification.\nSee module docstring for full experimental rationale.\n"
    )

    results = {}
    for config_name, feature_columns in CONFIGURATIONS.items():
        results[config_name] = run_configuration(
            dataset_root, config_name, feature_columns
        )

    # Print comparison table
    print("\n" + "=" * 65)
    print("Results Summary")
    print("=" * 65)
    print(f"{'Configuration':<20} {'AUROC':>8} {'F1':>8} {'Accuracy':>10}")
    print("-" * 65)
    for config_name, metrics in results.items():
        auroc = metrics.get("roc_auc", float("nan"))
        f1 = metrics.get("f1", float("nan"))
        acc = metrics.get("accuracy", float("nan"))
        print(f"{config_name:<20} {auroc:>8.3f} {f1:>8.3f} {acc:>10.3f}")
    print("=" * 65)
    print(
        "\nExpected finding: top2_paper should approach all_features "
        "performance,\nconfirming ACC_INDEX and HRV_HFD as dominant predictors "
        "(Wang et al. 2024)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation study for SleepWakeDetectionDREAMT task."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--root",
        type=str,
        help="Path to real DREAMT dataset root, e.g. .../dreamt/2.1.0/",
    )
    group.add_argument(
        "--synthetic",
        action="store_true",
        help="Run with synthetic data (no PhysioNet access required).",
    )
    args = parser.parse_args()

    if args.synthetic:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _make_synthetic_dreamt_root(tmp_dir)
            main(root)
    else:
        main(args.root)