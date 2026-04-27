"""Ablation study: SleepWakeDetectionDREAMT with varying feature configurations.

This script reproduces the core experimental comparison from the DREAMT paper
(Wang et al., CHIL 2024) within PyHealth, testing how different feature subsets
affect binary sleep-wake classification performance.

Experimental Setup:
We compare three feature configurations using PyHealth's RNN model:

  1. all_features: Full feature set from Wang et al. (ACC, TEMP, BVP, EDA,
     HR, ACC_INDEX, HRV_HFD) — replicates the paper's complete input space.
  2. acc_only: Accelerometry-derived features only — tests whether motion
     signals alone are sufficient for sleep-wake detection.
  3. top2_paper: ACC_INDEX + HRV_HFD — the two strongest predictors
     identified in the paper via feature importance analysis.

Hypothesis:
Based on the paper's findings, we expect top2_paper to perform comparably
to all_features, since ACC_INDEX and HRV_HFD were the dominant predictors.
acc_only should underperform, confirming that cardiac signals (HRV_HFD)
add meaningful information beyond motion alone.

Usage:
# With real DREAMT data:
python examples/dreamt_sleep_wake_detection_rnn.py --root /path/to/dreamt/2.1.0/

# With synthetic data (no PhysioNet access required):
python examples/dreamt_sleep_wake_detection_rnn.py --synthetic
"""

import argparse
import os
import tempfile

import numpy as np
import pandas as pd

from pyhealth.datasets import DREAMTDataset
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets.utils import get_dataloader
from pyhealth.models import RNN
from pyhealth.tasks.sleep_wake_detection_dreamt import (
    EPOCH_SAMPLES,
    FEATURE_COLUMNS,
    SleepWakeDetectionDREAMT,
)
from pyhealth.trainer import Trainer

# Feature configurations to compare
ACC_FEATURES = [c for c in FEATURE_COLUMNS if c.startswith("ACC")]
TOP2_FEATURES = ["ACC_INDEX", "HRV_HFD"]

CONFIGURATIONS = {
    "all_features": FEATURE_COLUMNS,
    "acc_only":     ACC_FEATURES,
    "top2_paper":   TOP2_FEATURES,
}


# Data helpers

def _make_synthetic_dreamt_root(tmp_dir: str, n_patients: int = 3) -> str:
    """Create a minimal synthetic DREAMT 2.1.0 directory for demo purposes.

    Generates participant_info.csv and one raw 64Hz CSV per patient with
    random signal values and cycling sleep stage labels.

    Args:
        tmp_dir: Directory to write synthetic files into.
        n_patients: Number of synthetic patients to generate.

    Returns:
        Path to the synthetic DREAMT root directory.
    """
    os.makedirs(os.path.join(tmp_dir, "data_64Hz"), exist_ok=True)

    sids = [f"S{i:03d}" for i in range(1, n_patients + 1)]

    participant_info = pd.DataFrame({
        "SID":           sids,
        "AGE":           np.random.randint(30, 75, n_patients),
        "GENDER":        ["M" if i % 2 == 0 else "F" for i in range(n_patients)],
        "BMI":           np.random.uniform(25, 40, n_patients).round(1),
        "OAHI":          np.random.uniform(0, 30, n_patients).round(1),
        "AHI":           np.random.uniform(0, 40, n_patients).round(1),
        "Mean_SaO2":     [f"{v:.1f}%" for v in np.random.uniform(90, 98, n_patients)],
        "Arousal Index": np.random.uniform(10, 50, n_patients).round(1),
        "MEDICAL_HISTORY": ["None"] * n_patients,
        "Sleep_Disorders": ["Sleep Apnea"] * n_patients,
    })
    participant_info.to_csv(os.path.join(tmp_dir, "participant_info.csv"), index=False)

    # 20 epochs per patient (enough to split train/val/test)
    n_epochs = 20
    n_rows = n_epochs * EPOCH_SAMPLES
    stages = ["W", "N1", "N2", "N3", "R"]
    stage_col = []
    for i in range(n_epochs):
        stage_col.extend([stages[i % len(stages)]] * EPOCH_SAMPLES)

    for sid in sids:
        df = pd.DataFrame({
            "TIMESTAMP":   np.linspace(0, n_epochs * 30, n_rows),
            "BVP":         np.random.randn(n_rows) * 2,
            "ACC_X":       np.random.randn(n_rows) * 10 + 30,
            "ACC_Y":       np.random.randn(n_rows) * 5 + 8,
            "ACC_Z":       np.random.randn(n_rows) * 8 + 55,
            "TEMP":        np.random.uniform(34, 37, n_rows),
            "EDA":         np.random.uniform(0.01, 0.5, n_rows),
            "HR":          np.random.uniform(50, 90, n_rows),
            "IBI":         np.random.uniform(600, 1200, n_rows),
            "Sleep_Stage": stage_col,
        })
        df.to_csv(
            os.path.join(tmp_dir, "data_64Hz", f"{sid}_whole_df.csv"),
            index=False,
        )

    return tmp_dir


# Core experiment runner

def run_configuration(
    dataset_root: str,
    config_name: str,
    feature_columns: list,
) -> dict:
    print(f"\n  Running: {config_name} ({len(feature_columns)} features)")

    dataset = DREAMTDataset(root=dataset_root)
    task = SleepWakeDetectionDREAMT(feature_columns=feature_columns)
    samples = dataset.set_task(task)

    train_ds, val_ds, test_ds = split_by_patient(samples, [0.6, 0.2, 0.2])
    train_loader = get_dataloader(train_ds, batch_size=16, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=16, shuffle=False)
    test_loader  = get_dataloader(test_ds,  batch_size=16, shuffle=False)

    # PyHealth 2.0 RNN: no feature_keys/label_key — inferred from dataset schema
    model = RNN(
        dataset=samples,
        embedding_dim=128,
        hidden_dim=128,
    )
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=5,
        monitor="roc_auc",
    )
    return trainer.evaluate(test_loader)



def main(dataset_root: str) -> None:
    """Run all configurations and print a comparison table.

    Args:
        dataset_root: Path to DREAMT dataset root directory.
    """
    print("=" * 65)
    print("Ablation Study: SleepWakeDetectionDREAMT Feature Configurations")
    print("=" * 65)

    results = {}
    for config_name, feature_columns in CONFIGURATIONS.items():
        results[config_name] = run_configuration(
            dataset_root, config_name, feature_columns
        )

    print("\n" + "=" * 65)
    print("Results Summary")
    print("=" * 65)
    print(f"{'Configuration':<20} {'AUROC':>8} {'F1':>8} {'Accuracy':>10}")
    print("-" * 65)
    for config_name, metrics in results.items():
        auroc = metrics.get("roc_auc", float("nan"))
        f1    = metrics.get("f1",      float("nan"))
        acc   = metrics.get("accuracy", float("nan"))
        print(f"{config_name:<20} {auroc:>8.3f} {f1:>8.3f} {acc:>10.3f}")
    print("=" * 65)
    print(
        "\nExpected: top2_paper approaches all_features, confirming "
        "ACC_INDEX and HRV_HFD as dominant predictors (Wang et al. 2024)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--root",      type=str,        help="Path to real DREAMT root")
    group.add_argument("--synthetic", action="store_true", help="Run with synthetic data")
    args = parser.parse_args()

    if args.synthetic:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = _make_synthetic_dreamt_root(tmp_dir)
            main(root)
    else:
        main(args.root)