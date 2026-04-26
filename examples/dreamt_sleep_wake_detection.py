"""
Sleep/Wake Detection on DREAMT Dataset
=======================================
Reproduces the binary sleep/wake classification pipeline from:

    Wang et al. (2024). Addressing wearable sleep tracking inequity:
    a new dataset and novel methods for a population with sleep disorders.
    CHIL 2024, PMLR 248:380-396.

    Dataset: https://physionet.org/content/dreamt/2.1.0/
    Original code: https://github.com/WillKeWang/DREAMT_FE

This example demonstrates:
    1. Loading the DREAMT dataset via PyHealth (or generating synthetic data)
    2. Applying the SleepWakeDetectionDREAMT task to generate epoch samples
    3. Training a baseline LightGBM classifier (epoch-by-epoch)
    4. Ablation: comparing LightGBM vs LightGBM + AHI vs LightGBM + BMI

Requirements:
    pip install lightgbm 
    pip install imbalanced-learn

Usage:
    # Run with synthetic demo data (no PhysioNet access needed):
    python examples/dreamt_sleep_wake_detection.py --demo

    # Run with real DREAMT data from PhysioNet:
    python examples/dreamt_sleep_wake_detection.py --root /path/to/dreamt/2.1.0
"""

import argparse
import numpy as np
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GroupShuffleSplit

from pyhealth.datasets import DREAMTDataset
from pyhealth.tasks import SleepWakeDetectionDREAMT
from pyhealth.metrics import binary_metrics_fn


def make_synthetic_samples(n_patients: int = 5, n_epochs_per_patient: int = 40):
    """Generate synthetic DREAMT-like epoch samples for demo/testing.

    Simulates the output of SleepWakeDetectionDREAMT.__call__() without
    needing real PhysioNet data. Signal values are random floats; labels
    are randomly assigned with a 75/25 sleep/wake split matching the
    approximate class balance in the real dataset.

    Args:
        n_patients: number of synthetic patients to generate
        n_epochs_per_patient: number of 30-second epochs per patient

    Returns:
        List of sample dicts matching SleepWakeDetectionDREAMT output format
    """
    np.random.seed(42)
    samples = []
    for i in range(n_patients):
        sid = f"S{i+1:03d}"
        ahi = np.random.uniform(5, 40)
        bmi = np.random.uniform(25, 45)
        for epoch_idx in range(n_epochs_per_patient):
            label = 1 if np.random.random() < 0.25 else 0
            signal = np.random.randn(96).astype(np.float32)
            samples.append({
                "patient_id": sid,
                "epoch_index": epoch_idx,
                "signal": signal.flatten(),
                "ahi": ahi,
                "bmi": bmi,
                "label": label,
            })
    return samples


def samples_to_arrays(samples):
    """Convert task samples into numpy arrays for sklearn.

    Args:
        samples: list of dicts from SleepWakeDetectionDREAMT

    Returns:
        X: feature matrix (n_samples, n_features)
        y: binary labels (n_samples,)
        groups: patient IDs for participant-level CV (n_samples,)
        ahi: AHI values (n_samples,)
        bmi: BMI values (n_samples,)
    """
    X, y, groups, ahi, bmi = [], [], [], [], []
    for s in samples:
        X.append(np.array(s["signal"]).flatten())
        y.append(s["label"])
        groups.append(s["patient_id"])
        ahi.append(s["ahi"])
        bmi.append(s["bmi"])
    return (
        np.array(X),
        np.array(y),
        np.array(groups),
        np.array(ahi),
        np.array(bmi),
    )


def evaluate(y_true, y_prob, label=""):
    """Print binary classification metrics using PyHealth metrics."""
    metrics = binary_metrics_fn(
        y_true,
        y_prob,
        metrics=["f1", "roc_auc", "pr_auc", "accuracy", "cohen_kappa"],
    )
    print(f"\n{'─' * 55}")
    print(label)
    print(f"\n{'─' * 55}")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.3f}")


def main(root: str = None, demo: bool = False):
    print("\n[1/4] Loading data...")
    if demo:
        print("Using synthetic data")
        print("      To use real data: --root /path/to/dreamt/2.1.0")
        all_samples = make_synthetic_samples(
            n_patients=5, n_epochs_per_patient=40
        )
    else:
        print("\nUsing real DREAMT dataset via PyHealth...")
        dataset = DREAMTDataset(root=root)
        task = SleepWakeDetectionDREAMT()
        all_samples = []
        for pid in dataset.unique_patient_ids:
            patient = dataset.get_patient(pid)
            all_samples.extend(task(patient))

    print(f"    Total epochs : {len(all_samples)}")
    wake = sum(s["label"] == 1 for s in all_samples)
    sleep = sum(s["label"] == 0 for s in all_samples)
    print(f"    Wake  (1)    : {wake}  ({100*wake/len(all_samples):.1f}%)")
    print(f"    Sleep (0)    : {sleep} ({100*sleep/len(all_samples):.1f}%)")

    print("\n[2/4] Extracting features...")
    X, y, groups, ahi, bmi = samples_to_arrays(all_samples)
    print(f"    Feature matrix: {X.shape}")

    print("\n[3/4] Splitting by participant (80/20)...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    ahi_train, ahi_test = ahi[train_idx], ahi[test_idx]
    bmi_train, bmi_test = bmi[train_idx], bmi[test_idx]

    # SMOTE balancing on training set (paper section 3.1)
    try:
        X_train, y_train = SMOTE(random_state=42).fit_resample(
            X_train, y_train
        )
        print("    Applied SMOTE balancing")
    except ImportError:
        print("    imbalanced-learn not installed, skipping SMOTE")

    # Ablation study 
    # Ablation A: Baseline LightGBM 
    clf_a = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    clf_a.fit(X_train, y_train)
    evaluate(y_test,
             clf_a.predict_proba(X_test)[:, 1],
             "Ablation A: Baseline LightGBM (no clinical metadata)"
    )

    # Ablation B: LightGBM + AHI 
    X_train_b = np.hstack([
        X_train,
        np.tile(ahi_train, (len(X_train)//len(ahi_train)+1))[:len(X_train)].reshape(-1, 1)
    ])
    X_test_b = np.hstack([X_test, ahi_test.reshape(-1, 1)])
    clf_b = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    clf_b.fit(X_train_b, y_train)
    evaluate(y_test,
             clf_b.predict_proba(X_test_b)[:, 1],
             "Ablation B: LightGBM + AHI (apnea severity)"
    )

    # Ablation C:LightGBM + BMI 
    X_train_c = np.hstack([
        X_train,
        np.tile(bmi_train, (len(X_train)//len(bmi_train)+1))[:len(X_train)].reshape(-1, 1)
    ])
    X_test_c = np.hstack([X_test, bmi_test.reshape(-1, 1)])
    clf_c = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    clf_c.fit(X_train_c, y_train)
    evaluate(y_test,
             clf_c.predict_proba(X_test_c)[:, 1],
             "Ablation C: LightGBM + BMI (obesity)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DREAMT sleep/wake detection — ablation study"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic data (no PhysioNet access needed)",
    )
    group.add_argument(
        "--root",
        type=str,
        help="Path to real DREAMT dataset, e.g. /path/to/dreamt/2.1.0",
    )
    args = parser.parse_args()
    main(root=args.root, demo=args.demo)