from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from pyhealth.datasets import (
    MIMIC4EHRDataset,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.models import CaliForest
from pyhealth.tasks import InHospitalMortalityMIMIC4


# Set your MIMIC-IV dataset path via environment variable before running:
# export MIMIC4_ROOT=/your/path/to/mimiciv/3.1
ROOT = os.getenv("MIMIC4_ROOT")


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute AUROC and Brier score."""
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def run_califorest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    calibration: str,
) -> dict[str, float]:
    """Train and evaluate CaliForest on tabularized features."""
    train_samples = []
    for i in range(len(X_train)):
        train_samples.append(
            {
                "patient_id": f"train-{i}",
                "visit_id": f"train-{i}",
                "features": X_train[i].tolist(),
                "label": int(y_train[i]),
            }
        )

    test_samples = []
    for i in range(len(X_test)):
        test_samples.append(
            {
                "patient_id": f"test-{i}",
                "visit_id": f"test-{i}",
                "features": X_test[i].tolist(),
                "label": int(y_test[i]),
            }
        )

    train_dataset = create_sample_dataset(
        samples=train_samples,
        input_schema={"features": "tensor"},
        output_schema={"label": "binary"},
        dataset_name=f"mimic4_train_tabular_{calibration}",
    )
    test_dataset = create_sample_dataset(
        samples=test_samples,
        input_schema={"features": "tensor"},
        output_schema={"label": "binary"},
        dataset_name=f"mimic4_test_tabular_{calibration}",
    )

    train_loader = get_dataloader(
        train_dataset, batch_size=len(train_dataset), shuffle=False
    )
    test_loader = get_dataloader(
        test_dataset, batch_size=len(test_dataset), shuffle=False
    )

    test_batch = next(iter(test_loader))

    model = CaliForest(
        dataset=train_dataset,
        n_estimators=100,
        calibration=calibration,
        random_state=42,
    )
    model.fit(train_loader)

    with torch.no_grad():
        ret = model(**test_batch)

    cali_probs = ret["y_prob"].detach().cpu().numpy().reshape(-1)
    return evaluate(y_test, cali_probs)


def main():
    if not ROOT:
        raise ValueError(
            "MIMIC4_ROOT is not set. Example:\n"
            "export MIMIC4_ROOT=/your/path/to/mimiciv/3.1"
        )

    print("=" * 80)
    print("Loading MIMIC-IV EHR dataset")
    print("=" * 80)

    dataset = MIMIC4EHRDataset(
        root=ROOT,
        tables=["diagnoses_icd", "procedures_icd", "labevents"],
    )

    task = InHospitalMortalityMIMIC4()
    sample_dataset = dataset.set_task(task)

    print(f"Total samples: {len(sample_dataset)}")

    subset_size = 2000
    raw_subset_samples = [sample_dataset[i] for i in range(subset_size)]

    clean_subset_samples = []
    for sample in raw_subset_samples:
        clean_subset_samples.append(
            {
                "patient_id": str(sample["patient_id"]),
                "visit_id": str(sample["admission_id"]),
                "labs": sample["labs"].tolist(),
                "mortality": int(sample["mortality"].item()),
            }
        )

    subset_dataset = create_sample_dataset(
        samples=clean_subset_samples,
        input_schema={"labs": "tensor"},
        output_schema={"mortality": "binary"},
        dataset_name="mimic4_mortality_subset",
    )

    loader = get_dataloader(subset_dataset, batch_size=subset_size, shuffle=False)
    batch = next(iter(loader))

    X = batch["labs"].detach().cpu().numpy()
    y = batch["mortality"].detach().cpu().numpy().reshape(-1)

    X = X.reshape(X.shape[0], -1)

    print("Flattened feature matrix:", X.shape)
    print("Labels:", y.shape)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("=" * 80)
    print("Baseline Random Forest")
    print("=" * 80)

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        bootstrap=True,
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate(y_test, rf_probs)
    print("RF metrics:", rf_metrics)

    print("=" * 80)
    print("CaliForest (isotonic calibration)")
    print("=" * 80)
    isotonic_metrics = run_califorest(
        X_train, y_train, X_test, y_test, calibration="isotonic"
    )
    print("CaliForest isotonic metrics:", isotonic_metrics)

    print("=" * 80)
    print("CaliForest (logistic calibration)")
    print("=" * 80)
    logistic_metrics = run_califorest(
        X_train, y_train, X_test, y_test, calibration="logistic"
    )
    print("CaliForest logistic metrics:", logistic_metrics)


if __name__ == "__main__":
    main()