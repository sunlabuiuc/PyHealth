"""
Example ablation script for MIMIC-III circulatory failure prediction.

This script compares different prediction windows (6h, 12h, 24h) and
feature settings using logistic regression. It is intended as an example
usage script for the standard PyHealth dataset → task → SampleDataset pipeline.

Usage:
    python mimic3_cf_circulatory_failure_logreg.py --root /path/to/mimic-iii
"""

import argparse

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import CirculatoryFailurePredictionTask

def samples_to_df(sample_dataset) -> pd.DataFrame:
    """Converts a SampleDataset into a pandas DataFrame."""
    rows = []
    for i in range(len(sample_dataset)):
        s = sample_dataset[i]
        rows.append(
            {
                "patient_id": s["patient_id"],
                "icustay_id": s["icustay_id"],
                "gender": s.get("gender"),
                "timestamp": s.get("timestamp"),
                "map": to_scalar(s["map"]),
                "map_diff": to_scalar(s["map_diff"]),
                "label": int(to_scalar(s["label"])),
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    balanced: bool = False,
) -> dict:
    if df.empty or df["label"].nunique() < 2:
        return {
            "n_samples": len(df),
            "accuracy": None,
            "roc_auc": None,
            "recall": None,
        }

    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced" if balanced else None,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "n_samples": len(df),
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "recall": recall_score(y_test, preds),
    }


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"n_samples: {metrics['n_samples']}")
    print(f"accuracy: {metrics['accuracy']}")
    print(f"roc_auc: {metrics['roc_auc']}")
    print(f"recall: {metrics['recall']}")

def to_scalar(x):
    """Converts scalar tensor-like values to Python scalars."""
    if hasattr(x, "item"):
        return x.item()
    return x

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIMIC-III circulatory failure prediction ablation study."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to the unzipped MIMIC-III database directory.",
    )
    args = parser.parse_args()

    dataset = MIMIC3Dataset(
        root=args.root,
        tables=["patients", "admissions", "icustays", "chartevents"],
    )

    # Task ablation: prediction windows
    for window in [6, 12, 24]:
        print(f"\n############################")
        print(f"Prediction window = {window}h")
        print(f"############################")

        task = CirculatoryFailurePredictionTask(prediction_window_hours=window)
        sample_dataset = dataset.set_task(task)
        df = samples_to_df(sample_dataset)

        print("\nSample preview:")
        print(df.head())

        # Baseline setting
        baseline_metrics = evaluate_model(
            df=df,
            feature_cols=["map"],
            balanced=False,
        )
        print_metrics("Baseline: LogisticRegression(map)", baseline_metrics)

        # Advanced setting
        advanced_metrics = evaluate_model(
            df=df,
            feature_cols=["map", "map_diff"],
            balanced=True,
        )
        print_metrics(
            "Advanced: LogisticRegression(map + map_diff, balanced)",
            advanced_metrics,
        )

        # Subgroup fairness
        for gender in ["M", "F"]:
            subgroup_df = df[df["gender"] == gender].copy()
            subgroup_metrics = evaluate_model(
                df=subgroup_df,
                feature_cols=["map", "map_diff"],
                balanced=True,
            )
            print_metrics(f"Advanced subgroup gender={gender}", subgroup_metrics)


if __name__ == "__main__":
    main()