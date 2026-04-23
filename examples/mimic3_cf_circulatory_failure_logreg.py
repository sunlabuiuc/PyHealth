"""
Example ablation script for MIMIC-III circulatory failure prediction.

This script compares different prediction windows (6h, 12h, 24h) and
feature settings using logistic regression. It is intended as an example
usage script for the dataset-task pipeline and ablation study.
"""

from pyhealth.datasets import MIMIC3CirculatoryFailureDataset
from pyhealth.tasks import CirculatoryFailurePredictionTask

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score


def samples_to_df(samples: list[dict]) -> pd.DataFrame:
    rows = []
    for s in samples:
        rows.append(
            {
                "patient_id": s["patient_id"],
                "icustay_id": s["icustay_id"],
                "gender": s["gender"],
                "timestamp": s["timestamp"],
                "map": s["features"]["map"],
                "label": s["label"],
            }
        )
    df = pd.DataFrame(rows)
    return df


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple temporal features for the advanced setting."""
    df = df.sort_values(["icustay_id", "timestamp"]).copy()
    df["map_prev"] = df.groupby("icustay_id")["map"].shift(1)
    df["map_diff"] = df["map"] - df["map_prev"]
    df["map_prev"] = df["map_prev"].fillna(df["map"])
    df["map_diff"] = df["map_diff"].fillna(0.0)
    return df


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


def main() -> None:
    dataset = MIMIC3CirculatoryFailureDataset(
        # path to the unzipped MIMIC-III database on your machine
        root="mimic-iii-dataset"
    )

    # task ablation: prediction windows
    for window in [6, 12, 24]:
        print(f"\n############################")
        print(f"Prediction window = {window}h")
        print(f"############################")

        task = CirculatoryFailurePredictionTask(prediction_window_hours=window)
        samples = dataset.set_task(task, max_patients=100)
        df = samples_to_df(samples)

        print("\nSample preview:")
        print(df.head())

        # baseline setting
        baseline_metrics = evaluate_model(
            df=df,
            feature_cols=["map"],
            balanced=False,
        )
        print_metrics("Baseline: LogisticRegression(map)", baseline_metrics)

        # advanced setting
        df_adv = add_advanced_features(df)
        advanced_metrics = evaluate_model(
            df=df_adv,
            feature_cols=["map", "map_diff"],
            balanced=True,
        )
        print_metrics(
            "Advanced: LogisticRegression(map + map_diff, balanced)",
            advanced_metrics,
        )

        # subgroup fairness
        for gender in ["M", "F"]:
            subgroup_df = df_adv[df_adv["gender"] == gender].copy()
            subgroup_metrics = evaluate_model(
                df=subgroup_df,
                feature_cols=["map", "map_diff"],
                balanced=True,
            )
            print_metrics(f"Advanced subgroup gender={gender}", subgroup_metrics)


if __name__ == "__main__":
    main()