from collections import Counter

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from pyhealth.datasets import DREAMTDataset
from pyhealth.tasks.sleep_wake_classification import SleepWakeClassification

# Configuration
DREAMT_ROOT = "REPLACE_WITH_DREAMT_ROOT"
TRAIN_PATIENT_IDS = ["S028", "S062", "S078"]
EVAL_PATIENT_IDS = ["S081", "S099"]
EPOCH_SECONDS = 30
SAMPLING_RATE = 64


def split_samples_by_patient_ids(X, y, groups):
    """Splits samples into train and evaluation sets using patient IDs.

    Args:
        X: Feature matrix.
        y: Binary label vector.
        groups: Patient identifier for each sample.

    Returns:
        Train and evaluation features, labels, and patient groups.
    """
    train_mask = np.isin(groups, TRAIN_PATIENT_IDS)
    eval_mask = np.isin(groups, EVAL_PATIENT_IDS)

    if not np.any(train_mask):
        raise ValueError("No samples found for TRAIN_PATIENT_IDS.")
    if not np.any(eval_mask):
        raise ValueError("No samples found for EVAL_PATIENT_IDS.")

    return (
        X[train_mask],
        X[eval_mask],
        y[train_mask],
        y[eval_mask],
        groups[train_mask],
        groups[eval_mask],
    )


def run_experiment(X, y, groups, name):
    # Split samples into train and evaluation sets
    X_train, X_test, y_train, y_test, g_train, g_test = split_samples_by_patient_ids(
        X,
        y,
        groups,
    )

    # Report dataset statistics
    print(f"\n=== {name} ===")
    print("train patients:", sorted(set(g_train)))
    print("evaluation patients:", sorted(set(g_test)))
    print("train size:", len(X_train))
    print("evaluation size:", len(X_test))

    # Remove features that are all NaN in the training set
    non_all_nan_cols = ~np.isnan(X_train).all(axis=0)
    X_train = X_train[:, non_all_nan_cols]
    X_test = X_test[:, non_all_nan_cols]

    print("kept features:", X_train.shape[1])

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Train a LightGBM model on the current feature subset.
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    y_prob = model.predict(X_test)
    y_pred = (y_prob >= 0.3).astype(int)

    # Report standard binary classification metrics.
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("AUROC:", roc_auc_score(y_test, y_prob))
    print("AUPRC:", average_precision_score(y_test, y_prob))


def run_model_comparison(X, y, groups):
    # Use the same predefined patient split to compare alternative models
    X_train, X_test, y_train, y_test, g_train, g_test = split_samples_by_patient_ids(
        X,
        y,
        groups,
    )

    print("\n=== Model comparison (ALL modalities + temporal) ===")
    print("train patients:", sorted(set(g_train)))
    print("evaluation patients:", sorted(set(g_test)))

    non_all_nan_cols = ~np.isnan(X_train).all(axis=0)
    X_train = X_train[:, non_all_nan_cols]
    X_test = X_test[:, non_all_nan_cols]

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Compare logistic regression and random forest on the full feature set.
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        y_pred = (y_prob >= 0.3).astype(int)

        print(f"\n{name}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("F1:", f1_score(y_test, y_pred))
        print("AUROC:", roc_auc_score(y_test, y_prob))
        print("AUPRC:", average_precision_score(y_test, y_prob))


def main():
    if DREAMT_ROOT == "REPLACE_WITH_DREAMT_ROOT":
        raise ValueError(
            "Please set DREAMT_ROOT in examples/sleep_wake_classification.py "
            "before running this example.",
        )

    dataset = DREAMTDataset(root=DREAMT_ROOT)
    task = SleepWakeClassification(
        epoch_seconds=EPOCH_SECONDS,
        sampling_rate=SAMPLING_RATE,
    )

    # Convert the selected DREAMT patients into epoch-level sleep/wake samples.
    all_samples = []
    selected_patient_ids = TRAIN_PATIENT_IDS + EVAL_PATIENT_IDS
    for patient_id in selected_patient_ids:
        patient = dataset.get_patient(patient_id)
        samples = task(patient)
        print(f"patient {patient_id}: {len(samples)} epoch samples")
        all_samples.extend(samples)

    print("total epoch samples:", len(all_samples))
    print("label counts:", Counter(s["label"] for s in all_samples))

    # Turn the task samples into arrays for training and evaluation.
    X_all = np.array([s["features"] for s in all_samples], dtype=float)
    y = np.array([s["label"] for s in all_samples], dtype=int)
    groups = np.array([s["patient_id"] for s in all_samples])

    print("X_all shape:", X_all.shape)

    # Keep only the base per-epoch features without temporal augmentation.
    X_base = X_all[:, :21]

    # Keep the full feature matrix, including temporal context features.
    X_temporal = X_all

    # Group feature indices by modality for the ablation experiments.
    acc_idx = list(range(0, 10))
    temp_idx = list(range(10, 14))
    bvp_idx = list(range(14, 17))
    eda_idx = list(range(17, 21))

    X_acc = X_base[:, acc_idx]
    X_acc_temp = X_base[:, acc_idx + temp_idx]
    X_acc_temp_bvp = X_base[:, acc_idx + temp_idx + bvp_idx]
    X_all_modalities = X_base[:, acc_idx + temp_idx + bvp_idx + eda_idx]

    # Run experiments using different features
    run_experiment(X_acc, y, groups, "ACC only")
    run_experiment(X_acc_temp, y, groups, "ACC + TEMP")
    run_experiment(X_acc_temp_bvp, y, groups, "ACC + TEMP + BVP")
    run_experiment(X_all_modalities, y, groups, "ALL modalities")
    run_experiment(X_temporal, y, groups, "ALL modalities + temporal")
    run_model_comparison(X_temporal, y, groups)


if __name__ == "__main__":
    main()
