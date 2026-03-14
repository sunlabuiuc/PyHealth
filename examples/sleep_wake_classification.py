from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import io
import logging
import warnings

import lightgbm as lgb
import numpy as np

from sklearn.exceptions import ConvergenceWarning
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

# Console formatting codes
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def format_section(title):
    """Formats a section title for console output.

    Args:
        title: Section title to format.

    Returns:
        A colorized section title string.
    """
    return f"\n{BOLD}{CYAN}{title}{RESET}"


def format_patient_ids(patient_ids):
    """Formats patient IDs for readable console output.

    Args:
        patient_ids: Iterable of patient identifiers.

    Returns:
        A comma-separated string of patient IDs.
    """
    return ", ".join(sorted(str(patient_id) for patient_id in set(patient_ids)))


def print_metric(name, value):
    """Prints a metric with consistent console formatting.

    Args:
        name: Metric name.
        value: Metric value.
    """
    print(f"  {name:<16}{value:.4f}")


def summarize_label_counts(labels):
    """Builds a readable sleep/wake label summary.

    Args:
        labels: Iterable of binary labels.

    Returns:
        A formatted label count string.
    """
    counts = Counter(labels)
    return (
        f"sleep (0): {counts.get(0, 0)}, "
        f"wake (1): {counts.get(1, 0)}"
    )


def configure_clean_output():
    """Suppresses noisy warnings and logs for a cleaner example run."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    logging.getLogger("pyhealth").setLevel(logging.ERROR)
    logging.getLogger("pyhealth.tasks.sleep_wake_classification").setLevel(
        logging.ERROR
    )


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
    print(format_section(f"Ablation: {name}"))
    print(f"{BOLD}Train patients:{RESET} {format_patient_ids(g_train)}")
    print(f"{BOLD}Eval patients:{RESET}  {format_patient_ids(g_test)}")
    print(f"{BOLD}Train samples:{RESET}  {len(X_train)}")
    print(f"{BOLD}Eval samples:{RESET}   {len(X_test)}")

    # Remove features that are all NaN in the training set
    non_all_nan_cols = ~np.isnan(X_train).all(axis=0)
    X_train = X_train[:, non_all_nan_cols]
    X_test = X_test[:, non_all_nan_cols]

    print(f"{BOLD}Feature count:{RESET} {X_train.shape[1]}")

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
    print_metric("Accuracy", accuracy_score(y_test, y_pred))
    print_metric("F1", f1_score(y_test, y_pred))
    print_metric("AUROC", roc_auc_score(y_test, y_prob))
    print_metric("AUPRC", average_precision_score(y_test, y_prob))


def run_model_comparison(X, y, groups):
    # Use the same predefined patient split to compare alternative models
    X_train, X_test, y_train, y_test, g_train, g_test = split_samples_by_patient_ids(
        X,
        y,
        groups,
    )

    print(format_section("Model Comparison: ALL modalities + temporal"))
    print(f"{BOLD}Train patients:{RESET} {format_patient_ids(g_train)}")
    print(f"{BOLD}Eval patients:{RESET}  {format_patient_ids(g_test)}")

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

        print(f"\n{YELLOW}{name}{RESET}")
        print_metric("Accuracy", accuracy_score(y_test, y_pred))
        print_metric("F1", f1_score(y_test, y_pred))
        print_metric("AUROC", roc_auc_score(y_test, y_prob))
        print_metric("AUPRC", average_precision_score(y_test, y_prob))


def main():
    configure_clean_output()

    if DREAMT_ROOT == "REPLACE_WITH_DREAMT_ROOT":
        raise ValueError(
            "Please set DREAMT_ROOT in examples/sleep_wake_classification.py "
            "before running this example.",
        )

    # Suppress verbose dataset initialization messages and print a cleaner summary.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        dataset = DREAMTDataset(root=DREAMT_ROOT)
    task = SleepWakeClassification(
        epoch_seconds=EPOCH_SECONDS,
        sampling_rate=SAMPLING_RATE,
    )

    print(format_section("DREAMT Sleep-Wake Classification Example"))
    print(f"{BOLD}Dataset root:{RESET} {DREAMT_ROOT}")
    print(
        f"{BOLD}Train patients:{RESET} {', '.join(TRAIN_PATIENT_IDS)}"
    )
    print(
        f"{BOLD}Eval patients:{RESET}  {', '.join(EVAL_PATIENT_IDS)}"
    )

    # Convert the selected DREAMT patients into epoch-level sleep/wake samples.
    all_samples = []
    selected_patient_ids = TRAIN_PATIENT_IDS + EVAL_PATIENT_IDS
    for patient_id in selected_patient_ids:
        patient = dataset.get_patient(patient_id)
        samples = task(patient)
        print(f"  patient {patient_id:<4} -> {len(samples)} epoch samples")
        all_samples.extend(samples)

    print(f"{BOLD}Total epoch samples:{RESET} {len(all_samples)}")
    print(
        f"{BOLD}Label counts:{RESET}      "
        f"{summarize_label_counts(sample['label'] for sample in all_samples)}"
    )

    # Turn the task samples into arrays for training and evaluation.
    X_all = np.array([s["features"] for s in all_samples], dtype=float)
    y = np.array([s["label"] for s in all_samples], dtype=int)
    groups = np.array([s["patient_id"] for s in all_samples])

    print(f"{BOLD}Feature matrix:{RESET}    {X_all.shape[0]} samples x {X_all.shape[1]} features")

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

    # Run experiments using different feature groups.
    run_experiment(X_acc, y, groups, "ACC only")
    run_experiment(X_acc_temp, y, groups, "ACC + TEMP")
    run_experiment(X_acc_temp_bvp, y, groups, "ACC + TEMP + BVP")
    run_experiment(X_all_modalities, y, groups, "ALL modalities")
    run_experiment(X_temporal, y, groups, "ALL modalities + temporal")
    run_model_comparison(X_temporal, y, groups)


if __name__ == "__main__":
    main()
