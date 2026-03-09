from collections import Counter

import numpy as np
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit

from pyhealth.datasets import DREAMTDataset
from pyhealth.tasks.sleep_wake_classification import SleepWakeClassification


def run_experiment(X, y, groups, name):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    g_train, g_test = groups[train_idx], groups[test_idx]

    print(f"\n=== {name} ===")
    print("train patients:", sorted(set(g_train)))
    print("test patients:", sorted(set(g_test)))
    print("train size:", len(X_train))
    print("test size:", len(X_test))

    non_all_nan_cols = ~np.isnan(X_train).all(axis=0)
    X_train = X_train[:, non_all_nan_cols]
    X_test = X_test[:, non_all_nan_cols]

    print("kept features:", X_train.shape[1])

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

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

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))
    print("AUROC:", roc_auc_score(y_test, y_prob))
    print("AUPRC:", average_precision_score(y_test, y_prob))


def main():
    root = r"C:\Users\faria\OneDrive - University of Illinois - Urbana\CS-598-DLH\dreamt-replication\data\DREAMT"

    dataset = DREAMTDataset(root=root)
    task = SleepWakeClassification()

    selected_patient_ids = ["S028", "S062", "S078", "S081", "S099"]

    all_samples = []
    for patient_id in selected_patient_ids:
        patient = dataset.get_patient(patient_id)
        samples = task(patient)
        print(patient_id, len(samples))
        all_samples.extend(samples)

    print("total samples:", len(all_samples))
    print("label counts:", Counter(s["label"] for s in all_samples))

    X_all = np.array([s["features"] for s in all_samples], dtype=float)
    y = np.array([s["label"] for s in all_samples], dtype=int)
    groups = np.array([s["patient_id"] for s in all_samples])

    print("X_all shape:", X_all.shape)

    # base only = first 21 features
    X_base = X_all[:, :21]

    # base + temporal = all features
    X_temporal = X_all

    acc_idx = list(range(0, 10))        # ACC_X, ACC_Y, ACC_Z, ACC_MAD
    temp_idx = list(range(10, 14))      # TEMP
    bvp_idx = list(range(14, 17))       # BVP
    eda_idx = list(range(17, 21))       # EDA

    X_acc = X_base[:, acc_idx]
    X_acc_temp = X_base[:, acc_idx + temp_idx]
    X_acc_temp_bvp = X_base[:, acc_idx + temp_idx + bvp_idx]
    X_all_modalities = X_base[:, acc_idx + temp_idx + bvp_idx + eda_idx]

    run_experiment(X_acc, y, groups, "ACC only")
    run_experiment(X_acc_temp, y, groups, "ACC + TEMP")
    run_experiment(X_acc_temp_bvp, y, groups, "ACC + TEMP + BVP")
    run_experiment(X_all_modalities, y, groups, "ALL modalities")
    run_experiment(X_temporal, y, groups, "ALL modalities + temporal")


if __name__ == "__main__":
    main()