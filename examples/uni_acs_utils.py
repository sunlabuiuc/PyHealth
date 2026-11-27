"""Shared helpers for the Uni-ACS MIMIC-IV reproduction examples.

This module contains experiment code, not part of the public UniACS API.
"""

import numpy as np
import pandas as pd
import shap
from scipy.special import logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


def load_mimic_data(csv_path):
    """Return the author's raw 70/30 split, before imputation.

    This reproduces the split in the author's Random Forest notebook;
    it does not use the CSV's separate 'testtrain' designation.
    """
    data = pd.read_csv(csv_path)
    data = data.drop(columns=["testtrain", "hadm_id"])
    data["gender"] = (data["gender"] == "M").astype(int)

    for col in ["marital_status", "ethnicity", "insurance", "ed_medgp_antibiotic_hrgp"]:
        prefix = "antibiotic" if col == "ed_medgp_antibiotic_hrgp" else col
        dummies = pd.get_dummies(data[col], prefix=prefix)
        data = pd.concat([data.drop(columns=[col]), dummies], axis=1)

    X = data.drop(columns=["icu_adm_flag", "hosdeath_flag"])
    y = data["hosdeath_flag"]
    return train_test_split(X, y, test_size=0.3, random_state=7)


def impute_data(X_train, *other_sets):
    """Fit the imputer on X_train only; transform every remaining set."""
    imputer = IterativeImputer(max_iter=10, random_state=0)
    columns = X_train.columns
    train = pd.DataFrame(imputer.fit_transform(X_train), columns=columns)
    others = [pd.DataFrame(imputer.transform(X), columns=columns) for X in other_sets]
    return (train, *others)


def train_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=260, max_depth=5, random_state=7)
    model.fit(X_train, y_train)
    return model


def train_calibrated_rf(X_train, y_train, n_estimators=260, cv=5):
    model = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=n_estimators, max_depth=5, random_state=7
        ),
        method="isotonic",
        cv=cv,
    )
    model.fit(X_train, y_train)
    return model


def aggregate_shap(model, X):
    """Average logit-transformed positive-class TreeSHAP over the CV forests.

    This is an approximation: applying a
    nonlinear logit transform to individual probability-scale SHAP values
    does not preserve SHAP additivity, and it does not explain isotonic output.
    """
    shap_list = []
    expected_list = []

    for i, fold in enumerate(model.calibrated_classifiers_, start=1):
        print(f"Calculating SHAP for fold {i}...")
        explainer = shap.TreeExplainer(
            fold.estimator, feature_perturbation="tree_path_dependent"
        )
        values = explainer.shap_values(X)
        if isinstance(values, list):
            values = values[1]
        elif values.ndim == 3:
            values = values[:, :, 1]

        base = float(np.asarray(explainer.expected_value)[1])
        base = np.clip(base, 0.0001, 0.9999)
        adjusted = np.clip(base + values, 0.0001, 0.9999)
        shap_list.append(logit(adjusted) - logit(base))
        expected_list.append(logit(base))

    return np.mean(shap_list, axis=0), float(np.mean(expected_list))


def evaluate(y_true, scores, probabilities=None):
    """Calculate ranking metrics; add Brier when probabilities are supplied."""
    result = {
        "AUROC": roc_auc_score(y_true, scores),
        "AUPRC": average_precision_score(y_true, scores),
    }
    if probabilities is not None:
        result["Brier"] = brier_score_loss(y_true, probabilities)
    return result


def print_metrics(label, metrics):
    print(label)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
