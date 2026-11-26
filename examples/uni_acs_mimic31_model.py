from pathlib import Path

import numpy as np
import pandas as pd
import shap

from scipy.special import logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split

from pyhealth.interpret.uni_acs import UniACS


DATA_PATH = Path.home() / "Desktop" / "mimiciv-3.1" / "derived"

FEATURES = [
    "age", "gender_male", "creatinine", "bun",
    "sodium", "potassium", "chloride", "bicarbonate",
    "glucose", "wbc", "hemoglobin", "platelets",
    "lactate", "bilirubin",
]


def evaluate(y, scores, label):
    print(label)
    print("AUROC:", round(roc_auc_score(y, scores), 4))
    print("AUPRC:", round(average_precision_score(y, scores), 4))


def main():
    data = pd.read_csv(DATA_PATH / "sepsis_24h_features.csv")

    X = data[FEATURES].copy()
    y = data["hospital_mortality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURES
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test), columns=FEATURES
    )

    print("MIMIC-IV 3.1: Sepsis mortality evaluation with a stratified random split")
    print("Train/test:", X_train.shape, X_test.shape)
    print("Mortality:", round(y_train.mean(), 4),
          round(y_test.mean(), 4))

    model = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=260,
            max_depth=5,
            random_state=7,
            n_jobs=-1,
        ),
        method="isotonic",
        cv=5,
    )
    model.fit(X_train, y_train)

    rf_probs = model.predict_proba(X_test)[:, 1]
    print("\nCalibrated RandomForest")
    evaluate(y_test, rf_probs, "Baseline:")
    print("Brier:", round(brier_score_loss(y_test, rf_probs), 4))

    # Start with 2,000 samples for SHAP
    X_sample = X_train.sample(n=2000, random_state=42)

    shap_list = []
    expected_list = []

    for i, clf in enumerate(model.calibrated_classifiers_):
        print(f"Calculating SHAP for fold {i + 1}...")

        explainer = shap.TreeExplainer(clf.estimator)
        values = explainer.shap_values(X_sample)

        if isinstance(values, list):
            values = values[1]
        elif values.ndim == 3:
            values = values[:, :, 1]

        base = float(np.asarray(explainer.expected_value)[1])
        base = np.clip(base, 0.0001, 0.9999)

        adjusted = np.clip(base + values, 0.0001, 0.9999)
        shap_list.append(logit(adjusted) - logit(base))
        expected_list.append(logit(base))

    shap_values = np.mean(shap_list, axis=0)
    expected_value = float(np.mean(expected_list))

    print("\nSHAP shape:", shap_values.shape)

    for k in [5, 10, 14]:
        score_model = UniACS(top_k=k, method="novel")
        score_model.fit(X_sample, shap_values, expected_value)

        scores = score_model.predict_score(X_test)
        probs = score_model.predict_proba(X_test)

        print(f"\nUniACS Top {k}")
        print("Categories:", len(score_model.get_score_table()))
        evaluate(y_test, scores, "Raw scores:")
        print("Brier (approx):",
              round(brier_score_loss(y_test, probs), 4))

    print("\nRandom-split evaluation complete.")


if __name__ == "__main__":
    main()
