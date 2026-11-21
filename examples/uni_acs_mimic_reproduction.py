"""Reproduce Uni-ACS experiments on the authors' preprocessed MIMIC-IV CSV.

Run from the PyHealth repository root:
    python examples/uni_acs_mimic_reproduction.py

The clinical CSV is private and remains outside this repository.
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from pyhealth.interpret.uni_acs import UniACS
from uni_acs_utils import (
    aggregate_shap,
    evaluate,
    impute_data,
    load_mimic_data,
    print_metrics,
    train_calibrated_rf,
    train_rf,
)

DATA_FILE = (
    Path.home() / "Desktop" / "Uni-ACS" / "mimiciv_sepsis_mortality"
    / "mimiciv_sepsis.csv"
)
TOP_K = (5, 10, 15)


def run_top_k(X_fit, shap_values, expected_value, X_test, y_test, calibrate=None):
    """Evaluate score tables. calibrate=(X_cal, y_cal) is optional."""
    for k in TOP_K:
        score = UniACS(top_k=k, method="novel")
        score.fit(X_fit, shap_values, expected_value)

        raw_scores = score.predict_score(X_test)
        approximate = score.predict_proba(X_test)
        print(f"\nTop {k} features (categories: {len(score.get_score_table())})")
        print_metrics("Raw score / approximate probability:",
                      evaluate(y_test, raw_scores, approximate))

        if calibrate is not None:
            X_cal, y_cal = calibrate
            score.calibrate_scores(X_cal, y_cal)
            calibrated = score.predict_proba(X_test)
            print_metrics("Independently calibrated probability:",
                          evaluate(y_test, calibrated, calibrated))


def run_d24(X_train_raw, X_test_raw, y_train, y_test):
    """D1.4 baseline and D2.4 5-fold SHAP + full-train Top-K study."""
    X_train, X_test = impute_data(X_train_raw, X_test_raw)
    print("\nD1.4: RandomForest baseline")
    baseline = train_rf(X_train, y_train)
    base_prob = baseline.predict_proba(X_test)[:, 1]
    print_metrics("Uncalibrated RF:", evaluate(y_test, base_prob, base_prob))

    print("\nD2.4: Calibrated RandomForest")
    model = train_calibrated_rf(X_train, y_train)
    rf_prob = model.predict_proba(X_test)[:, 1]
    print_metrics("5-fold calibrated RF:", evaluate(y_test, rf_prob, rf_prob))
    constant = np.full(len(y_test), y_train.mean())
    print("Constant Brier:", round(brier_score_loss(y_test, constant), 4))

    shap_values, expected_value = aggregate_shap(model, X_train)
    print("SHAP shape:", shap_values.shape)
    print("Expected log-odds:", round(expected_value, 4))
    print("\nD2.4: Uni-ACS Top-K Ablation")
    run_top_k(X_train, shap_values, expected_value, X_test, y_test)


def run_d25(X_train_raw, X_test_raw, y_train, y_test):
    """D2.5 separate score-training / calibration split (no test leakage)."""
    X_score_raw, X_cal_raw, y_score, y_cal = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    X_score, X_cal, X_test = impute_data(X_score_raw, X_cal_raw, X_test_raw)
    print("\nD2.5: Independent score-level calibration")
    print("Sizes (score/cal/test):", len(X_score), len(X_cal), len(X_test))

    model = train_calibrated_rf(X_score, y_score)
    rf_prob = model.predict_proba(X_test)[:, 1]
    print_metrics("D2.5 calibrated RF:", evaluate(y_test, rf_prob, rf_prob))

    shap_values, expected_value = aggregate_shap(model, X_score)
    print("SHAP shape:", shap_values.shape)
    run_top_k(X_score, shap_values, expected_value, X_test, y_test,
              calibrate=(X_cal, y_cal))


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"MIMIC CSV not found: {DATA_FILE}")
    X_train, X_test, y_train, y_test = load_mimic_data(DATA_FILE)
    print("Train/test:", X_train.shape, X_test.shape)
    print("Train/test mortality:", round(y_train.mean(), 4), round(y_test.mean(), 4))
    run_d24(X_train, X_test, y_train, y_test)
    run_d25(X_train, X_test, y_train, y_test)
    print("\nReproduction complete.")


if __name__ == "__main__":
    main()
