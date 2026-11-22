"""Compare Uni-ACS feature counts, discretization methods, and calibration.

Run with synthetic data:
    python examples/mimic4_mortality_uni_acs.py

Run with the authors' preprocessed MIMIC-IV data (requires scikit-learn and SHAP):
    python examples/mimic4_mortality_uni_acs.py --csv /path/to/mimiciv_sepsis.csv

See mimic4_mortality_uni_acs_results.md for the setup and measured results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from pyhealth.interpret.uni_acs import UniACS

FEATURE_NAMES = [
    "age",
    "lactate",
    "creatinine",
    "bilirubin",
    "bun",
    "heart_rate",
    "resp_rate",
    "sbp",
    "spo2",
    "wbc",
    "platelets",
    "sodium",
    "potassium",
    "glucose",
    "temperature",
]


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC via Mann-Whitney U statistic without external deps."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diffs = pos[:, None] - neg[None, :]
    return float((np.sum(diffs > 0) + 0.5 * np.sum(diffs == 0)) / (len(pos) * len(neg)))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUPRC without external dependencies."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    scores = y_score[order]
    # Evaluate precision once per score threshold, including all ties.
    threshold_ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    true_positives = np.cumsum(y_true[order])[threshold_ends]
    precision = true_positives / (threshold_ends + 1)
    recall_increase = np.diff(true_positives, prepend=0) / n_pos
    return float(np.dot(precision, recall_increase))


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score."""
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def generate_synthetic_clinical_data(
    n_samples: int = 1500,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    """Generate synthetic ICU mortality features and exact additive log-odds SHAP.

    Returns:
        Tuple ``(X, y, shap_values, expected_log_odds)``.
    """
    rng = np.random.default_rng(seed)
    X_mat = rng.normal(0.0, 1.0, size=(n_samples, len(FEATURE_NAMES)))
    X = pd.DataFrame(X_mat, columns=FEATURE_NAMES)

    # First 8 features carry clinical risk signal; remaining 7 are weak/noise
    weights = np.array(
        [
            0.85, 0.95, 0.65, 0.55, 0.45,
            0.40, -0.50, -0.45, 0.25, 0.20,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ]
    )
    # Include mild nonlinearity for lactate and heart_rate
    contrib = X_mat * weights
    contrib[:, 1] += 0.25 * np.maximum(X_mat[:, 1], 0.0) ** 2
    contrib[:, 5] += 0.15 * np.abs(X_mat[:, 5])
    # Center per-feature contributions to mirror SHAP zero-mean property
    shap_values = contrib - contrib.mean(axis=0, keepdims=True)

    expected_log_odds = -1.45
    probs = expit(expected_log_odds + shap_values.sum(axis=1))
    y = rng.binomial(1, probs).astype(int)
    return X, y, shap_values, expected_log_odds


def run_ablation(
    X_fit: pd.DataFrame,
    shap_fit: np.ndarray,
    expected_value: float,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    top_k_values: tuple[int, ...] = (3, 5, 10, 15),
) -> pd.DataFrame:
    """Run Uni-ACS ablation over top_k, binning method, and calibration."""
    try:
        import sklearn  # noqa: F401

        has_sklearn = True
    except ImportError:
        has_sklearn = False

    records = []
    for method in ("quantile", "novel"):
        for k in top_k_values:
            scorer = UniACS(top_k=k, threshold=0.05, method=method)
            scorer.fit(X_fit, shap_fit, expected_value)

            raw_scores = scorer.predict_score(X_test)
            approx_probs = scorer.predict_proba(X_test)
            n_categories = len(scorer.get_score_table())

            row = {
                "method": method,
                "top_k": k,
                "categories": n_categories,
                "auroc_raw": _roc_auc(y_test, raw_scores),
                "auprc_raw": _average_precision(y_test, raw_scores),
                "brier_approx": _brier_score(y_test, approx_probs),
            }

            if has_sklearn:
                scorer.calibrate_scores(X_cal, y_cal)
                cal_probs = scorer.predict_proba(X_test)
                row["auroc_cal"] = _roc_auc(y_test, cal_probs)
                row["brier_cal"] = _brier_score(y_test, cal_probs)

            records.append(row)

    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Uni-ACS clinical scoring ablation study."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to MIMIC-IV sepsis CSV. Uses synthetic data if omitted.",
    )
    args = parser.parse_args()

    if args.csv is not None:
        from sklearn.model_selection import train_test_split
        from uni_acs_utils import (
            aggregate_shap,
            impute_data,
            load_mimic_data,
            train_calibrated_rf,
        )

        X_train_raw, X_test_raw, y_train, y_test = load_mimic_data(args.csv)
        X_score_raw, X_cal_raw, y_score, y_cal = train_test_split(
            X_train_raw, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        X_fit, X_cal, X_test = impute_data(X_score_raw, X_cal_raw, X_test_raw)
        model = train_calibrated_rf(X_fit, y_score)
        shap_fit, expected_value = aggregate_shap(model, X_fit)
        y_cal = np.asarray(y_cal)
        y_test = np.asarray(y_test)
        source_label = f"MIMIC-IV CSV ({args.csv.name})"
    else:
        X, y, shap_vals, expected_value = generate_synthetic_clinical_data()
        X_fit, X_cal, X_test = X.iloc[:900], X.iloc[900:1200], X.iloc[1200:]
        shap_fit = shap_vals[:900]
        y_cal, y_test = y[900:1200], y[1200:]
        source_label = "Synthetic ICU Mortality Cohort (n=1500)"

    print(f"Uni-ACS Ablation Study — {source_label}")
    print(f"Split sizes (fit/cal/test): {len(X_fit)} / {len(X_cal)} / {len(X_test)}")
    print(f"Test event rate: {y_test.mean():.4f}\n")

    results = run_ablation(
        X_fit, shap_fit, expected_value, X_cal, y_cal, X_test, y_test
    )
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
