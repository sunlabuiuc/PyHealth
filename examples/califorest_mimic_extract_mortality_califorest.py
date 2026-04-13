# Authors: Cesar Jesus Giglio Badoino (cesarjg2@illinois.edu)
#          Arjun Tangella (avtange2@illinois.edu)
#          Tony Nguyen (tonyln2@illinois.edu)
# Paper: CaliForest: Calibrated Random Forest for Health Data
# Paper link: https://doi.org/10.1145/3368555.3384461
# Description: Ablation study for CaliForest hyperparameters
"""MIMIC-Extract CaliForest ablation study.

This script demonstrates the full CaliForest pipeline and runs
ablation experiments on:

1. **Calibration type**: CF-Iso vs CF-Logit vs RF-NoCal
2. **Prior sensitivity**: varying alpha0/beta0
3. **Ensemble size**: 50, 100, 200, 300 trees
4. **Prediction target**: mort_hosp, mort_icu, los_3, los_7

Usage with real MIMIC-Extract data::

    python mimic_extract_califorest_ablation.py \\
        --hdf5_path /path/to/all_hourly_data.h5

Usage with synthetic data (for testing)::

    python mimic_extract_califorest_ablation.py --synthetic

Paper: Y. Park and J. C. Ho. "CaliForest: Calibrated Random Forest
for Health Data." ACM CHIL, 2020.
https://doi.org/10.1145/3368555.3384461
"""

import argparse
import numpy as np
from sklearn.metrics import roc_auc_score


def make_synthetic_data(n=500, d=50, seed=42):
    """Generate synthetic binary classification data."""
    np.random.seed(seed)
    X = np.random.randn(n, d)
    logits = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
    y = (logits > 0).astype(int)
    split = int(0.7 * n)
    return X[:split], X[split:], y[:split], y[split:]


def evaluate(y_true, y_pred):
    """Compute AUROC and all six calibration metrics."""
    from pyhealth.metrics.califorest_calibration import (
        hosmer_lemeshow,
        reliability,
        scaled_brier_score,
        spiegelhalter,
    )

    auc = roc_auc_score(y_true, y_pred)
    brier, brier_scaled = scaled_brier_score(y_true, y_pred)
    hl = hosmer_lemeshow(y_true, y_pred)
    sp = spiegelhalter(y_true, y_pred)
    rel_s, rel_l = reliability(y_true, y_pred)
    return {
        "AUROC": auc,
        "Brier": brier,
        "Scaled Brier": brier_scaled,
        "HL p-value": hl,
        "Spiegelhalter p": sp,
        "Rel-small": rel_s,
        "Rel-large": rel_l,
    }


def run_califorest(X_train, y_train, X_test, y_test, **kwargs):
    """Train CaliForest and return predictions."""
    from pyhealth.models.califorest import CaliForest

    # Use the numpy-level API directly for efficiency
    model = CaliForest.__new__(CaliForest)
    model.n_estimators = kwargs.get("n_estimators", 300)
    model.max_depth = kwargs.get("max_depth", 10)
    model.min_samples_split = kwargs.get("min_samples_split", 3)
    model.min_samples_leaf = kwargs.get("min_samples_leaf", 1)
    model.ctype = kwargs.get("ctype", "isotonic")
    model.alpha0 = kwargs.get("alpha0", 100.0)
    model.beta0 = kwargs.get("beta0", 25.0)
    model.estimators_ = []
    model.calibrator_ = None
    model.is_fitted_ = False

    model._fit_numpy(X_train, y_train)
    y_pred = model._predict_proba_numpy(X_test)
    return y_pred


def run_rf_nocal(X_train, y_train, X_test, **kwargs):
    """Train uncalibrated Random Forest baseline."""
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=kwargs.get("n_estimators", 300),
        max_depth=kwargs.get("max_depth", 10),
        min_samples_split=kwargs.get("min_samples_split", 3),
        min_samples_leaf=kwargs.get("min_samples_leaf", 1),
    )
    rf.fit(X_train, y_train)
    return rf.predict_proba(X_test)[:, 1]


def print_results(name, metrics):
    """Pretty-print a single row of results."""
    parts = [f"{name:<20}"]
    for k, v in metrics.items():
        parts.append(f"{k}={v:.4f}")
    print("  ".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_path", type=str, default=None)
    parser.add_argument("--target", type=str, default="mort_hosp")
    parser.add_argument(
        "--synthetic", action="store_true", default=False
    )
    args = parser.parse_args()

    # ---- Load data ----
    if args.synthetic or args.hdf5_path is None:
        print("Using synthetic data\n")
        X_tr, X_te, y_tr, y_te = make_synthetic_data()
    else:
        from pyhealth.datasets.califorest_mimic_extract import (
            load_califorest_data,
        )

        X_tr, X_te, y_tr, y_te = load_califorest_data(
            args.hdf5_path, target=args.target
        )

    # ---- Ablation 1: Calibration type ----
    print("=" * 60)
    print("Ablation 1: Calibration Type")
    print("=" * 60)
    for ctype in ["isotonic", "logistic", "beta"]:
        y_pred = run_califorest(
            X_tr, y_tr, X_te, y_te, ctype=ctype
        )
        print_results(f"CF-{ctype}", evaluate(y_te, y_pred))
    y_pred = run_rf_nocal(X_tr, y_tr, X_te)
    print_results("RF-NoCal", evaluate(y_te, y_pred))

    # ---- Ablation 2: Prior sensitivity ----
    print(f"\n{'=' * 60}")
    print("Ablation 2: Prior Sensitivity (alpha0, beta0)")
    print("=" * 60)
    for alpha0, beta0 in [(50, 12.5), (100, 25), (200, 50), (10, 2.5)]:
        y_pred = run_califorest(
            X_tr, y_tr, X_te, y_te,
            alpha0=alpha0, beta0=beta0,
        )
        print_results(
            f"a0={alpha0},b0={beta0}", evaluate(y_te, y_pred)
        )

    # ---- Ablation 3: Ensemble size ----
    print(f"\n{'=' * 60}")
    print("Ablation 3: Number of Estimators")
    print("=" * 60)
    for n_est in [50, 100, 200, 300]:
        y_pred = run_califorest(
            X_tr, y_tr, X_te, y_te, n_estimators=n_est
        )
        print_results(f"n_est={n_est}", evaluate(y_te, y_pred))

    # ---- Ablation 4: Prediction targets (synthetic only) ----
    if args.synthetic or args.hdf5_path is None:
        print("\n(Skipping target ablation on synthetic data)")
    else:
        print(f"\n{'=' * 60}")
        print("Ablation 4: Prediction Targets")
        print("=" * 60)
        from pyhealth.datasets.califorest_mimic_extract import (
            load_califorest_data,
        )

        for tgt in ["mort_hosp", "mort_icu", "los_3", "los_7"]:
            Xr, Xe, yr, ye = load_califorest_data(
                args.hdf5_path, target=tgt
            )
            y_pred = run_califorest(Xr, yr, Xe, ye)
            print_results(tgt, evaluate(ye, y_pred))


if __name__ == "__main__":
    main()
