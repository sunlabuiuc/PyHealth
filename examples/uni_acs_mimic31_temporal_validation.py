"""Evaluate Uni-ACS across MIMIC-IV 3.1 sepsis admission eras.

Run without arguments for a synthetic demonstration, or pass
--mimic-dir /path/to/mimiciv-3.1 for the local ICD-defined sepsis experiment.
The real-data mode uses the derived ICD-defined sepsis cohort and lab features.
Add --audit-eras to print cohort counts and exit before reading labs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from pyhealth.interpret.uni_acs import UniACS


def assign_admission_eras(cohort: pd.DataFrame) -> pd.DataFrame:
    """Report era inclusion and exclusion counts and return eligible admissions."""
    if cohort["subject_id"].duplicated().any():
        raise ValueError("Expected one sepsis admission per patient.")

    # Anchor-year bounds are in real years. Shift both bounds by the
    # admission's year offset and retain only unambiguous era assignments.
    anchor_bounds = cohort["anchor_year_group"].str.extract(r"^(\d{4}) - (\d{4})$")
    year_offset = cohort["admittime"].dt.year - cohort["anchor_year"]
    first_year = anchor_bounds[0].astype(float) + year_offset
    last_year = anchor_bounds[1].astype(float) + year_offset
    valid_years = first_year.notna() & last_year.notna() & first_year.le(last_year)
    historical = valid_years & first_year.ge(2008) & last_year.le(2019)
    recent = valid_years & first_year.ge(2020) & last_year.le(2022)
    keep = historical | recent

    outside = valid_years & (first_year.lt(2008) | last_year.gt(2022))
    crosses_boundary = (
        valid_years & ~outside & first_year.lt(2020) & last_year.ge(2020)
    )
    counts = {
        "Historical (2008-2019)": int(historical.sum()),
        "Recent (2020-2022)": int(recent.sum()),
        "Estimated year range extends outside 2008-2022": int(outside.sum()),
        "Estimated year range crosses the 2020 boundary": int(crosses_boundary.sum()),
        "Missing or invalid year range": int((~valid_years).sum()),
    }
    if sum(counts.values()) != len(cohort):
        raise ValueError("Era audit did not account for every admission.")

    print("\nAdmission era audit")
    for label, count in counts.items():
        print(f"  {label}: {count:,}")
    print(f"  Total excluded: {(~keep).sum():,}")
    return cohort.loc[keep].assign(recent=recent.loc[keep])


def load_mimic31_temporal_cohort(mimic_dir: Path):
    """Collect labs available by 24 hours for the ICD-defined sepsis cohort."""
    from uni_acs_mimic31_features import LAB_NAMES, extract_labs, get_lab_items, load_cohort

    cohort = assign_admission_eras(load_cohort(mimic_dir / "derived"))
    labs = extract_labs(
        cohort,
        get_lab_items(mimic_dir),
        data_path=mimic_dir,
        require_available=True,
    )
    data = cohort.merge(labs, on="hadm_id", how="left", validate="one_to_one")
    data["gender_male"] = data["gender"].eq("M").astype(int)
    features = ["age", "gender_male", *LAB_NAMES.values()]
    X = data.reindex(columns=features)
    y = data["hospital_mortality"].to_numpy(dtype=int)
    recent = data["recent"].to_numpy()
    return X.loc[~recent], y[~recent], X.loc[recent], y[recent]


def generate_synthetic_temporal_cohorts(
    n_hist: int = 1200,
    n_temp: int = 400,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, float]:
    """Simulate 2008-2019 historical vs 2020-2022 shifted clinical cohorts."""
    rng = np.random.default_rng(seed)
    cols = [
        "anchor_age",
        "gender_m",
        "los",
        "lactate",
        "creatinine",
        "bilirubin",
        "bun",
        "wbc",
        "platelets",
        "sodium",
        "potassium",
        "glucose",
        "hematocrit",
        "bicarbonate",
        "aniongap",
    ]
    weights = np.array(
        [
            0.75, 0.15, 0.45, 0.90, 0.60,
            0.50, 0.40, 0.35, -0.35, 0.15,
            0.15, 0.10, -0.10, -0.25, 0.30,
        ]
    )

    X_hist_arr = rng.normal(0.0, 1.0, size=(n_hist, len(cols)))
    X_hist_arr[:, 1] = (X_hist_arr[:, 1] > 0).astype(float)
    contrib_hist = X_hist_arr * weights
    shap_hist = contrib_hist - contrib_hist.mean(axis=0, keepdims=True)
    base_log_odds = -1.5
    y_hist = rng.binomial(1, expit(base_log_odds + shap_hist.sum(axis=1))).astype(int)

    # Simulate mild 2020-2022 temporal shift (older patients, higher acuity)
    X_temp_arr = rng.normal(0.15, 1.05, size=(n_temp, len(cols)))
    X_temp_arr[:, 1] = (X_temp_arr[:, 1] > 0).astype(float)
    shifted_weights = weights * np.linspace(0.92, 1.08, len(cols))
    y_temp = rng.binomial(
        1, expit(base_log_odds + 0.15 + (X_temp_arr * shifted_weights).sum(axis=1))
    ).astype(int)

    return (
        pd.DataFrame(X_hist_arr, columns=cols),
        y_hist,
        shap_hist,
        pd.DataFrame(X_temp_arr, columns=cols),
        y_temp,
        base_log_odds,
    )


def evaluate_temporal_split(
    X_fit: pd.DataFrame,
    shap_fit: np.ndarray,
    expected_value: float,
    X_inperiod_test: pd.DataFrame,
    y_inperiod_test: np.ndarray,
    X_temporal_test: pd.DataFrame,
    y_temporal_test: np.ndarray,
    top_k_values: tuple[int, ...] = (5, 10, 15),
) -> pd.DataFrame:
    """Compare in-period (2008-2019) vs temporal (2020-2022) Uni-ACS metrics."""
    rows = []
    for method in ("quantile", "novel"):
        for k in top_k_values:
            k_eff = min(k, X_fit.shape[1])
            scorer = UniACS(top_k=k_eff, threshold=0.05, method=method)
            scorer.fit(X_fit, shap_fit, expected_value)

            in_scores = scorer.predict_score(X_inperiod_test)
            in_probs = scorer.predict_proba(X_inperiod_test)
            tp_scores = scorer.predict_score(X_temporal_test)
            tp_probs = scorer.predict_proba(X_temporal_test)

            in_auc = roc_auc_score(y_inperiod_test, in_scores)
            temporal_auc = roc_auc_score(y_temporal_test, tp_scores)
            rows.append(
                {
                    "method": method,
                    "top_k": k_eff,
                    "categories": len(scorer.get_score_table()),
                    "inperiod_auroc": in_auc,
                    "temporal_auroc": temporal_auc,
                    "auroc_delta": temporal_auc - in_auc,
                    "inperiod_auprc": average_precision_score(y_inperiod_test, in_scores),
                    "temporal_auprc": average_precision_score(y_temporal_test, tp_scores),
                    "inperiod_brier": brier_score_loss(y_inperiod_test, in_probs),
                    "temporal_brier": brier_score_loss(y_temporal_test, tp_probs),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Uni-ACS temporal validation (2008-2019 vs 2020-2022)."
    )
    parser.add_argument(
        "--mimic-dir",
        type=Path,
        default=None,
        help=(
            "Path to local MIMIC-IV 3.1 root directory. "
            "Uses synthetic shift if omitted."
        ),
    )
    parser.add_argument(
        "--audit-eras",
        action="store_true",
        help="Print admission era counts and exit before reading lab events.",
    )
    args = parser.parse_args()

    if args.audit_eras:
        if args.mimic_dir is None:
            parser.error("--audit-eras requires --mimic-dir.")
        from uni_acs_mimic31_features import load_cohort

        assign_admission_eras(load_cohort(args.mimic_dir / "derived"))
        return

    if args.mimic_dir is not None:
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from uni_acs_utils import aggregate_shap, evaluate, print_metrics, train_calibrated_rf

        X_hist_raw, y_hist, X_temp_raw, y_temp = load_mimic31_temporal_cohort(
            args.mimic_dir
        )
        X_fit_raw, X_in_raw, y_fit, y_in = train_test_split(
            X_hist_raw, y_hist, test_size=0.3, stratify=y_hist, random_state=42
        )
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_fit = pd.DataFrame(imputer.fit_transform(X_fit_raw), columns=X_fit_raw.columns)
        X_in = pd.DataFrame(imputer.transform(X_in_raw), columns=X_fit.columns)
        X_temp = pd.DataFrame(imputer.transform(X_temp_raw), columns=X_fit.columns)
        model = train_calibrated_rf(X_fit, y_fit)
        for period, X_test, y_test in (
            ("Historical holdout", X_in, y_in),
            ("Recent holdout", X_temp, y_temp),
        ):
            probs = model.predict_proba(X_test)[:, 1]
            print_metrics(f"Calibrated RandomForest - {period}", evaluate(y_test, probs, probs))

        # Match the random-split SHAP sample size using historical training only.
        X_fit = X_fit.sample(n=min(2000, len(X_fit)), random_state=42)
        shap_fit, expected_value = aggregate_shap(model, X_fit)
        n_train = len(X_fit_raw)
        label = "MIMIC-IV 3.1 ICD-defined sepsis (estimated admission eras)"
    else:
        X_hist, y_hist, shap_hist, X_temp, y_temp, expected_value = (
            generate_synthetic_temporal_cohorts()
        )
        split = int(0.7 * len(X_hist))
        X_fit, X_in = X_hist.iloc[:split], X_hist.iloc[split:]
        shap_fit = shap_hist[:split]
        y_in = y_hist[split:]
        label = "Synthetic Temporal Shift Cohort (2008-2019 vs 2020-2022)"

    print(f"Uni-ACS Temporal Validation — {label}")
    if args.mimic_dir is not None:
        print(f"RandomForest training admissions: {n_train}")
    print(f"Uni-ACS score-fitting admissions: {len(X_fit)}")
    print(f"Test admissions (historical / recent): {len(X_in)} / {len(X_temp)}")
    print(
        f"Mortality rates (in-period / temporal): "
        f"{y_in.mean():.4f} / {y_temp.mean():.4f}\n"
    )

    df = evaluate_temporal_split(
        X_fit, shap_fit, expected_value, X_in, y_in, X_temp, y_temp,
        top_k_values=(5, 10, 14) if args.mimic_dir else (5, 10, 15),
    )
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
