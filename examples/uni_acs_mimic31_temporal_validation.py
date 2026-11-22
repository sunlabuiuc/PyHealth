"""Temporal validation extension for Uni-ACS on MIMIC-IV 3.1 (2008-2019 vs 2020-2022).

Implements the temporal generalizability extension proposed for CSE6250:
    - Fits the predictive model, SHAP attributions, and Uni-ACS scoring table
      on the historical MIMIC-IV cohort (``anchor_year_group`` in 2008-2019).
    - Evaluates both in-period holdout performance (2008-2019 test split) and
      prospective temporal generalization on newer admissions (2020-2022).
    - Compares ``top_k`` feature-set sizes (5, 10, 15) and ``"quantile"`` vs
      ``"novel"`` discretization across both periods.

Usage:
    Run with synthetic temporal-shift data (no MIMIC-IV files required):
        python examples/uni_acs_mimic31_temporal_validation.py

    Run on local MIMIC-IV 3.1 tables (requires ``scikit-learn`` and ``shap``):
        python examples/uni_acs_mimic31_temporal_validation.py \\
            --mimic-dir ~/Desktop/mimiciv-3.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from pyhealth.interpret.uni_acs import UniACS

HISTORICAL_GROUPS = {
    "2008 - 2010",
    "2011 - 2013",
    "2014 - 2016",
    "2017 - 2019",
}
TEMPORAL_GROUPS = {"2020 - 2022"}

# Standard MIMIC-IV d_labitems itemids for first-24h ICU mortality features
MIMIC31_LAB_ITEMIDS = {
    50813: "lactate",
    50912: "creatinine",
    50885: "bilirubin",
    51006: "bun",
    51301: "wbc",
    51265: "platelets",
    50983: "sodium",
    50971: "potassium",
    50931: "glucose",
    51221: "hematocrit",
    50882: "bicarbonate",
    50868: "aniongap",
    50902: "chloride",
}


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diffs = pos[:, None] - neg[None, :]
    return float((np.sum(diffs > 0) + 0.5 * np.sum(diffs == 0)) / (len(pos) * len(neg)))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    ranks = np.arange(1, len(y_sorted) + 1)
    return float(np.sum((tp / ranks)[y_sorted == 1]) / n_pos)


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def load_mimic31_temporal_cohort(
    mimic_dir: Path,
    max_lab_chunks: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Build historical (2008-2019) and temporal (2020-2022) ICU cohorts.

    Extracts demographics and first-24-hour ICU lab summaries from raw
    MIMIC-IV 3.1 ``hosp/`` and ``icu/`` tables.
    """
    patients = pd.read_csv(
        mimic_dir / "hosp" / "patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year_group"],
    )
    admissions = pd.read_csv(
        mimic_dir / "hosp" / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime", "hospital_expire_flag"],
    )
    icustays = pd.read_csv(
        mimic_dir / "icu" / "icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "los"],
    )

    # Keep first ICU stay per hospital admission
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    first_icu = (
        icustays.sort_values("intime")
        .drop_duplicates(subset=["hadm_id"], keep="first")
    )

    cohort = first_icu.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    cohort = cohort.merge(patients, on="subject_id", how="inner")
    cohort["gender_m"] = (cohort["gender"] == "M").astype(float)

    hadm_intime = cohort.set_index("hadm_id")["intime"]
    valid_hadm = set(hadm_intime.index)
    target_itemids = set(MIMIC31_LAB_ITEMIDS.keys())

    # Stream labevents.csv.gz in chunks and keep first-24h measurements
    lab_chunks = []
    reader = pd.read_csv(
        mimic_dir / "hosp" / "labevents.csv.gz",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        chunksize=500_000,
    )
    for idx, chunk in enumerate(reader):
        if max_lab_chunks is not None and idx >= max_lab_chunks:
            break
        chunk = chunk[
            chunk["itemid"].isin(target_itemids)
            & chunk["hadm_id"].isin(valid_hadm)
            & chunk["valuenum"].notna()
        ]
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk["charttime"])
        intime_aligned = chunk["hadm_id"].map(hadm_intime)
        hours = (chunk["charttime"] - intime_aligned).dt.total_seconds() / 3600.0
        chunk = chunk[(hours >= -6.0) & (hours <= 24.0)]
        if not chunk.empty:
            lab_chunks.append(chunk[["hadm_id", "itemid", "valuenum"]])

    if lab_chunks:
        labs = pd.concat(lab_chunks, ignore_index=True)
        labs["feature"] = labs["itemid"].map(MIMIC31_LAB_ITEMIDS)
        lab_pivot = (
            labs.groupby(["hadm_id", "feature"])["valuenum"]
            .mean()
            .unstack("feature")
            .reset_index()
        )
        cohort = cohort.merge(lab_pivot, on="hadm_id", how="left")

    feature_cols = ["anchor_age", "gender_m", "los"] + [
        col for col in MIMIC31_LAB_ITEMIDS.values() if col in cohort.columns
    ]

    hist_mask = cohort["anchor_year_group"].isin(HISTORICAL_GROUPS)
    temp_mask = cohort["anchor_year_group"].isin(TEMPORAL_GROUPS)

    X_hist = cohort.loc[hist_mask, feature_cols].reset_index(drop=True)
    y_hist = cohort.loc[hist_mask, "hospital_expire_flag"].to_numpy(dtype=int)
    X_temp = cohort.loc[temp_mask, feature_cols].reset_index(drop=True)
    y_temp = cohort.loc[temp_mask, "hospital_expire_flag"].to_numpy(dtype=int)

    return X_hist, y_hist, X_temp, y_temp


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

            rows.append(
                {
                    "method": method,
                    "top_k": k_eff,
                    "categories": len(scorer.get_score_table()),
                    "inperiod_auroc": _roc_auc(y_inperiod_test, in_scores),
                    "temporal_auroc": _roc_auc(y_temporal_test, tp_scores),
                    "auroc_delta": (
                        _roc_auc(y_temporal_test, tp_scores)
                        - _roc_auc(y_inperiod_test, in_scores)
                    ),
                    "inperiod_brier": _brier(y_inperiod_test, in_probs),
                    "temporal_brier": _brier(y_temporal_test, tp_probs),
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
    args = parser.parse_args()

    if args.mimic_dir is not None:
        from sklearn.model_selection import train_test_split
        from uni_acs_utils import aggregate_shap, impute_data, train_calibrated_rf

        X_hist_raw, y_hist, X_temp_raw, y_temp = load_mimic31_temporal_cohort(
            args.mimic_dir
        )
        X_fit_raw, X_in_raw, y_fit, y_in = train_test_split(
            X_hist_raw, y_hist, test_size=0.3, stratify=y_hist, random_state=42
        )
        X_fit, X_in, X_temp = impute_data(X_fit_raw, X_in_raw, X_temp_raw)
        model = train_calibrated_rf(X_fit, y_fit)
        shap_fit, expected_value = aggregate_shap(model, X_fit)
        label = f"MIMIC-IV 3.1 ({args.mimic_dir})"
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
    print(
        f"Sizes (fit / 2008-2019 test / 2020-2022 test): "
        f"{len(X_fit)} / {len(X_in)} / {len(X_temp)}"
    )
    print(
        f"Mortality rates (in-period / temporal): "
        f"{y_in.mean():.4f} / {y_temp.mean():.4f}\n"
    )

    df = evaluate_temporal_split(
        X_fit, shap_fit, expected_value, X_in, y_in, X_temp, y_temp
    )
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
