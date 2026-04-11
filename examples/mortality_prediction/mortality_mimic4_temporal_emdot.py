"""
EMDOT Temporal Evaluation for In-Hospital Mortality on MIMIC-IV.

Reproduces the temporal evaluation framework from:
    Zhou, H.; Chen, Y.; and Lipton, Z. C. 2023. "Evaluating Model Performance
    in Medical Datasets Over Time." CHIL 2023. PMLR 209:498-508.

This script demonstrates that standard random train-test splits overestimate
real-world model performance by ignoring temporal distribution shift.

Two EMDOT training regimes are evaluated across simulated deployment years:
    - All-historical: train on all in-period data up to deployment year t
      (the simulated clinical deployment date), test on all data after t.
    - Sliding window: train only on the most recent `window` years before t,
      test on all data after t.

This example shows:
1. Loading MIMIC-IV data with chronological admission year tagging.
2. Applying the InHospitalMortalityTemporalMIMIC4 task to create temporal samples.
3. Establishing a baseline with simple random splits for comparison.
4. Evaluating logistic regression under the EMDOT all-historical specification.
5. Evaluating logistic regression under the EMDOT sliding window specification.
6. Comparing performance across temporal splits to highlight distribution shift effects.

Ablation study:
    - Regime comparison: all-historical vs sliding-window across deployment years
      using logistic regression as the classifier.
    - Window size ablation: sliding window with w = 1, 2, 3, 5 years to test
      the trade-off between recency (smaller window) and sample size (larger window).

Expected findings (from the EMDOT paper):
    - Random splits overestimate AUROC compared to temporal evaluation.
    - Performance degrades for later deployment years as distribution shift
      accumulates.
    - Sliding window can outperform all-historical when recent data is more
      representative of the test distribution, but undersized windows hurt
      due to insufficient training data.

Usage:
    python examples/mortality_prediction/mortality_mimic4_temporal_emdot.py
"""

import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

from pyhealth.datasets import MIMIC4Dataset, split_by_sample
from pyhealth.tasks import InHospitalMortalityTemporalMIMIC4


def temporal_split(
    samples: List[Dict],
    deployment_year: int,
    regime: str = "all_historical",
    window: int = 3,
) -> Tuple[List[Dict], List[Dict]]:
    """Splits samples into train and test sets using EMDOT regimes.

    Args:
        samples: flat list of dicts, each with an 'admission_year' field.
        deployment_year: simulated deployment date t which is the year at which
            the model would be released into a real clinical environment.
            Samples at or before this year form the in-period; samples
            after form the out-period.
        regime: 'all_historical' uses all in-period data for training.
            'sliding_window' restricts training to the most recent
            window years of the in-period.
        window: number of years to include in the sliding window.

    Returns:
        train_samples: list of samples used for training.
        test_samples: list of out-period samples used for evaluation.
    """
    in_period = [s for s in samples if s["admission_year"] <= deployment_year]
    out_period = [s for s in samples if s["admission_year"] > deployment_year]

    if regime == "all_historical":
        train_samples = in_period
    elif regime == "sliding_window":
        train_samples = [
            s
            for s in in_period
            if s["admission_year"] >= deployment_year - window
        ]
    else:
        raise ValueError(
            f"Unknown regime: {regime}. "
            "Use 'all_historical' or 'sliding_window'."
        )

    return train_samples, out_period


def encode_features(
    train_samples: List[Dict],
    test_samples: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encodes conditions, procedures, and drugs as bag-of-codes features.

    Fits a MultiLabelBinarizer on training data only to avoid leakage,
    then transforms both train and test sets.

    Args:
        train_samples: list of training sample dicts.
        test_samples: list of test sample dicts.

    Returns:
        X_train, X_test: feature matrices.
        y_train, y_test: binary mortality labels.
    """

    def get_codes(sample: Dict) -> List[str]:
        return (
            sample.get("conditions", [])
            + sample.get("procedures", [])
            + sample.get("drugs", [])
        )

    mlb = MultiLabelBinarizer(sparse_output=False)
    X_train = mlb.fit_transform([get_codes(s) for s in train_samples])
    X_test = mlb.transform([get_codes(s) for s in test_samples])

    y_train = np.array([s["mortality"] for s in train_samples])
    y_test = np.array([s["mortality"] for s in test_samples])

    return X_train, X_test, y_train, y_test


def run_emdot_evaluation(
    samples: List[Dict],
    regime: str,
    window: int = 3,
    min_train: int = 50,
    deployment_years: range = range(2012, 2020),
    seed: int = 42,
) -> Dict[int, float]:
    """Runs the EMDOT evaluation loop over all deployment years.

    For each deployment year t, trains a logistic regression model under the
    specified regime and evaluates AUROC on out-period data.

    Args:
        samples: full list of task samples with 'admission_year' field.
        regime: 'all_historical' or 'sliding_window'.
        window: sliding window size in years.
        min_train: minimum number of training samples to proceed.
        deployment_years: range of deployment years to evaluate.
        seed: random seed for reproducibility.

    Returns:
        dict mapping deployment_year -> AUROC on out-period data.
    """
    results = {}

    for t in deployment_years:
        train_samples, test_samples = temporal_split(
            samples, deployment_year=t, regime=regime, window=window
        )

        if len(train_samples) < min_train or len(test_samples) == 0:
            print(
                f"  t={t}: skipping "
                f"(train={len(train_samples)}, test={len(test_samples)})"
            )
            continue

        y_train_check = [s["mortality"] for s in train_samples]
        if len(set(y_train_check)) < 2:
            print(f"  t={t}: skipping (only one class in training set)")
            continue

        X_train, X_test, y_train, y_test = encode_features(
            train_samples, test_samples
        )

        model = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_prob)
        results[t] = auroc

        print(
            f"  t={t} | train={len(train_samples):5d} | "
            f"test={len(test_samples):5d} | "
            f"pos_rate={y_test.mean():.3f} | AUROC={auroc:.4f}"
        )

    return results


def run_window_ablation(
    samples: List[Dict],
    window_sizes: List[int],
    deployment_years: range = range(2012, 2020),
) -> Dict[int, Dict[int, float]]:
    """Ablation study over sliding window sizes.

    Tests how the choice of window size affects temporal performance,
    trading off recency (smaller window) against sample size (larger window).

    Args:
        samples: full list of task samples.
        window_sizes: list of window sizes in years to evaluate.
        deployment_years: range of deployment years to evaluate.

    Returns:
        dict mapping window_size -> {deployment_year -> AUROC}.
    """
    ablation_results = {}
    for w in window_sizes:
        print(f"\n  --- window={w} years ---")
        ablation_results[w] = run_emdot_evaluation(
            samples,
            regime="sliding_window",
            window=w,
            deployment_years=deployment_years,
        )
    return ablation_results


if __name__ == "__main__":
    t0 = time.perf_counter()

    # ── STEP 1: Load MIMIC-IV base dataset ────────────────────────
    print("=" * 60)
    print("STEP 1: Loading MIMIC-IV Dataset")
    print("=" * 60)
    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir="../benchmark_cache/mimic4_temporal/",
    )
    base_dataset.stats()

    # ── STEP 2: Apply temporal mortality task ─────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Applying InHospitalMortalityTemporalMIMIC4 Task")
    print("=" * 60)
    task = InHospitalMortalityTemporalMIMIC4()
    sample_dataset = base_dataset.set_task(task, num_workers=4)

    # Collect raw samples for sklearn-based evaluation
    all_samples = list(sample_dataset)
    years = [s["admission_year"] for s in all_samples]
    print(f"Total samples: {len(all_samples)}")
    print(f"Admission years: {min(years)} - {max(years)}")
    print(
        f"Mortality rate: "
        f"{np.mean([s['mortality'] for s in all_samples]):.3f}"
    )

    # ── STEP 3: Baseline — standard random split (time-agnostic) ──
    print("\n" + "=" * 60)
    print("STEP 3: Baseline -- Standard Random Split (Time-Agnostic)")
    print("=" * 60)
    train_dataset, _, test_dataset = split_by_sample(
        sample_dataset, ratios=[0.7, 0.1, 0.2], seed=42
    )
    train_samples_rand = [sample_dataset[i] for i in train_dataset.indices]
    test_samples_rand = [sample_dataset[i] for i in test_dataset.indices]

    X_train_r, X_test_r, y_train_r, y_test_r = encode_features(
        train_samples_rand, test_samples_rand
    )
    lr_baseline = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    lr_baseline.fit(X_train_r, y_train_r)
    baseline_auroc = roc_auc_score(
        y_test_r, lr_baseline.predict_proba(X_test_r)[:, 1]
    )
    print(f"Random split AUROC (optimistic estimate): {baseline_auroc:.4f}")

    # ── STEP 4: EMDOT all-historical regime ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: EMDOT -- All-Historical Regime (Logistic Regression)")
    print("=" * 60)
    results_ah = run_emdot_evaluation(
        all_samples, regime="all_historical"
    )

    # ── STEP 5: EMDOT sliding window regime ───────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: EMDOT -- Sliding Window Regime (window=3 years)")
    print("=" * 60)
    results_sw = run_emdot_evaluation(
        all_samples, regime="sliding_window"
    )

    # ── STEP 6: Ablation — vary window size ───────────────────────
    # This is the core ablation: how does window size affect temporal
    # performance? Smaller windows use more recent (relevant) data but
    # have fewer samples; larger windows have more data but include
    # older (potentially shifted) distributions.
    print("\n" + "=" * 60)
    print("STEP 6: Ablation -- Sliding Window Size")
    print("=" * 60)
    ablation = run_window_ablation(
        all_samples, window_sizes=[1, 2, 3, 5]
    )

    # ── STEP 7: Summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: Summary")
    print("=" * 60)
    print(f"\nRandom split AUROC (time-agnostic):   {baseline_auroc:.4f}")
    print(f"\nAUROC by deployment year (Logistic Regression):")
    print(f"{'Year':<8} {'All-Historical':<18} {'Sliding Window':<18}")
    for t in range(2012, 2020):
        print(
            f"{t:<8} "
            f"{results_ah.get(t, float('nan')):<18.4f} "
            f"{results_sw.get(t, float('nan')):<18.4f}"
        )

    print("\nWindow size ablation (mean AUROC across deployment years):")
    for w, res in ablation.items():
        if res:
            print(
                f"  window={w} years: "
                f"mean AUROC = {np.mean(list(res.values())):.4f}"
            )

    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")
    print("=" * 60)
