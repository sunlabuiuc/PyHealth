"""
EMDOT Temporal Evaluation for In-Hospital Mortality on MIMIC-IV

Reproduces the EMDOT framework from Zhou et al. (CHIL 2023) to show that
random train/test splits overestimate performance vs temporal splits.

We compare two training regimes across deployment years:
  - All-historical: train on everything up to year t, test on the rest
  - Sliding window: train on only the last `window` years before t

We also ablate over window sizes (1, 2, 3, 5 years) to see how the
recency vs sample-size tradeoff plays out. All experiments use logistic
regression with bag-of-codes features.

Run:
  python examples/mortality_prediction/mortality_mimic4_temporal_emdot.py
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

from pyhealth.datasets import MIMIC4Dataset, split_by_sample
from pyhealth.tasks import InHospitalMortalityTemporalMIMIC4


def temporal_split(samples, deployment_year, regime="all_historical", window=3):
    # split into in-period (<=t) and out-period (>t)
    in_period = [s for s in samples if s["admission_year"] <= deployment_year]
    out_period = [s for s in samples if s["admission_year"] > deployment_year]

    if regime == "all_historical":
        train = in_period
    elif regime == "sliding_window":
        train = [s for s in in_period
                 if s["admission_year"] >= deployment_year - window]
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return train, out_period


def encode_features(train_samples, test_samples):
    # concat all codes into one list per sample, then binarize
    def get_codes(s):
        return s.get("conditions", []) + s.get("procedures", []) + s.get("drugs", [])

    mlb = MultiLabelBinarizer(sparse_output=False)
    X_train = mlb.fit_transform([get_codes(s) for s in train_samples])
    X_test = mlb.transform([get_codes(s) for s in test_samples])
    y_train = np.array([s["mortality"] for s in train_samples])
    y_test = np.array([s["mortality"] for s in test_samples])
    return X_train, X_test, y_train, y_test


def run_emdot(samples, regime, window=3, min_train=50,
              deployment_years=range(2012, 2020), seed=42):
    # loop over deployment years, train LR, get AUROC
    results = {}
    for t in deployment_years:
        train_samp, test_samp = temporal_split(samples, t, regime=regime, window=window)

        if len(train_samp) < min_train or len(test_samp) == 0:
            print(f"  t={t}: skipping (train={len(train_samp)}, test={len(test_samp)})")
            continue
        if len(set(s["mortality"] for s in train_samp)) < 2:
            print(f"  t={t}: skipping (only one class)")
            continue

        X_train, X_test, y_train, y_test = encode_features(train_samp, test_samp)
        lr = LogisticRegression(max_iter=1000, solver="lbfgs",
                                class_weight="balanced", random_state=seed)
        lr.fit(X_train, y_train)
        auroc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
        results[t] = auroc
        print(f"  t={t} | train={len(train_samp):5d} | test={len(test_samp):5d} | "
              f"pos_rate={y_test.mean():.3f} | AUROC={auroc:.4f}")
    return results


if __name__ == "__main__":
    t0 = time.perf_counter()

    # load dataset
    print("=" * 60)
    print("Step 1: Loading MIMIC-IV dataset")
    print("=" * 60)
    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir="../benchmark_cache/mimic4_temporal/",
    )
    base_dataset.stats()

    # apply task
    print("\n" + "=" * 60)
    print("Step 2: Applying temporal mortality task")
    print("=" * 60)
    task = InHospitalMortalityTemporalMIMIC4()
    sample_dataset = base_dataset.set_task(task, num_workers=4)
    all_samples = list(sample_dataset)
    years = [s["admission_year"] for s in all_samples]
    print(f"Total samples: {len(all_samples)}")
    print(f"Year range: {min(years)}-{max(years)}")
    print(f"Mortality rate: {np.mean([s['mortality'] for s in all_samples]):.3f}")

    # random split baseline
    print("\n" + "=" * 60)
    print("Step 3: Random split baseline (time-agnostic)")
    print("=" * 60)
    train_ds, _, test_ds = split_by_sample(
        sample_dataset, ratios=[0.7, 0.1, 0.2], seed=42
    )
    train_rand = [sample_dataset[i] for i in train_ds.indices]
    test_rand = [sample_dataset[i] for i in test_ds.indices]
    X_tr, X_te, y_tr, y_te = encode_features(train_rand, test_rand)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs",
                            class_weight="balanced", random_state=42)
    lr.fit(X_tr, y_tr)
    baseline_auroc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
    print(f"Random split AUROC: {baseline_auroc:.4f}")

    # all-historical
    print("\n" + "=" * 60)
    print("Step 4: All-historical regime")
    print("=" * 60)
    results_ah = run_emdot(all_samples, regime="all_historical")

    # sliding window (default w=3)
    print("\n" + "=" * 60)
    print("Step 5: Sliding window regime (w=3)")
    print("=" * 60)
    results_sw = run_emdot(all_samples, regime="sliding_window")

    # window size ablation
    # trying different window sizes to see how it affects things
    print("\n" + "=" * 60)
    print("Step 6: Window size ablation")
    print("=" * 60)
    ablation = {}
    for w in [1, 2, 3, 5]:
        print(f"\n  --- w={w} ---")
        ablation[w] = run_emdot(all_samples, regime="sliding_window", window=w)

    # print results
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nRandom split AUROC: {baseline_auroc:.4f}")
    print(f"\nBy deployment year:")
    print("Year\tAll-Hist\tSliding(w=3)")
    for t in range(2012, 2020):
        ah = results_ah.get(t, float('nan'))
        sw = results_sw.get(t, float('nan'))
        print(f"{t}\t{ah:.4f}\t\t{sw:.4f}")

    print(f"\nWindow ablation (mean AUROC):")
    for w, res in ablation.items():
        if res:
            print(f"  w={w}: {np.mean(list(res.values())):.4f}")

    print(f"\nDone in {time.perf_counter() - t0:.1f}s")
