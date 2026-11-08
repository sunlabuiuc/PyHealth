"""Calibration metrics and patient-clustered bootstrap confidence intervals.

Reports what a clinical prediction paper needs for each model: discrimination
(PR-AUC, ROC-AUC) and calibration (Brier score, observed/expected ratio,
calibration slope and intercept), each with a 95% interval from a bootstrap
that resamples whole patients, plus the paired difference between two models.

Uses synthetic predictions (several samples per patient); no model training
or download needed. Runs in a few seconds.

Usage:
    python examples/calibration_and_bootstrap_ci.py
"""

import numpy as np

from pyhealth.metrics import binary_metrics_fn, bootstrap_ci, paired_bootstrap_diff

METRICS = ["pr_auc", "roc_auc", "brier", "oe_ratio", "calibration_slope", "calibration_intercept"]


def synthetic_predictions(n_patients=300, seed=0):
    rng = np.random.default_rng(seed)
    visits = rng.integers(1, 6, size=n_patients)  # 1-5 samples per patient
    patient_ids = np.repeat(np.arange(n_patients), visits)
    risk = np.repeat(rng.beta(1, 6, size=n_patients), visits)  # rare outcome
    y_true = (rng.uniform(size=risk.size) < risk).astype(int)
    calibrated = np.clip(risk + rng.normal(0, 0.03, risk.size), 0.001, 0.999)
    logit = np.log(calibrated / (1 - calibrated))
    overconfident = 1 / (1 + np.exp(-1.8 * logit))  # same ranking, too extreme
    return y_true, patient_ids, calibrated, overconfident


def main():
    y_true, patient_ids, model_a, model_b = synthetic_predictions()
    print(f"{y_true.size} samples from {np.unique(patient_ids).size} patients, "
          f"{y_true.mean():.1%} positive\n")

    for name, y_prob in (("model A", model_a), ("model B (A made overconfident)", model_b)):
        print(name)
        point = binary_metrics_fn(y_true, y_prob, metrics=METRICS)
        for metric in METRICS:
            ci = bootstrap_ci(y_true, y_prob, metric, groups=patient_ids, n_boot=500)
            print(f"  {metric:22s} {point[metric]:6.3f}  [{ci['lower']:.3f}, {ci['upper']:.3f}]")
        print()

    for metric in ("roc_auc", "brier"):
        diff = paired_bootstrap_diff(y_true, model_a, model_b, metric, groups=patient_ids, n_boot=500)
        print(f"A - B {metric:8s} {diff['estimate']:+.3f}  [{diff['lower']:+.3f}, {diff['upper']:+.3f}]")


if __name__ == "__main__":
    main()
