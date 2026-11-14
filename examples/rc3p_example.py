"""RC3P rank-calibrated class-conditional conformal prediction on synthetic data.

This script shows the RC3P workflow with no download or real patient
data:

1) A small MLP is trained on synthetic samples from three classes.
2) RC3P is calibrated: one part of the calibration data estimates each
   class's top-k error and rank cutoff, the other part computes each
   class's score threshold at the adjusted miscoverage rate.
3) Each predictor's prediction sets are evaluated on a test set:
   per-class empirical coverage (target 90%) and average set size. The
   plain class-conditional (LABEL) predictor is included as a baseline.

Usage (from the repo root):
    python examples/rc3p_example.py
"""

import numpy as np
import torch

from pyhealth.calib.predictionset import LABEL, RC3P
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP
from pyhealth.trainer import Trainer

ALPHA = 0.1
CLASS_MEANS = np.array([0.2, 0.5, 0.8])


def make_samples(rng, n_samples, start_id):
    """Synthetic samples: the first procedure value carries the signal.

    Each class draws it from a Gaussian centred on a different mean, so
    a small MLP can learn the classes.
    """
    labels = rng.integers(0, 3, size=n_samples)
    samples = []
    for i, label in enumerate(labels):
        signal = float(np.clip(rng.normal(CLASS_MEANS[label], 0.15), 0, 1))
        samples.append(
            {
                "patient_id": f"patient-{start_id + i}",
                "visit_id": f"visit-{start_id + i}",
                # Procedures only: no categorical feature, so no
                # per-split vocabulary mapping can scramble the inputs.
                "procedures": [
                    signal,
                    float(rng.uniform()),
                    float(rng.uniform()),
                    float(rng.uniform()),
                ],
                "label": int(label),
            }
        )
    return samples


def make_dataset(samples):
    return create_sample_dataset(
        samples=samples,
        input_schema={"procedures": "tensor"},
        output_schema={"label": "multiclass"},
        dataset_name="synthetic",
    )


def evaluate_sets(predictor, test_dl, num_classes=3):
    """Per-class coverage and average set size of a predictor."""
    predictor.eval()
    y_true_all, predset_all = [], []
    with torch.no_grad():
        for batch in test_dl:
            output = predictor(**batch)
            y_true_all.append(output["y_true"].cpu().numpy())
            predset_all.append(output["y_predset"].cpu().numpy())
    y_true = np.concatenate(y_true_all)
    predset = np.concatenate(predset_all)
    coverage = [
        predset[y_true == y, y].mean() for y in range(num_classes)
    ]
    return coverage, predset.sum(axis=1).mean()


def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    train_ds = make_dataset(make_samples(rng, 600, 0))
    cal_ds = make_dataset(make_samples(rng, 400, 1000))
    test_ds = make_dataset(make_samples(rng, 400, 2000))

    model = MLP(
        dataset=train_ds,
        feature_keys=["procedures"],
        label_key="label",
        mode="multiclass",
    )
    train_dl = get_dataloader(train_ds, batch_size=64, shuffle=True)
    Trainer(model=model, enable_logging=False).train(train_dl, epochs=8)

    test_dl = get_dataloader(test_ds, batch_size=64, shuffle=False)

    # Class-conditional baseline: one threshold per class.
    label = LABEL(model, np.array([ALPHA, ALPHA, ALPHA]))
    label.calibrate(cal_dataset=cal_ds)

    # RC3P: rank calibration plus adjusted per-class thresholds. The
    # calibration set is split internally, stratified by class, into
    # disjoint rank and score subsets.
    rc3p = RC3P(model, alpha=ALPHA, random_state=0)
    rc3p.calibrate(cal_dataset=cal_ds)

    print(f"{'predictor':<28s} {'cov c0':>7s} {'cov c1':>7s} {'cov c2':>7s} {'avg size':>8s}")
    for name, predictor in (
        ("Class-conditional (LABEL)", label),
        ("RC3P", rc3p),
    ):
        coverage, avg_size = evaluate_sets(predictor, test_dl)
        cov_str = " ".join(f"{c:>7.3f}" for c in coverage)
        print(f"{name:<28s} {cov_str} {avg_size:>8.2f}")
    print(f"\nTarget per-class coverage: {1 - ALPHA:.2f}")
    print(f"RC3P rank cutoffs: {rc3p.k_hat}")


if __name__ == "__main__":
    main()
