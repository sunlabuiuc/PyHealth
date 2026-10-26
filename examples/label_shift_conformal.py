"""Label-shift conformal prediction (LabelShift) on synthetic data.

This script shows the whole label-shift workflow with no download or
real patient data:

1) A small MLP is trained on synthetic samples whose labels follow a
   source mix.
2) The test set follows a different, shifted mix, so plain split
   conformal prediction undercovers.
3) LabelShift is calibrated three ways: with both true mixes
   (oracle, so the label ratios are fully known), with BBSE, and with
   EM/MLLS, where BBSE and EM estimate the target mix from the model's
   predictions alone, without test labels.
4) Each predictor's prediction sets are evaluated on the shifted test
   set: empirical coverage (target 90%) and average set size. The
   class-conditional (LABEL) predictor is included as a baseline that
   needs no target-mix estimate at all, since class-conditional
   coverage is preserved under label shift; the tradeoff is that it
   can need more calibration data per class and may give larger sets.

Usage (from the repo root):
    python examples/label_shift_conformal.py
"""

import numpy as np
import torch

from pyhealth.calib.predictionset import (
    LABEL,
    BaseConformal,
    LabelShift,
    bbse,
    em,
)
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MLP
from pyhealth.trainer import Trainer

ALPHA = 0.1
SOURCE_MIX = np.array([0.6, 0.3, 0.1])
TARGET_MIX = np.array([0.2, 0.3, 0.5])
CLASS_MEANS = np.array([0.2, 0.5, 0.8])


def make_samples(rng, n_samples, mix, start_id):
    """Synthetic samples: the first procedure value carries the signal.

    Each class draws it from a Gaussian centred on a different mean, so
    a model trained on the source mix can learn the classes, while the
    label proportions still shift between train/calibration and test.
    """
    labels = rng.choice(3, size=n_samples, p=mix)
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


def get_probs(model, dataloader):
    """True labels and predicted probabilities for a dataset."""
    model.eval()
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for batch in dataloader:
            output = model(**batch)
            y_true_all.append(output["y_true"].cpu().numpy())
            y_prob_all.append(output["y_prob"].cpu().numpy())
    return np.concatenate(y_true_all), np.concatenate(y_prob_all)


def evaluate_sets(predictor, test_dl):
    """Coverage and average set size of a calibrated predictor."""
    predictor.eval()
    y_true_all, predset_all = [], []
    with torch.no_grad():
        for batch in test_dl:
            output = predictor(**batch)
            y_true_all.append(output["y_true"].cpu().numpy())
            predset_all.append(output["y_predset"].cpu().numpy())
    y_true = np.concatenate(y_true_all)
    predset = np.concatenate(predset_all)
    coverage = predset[np.arange(len(y_true)), y_true].mean()
    return coverage, predset.sum(axis=1).mean()


def main():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    train_ds = make_dataset(make_samples(rng, 600, SOURCE_MIX, 0))
    cal_ds = make_dataset(make_samples(rng, 400, SOURCE_MIX, 1000))
    test_ds = make_dataset(make_samples(rng, 400, TARGET_MIX, 2000))

    model = MLP(
        dataset=train_ds,
        feature_keys=["procedures"],
        label_key="label",
        mode="multiclass",
    )
    train_dl = get_dataloader(train_ds, batch_size=64, shuffle=True)
    Trainer(model=model, enable_logging=False).train(train_dl, epochs=8)

    cal_dl = get_dataloader(cal_ds, batch_size=64, shuffle=False)
    test_dl = get_dataloader(test_ds, batch_size=64, shuffle=False)
    cal_true, cal_prob = get_probs(model, cal_dl)
    _, test_prob = get_probs(model, test_dl)

    bbse_mix = bbse(cal_true, cal_prob.argmax(axis=1), test_prob.argmax(axis=1))
    em_mix = em(cal_true, test_prob)
    print(f"source mix: {SOURCE_MIX}")
    print(f"target mix: {TARGET_MIX}")
    print(f"BBSE estimate: {np.round(bbse_mix, 3)}")
    print(f"EM estimate:   {np.round(em_mix, 3)}")
    print()

    predictors = []
    plain = BaseConformal(model, alpha=ALPHA)
    plain.calibrate(cal_dataset=cal_ds)
    predictors.append(("Plain BaseConformal", plain))
    # Class-conditional baseline: one threshold per class, no
    # target-mix estimate needed.
    label = LABEL(model, np.array([ALPHA, ALPHA, ALPHA]))
    label.calibrate(cal_dataset=cal_ds)
    predictors.append(("Class-conditional (LABEL)", label))
    oracle = LabelShift(model, alpha=ALPHA)
    oracle.calibrate(
        cal_dataset=cal_ds,
        target_label_dist=TARGET_MIX,
        source_label_dist=SOURCE_MIX,
    )
    predictors.append(("LabelShift (oracle)", oracle))
    for name, target_dist in (
        ("LabelShift (BBSE)", bbse_mix),
        ("LabelShift (EM)", em_mix),
    ):
        predictor = LabelShift(model, alpha=ALPHA)
        predictor.calibrate(
            cal_dataset=cal_ds, target_label_dist=target_dist
        )
        predictors.append((name, predictor))

    print(f"{'predictor':<24s} {'coverage':>8s} {'avg set size':>12s}")
    for name, predictor in predictors:
        coverage, avg_size = evaluate_sets(predictor, test_dl)
        print(f"{name:<24s} {coverage:>8.3f} {avg_size:>12.2f}")
    print(f"\nTarget coverage: {1 - ALPHA:.2f}")


if __name__ == "__main__":
    main()
