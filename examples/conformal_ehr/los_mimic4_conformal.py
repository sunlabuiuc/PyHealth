"""
Split conformal prediction for length-of-stay prediction on MIMIC-IV.

This example demonstrates:
1. Training a Transformer on the MIMIC-IV length-of-stay task (10 classes).
2. Wrapping the trained model with LABEL (split conformal prediction) to produce
   prediction sets with a user-specified coverage guarantee (1 - alpha).
3. Evaluating the prediction sets via overall coverage, average set size, and
   per-class miscoverage, averaged over multiple random seeds.

Coverage on any single calibration/test split is high-variance, so results are
averaged over several seeds and reported as mean +/- std. Per-class miscoverage is
also reported, since it exposes class imbalance that overall coverage can hide.

Usage:
    # Full dataset
    python los_mimic4_conformal.py --root /path/to/mimiciv/2.2

    # Quick smoke test on a subsampled dataset
    python los_mimic4_conformal.py --dev --epochs 1 --seeds 0
"""

from __future__ import annotations

import argparse
import logging
import random

import numpy as np
import torch

from pyhealth.calib.predictionset import LABEL
from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient_conformal,
)
from pyhealth.metrics.prediction_set import (
    miscoverage_overall_ps,
    miscoverage_ps,
    size,
)
from pyhealth.models import Transformer
from pyhealth.tasks import LengthOfStayPredictionMIMIC4
from pyhealth.trainer import Trainer

# Quiet PyHealth's per-init model summary logging.
logging.getLogger("pyhealth").setLevel(logging.WARNING)


def run_seed(samples, seed: int, alphas: list[float], epochs: int) -> dict:
    """Train the base model and run split conformal prediction for a single seed.

    Returns {alpha: (coverage, avg_set_size, per_class_miscoverage)} on the test split.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Train / validation / calibration / test split (calibration is conformal-specific).
    train_data, val_data, cal_data, test_data = split_by_patient_conformal(
        samples, ratios=[0.6, 0.1, 0.1, 0.2], seed=seed
    )
    train_loader = get_dataloader(train_data, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_data, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_data, batch_size=32, shuffle=False)

    model = Transformer(dataset=samples)
    Trainer(model=model).train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="accuracy",
    )

    results = {}
    for alpha in alphas:
        cal_model = LABEL(model, alpha=alpha)
        cal_model.calibrate(cal_dataset=cal_data)
        y_true, _, _, extra = Trainer(model=cal_model, enable_logging=False).inference(
            test_loader, additional_outputs=["y_predset"]
        )
        predset = extra["y_predset"]
        # The miscoverage metrics expect a 1D integer array of labels.
        y_true = np.asarray(y_true)
        coverage = 1 - miscoverage_overall_ps(predset, y_true)
        set_size = size(predset)
        class_miscov = miscoverage_ps(predset, y_true)
        results[alpha] = (coverage, set_size, class_miscov)
    return results


def main(
    root: str,
    seeds: list[int],
    alphas: list[float],
    epochs: int,
    dev: bool,
) -> None:
    dataset = MIMIC4Dataset(
        ehr_root=root,
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=dev,
    )
    samples = dataset.set_task(LengthOfStayPredictionMIMIC4())
    print(f"Samples: {len(samples)}")

    # Aggregate results across seeds.
    coverage = {a: [] for a in alphas}
    set_size = {a: [] for a in alphas}
    class_miscov = {a: [] for a in alphas}
    for seed in seeds:
        results = run_seed(samples, seed, alphas, epochs)
        for a in alphas:
            coverage[a].append(results[a][0])
            set_size[a].append(results[a][1])
            class_miscov[a].append(results[a][2])

    print(f"\nResults over {len(seeds)} seeds (mean +/- std):")
    print("alpha  target  coverage      avg_set_size")
    for a in alphas:
        cov = np.array(coverage[a])
        sizes = np.array(set_size[a])
        print(f"{a:.2f}    {1 - a:.0%}    {cov.mean():.2f} +/- {cov.std():.2f}   "
              f"{sizes.mean():.1f} +/- {sizes.std():.1f}")

    # Per-class miscoverage_ps (one value per LOS class), averaged over seeds.
    print(f"\nPer-class miscoverage_ps (mean over {len(seeds)} seeds):")
    for a in alphas:
        per_class = np.stack(class_miscov[a]).mean(0)
        print(f"alpha={a:.2f}: " + np.array2string(per_class, precision=2, floatmode="fixed"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split conformal prediction for MIMIC-IV length-of-stay prediction."
    )
    parser.add_argument(
        "--root",
        default="/srv/local/data/physionet.org/files/mimiciv/2.2",
        help="MIMIC-IV root (the folder containing hosp/).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per seed.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated random seeds to average over.",
    )
    parser.add_argument(
        "--alphas",
        default="0.2,0.1,0.05,0.01",
        help="Comma-separated target miscoverage rates, e.g. '0.2,0.1,0.05'.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use a subsampled dataset for a quick smoke test.",
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]
    main(args.root, seeds, alphas, args.epochs, args.dev)
