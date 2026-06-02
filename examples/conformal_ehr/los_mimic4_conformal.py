"""
Split conformal prediction for length of stay prediction on MIMIC-IV.

Trains a Transformer on the MIMIC-IV LOS task (10 classes), then wraps it with
LABEL (split conformal prediction) to use prediction sets instead of point predictions to
guarantee coverage?

Coverage on a single split is noisy here bc the demo only leaves  about 15 calibration
patients, so I averaged over a few seeds like the conformal eeg example.

Results on the MIMIC-IV demo (100 patients):

    alpha  target  coverage        avg_set_size
    0.20    80%    0.74 +/- 0.11    6.6 +/- 1.4
    0.10    90%    0.93 +/- 0.08    8.8 +/- 0.6
    0.05    95%    0.96 +/- 0.07    9.4 +/- 0.7

The targets are all within the error bars. Coverage is near the target
but noisy with so little patients. Sets shrink w/ less coverage.
On full MIMIC-IV the model would be stronger, so the sets get much smaller and coverage
tighter.

Run on the demo:
    python los_mimic4_conformal.py --root /path/to/mimic-iv-clinical-database-demo-2.2
"""

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
from pyhealth.models import Transformer
from pyhealth.tasks import LengthOfStayPredictionMIMIC4
from pyhealth.trainer import Trainer

# pyhealth logs the whole model at info on every Trainer init ( just want to quiet it down)
logging.getLogger("pyhealth").setLevel(logging.WARNING)


def run_seed(samples, seed, alphas):
    """Split, train, and run conformal for one seed.

    Returns {alpha: (coverage, avg_set_size)} measured on the test split.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # train / validation / calibration / test (calibration is the conformal specific one)
    train_data, val_data, cal_data, test_data = split_by_patient_conformal(
        samples, ratios=[0.6, 0.1, 0.1, 0.2], seed=seed
    )
    train_loader = get_dataloader(train_data, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_data, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_data, batch_size=32, shuffle=False)

    model = Transformer(dataset=samples)
    Trainer(model=model, enable_logging=False).train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=10,
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
        coverage = predset[np.arange(len(y_true)), y_true].mean()
        set_size = predset.sum(axis=1).mean()
        results[alpha] = (coverage, set_size)
    return results


def main(root):
    dataset = MIMIC4Dataset(
        ehr_root=root,
        ehr_tables=["patients", "admissions", "diagnoses_icd",
                    "procedures_icd", "prescriptions"],
        dev=False,
    )
    samples = dataset.set_task(LengthOfStayPredictionMIMIC4())
    print("samples:", len(samples))

    alphas = [0.2, 0.1, 0.05]
    seeds = [0, 1, 2, 3, 4]

    # average seeds
    coverage = {a: [] for a in alphas}
    set_size = {a: [] for a in alphas}
    for seed in seeds:
        results = run_seed(samples, seed, alphas)
        for a in alphas:
            coverage[a].append(results[a][0])
            set_size[a].append(results[a][1])

    print(f"\nresults over {len(seeds)} seeds (mean +/- std)")
    print("alpha  target  coverage      avg_set_size")
    for a in alphas:
        cov = np.array(coverage[a])
        size = np.array(set_size[a])
        print(f"{a:.2f}    {1 - a:.0%}    {cov.mean():.2f} +/- {cov.std():.2f}   "
              f"{size.mean():.1f} +/- {size.std():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/srv/local/data/physionet.org/files/mimiciv/2.2",
        help="MIMIC-IV root (the folder containing hosp/)",
    )
    args = parser.parse_args()
    main(args.root)
