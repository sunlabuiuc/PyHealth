"""Update a binary model for a new site with logistic recalibration.

A model trained at one site often ranks patients well at a new site but
predicts the wrong event rate there. This script trains a logistic regression
on a synthetic "site A", applies it to a "site B" with a higher event rate,
and then refits the calibration intercept (and slope) on a small site B
calibration set. Runs on CPU in a few seconds; no data download.

Usage:
    python examples/logistic_recalibration_new_site.py
"""

import random

import torch

from pyhealth.calib.calibration import LogisticRecalibration
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import LogisticRegression
from pyhealth.trainer import Trainer

CODES = [f"dx-{i}" for i in range(8)]
RISK_CODES = {"dx-0", "dx-1"}


def make_samples(n, base_rate, seed, prefix):
    """Codes drive risk the same way at both sites. Only the base rate differs."""
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        codes = rng.sample(CODES, k=rng.randint(1, 4))
        n_risk = len(RISK_CODES.intersection(codes))
        p = min(0.95, base_rate * (1 + 2 * n_risk))
        samples.append(
            {
                "patient_id": f"{prefix}-{i}",
                "visit_id": f"{prefix}-{i}",
                "conditions": codes,
                "label": int(rng.random() < p),
            }
        )
    return samples


def make_dataset(samples, name, reference=None):
    kwargs = {}
    if reference is not None:
        kwargs = {
            "input_processors": reference.input_processors,
            "output_processors": reference.output_processors,
        }
    return create_sample_dataset(
        samples=samples,
        input_schema={"conditions": "sequence"},
        output_schema={"label": "binary"},
        dataset_name=name,
        **kwargs,
    )


def evaluate(model, dataset):
    loader = get_dataloader(dataset, batch_size=256, shuffle=False)
    trainer = Trainer(model=model, metrics=["roc_auc", "ECE"], enable_logging=False)
    return trainer.evaluate(loader)


def main():
    torch.manual_seed(0)
    site_a = make_dataset(make_samples(2000, 0.10, seed=0, prefix="a"), "site_a")
    site_b_cal = make_dataset(
        make_samples(150, 0.25, seed=1, prefix="bc"), "site_b_cal", site_a
    )
    site_b_test = make_dataset(
        make_samples(2000, 0.25, seed=2, prefix="bt"), "site_b_test", site_a
    )

    model = LogisticRegression(dataset=site_a)
    Trainer(model=model, enable_logging=False).train(
        train_dataloader=get_dataloader(site_a, batch_size=64, shuffle=True),
        epochs=10,
    )

    print("original model   ", evaluate(model, site_b_test))
    for method in ("intercept", "intercept_slope"):
        cal_model = LogisticRecalibration(model, method=method)
        cal_model.calibrate(cal_dataset=site_b_cal)
        print(
            f"{method:17s}",
            evaluate(cal_model, site_b_test),
            "intercept",
            cal_model.intercept.tolist(),
            "slope",
            cal_model.slope.tolist(),
        )


if __name__ == "__main__":
    main()
