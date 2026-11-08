"""Ablation study for TPC on synthetic ICU data (runs on CPU in about a minute).

Paper: Rocheteau et al. (2021), "Temporal Pointwise Convolutional Networks for
Length of Stay Prediction in the Intensive Care Unit",
https://doi.org/10.1145/3450439.3451860

What this script does
---------------------
1. Builds a small synthetic ICU cohort: hourly vitals/labs with different
   sampling rates, forward-filling, and decay indicators (as in Section 4).
2. Trains TPC under several hyperparameter settings and reports MAD, MAPE,
   MSLE and R^2 on a held-out split, as in Table 2 of the paper.

Ablations (hyperparameter variations, not repeated from the paper)
------------------------------------------------------------------
* kernel size k and number of layers N (receptive field)
* temporal channels Y and pointwise channels Z (capacity)
* main dropout
* MSLE vs MSE loss (sanity check of the paper's loss claim)

To use real data, replace ``make_synthetic_samples`` with samples produced by
an eICU or MIMIC-IV task that returns ``time_series``, ``decay`` and a scalar
remaining-LoS label in days. Keep the same ``input_schema``/``output_schema``.

The numbers printed here come from synthetic data and say nothing about the
paper's eICU/MIMIC-IV results.
"""

import time
from typing import Dict, List

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.tpc import TPC, compute_decay_indicators

STEPS = 24  # hours of history; we predict remaining LoS at the last hour
NUM_FEATURES = 8
# Probability a feature is measured in a given hour (labs are sparse).
SAMPLE_RATES = [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.1, 0.1]


def make_synthetic_samples(n: int, seed: int) -> List[dict]:
    """Creates synthetic ICU windows with a hidden severity driving LoS."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        severity = rng.normal()
        trend = rng.normal(scale=0.1)
        base = severity * rng.uniform(0.5, 1.5, NUM_FEATURES)
        t = np.arange(STEPS)[:, None]
        raw = base + trend * t + rng.normal(scale=0.3, size=(STEPS, NUM_FEATURES))
        observed = rng.random((STEPS, NUM_FEATURES)) < np.array(SAMPLE_RATES)
        observed[0] = True  # every feature seen at least once

        # Forward-fill unobserved hours with the last recorded value.
        filled = raw.copy()
        for step in range(1, STEPS):
            filled[step] = np.where(observed[step], raw[step], filled[step - 1])
        decay = compute_decay_indicators(
            torch.tensor(observed[None], dtype=torch.float32)
        )[0]

        los = float(np.exp(0.7 * severity + 0.8 + rng.normal(scale=0.3)))
        samples.append(
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "time_series": filled.astype(np.float32).tolist(),
                "decay": decay.tolist(),
                "los": max(los, 1.0 / 48.0),
            }
        )
    return samples


def build_dataset(samples: List[dict], name: str):
    return create_sample_dataset(
        samples=samples,
        input_schema={"time_series": "tensor", "decay": "tensor"},
        output_schema={"los": "regression"},
        dataset_name=name,
    )


def los_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAD, MAPE (4h floor on the divisor), MSLE and R^2 as in the paper."""
    err = y_true - y_pred
    mad = np.abs(err).mean()
    mape = (np.abs(err) / np.maximum(y_true, 4 / 24)).mean() * 100
    msle = ((np.log(y_pred) - np.log(y_true)) ** 2).mean()
    r2 = 1 - (err**2).sum() / ((y_true - y_true.mean()) ** 2).sum()
    return {"MAD": mad, "MAPE": mape, "MSLE": msle, "R2": r2}


def train_and_eval(train_ds, test_ds, epochs: int, lr: float, **model_kwargs):
    torch.manual_seed(0)
    model = TPC(train_ds, value_key="time_series", decay_key="decay", **model_kwargs)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    test_loader = get_dataloader(test_ds, batch_size=64, shuffle=False)

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            model(**batch)["loss"].backward()
            optim.step()

    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(**batch)
            ys.append(out["y_true"].squeeze(-1))
            preds.append(out["y_prob"].squeeze(-1))
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    return los_metrics(y_true, y_pred)


# Small base config so the whole sweep finishes quickly on CPU.
BASE = dict(
    temporal_channels=4,
    pointwise_channels=4,
    num_layers=3,
    kernel_size=3,
    main_dropout=0.1,
    temporal_dropout=0.05,
    final_fc_size=16,
    loss="msle",
)

ABLATIONS = {
    "base": {},
    "kernel k=2": {"kernel_size": 2},
    "kernel k=5": {"kernel_size": 5},
    "layers N=1": {"num_layers": 1},
    "layers N=5": {"num_layers": 5},
    "temporal Y=2": {"temporal_channels": 2},
    "temporal Y=8": {"temporal_channels": 8},
    "pointwise Z=2": {"pointwise_channels": 2},
    "pointwise Z=8": {"pointwise_channels": 8},
    "dropout 0.0": {"main_dropout": 0.0},
    "dropout 0.4": {"main_dropout": 0.4},
    "loss MSE": {"loss": "mse"},
}


def main(n_train: int = 300, n_test: int = 100, epochs: int = 8, lr: float = 3e-3):
    train_ds = build_dataset(make_synthetic_samples(n_train, seed=1), "syn_train")
    test_ds = build_dataset(make_synthetic_samples(n_test, seed=2), "syn_test")

    print(f"{'config':<16}{'MAD':>8}{'MAPE':>9}{'MSLE':>8}{'R2':>8}{'sec':>7}")
    for name, override in ABLATIONS.items():
        start = time.time()
        res = train_and_eval(train_ds, test_ds, epochs, lr, **{**BASE, **override})
        print(
            f"{name:<16}{res['MAD']:>8.3f}{res['MAPE']:>9.1f}"
            f"{res['MSLE']:>8.3f}{res['R2']:>8.3f}{time.time() - start:>7.1f}"
        )


if __name__ == "__main__":
    main()
