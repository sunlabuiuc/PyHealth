"""Matched multi-seed TPC ablations on synthetic day-one ICU windows.

Paper: Rocheteau et al. (CHIL 2021), https://arxiv.org/abs/2007.09483.
Contribution: Model (Option 2), UIUC CS 598 DLH.

Run from the repository root::

    python -m examples.synthetic_icu_los_tpc

Protocol
--------
The default run uses seeds 0, 1, 2, 300 training patients, 150 disjoint held-out
patients, eight epochs, batch size 32, Adam, and one CPU thread. Each sample
contains the first 24 hours; the target is remaining LoS at hour 24. Thus bias
measures day-one prediction bias, not a separate short-stay subgroup. Within
each seed, configurations share raw cohorts, labels, split, initialization seed,
and shuffle seed. Preprocessing changes only the named factor. Processors are
fitted on training data and reused on held-out data. No test-based selection or
early stopping is used. Dataset-size runs use nested training cohorts.

Baseline: N=3, k=4, Y=4, Z=4, main dropout=0.2, temporal dropout=0.05,
final width=16, learning rate=0.0023, official dilation, MSLE, decay base=0.75,
lab forward-fill enabled. Layers (1,3,5,9), kernels (2,3,4,5), each integer
channel count 4 through 16, dropout (0,0.2,0.45), and learning rates
(0.001,0.0023,0.01) form one-factor sweeps, not a Cartesian product.

Extensions compare constant/linear/exponential dilation, asymmetric MSLE
(underprediction weight 2), Huber log-loss (delta 1), decay bases (0.5,0.75,0.9),
and zeroing missing lab features without changing vital-sign forward-fill.
Linear equals the paper schedule; it is not an independent novel architecture.
The baseline kernel of 4 makes official dilations (1,3,6) distinct from
exponential (1,2,4), avoiding a degenerate extension comparison.
MSE vs. MSLE is explicitly a Sanity Check of the paper's loss result.
Training sizes (150,300,600) are a Compute-Constraint Demonstration, not novelty.
Hyperparameter sweeps are sensitivity analyses, not claimed novel extensions.

Metrics: MAD (days), MAPE (percent, target denominator floored at four hours),
MSLE (log days, not log1p), and R2. Bias is mean(predicted - true) in days;
negative means underprediction. LongMAD is MAD for true remaining LoS >=5 days;
LongBias is the corresponding signed bias. Each value reports mean +/- sample
standard deviation across three seeds. LongN reports subgroup counts per seed.
Results are synthetic feasibility evidence, not reproduction of clinical
eICU/MIMIC benchmark scores. Standard BatchNorm uses running statistics at eval;
training statistics pool over time, as noted in the model documentation.

Measured results
----------------
Default command, CPU, seeds 0/1/2, eight epochs; elapsed 157.6 seconds.
The following Markdown table is copied exactly from the measured output.

| Configuration | MAD | MAPE | MSLE | R2 | Bias | LongMAD | LongBias | LongN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.803 +/- 0.109 | 28.894 +/- 2.720 | 0.141 +/- 0.022 | 0.638 +/- 0.112 | -0.416 +/- 0.264 | 2.706 +/- 0.967 | -2.302 +/- 1.146 | 19/20/16 |
| layers=1 | 1.274 +/- 0.206 | 52.183 +/- 5.595 | 0.386 +/- 0.095 | 0.123 +/- 0.155 | -0.591 +/- 0.371 | 4.204 +/- 0.684 | -4.153 +/- 0.599 | 19/20/16 |
| layers=5 | 0.781 +/- 0.123 | 27.412 +/- 1.461 | 0.119 +/- 0.025 | 0.591 +/- 0.214 | -0.253 +/- 0.334 | 2.677 +/- 0.961 | -1.373 +/- 1.616 | 19/20/16 |
| layers=9 | 0.808 +/- 0.082 | 27.589 +/- 1.152 | 0.130 +/- 0.013 | 0.579 +/- 0.070 | -0.491 +/- 0.229 | 3.111 +/- 0.713 | -3.026 +/- 0.811 | 19/20/16 |
| kernel=2 | 1.261 +/- 0.160 | 51.513 +/- 4.544 | 0.378 +/- 0.067 | 0.195 +/- 0.049 | -0.676 +/- 0.289 | 4.451 +/- 0.654 | -4.451 +/- 0.654 | 19/20/16 |
| kernel=3 | 1.003 +/- 0.134 | 34.680 +/- 1.077 | 0.201 +/- 0.008 | 0.347 +/- 0.249 | -0.372 +/- 0.249 | 3.677 +/- 0.745 | -1.733 +/- 1.955 | 19/20/16 |
| kernel=5 | 0.776 +/- 0.033 | 27.610 +/- 1.858 | 0.129 +/- 0.007 | 0.655 +/- 0.050 | -0.387 +/- 0.240 | 2.609 +/- 0.514 | -2.232 +/- 1.021 | 19/20/16 |
| temporal=5 | 0.769 +/- 0.049 | 27.593 +/- 1.991 | 0.125 +/- 0.012 | 0.646 +/- 0.058 | -0.406 +/- 0.162 | 2.625 +/- 0.479 | -2.397 +/- 0.872 | 19/20/16 |
| temporal=6 | 0.796 +/- 0.120 | 28.045 +/- 1.030 | 0.133 +/- 0.022 | 0.626 +/- 0.166 | -0.467 +/- 0.257 | 2.703 +/- 0.891 | -2.477 +/- 1.069 | 19/20/16 |
| temporal=7 | 0.776 +/- 0.085 | 27.020 +/- 2.074 | 0.128 +/- 0.018 | 0.654 +/- 0.086 | -0.376 +/- 0.185 | 2.613 +/- 0.689 | -2.001 +/- 0.511 | 19/20/16 |
| temporal=8 | 0.794 +/- 0.100 | 28.072 +/- 1.032 | 0.124 +/- 0.011 | 0.654 +/- 0.089 | -0.403 +/- 0.208 | 2.717 +/- 0.816 | -2.638 +/- 0.813 | 19/20/16 |
| temporal=9 | 0.789 +/- 0.036 | 27.391 +/- 1.771 | 0.126 +/- 0.009 | 0.633 +/- 0.026 | -0.467 +/- 0.131 | 2.776 +/- 0.339 | -2.677 +/- 0.257 | 19/20/16 |
| temporal=10 | 0.811 +/- 0.040 | 29.310 +/- 3.827 | 0.126 +/- 0.003 | 0.620 +/- 0.067 | -0.195 +/- 0.470 | 2.608 +/- 0.658 | -1.625 +/- 2.037 | 19/20/16 |
| temporal=11 | 0.773 +/- 0.042 | 28.041 +/- 1.442 | 0.126 +/- 0.005 | 0.662 +/- 0.047 | -0.360 +/- 0.200 | 2.520 +/- 0.489 | -2.302 +/- 0.375 | 19/20/16 |
| temporal=12 | 0.818 +/- 0.125 | 28.345 +/- 1.177 | 0.137 +/- 0.016 | 0.604 +/- 0.114 | -0.462 +/- 0.331 | 2.834 +/- 1.051 | -2.775 +/- 1.142 | 19/20/16 |
| temporal=13 | 0.737 +/- 0.045 | 27.392 +/- 2.269 | 0.114 +/- 0.007 | 0.690 +/- 0.046 | -0.155 +/- 0.141 | 1.971 +/- 0.356 | -1.322 +/- 0.372 | 19/20/16 |
| temporal=14 | 0.826 +/- 0.093 | 28.128 +/- 3.821 | 0.140 +/- 0.043 | 0.572 +/- 0.148 | -0.421 +/- 0.120 | 2.897 +/- 0.589 | -2.408 +/- 1.034 | 19/20/16 |
| temporal=15 | 0.744 +/- 0.036 | 26.605 +/- 2.353 | 0.111 +/- 0.003 | 0.710 +/- 0.019 | -0.314 +/- 0.193 | 2.318 +/- 0.361 | -1.868 +/- 0.693 | 19/20/16 |
| temporal=16 | 0.775 +/- 0.102 | 27.526 +/- 0.386 | 0.126 +/- 0.021 | 0.673 +/- 0.077 | -0.423 +/- 0.265 | 2.564 +/- 0.742 | -2.424 +/- 0.879 | 19/20/16 |
| pointwise=5 | 0.828 +/- 0.079 | 29.125 +/- 0.371 | 0.139 +/- 0.014 | 0.578 +/- 0.068 | -0.438 +/- 0.200 | 2.952 +/- 0.628 | -2.824 +/- 0.775 | 19/20/16 |
| pointwise=6 | 0.803 +/- 0.020 | 28.501 +/- 3.457 | 0.131 +/- 0.013 | 0.628 +/- 0.028 | -0.340 +/- 0.282 | 2.671 +/- 0.324 | -2.047 +/- 0.854 | 19/20/16 |
| pointwise=7 | 0.821 +/- 0.092 | 29.304 +/- 1.267 | 0.135 +/- 0.017 | 0.623 +/- 0.113 | -0.376 +/- 0.316 | 2.689 +/- 0.937 | -2.415 +/- 1.278 | 19/20/16 |
| pointwise=8 | 0.769 +/- 0.048 | 27.855 +/- 1.201 | 0.129 +/- 0.009 | 0.676 +/- 0.080 | -0.391 +/- 0.205 | 2.473 +/- 0.536 | -2.305 +/- 0.747 | 19/20/16 |
| pointwise=9 | 0.768 +/- 0.051 | 28.242 +/- 0.948 | 0.130 +/- 0.012 | 0.676 +/- 0.007 | -0.397 +/- 0.150 | 2.429 +/- 0.169 | -2.064 +/- 0.454 | 19/20/16 |
| pointwise=10 | 0.783 +/- 0.130 | 27.589 +/- 0.574 | 0.124 +/- 0.023 | 0.666 +/- 0.117 | -0.366 +/- 0.310 | 2.606 +/- 1.044 | -2.274 +/- 1.263 | 19/20/16 |
| pointwise=11 | 0.797 +/- 0.072 | 29.796 +/- 3.075 | 0.129 +/- 0.016 | 0.648 +/- 0.067 | -0.204 +/- 0.427 | 2.572 +/- 0.748 | -1.582 +/- 2.113 | 19/20/16 |
| pointwise=12 | 0.844 +/- 0.078 | 29.388 +/- 2.173 | 0.140 +/- 0.004 | 0.598 +/- 0.069 | -0.416 +/- 0.369 | 2.848 +/- 0.961 | -2.529 +/- 1.450 | 19/20/16 |
| pointwise=13 | 0.779 +/- 0.044 | 28.666 +/- 1.079 | 0.125 +/- 0.008 | 0.660 +/- 0.065 | -0.341 +/- 0.023 | 2.651 +/- 0.284 | -2.287 +/- 0.091 | 19/20/16 |
| pointwise=14 | 0.814 +/- 0.018 | 29.503 +/- 2.129 | 0.134 +/- 0.003 | 0.638 +/- 0.039 | -0.340 +/- 0.250 | 2.756 +/- 0.355 | -1.978 +/- 1.544 | 19/20/16 |
| pointwise=15 | 0.839 +/- 0.048 | 27.892 +/- 1.558 | 0.140 +/- 0.009 | 0.576 +/- 0.044 | -0.548 +/- 0.226 | 3.064 +/- 0.458 | -2.818 +/- 0.883 | 19/20/16 |
| pointwise=16 | 0.823 +/- 0.054 | 28.127 +/- 2.891 | 0.138 +/- 0.016 | 0.581 +/- 0.041 | -0.484 +/- 0.121 | 2.923 +/- 0.149 | -2.916 +/- 0.152 | 19/20/16 |
| dropout=0.0 | 0.817 +/- 0.018 | 30.474 +/- 4.172 | 0.138 +/- 0.017 | 0.662 +/- 0.032 | -0.176 +/- 0.203 | 2.310 +/- 0.654 | -1.437 +/- 0.438 | 19/20/16 |
| dropout=0.45 | 0.835 +/- 0.083 | 29.953 +/- 2.093 | 0.154 +/- 0.008 | 0.602 +/- 0.106 | -0.509 +/- 0.191 | 2.823 +/- 0.815 | -2.579 +/- 1.170 | 19/20/16 |
| lr=0.001 | 0.856 +/- 0.119 | 30.598 +/- 1.631 | 0.158 +/- 0.023 | 0.577 +/- 0.130 | -0.368 +/- 0.238 | 2.992 +/- 0.926 | -2.009 +/- 1.443 | 19/20/16 |
| lr=0.01 | 0.768 +/- 0.048 | 31.020 +/- 3.914 | 0.128 +/- 0.009 | 0.655 +/- 0.118 | -0.067 +/- 0.181 | 2.233 +/- 0.793 | -1.375 +/- 0.410 | 19/20/16 |
| dilation=paper | 0.903 +/- 0.101 | 33.762 +/- 0.365 | 0.173 +/- 0.032 | 0.587 +/- 0.069 | -0.355 +/- 0.384 | 2.922 +/- 0.696 | -2.248 +/- 1.379 | 19/20/16 |
| dilation=constant | 1.080 +/- 0.071 | 45.342 +/- 4.939 | 0.282 +/- 0.013 | 0.435 +/- 0.048 | -0.441 +/- 0.315 | 3.332 +/- 0.537 | -3.083 +/- 0.687 | 19/20/16 |
| dilation=linear | 0.903 +/- 0.101 | 33.762 +/- 0.365 | 0.173 +/- 0.032 | 0.587 +/- 0.069 | -0.355 +/- 0.384 | 2.922 +/- 0.696 | -2.248 +/- 1.379 | 19/20/16 |
| dilation=exponential | 0.976 +/- 0.086 | 33.076 +/- 1.453 | 0.193 +/- 0.020 | 0.440 +/- 0.058 | -0.611 +/- 0.211 | 3.771 +/- 0.570 | -3.500 +/- 0.800 | 19/20/16 |
| loss=asymmetric_msle | 0.790 +/- 0.056 | 30.384 +/- 3.355 | 0.129 +/- 0.015 | 0.675 +/- 0.070 | -0.121 +/- 0.232 | 2.518 +/- 0.764 | -1.454 +/- 1.200 | 19/20/16 |
| loss=huber_log | 0.789 +/- 0.130 | 28.304 +/- 2.012 | 0.134 +/- 0.025 | 0.639 +/- 0.151 | -0.382 +/- 0.288 | 2.751 +/- 1.069 | -2.165 +/- 1.381 | 19/20/16 |
| Sanity Check: MSE | 0.998 +/- 0.087 | 37.390 +/- 5.658 | 0.228 +/- 0.040 | 0.464 +/- 0.036 | -0.681 +/- 0.079 | 3.638 +/- 0.282 | -3.627 +/- 0.299 | 19/20/16 |
| decay=0.5 | 0.797 +/- 0.091 | 28.691 +/- 2.286 | 0.134 +/- 0.015 | 0.652 +/- 0.068 | -0.396 +/- 0.168 | 2.694 +/- 0.717 | -2.276 +/- 0.775 | 19/20/16 |
| decay=0.9 | 0.791 +/- 0.083 | 28.380 +/- 1.944 | 0.135 +/- 0.016 | 0.657 +/- 0.099 | -0.395 +/- 0.304 | 2.673 +/- 0.800 | -2.280 +/- 1.250 | 19/20/16 |
| lab no-forward-fill | 0.915 +/- 0.157 | 32.067 +/- 2.196 | 0.154 +/- 0.021 | 0.468 +/- 0.239 | -0.246 +/- 0.316 | 3.250 +/- 1.144 | -1.946 +/- 1.169 | 19/20/16 |
| Compute-Constraint: n=150 | 0.961 +/- 0.181 | 32.989 +/- 4.492 | 0.205 +/- 0.069 | 0.445 +/- 0.231 | -0.661 +/- 0.286 | 3.434 +/- 0.893 | -3.360 +/- 0.992 | 19/20/16 |
| Compute-Constraint: n=600 | 0.728 +/- 0.043 | 27.918 +/- 1.373 | 0.115 +/- 0.002 | 0.708 +/- 0.037 | -0.134 +/- 0.164 | 2.132 +/- 0.622 | -1.559 +/- 0.364 | 19/20/16 |

Analysis
--------
Asymmetric MSLE shifts day-one bias from -0.416 to -0.121 days and long-stay
bias from -2.302 to -1.454 days, while MAD is similar (0.803 vs 0.790).
It does not eliminate underprediction and raises MAPE (28.894 to 30.384).
Huber has similar MAD (0.789) but little bias improvement (-0.382).
Official dilation has lower average MSLE than constant (0.282), linear
(0.173), or exponential (0.193). Linear and paper match exactly as expected.
Decay-base changes yield small differences relative to seed variation; no
clear best base is established. Removing lab forward-fill increases MAD
(0.803 to 0.915) and long-stay MAD (2.706 to 3.250), consistent with this
synthetic generator's persistent severity signal, not a clinical conclusion.
Sensitivity to capacity is not monotonic: temporal=15 has low MSLE (0.111),
but increasing depth to 9 does not reliably improve MAD over the baseline.
The Sanity Check favors MSLE over MSE (0.141 vs 0.228 evaluation MSLE).
The Compute-Constraint Demonstration improves MAD with more training samples
(0.961, 0.803, 0.728 for 150, 300, 600); fixed epochs also mean more optimizer
steps, so this does not isolate dataset size from compute. Three seeds and
19/20/16 long-stay patients are insufficient for significance or clinical claims.
"""

import argparse
import math
import time
from typing import Any

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.datasets.sample_dataset import InMemorySampleDataset
from pyhealth.models.tpc import MAX_LOS_DAYS, MIN_LOS_DAYS, TPC, prepare_time_series

STEPS = 24
NUM_FEATURES = 8
SAMPLE_RATES = [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.1, 0.1]
LAB_INDICES = [4, 5, 6, 7]
SEEDS = (0, 1, 2)
METRICS = ("MAD", "MAPE", "MSLE", "R2", "Bias", "LongMAD", "LongBias")


def make_synthetic_samples(
    n: int,
    seed: int,
    decay_base: float = 0.75,
    forward_fill_labs: bool = True,
    prefix: str = "patient",
) -> list[dict[str, Any]]:
    """Creates day-one windows; missing values never reveal raw hidden values.

    Args:
        n: Number of independent patients.
        seed: Cohort generation seed.
        decay_base: Measurement decay base.
        forward_fill_labs: Whether missing labs carry their last measurement.
        prefix: Patient/visit namespace to keep train and test disjoint.

    Returns:
        Dense samples containing filled values, decay and remaining-LoS labels.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        severity = rng.normal()
        trend = rng.normal(scale=0.1)
        baseline = severity * rng.uniform(0.5, 1.5, NUM_FEATURES)
        t = np.arange(STEPS)[:, None]
        raw = baseline + trend * t + rng.normal(scale=0.3, size=(STEPS, NUM_FEATURES))
        observed = rng.random((STEPS, NUM_FEATURES)) < np.array(SAMPLE_RATES)
        observed[0] = True
        filled, decay = prepare_time_series(
            torch.tensor(raw[None], dtype=torch.float32),
            torch.tensor(observed[None]),
            decay_base,
            LAB_INDICES,
            forward_fill_labs,
        )
        los = float(np.exp(0.7 * severity + 0.8 + rng.normal(scale=0.3)))
        samples.append(
            {
                "patient_id": f"{prefix}{i}",
                "visit_id": f"{prefix}{i}",
                "time_series": filled[0].tolist(),
                "decay": decay[0].tolist(),
                "los": float(np.clip(los, MIN_LOS_DAYS, MAX_LOS_DAYS)),
            }
        )
    return samples


def build_dataset(
    samples: list[dict[str, Any]],
    name: str,
    training: InMemorySampleDataset | None = None,
) -> InMemorySampleDataset:
    """Builds an in-memory dataset; held-out data reuse training processors."""
    return create_sample_dataset(
        samples=samples,
        input_schema={"time_series": "tensor", "decay": "tensor"},
        output_schema={"los": "regression"},
        dataset_name=name,
        input_processors=training.input_processors if training is not None else None,
        output_processors=training.output_processors if training is not None else None,
        in_memory=True,
    )


def los_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Reports paper-style errors plus day-one and >=5-day subgroup bias.

    Args:
        y_true: Positive remaining-LoS labels in days, one-dimensional.
        y_pred: Positive finite predictions with the same shape.

    Returns:
        Seven metrics and the long-stay subgroup count.

    Raises:
        ValueError: If inputs are invalid, R2 undefined, or no long stays exist.
    """
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or y_true.size < 2:
        raise ValueError("Metrics need matching vectors with at least two samples.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metrics require finite labels and predictions.")
    if (y_true <= 0).any() or (y_pred <= 0).any():
        raise ValueError("LoS must be positive.")
    centered_ss = ((y_true - y_true.mean()) ** 2).sum()
    long = y_true >= 5.0
    if centered_ss == 0 or not long.any():
        raise ValueError("R2 needs varying labels and LongMAD needs >=5-day stays.")
    err = y_pred - y_true
    return {
        "MAD": float(np.abs(err).mean()),
        "MAPE": float((np.abs(err) / np.maximum(y_true, 4 / 24)).mean() * 100),
        "MSLE": float(((np.log(y_pred) - np.log(y_true)) ** 2).mean()),
        "R2": float(1 - (err**2).sum() / centered_ss),
        "Bias": float(err.mean()),
        "LongMAD": float(np.abs(err[long]).mean()),
        "LongBias": float(err[long].mean()),
        "LongN": float(long.sum()),
    }


def train_and_eval(
    train_ds: InMemorySampleDataset,
    test_ds: InMemorySampleDataset,
    epochs: int,
    lr: float,
    seed: int,
    **model_kwargs: Any,
) -> dict[str, float]:
    """Trains a fixed epoch budget and evaluates once on held-out patients."""
    torch.manual_seed(seed)
    model = TPC(train_ds, value_key="time_series", decay_key="decay", **model_kwargs)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True, seed=seed)
    test_loader = get_dataloader(test_ds, batch_size=64, shuffle=False)
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            loss = model(**batch)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss for seed {seed}.")
            loss.backward()
            optim.step()
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(**batch)
            ys.append(out["y_true"].squeeze(-1))
            preds.append(out["y_prob"].squeeze(-1))
    return los_metrics(torch.cat(ys).numpy(), torch.cat(preds).numpy())


BASE = {
    "temporal_channels": 4,
    "pointwise_channels": 4,
    "num_layers": 3,
    "kernel_size": 4,
    "main_dropout": 0.2,
    "temporal_dropout": 0.05,
    "final_fc_size": 16,
    "loss": "msle",
    "dilation_schedule": "official",
}


def ablation_configs() -> dict[str, dict[str, Any]]:
    """Returns one-factor configurations; baseline covers unchanged levels."""
    configs: dict[str, dict[str, Any]] = {"baseline": {}}
    for title, key, levels in (
        ("layers", "num_layers", (1, 3, 5, 9)),
        ("kernel", "kernel_size", (2, 3, 4, 5)),
        ("temporal", "temporal_channels", range(4, 17)),
        ("pointwise", "pointwise_channels", range(4, 17)),
        ("dropout", "main_dropout", (0.0, 0.2, 0.45)),
        ("lr", "lr", (0.001, 0.0023, 0.01)),
    ):
        baseline = 0.0023 if key == "lr" else BASE[key]
        for level in levels:
            if level != baseline:
                configs[f"{title}={level}"] = {key: level}
    for schedule in ("paper", "constant", "linear", "exponential"):
        configs[f"dilation={schedule}"] = {"dilation_schedule": schedule}
    configs["loss=asymmetric_msle"] = {"loss": "asymmetric_msle"}
    configs["loss=huber_log"] = {"loss": "huber_log"}
    configs["Sanity Check: MSE"] = {"loss": "mse"}
    for base in (0.5, 0.9):
        configs[f"decay={base}"] = {"decay_base": base}
    configs["lab no-forward-fill"] = {"forward_fill_labs": False}
    for size in (150, 600):
        configs[f"Compute-Constraint: n={size}"] = {"n_train": size}
    return configs


def main(
    n_train: int = 300,
    n_test: int = 150,
    epochs: int = 8,
    lr: float = 0.0023,
    seeds: tuple[int, ...] = SEEDS,
) -> None:
    """Runs the complete sweep and prints exact Markdown mean +/- std tables."""
    if len(set(seeds)) < 3:
        raise ValueError("At least three distinct random seeds are required.")
    if n_train < 2 or n_test < 2 or epochs < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError(
            "Positive epochs/lr and at least two train/test samples required."
        )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        print(
            f"Protocol: seeds={seeds}; train={n_train}; test={n_test}; "
            f"epochs={epochs}; baseline lr={lr}; CPU threads=1",
            flush=True,
        )
        print("| Configuration | " + " | ".join(METRICS) + " | LongN |", flush=True)
        print("| --- | " + " | ".join(["---:"] * (len(METRICS) + 1)) + " |", flush=True)
        start = time.perf_counter()
        for name, override in ablation_configs().items():
            params = {**BASE, **override}
            run_lr = params.pop("lr", lr)
            size = params.pop("n_train", n_train)
            decay_base = params.pop("decay_base", 0.75)
            fill_labs = params.pop("forward_fill_labs", True)
            results = []
            for seed in seeds:
                train_ds = build_dataset(
                    make_synthetic_samples(
                        size, seed + 100, decay_base, fill_labs, prefix="train"
                    ),
                    f"tpc_train_{seed}",
                )
                test_ds = build_dataset(
                    make_synthetic_samples(
                        n_test, seed + 10000, decay_base, fill_labs, prefix="test"
                    ),
                    f"tpc_test_{seed}",
                    training=train_ds,
                )
                results.append(
                    train_and_eval(train_ds, test_ds, epochs, run_lr, seed, **params)
                )
            cells = []
            for metric in METRICS:
                values = np.array([result[metric] for result in results])
                cells.append(f"{values.mean():.3f} +/- {values.std(ddof=1):.3f}")
            counts = "/".join(str(int(result["LongN"])) for result in results)
            print(f"| {name} | " + " | ".join(cells) + f" | {counts} |", flush=True)
        print(f"Elapsed: {time.perf_counter() - start:.1f} seconds.", flush=True)
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0023)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    main(args.n_train, args.n_test, args.epochs, args.lr, tuple(args.seeds))
