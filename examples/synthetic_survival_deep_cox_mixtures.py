"""Ablation example: Deep Cox Mixtures on synthetic survival data.

This script reproduces the core claim of the Deep Cox Mixtures paper on a
small, self-contained synthetic benchmark. It generates 500 samples from two
latent patient subgroups with different baseline hazards and different linear
risk directions, then trains :class:`pyhealth.models.DeepCoxMixtures` with
``k in {1, 2, 3}`` components, reporting Harrell's concordance index on a
held-out split.

Reference:
    Nagpal, C., Yadlowsky, S., Rostamzadeh, N., & Heller, K. (2021).
    Deep Cox Mixtures for Survival Regression.
    Proceedings of Machine Learning for Healthcare.
    https://proceedings.mlr.press/v149/nagpal21a/nagpal21a.pdf

Hypothesis (from the paper): when the true data-generating process is a
mixture of proportional-hazards subgroups, a single Cox expert (``k=1``)
is mis-specified and a mixture recovers the subgroups.

Expected result (qualitative, seeded): ``k=1`` trails ``k=2`` on C-index
because a single Cox expert cannot represent two subgroups with opposite
linear risk; ``k=2`` matches the latent structure and wins; ``k=3`` is
slightly over-parameterised and regresses toward ``k=1``.

Observed with default flags (seed=0, 400 train / 100 test, 20 epochs)::

    k |    C-index
   --------------------
    1 |     0.6209
    2 |     0.6499   (best)
    3 |     0.6365

Usage:
    python examples/synthetic_survival_deep_cox_mixtures.py

Runtime: ~30 seconds on a laptop CPU.
"""

import argparse
import random
from typing import List

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DeepCoxMixtures


def generate_two_group_survival(
    n: int, feature_dim: int, seed: int
) -> List[dict]:
    """Synthesise ``n`` survival samples drawn from two latent subgroups.

    Subgroup membership is determined by the sign of the first feature, and
    each subgroup uses an opposite linear risk direction so that a single Cox
    expert is mis-specified and a two-component mixture should improve fit.
    """
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n, feature_dim)).astype(np.float32)
    group = (features[:, 0] > 0).astype(int)
    beta = np.zeros(feature_dim, dtype=np.float64)
    beta[1] = 1.0
    beta[2] = -0.5
    log_hr = np.clip(
        np.where(group == 0, features @ beta, features @ (-beta)), -4.0, 4.0
    )
    baseline = np.where(group == 0, 10.0, 4.0)
    u = rng.uniform(0.05, 0.95, size=n)
    true_time = -np.log(u) * baseline / np.exp(log_hr)
    censor_time = rng.exponential(scale=12.0, size=n)
    observed_time = np.minimum(true_time, censor_time)
    event = (true_time <= censor_time).astype(int)
    return [
        {
            "patient_id": f"p{i}",
            "features": features[i].tolist(),
            "time": float(observed_time[i]),
            "event": int(event[i]),
        }
        for i in range(n)
    ]


def harrell_c_index(
    risk: np.ndarray, time: np.ndarray, event: np.ndarray
) -> float:
    """Harrell's concordance index for right-censored survival data."""
    n = risk.shape[0]
    concordant = 0.0
    comparable = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if time[i] < time[j] and event[i] == 1:
                comparable += 1
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return concordant / max(comparable, 1)


def run_one(
    train_ds, test_batch, k: int, seed: int, epochs: int, horizon: float
) -> float:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader = get_dataloader(train_ds, batch_size=64, shuffle=True)
    model = DeepCoxMixtures(dataset=train_ds, k=k, hidden_dims=(32, 16))
    model.fit(loader, epochs=epochs, learning_rate=5e-3, verbose=False)
    # 1 - S(horizon) is used as the risk score so that differences in both
    # per-component hazard ratios and per-component baseline hazards drive
    # the ranking.
    surv = model.predict_survival_curve(test_batch["features"], [horizon])
    return harrell_c_index(
        risk=1.0 - surv[:, 0],
        time=test_batch["time"].view(-1).cpu().numpy(),
        event=test_batch["event"].view(-1).cpu().numpy(),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--feature-dim", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Mixture-component counts to ablate over.",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=5.0,
        help="Evaluation time horizon for the 1-S(t) risk score.",
    )
    args = parser.parse_args(argv)

    train_samples = generate_two_group_survival(
        n=args.n_train, feature_dim=args.feature_dim, seed=args.seed
    )
    test_samples = generate_two_group_survival(
        n=args.n_test, feature_dim=args.feature_dim, seed=args.seed + 1
    )
    train_ds = create_sample_dataset(
        samples=train_samples,
        input_schema={"features": "tensor"},
        output_schema={"time": "regression", "event": "binary"},
        dataset_name="synthetic_survival_train",
    )
    test_ds = create_sample_dataset(
        samples=test_samples,
        input_schema={"features": "tensor"},
        output_schema={"time": "regression", "event": "binary"},
        dataset_name="synthetic_survival_test",
    )
    test_loader = get_dataloader(test_ds, batch_size=len(test_ds), shuffle=False)
    test_batch = next(iter(test_loader))

    print("== DeepCoxMixtures ablation (C-index, higher is better) ==")
    print(f"{'k':>4} | {'C-index':>10}")
    print("-" * 20)
    results = {}
    for k in args.k_values:
        c = run_one(
            train_ds,
            test_batch,
            k=k,
            seed=args.seed,
            epochs=args.epochs,
            horizon=args.horizon,
        )
        results[k] = c
        print(f"{k:>4} | {c:>10.4f}")
    print()
    best_k = max(results, key=results.get)
    print(f"Best: k={best_k} with C-index={results[best_k]:.4f}")
    print(
        "Experimental setup: 400/100 train/test synthetic samples drawn from "
        "two latent subgroups with different Weibull baselines and different "
        "linear risk directions. Model is a shared MLP embedding with K Cox "
        "experts; Breslow baselines are refit each epoch."
    )


if __name__ == "__main__":
    main()
