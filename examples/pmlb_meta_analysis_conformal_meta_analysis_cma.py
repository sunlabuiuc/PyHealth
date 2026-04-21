"""Reproduction of Simulations 1-4 from Kaul & Gordon (2024) on PMLB.

This script reproduces the four simulations from Figure 2 of
Kaul, S. and Gordon, G. J. 2024. "Meta-Analysis with Untrusted
Data." Proceedings of Machine Learning Research, 259:563-593.

The simulations compare conformal meta-analysis (CMA, the paper's
method) to the Hartung-Knapp-Sidik-Jonkman (HKSJ) baseline on
partially-synthetic data derived from a PMLB regression dataset:

    Simulation 1: Interval width vs sample size ``n``, across three
        settings of prior quality (bad / okay / good).
    Simulation 2: Coverage vs effect noise; CMA should hold the
        95% target while HKSJ drops below it.
    Simulation 3: Coverage for ``eta=0`` vs ``eta>0``; the
        noise-correction parameter.
    Simulation 4: Interval width vs prior error, comparing CMA /
        HKSJ / Prior-only baselines.

Each simulation saves a PNG next to this script. Total runtime is a
few minutes on a CPU; no GPU is required.

Expected findings (the claims each simulation is designed to
validate from the paper):
    - Sim 1: CMA produces narrower intervals than HKSJ at small
      ``n`` when the prior is good, and tracks HKSJ as ``n`` grows
      or the prior degrades.
    - Sim 2: CMA maintains the target coverage ``1 - alpha`` as
      effect noise grows, while HKSJ drops below target once noise
      exceeds the within-trial variance scale.
    - Sim 3: Setting ``eta > 0`` (noise correction) restores
      coverage at high noise levels, at the cost of wider intervals.
    - Sim 4: CMA interval widths stay bounded as the prior
      degrades, whereas the Prior-only baseline diverges and HKSJ
      is unchanged (it ignores the prior).

Usage:
    python pmlb_meta_analysis_conformal_meta_analysis_cma.py

Dependencies:
    pyhealth (with this PR's additions), pmlb, scipy, matplotlib,
    numpy, torch.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as t_dist

from pyhealth.datasets import get_dataloader
from pyhealth.datasets.pmlb_meta_analysis_dataset import (
    PMLBMetaAnalysisDataset,
)
from pyhealth.models.conformal_meta_analysis_krr import (
    ConformalMetaAnalysisModel,
)
from pyhealth.tasks.conformal_meta_analysis import (
    ConformalMetaAnalysisTask,
)


# ---------------------------------------------------------------------
# HKSJ baseline (Proposition 10 in the paper, Appendix A.6)
# ---------------------------------------------------------------------
def hksj_interval(
    Y: np.ndarray, V: np.ndarray, alpha: float = 0.1,
) -> Tuple[float, float]:
    """Hartung-Knapp-Sidik-Jonkman prediction interval.

    Iteratively estimates heterogeneity ``nu`` and the weighted ATE,
    then returns a Student-t prediction interval at confidence
    ``1 - alpha``.

    Args:
        Y: Observed effects, shape ``(n,)``.
        V: Within-trial variances, shape ``(n,)``.
        alpha: Significance level.

    Returns:
        Tuple ``(lower, upper)`` of the prediction interval.
    """
    n = len(Y)
    nu = 0.0
    for _ in range(1000):
        w = 1.0 / (V + nu)
        ate = np.sum(w * Y) / np.sum(w)
        nu_new = max(
            0.0,
            np.sum(w ** 2 * ((Y - ate) ** 2 - V)) / np.sum(w ** 2)
            + 1.0 / np.sum(w),
        )
        if abs(nu_new - nu) < 1e-8:
            break
        nu = nu_new
    w = 1.0 / (V + nu)
    ate = np.sum(w * Y) / np.sum(w)
    var_ate = np.sum((Y - ate) ** 2 * w) / ((n - 1) * np.sum(w))
    half = t_dist.ppf(1 - alpha / 2, df=n - 1) * np.sqrt(nu + var_ate)
    return float(ate - half), float(ate + half)


# ---------------------------------------------------------------------
# Single-run harness: returns CMA / HKSJ / Prior widths and coverages.
# ---------------------------------------------------------------------
def run_one(
    n_train: int,
    prior_error: float,
    effect_noise: float,
    alpha: float = 0.1,
    eta: float = 0.0,
    seed: int = 0,
    n_samples: int = 2000,
    n_test: int = 200,
) -> Dict[str, float]:
    """Run one batch through CMA, HKSJ, and a fixed-prior baseline.

    Uses a train/holdout split: ``n_train`` trials feed the
    training context and ``n_test`` held-out trials are scored.
    """
    dataset = PMLBMetaAnalysisDataset(
        root=f"./data/pmlb_pe{prior_error}_en{effect_noise}_s{seed}",
        pmlb_dataset_name="1196_BNG_pharynx",
        synthesize_noise=True,
        prior_error=prior_error,
        effect_noise=effect_noise,
        seed=seed,
        n_samples=n_samples,
    )
    samples = dataset.set_task(ConformalMetaAnalysisTask())
    model = ConformalMetaAnalysisModel(
        dataset=samples, alpha=alpha, eta=eta, kernel_type="gaussian",
    )

    # Pull combined batch
    loader = get_dataloader(samples, batch_size=n_train + n_test, shuffle=True)
    batch = next(iter(loader))

    # Unpack tensors
    X_all = batch["features"].cpu().numpy().astype(np.float64)
    Y_all = batch["observed_effect"].cpu().numpy().astype(np.float64).ravel()
    V_all = batch["variance"].cpu().numpy().astype(np.float64).ravel()
    M_all = batch["prior_mean"].cpu().numpy().astype(np.float64).ravel()
    U_all = batch["true_effect"].cpu().numpy().astype(np.float64).ravel()

    # Strict Train/Test Split
    X_train = X_all[:n_train]
    Y_train = Y_all[:n_train]
    V_train = V_all[:n_train]
    M_train = M_all[:n_train]
    X_test = X_all[n_train:]
    M_test = M_all[n_train:]
    U_test = U_all[n_train:]

    lowers = np.zeros(n_test, dtype=np.float64)
    uppers = np.zeros(n_test, dtype=np.float64)

    # Predict explicitly on the holdout set. We call _predict_one
    # directly because ConformalMetaAnalysisModel.forward() uses
    # leave-one-out over the whole batch — for this evaluation we
    # want a clean train/holdout split instead, so the test trials
    # never appear in the training context.
    for i in range(n_test):
        lo, hi = model._predict_one(
            X_train=X_train,
            Y_train=Y_train,
            V_train=V_train,
            M_train=M_train,
            x_test=X_test[i],
            m_test=float(M_test[i]),
        )
        lowers[i] = lo
        uppers[i] = hi

    # Calculate metrics on test bounds against ground truth U_test
    finite = np.isfinite(lowers) & np.isfinite(uppers)
    if finite.any():
        cma_width = float(np.mean(uppers[finite] - lowers[finite]))
    else:
        cma_width = np.nan
    cma_cov = float(np.mean((U_test >= lowers) & (U_test <= uppers)))

    # HKSJ evaluation
    hlo, hhi = hksj_interval(Y_train, V_train, alpha=alpha)
    hksj_width = hhi - hlo
    hksj_cov = float(np.mean((U_test >= hlo) & (U_test <= hhi)))

    # Fixed-prior evaluation
    residuals = Y_train - M_train
    resid_sd = float(np.std(residuals))
    half = t_dist.ppf(1 - alpha / 2, df=n_train - 1) * resid_sd
    prior_width = 2 * half
    prior_cov = float(np.mean(np.abs(U_test - M_test) <= half))

    return {
        "cma_width": cma_width,
        "cma_cov": cma_cov,
        "hksj_width": hksj_width,
        "hksj_cov": hksj_cov,
        "prior_width": prior_width,
        "prior_cov": prior_cov,
    }


# ---------------------------------------------------------------------
# Simulation 1: Width vs n across prior quality
# ---------------------------------------------------------------------
def simulation_1(
    n_values: Iterable[int] = (20, 30, 40, 50),
    seeds: Iterable[int] = range(32),
    ylim: Tuple[float, float] = (0, 2500),
) -> Dict:
    """Reproduce Simulation 1: width vs n for bad/okay/good priors.

    Uses 32 seeds (more than sims 2-4, which use 2) because this
    plot has the smallest effect size between methods and benefits
    from tighter error bars. Runtime dominates the script but stays
    under a few minutes on a CPU.
    """
    priors = {"bad": 3.0, "okay": 0.9, "good": 0.2}
    results = {label: {n: [] for n in n_values} for label in priors}

    for label, pe in priors.items():
        for n in n_values:
            for s in seeds:
                r = run_one(n, pe, effect_noise=0.5, seed=s)
                results[label][n].append(r)
                print(
                    f"[Sim1] prior={label} n={n} seed={s} "
                    f"cma={r['cma_width']:.1f} hksj={r['hksj_width']:.1f}"
                )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (label, pe) in zip(axes, priors.items()):
        n_vals = sorted(results[label].keys())
        cma = [
            np.nanmean([r["cma_width"] for r in results[label][n]])
            for n in n_vals
        ]
        hksj = [
            np.nanmean([r["hksj_width"] for r in results[label][n]])
            for n in n_vals
        ]
        ax.plot(n_vals, cma, "-o", label="CMA", color="crimson")
        ax.plot(n_vals, hksj, "-o", label="HKSJ", color="teal")
        ax.set_xscale("log")
        ax.set_xlabel("n")
        ax.set_title(f"Simulation 1 ({label} prior)")
        ax.set_ylim(*ylim)
        ax.legend()
    axes[0].set_ylabel("interval width")
    plt.tight_layout()
    plt.savefig("simulation_1.png", dpi=150)
    plt.show()
    return results


# ---------------------------------------------------------------------
# Simulation 2: Coverage vs effect noise (CMA vs HKSJ)
# ---------------------------------------------------------------------
def simulation_2(
    noise_values: Iterable[float] = (1.0, 50.0, 100.0, 200.0, 400.0),
    n_values: Iterable[int] = (50, 200),
    seeds: Iterable[int] = (0, 1),
    alpha: float = 0.05,
    ylim: Tuple[float, float] = (0.80, 1.02),
) -> Dict:
    """Reproduce Simulation 2: coverage vs effect noise."""
    # Normalize to a reusable sequence so callers can pass a
    # generator without exhausting it across the len()/zip() calls
    # below.
    n_values = tuple(n_values)
    noise_values = tuple(noise_values)
    results = {
        n: {noise: [] for noise in noise_values} for n in n_values
    }
    for n in n_values:
        for noise in noise_values:
            for s in seeds:
                r = run_one(
                    n, prior_error=0.2, effect_noise=noise,
                    alpha=alpha, seed=s,
                )
                results[n][noise].append(r)
                print(
                    f"[Sim2] n={n} noise={noise} seed={s} "
                    f"cma_cov={r['cma_cov']:.3f} "
                    f"hksj_cov={r['hksj_cov']:.3f}"
                )

    n_panels = len(n_values)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6 * n_panels, 4), sharey=True,
    )
    if n_panels == 1:
        axes = [axes]
    for ax, n in zip(axes, n_values):
        noise_vals = sorted(results[n].keys())
        cma = [
            np.nanmean([r["cma_cov"] for r in results[n][noise]])
            for noise in noise_vals
        ]
        hksj = [
            np.nanmean([r["hksj_cov"] for r in results[n][noise]])
            for noise in noise_vals
        ]
        ax.plot(noise_vals, cma, "-o", label="CMA", color="crimson")
        ax.plot(noise_vals, hksj, "-o", label="HKSJ", color="teal")
        ax.axhline(
            1 - alpha, ls="--", color="black", alpha=0.5,
            label=f"target {1 - alpha:.2f}",
        )
        ax.set_xlabel("effect noise")
        size_label = "small n" if n < 100 else "large n"
        ax.set_title(f"Simulation 2 ({size_label}, n={n})")
        ax.set_ylim(*ylim)
        ax.legend()
    axes[0].set_ylabel("coverage")
    plt.tight_layout()
    plt.savefig("simulation_2.png", dpi=150)
    plt.show()
    return results


# ---------------------------------------------------------------------
# Simulation 3: Coverage with eta=0 vs eta>0
# ---------------------------------------------------------------------
def simulation_3(
    noise_values: Iterable[float] = (1.0, 100.0, 500.0, 1000.0),
    n_values: Iterable[int] = (50, 200),
    seeds: Iterable[int] = (0, 1),
    alpha: float = 0.1,
    ylim: Tuple[float, float] = (0.85, 1.02),
) -> Dict:
    """Reproduce Simulation 3: eta=0 vs eta>0 coverage.

    Compares two instantiations of Algorithm 2 from the paper:
    ``eta=0`` (no noise correction) and ``eta=0.4015`` (the value
    used in the paper to target ~2*alpha confidence loss).
    """
    # Normalize to reusable sequences; see simulation_2 rationale.
    n_values = tuple(n_values)
    noise_values = tuple(noise_values)
    eta_values = {"eta=0": 0.0, "eta>0": 0.4015}

    results = {
        n: {name: {noise: [] for noise in noise_values}
            for name in eta_values}
        for n in n_values
    }
    for n in n_values:
        for name, eta in eta_values.items():
            for noise in noise_values:
                for s in seeds:
                    r = run_one(
                        n, prior_error=0.1, effect_noise=noise,
                        alpha=alpha, eta=eta, seed=s,
                    )
                    results[n][name][noise].append(r)
                    print(
                        f"[Sim3] n={n} {name} noise={noise} "
                        f"cov={r['cma_cov']:.3f}"
                    )

    n_panels = len(n_values)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6 * n_panels, 4), sharey=True,
    )
    if n_panels == 1:
        axes = [axes]
    for ax, n in zip(axes, n_values):
        noise_vals = sorted(noise_values)
        for name, color in [("eta=0", "salmon"), ("eta>0", "purple")]:
            covs = [
                np.nanmean([r["cma_cov"] for r in results[n][name][noise]])
                for noise in noise_vals
            ]
            ax.plot(noise_vals, covs, "-o", label=name, color=color)
        ax.axhline(
            1 - alpha, ls="--", color="black", alpha=0.5,
            label=f"target {1 - alpha:.2f}",
        )
        size_label = "small n" if n < 100 else "large n"
        ax.set_xlabel("effect noise")
        ax.set_title(f"Simulation 3 ({size_label}, n={n})")
        ax.set_ylim(*ylim)
        ax.legend()
    axes[0].set_ylabel("coverage")
    plt.tight_layout()
    plt.savefig("simulation_3.png", dpi=150)
    plt.show()
    return results


# ---------------------------------------------------------------------
# Simulation 4: Width comparison vs prior quality
# ---------------------------------------------------------------------
def simulation_4(
    prior_values: Iterable[float] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    n_values: Iterable[int] = (16, 200),
    seeds: Iterable[int] = (0, 1),
    alpha: float = 0.1,
    ylim: Tuple[float, float] = (0, 5000),
) -> Dict:
    """Reproduce Simulation 4: CMA / HKSJ / Prior widths vs prior error."""
    # Normalize to reusable sequences; see simulation_2 rationale.
    n_values = tuple(n_values)
    prior_values = tuple(prior_values)
    results = {n: {pe: [] for pe in prior_values} for n in n_values}
    for n in n_values:
        for pe in prior_values:
            for s in seeds:
                r = run_one(
                    n, prior_error=pe, effect_noise=0.02,
                    alpha=alpha, seed=s,
                )
                results[n][pe].append(r)
                print(
                    f"[Sim4] n={n} pe={pe} seed={s} "
                    f"cma={r['cma_width']:.1f} "
                    f"hksj={r['hksj_width']:.1f} "
                    f"prior={r['prior_width']:.1f}"
                )

    n_panels = len(n_values)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6 * n_panels, 4), sharey=True,
    )
    if n_panels == 1:
        axes = [axes]
    for ax, n in zip(axes, n_values):
        pe_vals = sorted(results[n].keys())
        cma = [
            np.nanmean([r["cma_width"] for r in results[n][pe]])
            for pe in pe_vals
        ]
        hksj = [
            np.nanmean([r["hksj_width"] for r in results[n][pe]])
            for pe in pe_vals
        ]
        prior = [
            np.nanmean([r["prior_width"] for r in results[n][pe]])
            for pe in pe_vals
        ]
        ax.plot(pe_vals, cma, "-o", label="CMA", color="crimson")
        ax.plot(pe_vals, hksj, "-o", label="HKSJ", color="teal")
        ax.plot(pe_vals, prior, "-o", label="Prior", color="olive")
        size_label = "small n" if n < 100 else "large n"
        ax.set_xlabel("prior error")
        ax.set_title(f"Simulation 4 ({size_label}, n={n})")
        ax.set_ylim(*ylim)
        ax.legend()
    axes[0].set_ylabel("interval width")
    plt.tight_layout()
    plt.savefig("simulation_4.png", dpi=150)
    plt.show()
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Reproducing Kaul & Gordon (2024) Simulations 1-4 on PMLB")
    print("=" * 60)

    print("\n>>> Simulation 1 (width vs n)")
    sim1 = simulation_1()

    print("\n>>> Simulation 2 (coverage vs effect noise)")
    sim2 = simulation_2()

    print("\n>>> Simulation 3 (eta=0 vs eta>0)")
    sim3 = simulation_3()

    print("\n>>> Simulation 4 (width vs prior error)")
    sim4 = simulation_4()

    print("\nDone. Figures saved as simulation_1.png .. simulation_4.png")