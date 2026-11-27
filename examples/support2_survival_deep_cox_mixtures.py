"""SUPPORT2 survival with Deep Cox Mixtures: mixture-size ablation.

Contributor: <AUTHOR NAME> (NetID: liu323)
Paper: Nagpal, Yadlowsky, Rostamzadeh and Heller. Deep Cox Mixtures for
Survival Regression. MLHC 2021. https://proceedings.mlr.press/v149/nagpal21a.html

Pipeline
--------
``Support2Dataset`` -> ``SurvivalTimeToEventSupport2DCM`` (9,105 patients, 44
features) -> mean imputation and standardization fit on training patients ->
``DeepCoxMixtures.fit`` (Monte Carlo EM) -> survival at the paper's horizons
(14, 58 and 252 days: the 25th/50th/75th event-time quantiles) -> the paper's
metrics: KM-adjusted expected calibration error (ECE, 20 equal-frequency bins,
out-of-fold predictions pooled), IPCW Brier score, IPCW time-dependent
concordance (Ctd, truncated at the horizon) and cumulative/dynamic AUC, the
last three averaged over folds with censoring weights from each fold's
training patients.

Protocol: the paper's 5 outer folds (row order, ``fold = (sno - 1) % 5``); in
each, a random 15% of the training patients is used for early stopping. The
model uses the paper's settings (Adam, learning rate 1e-3, batch size 128, up
to 50 epochs, SELU encoder without bias). Validation and test patients are
never used to fit preprocessing, baselines or weights.

Ablation
--------
The number of latent groups ``K`` is varied with everything else fixed
(encoder ``[100, 100]``, same folds, seed and splits). ``K = 1`` is a neural
Cox model with a Breslow baseline, so the comparison isolates the mixture.
``--train-censoring`` adds the project's extension: extra independent
exponential censoring of the *training* patients only (validation and test
unchanged), with the rate solved so that the training censoring rate rises by
the given number of percentage points.

Usage
-----
The paper's metrics use scikit-survival (``pip install scikit-survival``),
which this example needs in both modes.

Synthetic data (a two-group population with crossing survival curves, so a
mixture should help; under a minute on a CPU)::

    python examples/support2_survival_deep_cox_mixtures.py --synthetic

SUPPORT2 (``root`` holds ``support2.csv`` with an ``sno`` column; the UCI file
https://archive.ics.uci.edu/dataset/880/support2 calls it ``id``)::

    python examples/support2_survival_deep_cox_mixtures.py --root data/support2
    python examples/support2_survival_deep_cox_mixtures.py --root data/support2 \
        --components 1,3 --train-censoring 0,10,20,30

Measured results (SUPPORT2, 5 folds, seed 0, CPU, about 1-2 minutes per run)
----------------------------------------------------------------------------
K ablation (no extra censoring), all patients | non-White:

    K   ECE@252  Ctd@252  Brier@252  Ctd@58 | ECE@252  Ctd@252
    1   0.0173   0.6646   0.2146     0.6665 | 0.0419   0.6887
    2   0.0214   0.6717   0.2131     0.7053 | 0.0519   0.6990
    3   0.0142   0.6703   0.2129     0.7041 | 0.0400   0.6915
    6   0.0151   0.6733   0.2124     0.7063 | 0.0376   0.6977

Extra training censoring (one censoring draw), all patients:

    +pp  K=1 ECE@252  K=3 ECE@252  K=1 Brier@252  K=3 Brier@252
      0  0.0173       0.0142       0.2146         0.2129
     10  0.0201       0.0123       0.2136         0.2130
     20  0.0182       0.0175       0.2144         0.2130
     30  0.0192       0.0159       0.2171         0.2144

Findings: mixtures (K >= 2) discriminate better than the single-component
neural Cox model, most clearly at 58 days (Ctd 0.667 -> 0.704-0.706) and less
at 252 days (0.665 -> 0.670-0.673), with a slightly lower Brier score.
Calibration differences between K values are small and not consistent across
K (ECE@252 0.014-0.021), so this run does not show that more components
improve calibration. With extra training censoring, K = 3 kept an ECE@252
no higher than K = 1 at every level in this single draw. In the project's
full study (3 seeds / 3 censoring draws, paired bootstrap), the
discrimination gain of K >= 2 over K = 1 was significant, while ECE
differences were within sampling noise. The synthetic run (two groups with
crossing curves) shows the expected calibration gain of K >= 2
(ECE@252 0.056 for K = 1 vs 0.029-0.033).
"""

import argparse
import itertools
import json
import math
import time

import numpy as np
import torch

from pyhealth.models import DeepCoxMixtures

HORIZONS = (14.0, 58.0, 252.0)


def load_support2(root: str) -> dict[str, np.ndarray]:
    """Load SUPPORT2 through PyHealth's dataset and the DCM-protocol task.

    Args:
        root: Folder containing ``support2.csv``.

    Returns:
        Raw features (NaN where missing), durations, events, race and the
        paper's outer fold of every patient, in ``sno`` order.
    """
    from pyhealth.datasets import Support2Dataset
    from pyhealth.tasks import SurvivalTimeToEventSupport2DCM

    dataset = Support2Dataset(root=root, tables=["support2"])
    samples = dataset.set_task(SurvivalTimeToEventSupport2DCM(), num_workers=1)
    rows = sorted(
        (samples[i] for i in range(len(samples))), key=lambda s: int(s["patient_id"])
    )
    sno = np.array([int(s["patient_id"]) for s in rows])
    return {
        "x": torch.stack([s["features"] for s in rows]).double().numpy(),
        "t": torch.cat([s["duration"] for s in rows]).double().numpy(),
        "e": torch.cat([s["event"] for s in rows]).double().numpy(),
        "race": np.array([s["race"] for s in rows]),
        "folds": (sno - 1) % 5,
    }


def synthetic(n: int = 3000, seed: int = 0) -> dict[str, np.ndarray]:
    """Two latent groups with crossing survival curves (non-PH overall).

    Args:
        n: Number of patients.
        seed: Random seed.

    Returns:
        The same fields as :func:`load_support2`, times in arbitrary units.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    early = x[:, 0] > 0
    shape = np.where(early, 0.5, 3.0)
    scale = np.where(early, 60.0, 200.0)
    t_event = scale * (-np.log(rng.uniform(size=n)) / np.exp(0.7 * x[:, 1])) ** (
        1 / shape
    )
    t_cens = rng.exponential(300.0, size=n)
    return {
        "x": x,
        "t": np.minimum(t_event, t_cens) + 1e-3,
        "e": (t_event <= t_cens).astype(float),
        "race": np.where(x[:, 2] > 1, "other", "white"),
        "folds": np.arange(n) % 5,
    }


def standardize(x: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Mean-impute then standardize, with statistics from ``fit`` rows only.

    Args:
        x: Raw features, NaN where missing.
        fit: Boolean mask of the rows used to estimate the statistics.

    Returns:
        Float32 features with no NaN.
    """
    mean = np.nanmean(x[fit], axis=0)
    x = np.where(np.isnan(x), mean, x)
    scale = x[fit].std(axis=0)
    scale[scale == 0] = 1.0
    return ((x - mean) / scale).astype(np.float32)


def extra_censoring(
    t: np.ndarray, e: np.ndarray, points: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Add independent exponential censoring to raise censoring by ``points``.

    The rate solves ``mean(e * exp(-rate * t)) = 1 - (base + points / 100)``.

    Args:
        t: Training durations.
        e: Training event indicators.
        points: Extra censoring in percentage points (0 returns the input).
        rng: Random generator for the censoring times.

    Returns:
        ``(t', e')`` with ``t' = min(t, C)`` and ``e' = e * 1[t <= C]``.
    """
    if points == 0:
        return t, e
    target = 1.0 - (1.0 - e.mean() + points / 100.0)
    lo, hi = 0.0, 1.0
    while np.mean(e * np.exp(-hi * t)) > target:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.mean(e * np.exp(-mid * t)) > target else (lo, mid)
    c = rng.exponential(1.0 / hi, size=len(t))
    return np.minimum(t, c), e * (t <= c)


def km_survival(t: np.ndarray, e: np.ndarray, horizon: float) -> float:
    """Kaplan-Meier survival probability at ``horizon``."""
    taus, deaths = np.unique(t[(e == 1) & (t <= horizon)], return_counts=True)
    at_risk = len(t) - np.searchsorted(np.sort(t), taus, side="left")
    return float(np.prod(1.0 - deaths / at_risk)) if len(taus) else 1.0


def ece(
    surv: np.ndarray, t: np.ndarray, e: np.ndarray, horizon: float, bins: int = 20
) -> float:
    """The paper's KM-adjusted ECE at one horizon (equal-frequency bins)."""
    edges = np.quantile(surv, np.linspace(0, 1, bins + 1))
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        member = (surv >= lo) & (surv <= hi)
        total += member.mean() * abs(
            surv[member].mean() - km_survival(t[member], e[member], horizon)
        )
    return float(total)


def evaluate(surv: np.ndarray, data: dict, subset: np.ndarray) -> dict:
    """Paper-protocol metrics for out-of-fold predictions on a subset.

    Args:
        surv: Out-of-fold survival at ``HORIZONS``, shape ``(n, 3)``.
        data: Output of :func:`load_support2`.
        subset: Boolean mask of the patients to score.

    Returns:
        ``{"ece@252": ..., "ctd@252": ..., ...}`` for every horizon.
    """
    from sksurv.metrics import (
        brier_score,
        concordance_index_ipcw,
        cumulative_dynamic_auc,
    )
    from sksurv.util import Surv

    t, e, folds = data["t"], data["e"], data["folds"]
    out = {}
    for j, h in enumerate(HORIZONS):
        name = f"@{h:g}"
        out["ece" + name] = ece(surv[subset, j], t[subset], e[subset], h)
        scores = {"brier": [], "ctd": [], "auc": []}
        for f in np.unique(folds[subset]):
            train = (folds != f) & subset
            test = (folds == f) & subset & (t <= t[train].max())
            y_tr = Surv.from_arrays(e[train].astype(bool), t[train])
            y_te = Surv.from_arrays(e[test].astype(bool), t[test])
            s = surv[test, j]
            scores["brier"].append(brier_score(y_tr, y_te, s, h)[1][0])
            scores["ctd"].append(concordance_index_ipcw(y_tr, y_te, 1 - s, tau=h)[0])
            scores["auc"].append(cumulative_dynamic_auc(y_tr, y_te, 1 - s, [h])[0][0])
        out.update({k + name: float(np.mean(v)) for k, v in scores.items()})
    return out


def cross_validate(
    data: dict, k: int, hidden: tuple[int, ...], args, censoring_points: float
) -> np.ndarray:
    """Out-of-fold survival predictions of one configuration.

    Args:
        data: Output of :func:`load_support2` or :func:`synthetic`.
        k: Number of mixture components.
        hidden: Encoder hidden layer widths.
        args: Parsed command-line arguments.
        censoring_points: Extra training censoring (percentage points).

    Returns:
        Survival at ``HORIZONS`` for every patient, shape ``(n, 3)``.
    """
    n = len(data["t"])
    surv = np.zeros((n, len(HORIZONS)))
    for f in range(args.folds):
        outer = np.flatnonzero(data["folds"] != f)
        rng = np.random.default_rng(10_000 + f)
        val = np.zeros(n, bool)
        val[rng.choice(outer, size=int(0.15 * len(outer)), replace=False)] = True
        train = (data["folds"] != f) & ~val
        test = data["folds"] == f
        x = standardize(data["x"], train)
        t_tr, e_tr = extra_censoring(
            data["t"][train],
            data["e"][train],
            censoring_points,
            np.random.default_rng(1_000 + f),
        )
        torch.manual_seed(args.seed)
        model = DeepCoxMixtures(
            None, input_dim=x.shape[1], num_components=k, hidden_dims=hidden
        )
        model.fit(
            (x[train], t_tr, e_tr),
            (x[val], data["t"][val], data["e"][val]),
            epochs=args.epochs,
            batch_size=128,
            learning_rate=1e-3,
            patience=3,
            seed=args.seed,
        )
        surv[test] = model.predict_survival(torch.from_numpy(x[test]), HORIZONS).numpy()
    return surv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", help="folder with support2.csv")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--components", default="1,2,3,6")
    parser.add_argument("--hidden", default="100,100")
    parser.add_argument(
        "--train-censoring",
        default="0",
        help="extra training censoring, percentage points",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="outer folds to run (5 for the full protocol)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="dcm_support2_results.json")
    args = parser.parse_args()
    if not args.synthetic and not args.root:
        parser.error("pass --root DIR or --synthetic")
    data = synthetic() if args.synthetic else load_support2(args.root)
    hidden = tuple(int(h) for h in args.hidden.split(",") if h)
    scored = np.isin(data["folds"], np.arange(args.folds))
    groups = {"overall": scored, "non-White": scored & (data["race"] != "white")}
    results = []
    for points in (float(p) for p in args.train_censoring.split(",")):
        for k in (int(c) for c in args.components.split(",")):
            start = time.time()
            surv = cross_validate(data, k, hidden, args, points)
            for name, subset in groups.items():
                metrics = evaluate(surv, data, subset)
                results.append(
                    {
                        "K": k,
                        "extra_censoring_pp": points,
                        "group": name,
                        **metrics,
                        "seconds": round(time.time() - start, 1),
                    }
                )
                print(
                    f"K={k} +{points:g}pp {name:9s} "
                    + "  ".join(
                        f"{m}={metrics[m]:.4f}"
                        for m in ("ece@252", "ctd@252", "brier@252", "auc@252")
                    )
                )
    with open(args.output, "w") as stream:
        json.dump(results, stream, indent=1)
    print(f"wrote {args.output}")
    assert all(math.isfinite(r["ece@252"]) for r in results)


if __name__ == "__main__":
    main()
