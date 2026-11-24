# Contributor: César Gutiérrez Corea (cesarfg2@illinois.edu)
"""METABRIC survival prediction with Survival MDN, with task and model ablations.

Paper: Han, Goldstein and Ranganath. Survival Mixture Density Networks.
MLHC 2022. https://proceedings.mlr.press/v182/han22a.html

Pipeline: ``MetabricDataset`` -> ``MetabricSurvivalPrediction`` (fitted on
the training patients of one of the ten published splits) -> ``SurvivalMDN``
-> concordance, IPCW Brier score and censored test NLL.

Ablations (``--ablation``):

- ``features``: all nine features vs the five clinical features vs the four
  gene-expression values (task-level, new relative to the paper, which uses
  all nine).
- ``components``: number of mixture components K = 1, 5, 10, 20 with all
  features (model-level).
- ``horizon``: administrative censoring of follow-up at 60 and 120 months
  vs full follow-up, evaluated on uncut test labels (task-level).
- ``scale``: sensitivity of PyHealth's ``SurvivalMDN`` to a lower bound on
  the mixture standard deviations (model-level). ``SurvivalMDN`` computes
  ``softplus(raw).clamp_min(min_scale)``; the arms are ``min_scale=1e-8``
  (no effective floor, the PyHealth default) and ``min_scale=softplus(0.01)``
  (about 0.698). The second arm matches the floor of the authors' released
  code, ``softplus(clamp(raw, 0.01, 100))`` in ``mdn_models.py``: for raw
  values below 100 it gives the same scales and the same zero gradient below
  the floor, but without the upper cap of 100. Neither arm reproduces the
  ``exp`` parameterization the paper describes (Sec. 5.4), so this measures
  how much the floor matters in PyHealth, not paper versus code.

``--paper-metrics`` also scores each run with the paper's metrics: the
IPCW time-dependent concordance C^td, the integrated Brier score (IBS) and
the integrated binomial log-likelihood (IBLL), at the near-untruncated
horizon (censoring-survival threshold 1e-8 for C^td). It reuses
``paper_metrics`` from ``examples/survival_mdn_metrics.py`` (PR #1279), which
keeps the authors' evaluation conventions, so METABRIC is scored the same
way as the SUPPORT reproduction there. Run the script from the repository
root so that module is importable.

Every configuration uses the same splits, seeds, optimizer and stopping rule,
so only the ablated setting changes.

Run on synthetic data (seconds, demonstrates execution only)::

    python examples/metabric_survival_survival_mdn.py --synthetic

Run on the real data (downloads the Survival MDN split files once)::

    python examples/metabric_survival_survival_mdn.py --root ~/data/metabric \\
        --download --ablation features components --splits 1 2 3 4 5 6 7 8 9 10

The "Findings" section below reports results on the real data. Synthetic
numbers carry no meaning.

Findings (real METABRIC, ten published splits, one seed, no tuning):
    Hidden width 32, three layers, RMSprop lr 1e-3, batch 128, early stopping
    on validation NLL (patience 10). Mean ± standard error over splits.
    C is Harrell's concordance of 1 - S(median event time), used to compare
    the ablation arms with each other. It is not the paper's metric; see
    "Paper comparison" below for C^td.

    =========  ========  ==  =======  =============  =============
    ablation   features  K   horizon  C-index        mean Brier
    =========  ========  ==  =======  =============  =============
    features   all       10  full     0.643 ± 0.004  0.179 ± 0.002
    features   clinical  10  full     0.628 ± 0.006  0.182 ± 0.002
    features   genes     10  full     0.595 ± 0.004  0.189 ± 0.002
    K          all       1   full     0.635 ± 0.004  0.184 ± 0.002
    K          all       5   full     0.641 ± 0.004  0.180 ± 0.002
    K          all       20  full     0.645 ± 0.004  0.178 ± 0.002
    horizon    all       10  120 mo   0.642 ± 0.004  0.180 ± 0.003
    horizon    all       10  60 mo    0.636 ± 0.005  0.183 ± 0.003
    =========  ========  ==  =======  =============  =============

    - Clinical features carry most of the signal; genes alone lose about
      0.05 C, and the two together beat either.
    - One component (a single Gaussian after softplus) is clearly worse;
      beyond five components the gains are within one standard error.
    - Cutting follow-up at 60 months costs little ranking accuracy on the
      full-follow-up test labels.

    Scale floor (all features, K = 10, ten splits x three seeds, paired by
    split and seed; differences are softplus(0.01) minus 1e-8, with standard
    errors over the ten split-level means):

    ==============  =====  =====  ==========  ==========
    min_scale       C      Brier  test NLL    best epoch
    ==============  =====  =====  ==========  ==========
    1e-8            0.643  0.179  0.141       8.6
    softplus(0.01)  0.645  0.178  0.135       9.9
    ==============  =====  =====  ==========  ==========

    - The floor helps slightly: C +0.002 (SE 0.001, higher on 8 of 10
      splits) and test NLL -0.006 (SE 0.002, lower on 9 of 10 splits), with
      no change in Brier score (-0.0004, SE 0.0004).
    - The effect is small next to the feature-set and K = 1 effects. This
      is a sensitivity analysis within PyHealth's parameterization; it does
      not test the paper's ``exp`` scales.

    Paper comparison (``--paper-metrics``, near-untruncated horizon with a
    censoring-survival threshold of 1e-8 for C^td, all features, K = 10, ten
    splits, one seed, no hyperparameter search; mean ± standard error):

    ========  =============  =============  ==============
    features  C^td           IBS            IBLL
    ========  =============  =============  ==============
    all       0.654 ± 0.008  0.169 ± 0.003  -0.501 ± 0.008
    clinical  0.642 ± 0.007  0.169 ± 0.002  -0.502 ± 0.006
    genes     0.581 ± 0.006  0.185 ± 0.002  -0.545 ± 0.004
    ========  =============  =============  ==============

    - With the same metric, the untuned baseline's C^td of 0.654 is 0.013
      below the paper's 0.667 ± 0.004 for Survival MDN on METABRIC
      (Table 3), which used a 100-trial hyperparameter search per dataset.
    - The feature-set ranking is the same under C^td as under Harrell's C.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import MetabricDataset
from pyhealth.metrics.survival import concordance_index, ipcw_brier_score
from pyhealth.models import SurvivalMDN
from pyhealth.tasks import MetabricSurvivalPrediction

#: Scale floor of the authors' released code: softplus(clamp(raw, 0.01, 100)).
CODE_SCALE_FLOOR = math.log1p(math.exp(0.01))


@dataclass
class Part:
    """Dense tensors for one part of a split."""

    x: torch.Tensor
    duration: torch.Tensor
    event: torch.Tensor


def to_part(samples) -> Part:
    """Stacks a processed ``SampleDataset`` into tensors."""
    items = [samples[i] for i in range(len(samples))]
    return Part(
        x=torch.stack([item["features"] for item in items]).float(),
        duration=torch.cat([item["duration"].reshape(-1) for item in items]).float(),
        event=torch.cat([item["event"].reshape(-1) for item in items]).float(),
    )


def write_synthetic(root: Path, n: int = 120, seed: int = 0) -> None:
    """Writes synthetic Survival MDN style split files (for the smoke run).

    Event times depend on age, MKI67 and ER status so that the ablations
    have something to find; nothing here resembles real METABRIC values.
    """
    rng = np.random.default_rng(seed)
    genes = rng.normal(6.0, 1.0, size=(n, 4))
    age = rng.uniform(30, 90, size=(n, 1))
    binary = rng.integers(0, 2, size=(n, 4)).astype(float)
    raw = np.hstack([genes, age, binary]).astype(np.float32)
    hazard = np.exp(
        0.03 * (age[:, 0] - 60) + 0.5 * (genes[:, 0] - 6) - 0.5 * binary[:, 3]
    )
    event_months = rng.exponential(120.0 / hazard)
    censor_months = rng.uniform(1, 300, size=n)
    months = np.minimum(event_months, censor_months)
    event = (event_months <= censor_months).astype(np.float32)
    labels = np.stack([months / 365.25, event], axis=1).astype(np.float32)
    for split_id in range(1, 11):
        order = np.random.default_rng(split_id).permutation(n)
        cut1, cut2 = int(0.6 * n), int(0.8 * n)
        for part, rows in zip(
            ("train", "valid", "test"), (order[:cut1], order[cut1:cut2], order[cut2:])
        ):
            np.savez(
                root / f"{part}_{split_id}.npz", raw[rows], labels[rows], raw[rows]
            )


def censored_nll(model: SurvivalMDN, part: Part) -> float:
    """Mean censored negative log-likelihood of ``part``."""
    model.eval()
    with torch.no_grad():
        params = model.network.mixture_parameters(part.x)
        return float(model.network.censored_nll(params, part.duration, part.event))


def train_model(
    train_samples,
    train: Part,
    valid: Part,
    num_components: int,
    seed: int,
    args,
    min_scale: float = 1e-8,
) -> tuple[SurvivalMDN, int, float]:
    """Trains one Survival MDN with early stopping on validation NLL.

    ``min_scale`` is the lower bound on the mixture standard deviations.

    Returns:
        The best model, its epoch and the training time in seconds.
    """
    torch.manual_seed(seed)
    grid_max = float(train.duration.max()) * 1.05
    model = SurvivalMDN(
        train_samples,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_layers,
        num_components=num_components,
        min_scale=min_scale,
        time_grid=torch.linspace(grid_max / 200, grid_max, 200),
    )
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(seed)
    batch = min(args.batch_size, len(train.x))
    best, best_epoch, best_state, waited = math.inf, 0, None, 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train.x))
        # Drop an incomplete last batch: BatchNorm cannot train on one sample.
        for begin in range(0, len(order) - batch + 1, batch):
            idx = torch.as_tensor(order[begin : begin + batch])
            optimizer.zero_grad()
            loss = model(
                features=train.x[idx],
                duration=train.duration[idx],
                event=train.event[idx],
            )["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()
        valid_nll = censored_nll(model, valid)
        if valid_nll < best:
            best, best_epoch, waited = valid_nll, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, best_epoch, time.perf_counter() - start


def evaluate(model: SurvivalMDN, train: Part, test: Part) -> dict[str, float]:
    """Test metrics at the 25th, 50th and 75th percentiles of training event times.

    ``train`` must carry the full follow-up, so that the horizons do not move
    with the administrative censoring ablation.

    Risk for concordance is ``1 - S(t | x)`` at the median horizon, a
    per-horizon ranking as in the time-dependent concordance of the paper;
    Harrell's C is reported here, so values are close to but not the same as
    the paper's censoring-weighted C^td.
    """
    event_times = train.duration[train.event == 1].numpy()
    horizons = np.quantile(event_times, [0.25, 0.5, 0.75])
    survival = model.predict_survival(
        test.x, torch.as_tensor(horizons, dtype=torch.float32)
    ).numpy()
    duration, event = test.duration.numpy(), test.event.numpy()
    out = {
        "c_index": concordance_index(duration, event, 1.0 - survival[:, 1]),
        "test_nll": censored_nll(model, test),
    }
    briers = [
        ipcw_brier_score(duration, event, survival[:, k], float(t))
        for k, t in enumerate(horizons)
    ]
    for name, value in zip(("brier_q25", "brier_q50", "brier_q75"), briers):
        out[name] = value
    out["brier_mean"] = float(np.mean(briers))
    return out


def paper_scores(model: SurvivalMDN, test: Part) -> dict[str, float]:
    """The paper's C^td, IBS and IBLL on the test part, near-untruncated.

    Uses the evaluator of the SUPPORT reproduction (PR #1279), which keeps the
    authors' conventions, so the numbers are comparable with Table 3.
    """
    from survival_mdn_metrics import paper_metrics  # examples/, PR #1279

    scores = paper_metrics(model, test)
    return {"c_td": scores["c"], "ibs": scores["ibs"], "ibll": scores["ibll"]}


def configurations(ablations: list[str]) -> list[dict]:
    """Lists the ablation arms; each changes one setting from the baseline."""
    arms = []
    if "features" in ablations:
        for name in ("all", "clinical", "genes"):
            arms.append({"ablation": "features", "feature_set": name, "k": 10})
    if "components" in ablations:
        for k in (1, 5, 10, 20):
            arms.append({"ablation": "components", "feature_set": "all", "k": k})
    if "horizon" in ablations:
        for horizon in (None, 120.0, 60.0):
            arms.append(
                {
                    "ablation": "horizon",
                    "feature_set": "all",
                    "k": 10,
                    "horizon": horizon,
                }
            )
    if "scale" in ablations:
        for min_scale in (1e-8, CODE_SCALE_FLOOR):
            arms.append(
                {
                    "ablation": "scale",
                    "feature_set": "all",
                    "k": 10,
                    "min_scale": min_scale,
                }
            )
    return arms


def run(args: argparse.Namespace) -> list[dict]:
    """Runs every configuration on every split and seed."""
    dataset = MetabricDataset(root=str(args.root), download=args.download)
    results = []
    for arm in configurations(args.ablation):
        horizon = arm.get("horizon")
        task = MetabricSurvivalPrediction(
            feature_set=arm["feature_set"], time_unit="paper", horizon_months=horizon
        )
        # Test labels always come from the full follow-up.
        full_task = MetabricSurvivalPrediction(
            feature_set=arm["feature_set"], time_unit="paper"
        )
        for split_id in args.splits:
            split = dataset.paper_split(split_id)
            train_samples, valid_samples, _ = dataset.set_task(task, split=split)
            full_train_samples, _, test_samples = dataset.set_task(
                full_task, split=split
            )
            full_train = to_part(full_train_samples)
            train, valid, test = (
                to_part(train_samples),
                to_part(valid_samples),
                to_part(test_samples),
            )
            for seed in args.seeds:
                model, epoch, seconds = train_model(
                    train_samples,
                    train,
                    valid,
                    arm["k"],
                    seed,
                    args,
                    min_scale=arm.get("min_scale", 1e-8),
                )
                row = {
                    "ablation": arm["ablation"],
                    "feature_set": arm["feature_set"],
                    "num_components": arm["k"],
                    "horizon_months": "full" if horizon is None else horizon,
                    "min_scale": arm.get("min_scale", 1e-8),
                    "split": split_id,
                    "seed": seed,
                    "best_epoch": epoch,
                    "train_seconds": round(seconds, 2),
                    **evaluate(model, full_train, test),
                    **(paper_scores(model, test) if args.paper_metrics else {}),
                }
                results.append(row)
                print(
                    f"{row['ablation']:10s} features={row['feature_set']:8s} "
                    f"K={row['num_components']:2d} "
                    f"horizon={row['horizon_months']!s:5s} "
                    f"min_scale={row['min_scale']:g} "
                    f"split={split_id} seed={seed} C={row['c_index']:.3f} "
                    f"IBS~{row['brier_mean']:.3f} NLL={row['test_nll']:.3f}"
                )
    return results


def summarize(results: list[dict]) -> None:
    """Prints, for each arm, the mean and standard error over split-level means.

    Seeds are averaged within each split first, since runs on the same split
    share patients and are not independent.
    """
    keys = (
        "ablation",
        "feature_set",
        "num_components",
        "horizon_months",
        "min_scale",
    )
    groups: dict[tuple, list[dict]] = {}
    for row in results:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    metrics = ["c_index", "brier_mean", "test_nll"]
    header = "C-index        mean Brier     test NLL"
    if "c_td" in results[0]:
        metrics += ["c_td", "ibs", "ibll"]
        header += "       C^td           IBS            IBLL"
    print("\nablation   features  K   horizon  min_scale  " + header)
    for group, rows in groups.items():
        cells = []
        for metric in metrics:
            # Seeds on the same split share patients: average them first, so
            # the standard error is over split-level means.
            split_values: dict[int, list[float]] = {}
            for row in rows:
                split_values.setdefault(row["split"], []).append(row[metric])
            values = np.array([np.mean(v) for v in split_values.values()])
            se = values.std(ddof=1) / math.sqrt(len(values)) if len(values) > 1 else 0
            cells.append(f"{values.mean():.3f} ± {se:.3f}")
        print(
            f"{group[0]:10s} {group[1]:9s} {group[2]:<3d} {group[3]!s:8s} "
            f"{group[4]:<10.3g} " + "  ".join(cells)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, help="Folder for metabric.csv")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use generated data in a temporary folder",
    )
    parser.add_argument(
        "--ablation",
        nargs="+",
        default=["features", "components"],
        choices=["features", "components", "horizon", "scale"],
    )
    parser.add_argument("--splits", nargs="+", type=int, default=[1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument(
        "--paper-metrics",
        action="store_true",
        help="Also report the paper's C^td, IBS and IBLL (slower)",
    )
    parser.add_argument("--output", type=Path, help="Optional CSV of all runs")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        if args.synthetic:
            if args.download or args.root is not None:
                parser.error("--synthetic cannot be combined with --root or --download")
            args.root = Path(tmp)
            write_synthetic(args.root)
            args.epochs = min(args.epochs, 5)
            args.batch_size = min(args.batch_size, 32)
        elif args.root is None:
            parser.error("give --root (and --download the first time) or --synthetic")
        results = run(args)
    summarize(results)
    if args.output is not None:
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote {len(results)} runs to {args.output}")


if __name__ == "__main__":
    main()
