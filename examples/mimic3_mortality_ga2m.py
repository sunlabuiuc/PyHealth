"""GA2M on 48h MIMIC-III in-hospital mortality: the experiments and
ablations of Hegselmann et al. (2020).

Stages (``--stages``, default all):

- ``bins``: bin-count ablation, 1D-only models at 32/64/128/256 bins.
- ``pairs``: the paper's pair selection. Train all 561 pairs at coarse bins,
  rank by contribution variance on the training split, keep the top 34,
  and retrain.
- ``pruning``: retrain without the terms clinicians flagged (pH risk
  curves, interactions involving standard-deviation features).
- ``lr``: logistic-regression baseline on the same 34 features.

Every GA2M is trained with Adam from the same seed, with a smoothness
penalty on its risk curves (``--smoothness``), and keeps the weights from
the epoch with the best validation ROC-AUC (early stopping).

Synthetic data (default, no MIMIC-III access needed)::

    python examples/mimic3_mortality_ga2m.py

Real MIMIC-III v1.4 (needs PhysioNet credentialing)::

    python examples/mimic3_mortality_ga2m.py --root path/to/mimic-iii/1.4

The synthetic cohort comes from one latent "severity" value per patient
that drives the 17 clinical variables (with missingness and noise) and the
label. It only shows the pipeline runs end to end; it says nothing about
real MIMIC-III. Real-data results are in
``examples/mimic3_mortality_ga2m_results.md``.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader, split_by_patient
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import GA2M
from pyhealth.models.ga2m import all_pairs, rank_pairs_by_contribution_variance
from pyhealth.tasks import InHospitalMortality48hMIMIC3
from pyhealth.tasks.in_hospital_mortality_48h_mimic3 import FEATURE_NAMES, MISSING

N_VARIABLES = len(FEATURE_NAMES) // 2
STAGES = ("bins", "pairs", "pruning", "lr")

#: Terms the paper's clinicians flagged as hard to interpret or implausible:
#: the pH risk curves, and interactions involving a standard-deviation
#: feature. The pruning ablation retrains without them.
PH_FEATURES = [FEATURE_NAMES.index("ph_mean"), FEATURE_NAMES.index("ph_std")]
STD_FEATURES = {i for i, name in enumerate(FEATURE_NAMES) if name.endswith("_std")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root", default=None, help="MIMIC-III folder (1.4); omit for synthetic data"
    )
    parser.add_argument("--cache-dir", default=None, help="Folder for processed data")
    parser.add_argument("--n-patients", type=int, default=600, help="Synthetic only")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument(
        "--smoothness", type=float, default=100.0,
        help="Weight of GA2M.smoothness_penalty() in the training loss",
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early-stopping patience in epochs, on validation ROC-AUC",
    )
    parser.add_argument(
        "--bins", type=int, nargs="+", default=[32, 64, 128, 256],
        help="Bin counts to ablate over (paper: 32, 64, 128, 256)",
    )
    parser.add_argument(
        "--coarse-bins", type=int, default=8,
        help="Bin count for the all-pairs pair-selection phase",
    )
    parser.add_argument("--n-pairs", type=int, default=34, help="Pairs to keep")
    parser.add_argument(
        "--pair-bins", type=int, default=8,
        help="Bins per feature in the final models' 2D interaction tables",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stages", nargs="+", choices=STAGES, default=list(STAGES),
        help="Which experiments to run (default: all)",
    )
    return parser.parse_args()


def make_synthetic_samples(n_patients: int, seed: int) -> list[dict]:
    """A synthetic cohort shaped like InHospitalMortality48hMIMIC3 output.

    Each patient has a latent "severity" in [0, 1] that shifts every
    variable's mean upward and its mortality probability upward, so the
    model has a real (if simplistic) signal to learn. ~15% of variable
    readings are missing per sample (:data:`MISSING`, ``-1.0``), mirroring
    real ICU data sparsity.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n_patients):
        severity = rng.uniform(0, 1)
        means = 10.0 * severity + rng.normal(0, 1, size=N_VARIABLES)
        stds = np.abs(rng.normal(1, 0.3, size=N_VARIABLES))
        missing = rng.uniform(size=N_VARIABLES) < 0.15
        means[missing] = MISSING
        stds[missing] = MISSING
        mortality = int(rng.uniform() < 0.1 + 0.6 * severity)
        samples.append(
            {
                "patient_id": f"synthetic-{i}",
                "icustay_id": i,
                "features": list(means) + list(stds),
                "mortality": mortality,
            }
        )
    return samples


def load_splits(args: argparse.Namespace):
    """Returns (train, val, test) SampleDataset splits."""
    if args.root:
        from pyhealth.datasets import MIMIC3Dataset

        dataset = MIMIC3Dataset(
            root=args.root,
            tables=["chartevents", "labevents"],
            cache_dir=args.cache_dir,
        )
        sample_dataset = dataset.set_task(InHospitalMortality48hMIMIC3())
    else:
        samples = make_synthetic_samples(args.n_patients, args.seed)
        sample_dataset = create_sample_dataset(
            samples=samples,
            input_schema={"features": "tensor"},
            output_schema={"mortality": "binary"},
            dataset_name="mimic3_mortality_ga2m_synthetic",
        )
    return split_by_patient(sample_dataset, [0.7, 0.1, 0.2], seed=args.seed)


def train(model: GA2M, train_data, args: argparse.Namespace, val_data=None,
          epochs: int | None = None) -> float:
    """Adam on the training split. With ``val_data``, keeps the weights from
    the epoch with the best validation ROC-AUC and stops after
    ``args.patience`` epochs without improvement (early stopping)."""
    optimizer_cls = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    optimizer = optimizer_cls[args.optimizer](
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loader = get_dataloader(train_data, batch_size=args.batch_size, shuffle=True)
    best_score, best_state, stale = -1.0, None, 0
    for _ in range(epochs or args.epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            if args.smoothness:
                loss = loss + args.smoothness * model.smoothness_penalty()
            loss.backward()
            optimizer.step()
        if val_data is None:
            continue
        score = evaluate(model, val_data, args.batch_size)["roc_auc"]
        if score > best_score:
            best_score, stale = score, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_score


def evaluate(model: GA2M, data, batch_size: int) -> dict:
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in get_dataloader(data, batch_size=batch_size):
            output = model(**batch)
            y_true.append(output["y_true"].cpu().numpy())
            y_prob.append(output["y_prob"].cpu().numpy())
    y_true = np.concatenate(y_true).ravel()
    y_prob = np.concatenate(y_prob).ravel()
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return binary_metrics_fn(y_true, y_prob, metrics=["roc_auc", "pr_auc"])


def select_top_pairs(
    train_data, val_data, args: argparse.Namespace, n_features: int
) -> list[tuple[int, int]]:
    """The paper's pair-selection procedure: train every candidate pair at
    coarse bins, rank by contribution variance on train, keep the top
    ``args.n_pairs``."""
    candidates = all_pairs(n_features)
    # dataset=train_data: __init__'s auto-fit_bins already uses the train
    # split only, matching the "quantile bin edges from train split only"
    # design (fit_bins is still callable again later, e.g. to re-fit after
    # changing the data, but isn't needed here).
    coarse = GA2M(dataset=train_data, n_bins=args.coarse_bins, pairs=candidates)
    train(coarse, train_data, args, val_data, epochs=max(args.epochs // 3, 1))
    ranked = rank_pairs_by_contribution_variance(
        coarse, get_dataloader(train_data, batch_size=args.batch_size, shuffle=False)
    )
    return [pair for pair, _ in ranked[: args.n_pairs]]


def fit_and_evaluate(args, splits, label, **model_kwargs) -> None:
    """Trains one GA2M from a fixed seed (so results do not depend on which
    stages ran before) and prints its test metrics."""
    train_data, val_data, test_data = splits
    torch.manual_seed(args.seed)
    model = GA2M(dataset=train_data, **model_kwargs)
    val_roc_auc = train(model, train_data, args, val_data)
    metrics = evaluate(model, test_data, args.batch_size)
    print(
        f"{label}: roc_auc={metrics['roc_auc']:.3f} pr_auc={metrics['pr_auc']:.3f}"
        f" (val roc_auc={val_roc_auc:.3f})",
        flush=True,
    )


def to_numpy(data) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for batch in get_dataloader(data, batch_size=1024):
        xs.append(batch["features"].numpy())
        ys.append(batch["mortality"].numpy())
    return np.concatenate(xs), np.concatenate(ys).ravel()


def logistic_regression_baseline(train_data, test_data) -> dict:
    """Linear baseline on the same 34 features: missing values imputed with
    the train mean plus a missing indicator per feature, standardized."""
    from sklearn.linear_model import LogisticRegression

    x_train, y_train = to_numpy(train_data)
    x_test, y_test = to_numpy(test_data)
    observed = x_train != MISSING
    fill = np.array(
        [x_train[observed[:, j], j].mean() if observed[:, j].any() else 0.0
         for j in range(x_train.shape[1])]
    )

    def design(x):
        missing = x == MISSING
        x = np.where(missing, fill, x)
        return np.hstack([x, missing.astype(float)])

    d_train, d_test = design(x_train), design(x_test)
    mu, sd = d_train.mean(0), d_train.std(0) + 1e-8
    clf = LogisticRegression(max_iter=2000)
    clf.fit((d_train - mu) / sd, y_train)
    y_prob = clf.predict_proba((d_test - mu) / sd)[:, 1]
    return binary_metrics_fn(y_test, y_prob, metrics=["roc_auc", "pr_auc"])


def main() -> None:
    args = parse_args()
    splits = load_splits(args)
    train_data, val_data, test_data = splits
    print(f"samples train/val/test: {len(train_data)}/{len(val_data)}/{len(test_data)}")
    # The model's input width: all 34 raw features (17 means + 17 stds),
    # not the 17 clinical variables -- "top 34 pairs" means pairs among
    # these 34 features (C(34, 2) = 561 candidates), per the paper.
    n_features = len(FEATURE_NAMES)
    n_bins = args.bins[-1]

    if "bins" in args.stages:
        print("\n== Bin-count ablation (1D terms only) ==", flush=True)
        for bins in args.bins:
            # dataset=train_data: bin edges are fit from the train split only.
            fit_and_evaluate(
                args, splits, f"bins={bins:>3}", n_bins=bins
            )

    top_pairs = []
    if {"pairs", "pruning"} & set(args.stages):
        print(
            f"\n== Pairwise-interaction selection "
            f"(all {len(all_pairs(n_features))} pairs @ {args.coarse_bins} coarse "
            f"bins -> top {args.n_pairs} -> retrain @ {n_bins} bins, "
            f"pairs @ {args.pair_bins}) ==",
            flush=True,
        )
        torch.manual_seed(args.seed)
        top_pairs = select_top_pairs(train_data, val_data, args, n_features)
        for rank, (i, j) in enumerate(top_pairs, 1):
            print(f"  {rank:>2}. {FEATURE_NAMES[i]} x {FEATURE_NAMES[j]}")
        fit_and_evaluate(
            args, splits,
            f"1D + top-{len(top_pairs)}-pairs @ {n_bins} bins",
            n_bins=n_bins, pairs=top_pairs, pair_bins=args.pair_bins,
        )

    if "pruning" in args.stages:
        print(
            "\n== Clinician-pruning ablation (retrain without flagged terms) ==",
            flush=True,
        )
        no_ph = [p for p in top_pairs if not set(p) & set(PH_FEATURES)]
        no_std = [p for p in top_pairs if not set(p) & STD_FEATURES]
        neither = [p for p in no_ph if p in no_std]
        variants = [
            ("without pH terms", PH_FEATURES, no_ph),
            ("without std-feature pairs", [], no_std),
            ("without both", PH_FEATURES, neither),
        ]
        for name, excluded, pairs in variants:
            fit_and_evaluate(
                args, splits,
                f"{name} ({len(pairs)} pairs)",
                n_bins=n_bins, pairs=pairs, pair_bins=args.pair_bins,
                excluded_features=excluded,
            )

    if "lr" in args.stages:
        print("\n== Logistic-regression baseline (same 34 features) ==", flush=True)
        metrics = logistic_regression_baseline(train_data, test_data)
        print(
            f"logistic regression: "
            f"roc_auc={metrics['roc_auc']:.3f} pr_auc={metrics['pr_auc']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
