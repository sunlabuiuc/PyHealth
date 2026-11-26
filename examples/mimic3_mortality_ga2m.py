"""GA2M on 48h MIMIC-III in-hospital mortality: bin-count ablation and the
paper's two-phase pairwise-interaction selection.

Reproduces (the mechanism of) Hegselmann et al. (2020): quantile-binned
additive terms trained by SGD, a reserved missing bin per feature, and
pairwise-interaction selection by training every candidate pair at coarse
bins, ranking by contribution variance, and retraining with only the top
pairs.

Synthetic fallback (default, no MIMIC-III access needed)::

    python examples/mimic3_mortality_ga2m.py

Real MIMIC-III (requires PhysioNet credentialing; the team's official
reproduction numbers against the mimic3-benchmarks cohort are computed
locally and reported separately, outside this PyHealth contribution)::

    python examples/mimic3_mortality_ga2m.py --root path/to/mimic-iii/1.4

The synthetic cohort is generated from a single latent "severity" variable
that drives both the 17 clinical variables (with injected missingness and
noise) and the mortality label, so there IS a learnable signal, but it is
not a claim about the real MIMIC-III cohort's statistics or difficulty --
it only demonstrates that the training/evaluation/ablation pipeline runs
end to end. See ``examples/mimic3_mortality_ga2m_results.md`` for more
discussion; these are not the team's real-MIMIC-III reproduction numbers.

Last synthetic run (default args, seed=0; 420/59/121 train/val/test
patients, ~22s on CPU)::

    == Bin-count ablation (1D terms only) ==
    bins= 32: roc_auc=0.658 pr_auc=0.474
    bins= 64: roc_auc=0.688 pr_auc=0.521
    bins=128: roc_auc=0.656 pr_auc=0.502
    bins=256: roc_auc=0.610 pr_auc=0.461

    == Pairwise-interaction selection (all 561 pairs @ 8 coarse bins ->
    top 34 -> retrain @ 256 bins) ==
    1D + top-34-pairs @ 256 bins: roc_auc=0.601 pr_auc=0.466
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader, split_by_patient
from pyhealth.metrics import binary_metrics_fn
from pyhealth.models import GA2M
from pyhealth.models.ga2m import all_pairs, rank_pairs_by_contribution_variance
from pyhealth.tasks import InHospitalMortality48hMIMIC3
from pyhealth.tasks.in_hospital_mortality_48h_mimic3 import FEATURE_NAMES, MISSING

N_VARIABLES = len(FEATURE_NAMES) // 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root", default=None, help="MIMIC-III folder (1.4); omit for synthetic data"
    )
    parser.add_argument("--cache-dir", default=None, help="Folder for processed data")
    parser.add_argument("--n-patients", type=int, default=600, help="Synthetic only")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--bins", type=int, nargs="+", default=[32, 64, 128, 256],
        help="Bin counts to ablate over (paper: 32, 64, 128, 256)",
    )
    parser.add_argument(
        "--coarse-bins", type=int, default=8,
        help="Bin count for the all-pairs pair-selection phase",
    )
    parser.add_argument("--n-pairs", type=int, default=34, help="Pairs to keep")
    parser.add_argument("--seed", type=int, default=0)
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


def train(model: GA2M, train_data, epochs: int, batch_size: int, lr: float) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = get_dataloader(train_data, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()


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
    train_data, args: argparse.Namespace, n_features: int
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
    train(
        coarse,
        train_data,
        epochs=max(args.epochs // 3, 1),
        batch_size=args.batch_size,
        lr=args.lr,
    )
    ranked = rank_pairs_by_contribution_variance(
        coarse, get_dataloader(train_data, batch_size=args.batch_size, shuffle=False)
    )
    return [pair for pair, _ in ranked[: args.n_pairs]]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    train_data, val_data, test_data = load_splits(args)
    print(f"samples train/val/test: {len(train_data)}/{len(val_data)}/{len(test_data)}")
    # The model's input width: all 34 raw features (17 means + 17 stds),
    # not the 17 clinical variables -- "top 34 pairs" means pairs among
    # these 34 features (C(34, 2) = 561 candidates), per the paper.
    n_features = len(FEATURE_NAMES)

    print("\n== Bin-count ablation (1D terms only) ==")
    for n_bins in args.bins:
        # dataset=train_data: bin edges are fit from the train split only.
        model = GA2M(dataset=train_data, n_bins=n_bins)
        train(model, train_data, args.epochs, args.batch_size, args.lr)
        metrics = evaluate(model, test_data, args.batch_size)
        print(
            f"bins={n_bins:>3}: "
            f"roc_auc={metrics['roc_auc']:.3f} pr_auc={metrics['pr_auc']:.3f}"
        )

    print(
        f"\n== Pairwise-interaction selection "
        f"(all {len(all_pairs(n_features))} pairs @ {args.coarse_bins} coarse "
        f"bins -> top {args.n_pairs} -> retrain @ {args.bins[-1]} bins) =="
    )
    top_pairs = select_top_pairs(train_data, args, n_features)
    final_model = GA2M(dataset=train_data, n_bins=args.bins[-1], pairs=top_pairs)
    train(final_model, train_data, args.epochs, args.batch_size, args.lr)
    metrics = evaluate(final_model, test_data, args.batch_size)
    print(
        f"1D + top-{len(top_pairs)}-pairs @ {args.bins[-1]} bins: "
        f"roc_auc={metrics['roc_auc']:.3f} pr_auc={metrics['pr_auc']:.3f}"
    )


if __name__ == "__main__":
    main()
