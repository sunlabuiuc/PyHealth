"""GA2M In-Hospital Mortality Prediction: Full Pipeline + Ablation Study.

Reproduces the mortality prediction pipeline from:
    Hegselmann et al., "An Evaluation of the Doctor-Interpretability of
    Generalized Additive Models with Interactions", MLHC 2020.
    https://proceedings.mlr.press/v126/hegselmann20a.html

This script runs three ablations matching the project proposal:
    1. Full GA2M (main effects + top-34 interactions)
    2. Main effects only (use_interactions=False)
    3. Reduced feature set (mean features only, no std)

Metrics: AUC-ROC and AUC-PR (paper Section 2.2).

Use:
    python examples/mimic4_mortality_ga2m.py \
        --data_root data/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2

Note:
    Full results require credentialed MIMIC-III access. This script uses
    MIMIC-IV demo (100 patients) for development and testing. Results on
    the demo will be different from the paper due to the small sample size.
"""

import argparse
import sys
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

#  check if pyhealth is installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.ga2m import GA2M
from pyhealth.datasets.mimic4_icu_mortality import build_mortality_samples, UNKNOWN_SENTINEL


# helpers

def evaluate(model: GA2M, loader, device: str = "cpu") -> dict:
    # auc-roc and auc-pr eval
    # return dict of metrics
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            # out = model(**{k: v.to(device) for k, v in batch.items()})
            out = model(**{k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)})
            all_probs.append(out["y_prob"].cpu().numpy())
            all_labels.append(out["y_true"].cpu().numpy())

    y_prob = np.concatenate(all_probs).squeeze()
    y_true = np.concatenate(all_labels).squeeze()

    # if only one class, metrics would not be defined (NaN)
    if len(np.unique(y_true)) < 2:
        return {"auc_roc": float("nan"), "auc_pr": float("nan")}

    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr":  average_precision_score(y_true, y_prob),
    }


def make_dataset_and_loaders(samples, test_size=0.2, batch_size=32, seed=42):
    # split samples
    # create SampleDataset for train and test
    # create DataLoader for train and test
    train_samples, test_samples = train_test_split(
        samples,
        test_size=test_size,
        random_state=seed,
        stratify=[s["label"] for s in samples],
    )

    train_ds = create_sample_dataset(
        samples=train_samples,
        input_schema={"features": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="mimic4_train",
        in_memory=True,
    )
    test_ds = create_sample_dataset(
        samples=test_samples,
        input_schema={"features": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="mimic4_test",
        in_memory=True,
    )

    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = get_dataloader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_ds, test_ds, train_loader, test_loader


def run_experiment(
    name: str,
    samples,
    n_bins: int = 32,
    top_k_interactions: int = 10,
    use_interactions: bool = True,
    stage1_epochs: int = 20,
    stage2_epochs: int = 20,
    lr: float = 1e-2,
    seed: int = 42,
):
    # one GA2M experiment & return metrics
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"  n_bins={n_bins}, top_k={top_k_interactions}, "
          f"use_interactions={use_interactions}")
    print(f"{'='*60}")

    torch.manual_seed(seed)

    train_ds, test_ds, train_loader, test_loader = make_dataset_and_loaders(
        samples, seed=seed
    )

    model = GA2M(
        dataset=train_ds,
        n_bins=n_bins,
        top_k_interactions=top_k_interactions,
        use_interactions=use_interactions,
    )

    # 1. train main effects
    model.fit_bins(train_loader)
    model.fit_main_effects(train_loader, epochs=stage1_epochs, lr=lr)

    # 2. select interactions and train full model
    if use_interactions:
        model.select_top_interactions()

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(stage2_epochs):
        total_loss = 0.0
        for batch in train_loader:
            optimiser.zero_grad()
            out = model(**batch)
            out["loss"].backward()
            optimiser.step()
            total_loss += out["loss"].item()
        if (epoch + 1) % 5 == 0:
            print(f"  [Stage 2] Epoch {epoch+1}/{stage2_epochs}  "
                  f"loss={total_loss/len(train_loader):.4f}")

    metrics = evaluate(model, test_loader)
    print(f"\n  AUC-ROC : {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR  : {metrics['auc_pr']:.4f}")
    return metrics, model


# main

def main():
    parser = argparse.ArgumentParser(
        description="GA2M mortality prediction ablation on MIMIC-IV"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2",
        help="Path to MIMIC-IV root directory",
    )
    parser.add_argument("--n_bins", type=int, default=32,
                        help="Number of bins (paper uses 256; use 32 for demo)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-K interactions (paper uses 34)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    print(f"Loading MIMIC-IV data from: {args.data_root}")
    all_samples = build_mortality_samples(root=args.data_root)
    print(f"Loaded {len(all_samples)} ICU stays  "
          f"({sum(s['label'] for s in all_samples)} deaths)")

    results = {}

    # ablation 1: Full GA2M (main effects + interactions)
    metrics, full_model = run_experiment(
        name="Full GA2M (main effects + interactions)",
        samples=all_samples,
        n_bins=args.n_bins,
        top_k_interactions=args.top_k,
        use_interactions=True,
        stage1_epochs=args.epochs,
        stage2_epochs=args.epochs,
        lr=args.lr,
    )
    results["Full GA2M"] = metrics

    # ablation 2: Main effects only (no interactions)
    metrics, _ = run_experiment(
        name="Main Effects Only (no interactions)",
        samples=all_samples,
        n_bins=args.n_bins,
        top_k_interactions=0,
        use_interactions=False,
        stage1_epochs=args.epochs,
        stage2_epochs=args.epochs,
        lr=args.lr,
    )
    results["Main Effects Only"] = metrics

    # ablation 3: Reduced feature set (mean features only, std masked)
    # w/ only mean features
    #   zero out std positions
    reduced_samples = []
    for s in all_samples:
        feats = list(s["features"])
        # std features are at odd indices (1, 3, 5, ...)
        for i in range(1, len(feats), 2):
            feats[i] = UNKNOWN_SENTINEL
        reduced_samples.append({**s, "features": feats})

    metrics, _ = run_experiment(
        name="Reduced Features (mean only, std masked)",
        samples=reduced_samples,
        n_bins=args.n_bins,
        top_k_interactions=args.top_k,
        use_interactions=True,
        stage1_epochs=args.epochs,
        stage2_epochs=args.epochs,
        lr=args.lr,
    )
    results["Mean Features Only"] = metrics

    # summary of results
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Experiment':<35} {'AUC-ROC':>8} {'AUC-PR':>8}")
    print(f"{'-'*55}")
    for name, m in results.items():
        print(f"{name:<35} {m['auc_roc']:>8.4f} {m['auc_pr']:>8.4f}")

    print(f"\nNote: Results on demo (100 patients) will differ from the paper")
    print(f"(paper uses ~14k training stays from full MIMIC-III).")

    # visualise shape function for heart rate
    print(f"\nHeart rate (feature 0) risk function range:")
    midpoints, risks = full_model.get_shape_function(0)
    finite = np.isfinite(midpoints)
    for mp, r in zip(midpoints[finite][:5], risks[finite][:5]):
        print(f"  bin midpoint={mp:.2f}  risk={r:.4f}")


if __name__ == "__main__":
    main()