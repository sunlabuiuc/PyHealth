"""Ablation Study: Attention Weight Consistency Across Feature Dimensionalities.

This script reproduces the core attention-consistency experiment from Yadav &
Subbian (2025), "When Attention Fails: Pitfalls of Attention-based Model
Interpretability for High-dimensional Clinical Time-Series" (CHIL 2025), and
evaluates how clinical feature dimensionality affects attention stability.

Experimental Setup:

    - Task: Binary ICU mortality prediction over 288 time steps (24h at 5-min
      intervals) using synthetic high-dimensional clinical time-series data.
    - Configurations Tested: Feature dimensions D in {231, 100, 50}, matching
      the paper's full 231-feature HiRID setup alongside reduced feature sets.
    - Variants per Configuration: 5 independent random seeds ({0, 1, 2, 3, 4}).
    - Metrics Evaluated:

        1. Predictive Performance: Mean +/- Std AUROC and AUPRC on the test set.
        2. Attention Consistency: Intersection count of the top-100 highest
           attention feature-time pairs for a positive test patient across all
           5 independently initialized model variants.

Key Findings:

    Actual CPU run summary:

    ======================================================================
    FeatureAttentionLSTM Ablation: Feature Dimensionality vs. Seed Stability
    ======================================================================
    Features (D)   | AUROC (Mean+/-Std)   | AUPRC (Mean+/-Std)   | Common Top-100
    --------------------------------------------------------------------------
    231            | 0.706 +/- 0.076      | 0.676 +/- 0.104      | 0 / 100
    100            | 0.801 +/- 0.081      | 0.740 +/- 0.078      | 0 / 100
    50             | 0.983 +/- 0.028      | 0.985 +/- 0.024      | 0 / 100
"""

from typing import List, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from pyhealth.models import FeatureAttentionLSTM


class SyntheticTimeSeriesDataset:
    """Lightweight dataset metadata wrapper for FeatureAttentionLSTM."""

    def __init__(self, feature_dim: int, output_dim: int = 2) -> None:
        self.input_info = {"ts_features": {"input_dim": feature_dim}}
        self.output_info = {"label": {"output_dim": output_dim}}
        self.input_schema = {"ts_features": "timeseries"}
        self.output_schema = {"label": "binary"}


def generate_synthetic_icu_data(
    num_samples: int = 240,
    time_steps: int = 288,
    feature_dim: int = 231,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create deterministic synthetic ICU train and test loaders.

    A late-window physiological signal is injected into positive mortality
    cases so models learn meaningful, non-trivial predictions.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(
        0.0,
        1.0,
        size=(num_samples, time_steps, feature_dim),
    ).astype(np.float32)
    y = rng.integers(0, 2, size=(num_samples,)).astype(np.int64)

    for idx in range(num_samples):
        if y[idx] == 1:
            x[idx, -24:, :5] += 0.85

    split_idx = int(0.8 * num_samples)
    train_ds = TensorDataset(
        torch.tensor(x[:split_idx]),
        torch.tensor(y[:split_idx]),
    )
    test_ds = TensorDataset(
        torch.tensor(x[split_idx:]),
        torch.tensor(y[split_idx:]),
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    return train_loader, test_loader


def train_and_evaluate_variant(
    feature_dim: int,
    seed: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 6,
    lr: float = 1e-3,
) -> Tuple[float, float, Set[int]]:
    """Train one seed and return AUROC, AUPRC, and top-100 indices."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mock_ds = SyntheticTimeSeriesDataset(
        feature_dim=feature_dim,
        output_dim=2,
    )
    model = FeatureAttentionLSTM(
        dataset=mock_ds,
        feature_keys=["ts_features"],
        label_key="label",
        mode="binary",
        hidden_dim=64,
        num_layers=1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(ts_features=batch_x, label=batch_y)
            output["loss"].backward()
            optimizer.step()

    model.eval()
    all_probs: List[float] = []
    all_labels: List[int] = []
    sample_attention = None

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = model(ts_features=batch_x, label=batch_y)
            probs = torch.softmax(output["y_hat"], dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(batch_y.cpu().numpy().tolist())

            if sample_attention is None:
                positive_indices = (batch_y == 1).nonzero(as_tuple=True)[0]
                if len(positive_indices) > 0:
                    first_positive = int(positive_indices[0].item())
                    sample_attention = output["attention_weights"][first_positive].cpu()

    labels_array = np.asarray(all_labels, dtype=np.int64)
    probs_array = np.asarray(all_probs, dtype=np.float64)

    positive_scores = probs_array[labels_array == 1]
    negative_scores = probs_array[labels_array == 0]
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        auroc = 0.0
    else:
        concordant_pairs = (positive_scores[:, None] > negative_scores[None, :]).sum()
        tied_pairs = (positive_scores[:, None] == negative_scores[None, :]).sum()
        auroc = float(
            (concordant_pairs + 0.5 * tied_pairs)
            / (len(positive_scores) * len(negative_scores))
        )

    order = np.argsort(-probs_array)
    sorted_labels = labels_array[order]
    positives = np.cumsum(sorted_labels)
    positions = np.arange(1, len(sorted_labels) + 1)
    positive_precision = positives / positions
    positive_count = int(sorted_labels.sum())
    auprc = (
        float(positive_precision[sorted_labels == 1].sum() / positive_count)
        if positive_count > 0
        else 0.0
    )

    if sample_attention is None:
        raise RuntimeError("No positive test patient was found")

    flat_weights = sample_attention.flatten().numpy()
    top_100_indices = set(np.argsort(flat_weights)[-100:].tolist())
    return auroc, auprc, top_100_indices


def run_ablation() -> None:
    """Execute dimensionality and random-seed consistency ablation study."""
    feature_dims = [231, 100, 50]
    seeds = [0, 1, 2, 3, 4]

    print("=" * 74)
    print("FeatureAttentionLSTM Ablation: Feature Dimensionality vs. Seed Stability")
    print("=" * 74)
    print(
        f"{'Features (D)':<14} | {'AUROC (Mean+/-Std)':<20} | "
        f"{'AUPRC (Mean+/-Std)':<20} | {'Common Top-100'}"
    )
    print("-" * 74)

    for dim in feature_dims:
        train_loader, test_loader = generate_synthetic_icu_data(
            num_samples=240,
            time_steps=288,
            feature_dim=dim,
            seed=42,
        )
        aurocs: List[float] = []
        auprcs: List[float] = []
        common_indices: Set[int] = set()

        for idx, seed in enumerate(seeds):
            auroc, auprc, top_indices = train_and_evaluate_variant(
                feature_dim=dim,
                seed=seed,
                train_loader=train_loader,
                test_loader=test_loader,
            )
            aurocs.append(auroc)
            auprcs.append(auprc)
            if idx == 0:
                common_indices = top_indices
            else:
                common_indices = common_indices.intersection(top_indices)

        auroc_str = f"{np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}"
        auprc_str = f"{np.mean(auprcs):.3f} +/- {np.std(auprcs):.3f}"
        print(
            f"{dim:<14} | {auroc_str:<20} | {auprc_str:<20} | "
            f"{len(common_indices)} / 100"
        )


if __name__ == "__main__":
    run_ablation()
