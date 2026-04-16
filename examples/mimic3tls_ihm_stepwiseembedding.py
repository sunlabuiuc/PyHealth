"""Step-wise Embedding ablation study for IHM on MIMIC-III (TLS preprocessed).

Paper: "On the Importance of Step-wise Embeddings for Heterogeneous
Clinical Time-Series" (Kuznetsova et al., JMLR 2023).

This script demonstrates the full pipeline: dataset loading, task
construction, model training, and evaluation with ablations across
three dimensions:

1. **Dataset/input ablations**: No grouping vs. type grouping vs.
   organ grouping (controls how features are split before embedding).
2. **Task ablations**: 48-hour vs. 24-hour observation window
   (controls the length of the input time-series).
3. **Model ablations**: Backbone-only, FTT direct, FTT with type
   groups, FTT with organ groups, linear direct, MLP direct.

Usage:
    # With real TLS-preprocessed data:
    python mimic3tls_ihm_stepwiseembedding.py --data_root /path/to/tls/

    # With synthetic data (for testing):
    python mimic3tls_ihm_stepwiseembedding.py --synthetic
"""

import argparse
import os
import time

import numpy as np
import torch

from pyhealth.datasets import (
    create_sample_dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.datasets.mimic3_tls import MIMIC3TLSDataset
from pyhealth.models import StepwiseEmbedding


# -----------------------------------------------------------------------
# Ablation configurations
# -----------------------------------------------------------------------

# Model ablation configs: (name, model_kwargs)
MODEL_ABLATIONS = {
    "backbone_only": {
        "embedding_type": None,
        "group_indices": None,
    },
    "linear_direct": {
        "embedding_type": "linear",
        "group_indices": None,
    },
    "mlp_direct": {
        "embedding_type": "mlp",
        "group_indices": None,
    },
    "ftt_direct": {
        "embedding_type": "ftt",
        "group_indices": None,
    },
    "ftt_type_groups": {
        "embedding_type": "ftt",
        "group_indices": MIMIC3TLSDataset.TYPE_GROUPS_INDICES,
        "aggregation": "mean",
    },
    "ftt_organ_groups": {
        "embedding_type": "ftt",
        "group_indices": MIMIC3TLSDataset.ORGAN_GROUPS_INDICES,
        "aggregation": "mean",
    },
}

# Task ablation configs: observation window in hours
TASK_ABLATIONS = [48, 24]


def create_synthetic_dataset(
    n_patients: int = 100,
    observation_hours: int = 48,
    n_features: int = 42,
    seed: int = 42,
):
    """Create a synthetic dataset mimicking TLS output for testing.

    Args:
        n_patients: Number of synthetic patients.
        observation_hours: Timesteps per patient.
        n_features: Number of features per timestep.
        seed: Random seed for reproducibility.

    Returns:
        A :class:`~pyhealth.datasets.SampleDataset` with synthetic data.
    """
    np.random.seed(seed)
    samples = [
        {
            "patient_id": f"patient-{i}",
            "time_series": np.random.randn(
                observation_hours, n_features
            ).tolist(),
            "ihm": int(np.random.binomial(1, 0.15)),
        }
        for i in range(n_patients)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"time_series": "tensor"},
        output_schema={"ihm": "binary"},
        dataset_name="synthetic_tls",
    )


def train_and_evaluate(
    model: StepwiseEmbedding,
    train_loader,
    val_loader,
    n_epochs: int = 5,
    lr: float = 1e-3,
):
    """Train the model and evaluate on the validation set.

    Args:
        model: The StepwiseEmbedding model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        n_epochs: Number of training epochs.
        lr: Learning rate.

    Returns:
        Dictionary with training and evaluation metrics.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(**batch)
            output["loss"].backward()
            optimizer.step()
            epoch_loss += output["loss"].item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{n_epochs} - Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            output = model(**batch)
            all_probs.append(output["y_prob"].cpu())
            all_labels.append(output["y_true"].cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Compute AUROC and AUPRC
    try:
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
        )

        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except (ValueError, ImportError):
        auroc = float("nan")
        auprc = float("nan")

    return {"auroc": auroc, "auprc": auprc}


def main():
    parser = argparse.ArgumentParser(
        description="Step-wise Embedding ablation study for IHM"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to TLS-preprocessed MIMIC-III data directory.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing.",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=5, help="Training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size."
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate."
    )
    args = parser.parse_args()

    if args.synthetic or args.data_root is None:
        print("=" * 60)
        print("Running with SYNTHETIC data (for demonstration)")
        print("=" * 60)
        use_synthetic = True
    else:
        use_synthetic = False

    results = []

    # ---------------------------------------------------------------
    # Task ablation: different observation windows
    # ---------------------------------------------------------------
    for obs_hours in TASK_ABLATIONS:
        print(f"\n{'=' * 60}")
        print(f"Task ablation: observation_hours={obs_hours}")
        print(f"{'=' * 60}")

        if use_synthetic:
            sample_dataset = create_synthetic_dataset(
                n_patients=100, observation_hours=obs_hours
            )
        else:
            from pyhealth.tasks import InHospitalMortalityTLS

            dataset = MIMIC3TLSDataset(root=args.data_root)
            task = InHospitalMortalityTLS(
                observation_hours=obs_hours
            )
            sample_dataset = dataset.set_task(task)

        input_dim = 42

        # Split dataset
        train_ds, val_ds, test_ds = split_by_patient(
            sample_dataset, [0.7, 0.15, 0.15]
        )
        train_loader = get_dataloader(
            train_ds, batch_size=args.batch_size, shuffle=True
        )
        val_loader = get_dataloader(
            val_ds, batch_size=args.batch_size, shuffle=False
        )

        # -----------------------------------------------------------
        # Model ablation: different embedding configurations
        # -----------------------------------------------------------
        for config_name, model_kwargs in MODEL_ABLATIONS.items():
            print(f"\n  Model config: {config_name}")
            print(f"  {'-' * 40}")

            start = time.time()
            model = StepwiseEmbedding(
                dataset=sample_dataset,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                backbone_depth=1,
                backbone_heads=1,
                **model_kwargs,
            )

            metrics = train_and_evaluate(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=args.n_epochs,
                lr=args.lr,
            )
            elapsed = time.time() - start

            results.append({
                "obs_hours": obs_hours,
                "config": config_name,
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "time_s": elapsed,
            })

            print(
                f"  AUROC: {metrics['auroc']:.4f} | "
                f"AUPRC: {metrics['auprc']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"{'Obs Hours':<12} {'Config':<22} "
        f"{'AUROC':<10} {'AUPRC':<10} {'Time (s)':<10}"
    )
    print("-" * 64)
    for r in results:
        print(
            f"{r['obs_hours']:<12} {r['config']:<22} "
            f"{r['auroc']:<10.4f} {r['auprc']:<10.4f} "
            f"{r['time_s']:<10.1f}"
        )


if __name__ == "__main__":
    main()
