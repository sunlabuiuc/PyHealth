"""EEG-GCNN Neurological Disease Detection — Ablation Study.

This script reproduces the ablation experiments from:

    Wagh, N. & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
    Electroencephalogram-based Neurological Disease Diagnosis using a
    Domain-guided Graph Convolutional Neural Network. ML4H @ NeurIPS 2020.
    https://proceedings.mlr.press/v136/wagh20a.html

Three ablation experiments are included:
    1. Adjacency type: combined vs spatial vs functional vs none
    2. Frequency band ablation: individual bands & progressive combinations
    3. Connectivity measure: coherence vs WPLI

Usage:
    python examples/eeg_gcnn_nd_detection_gcn.py --root /path/to/eeg-gcnn-data
"""

import argparse
import json
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from pyhealth.datasets import EEGGCNNDataset, get_dataloader, split_by_patient
from pyhealth.models.gnn import GCN
from pyhealth.tasks import EEGGCNNDiseaseDetection
from pyhealth.tasks.eeg_gcnn_nd_detection import DEFAULT_BANDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Training / evaluation helpers
# -------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        optimizer.zero_grad()
        output = model(**batch)
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model and return AUC and loss."""
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        output = model(**batch)
        total_loss += output["loss"].item()
        n_batches += 1
        probs = output["y_prob"].cpu().numpy()
        labels = output["y_true"].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except ValueError:
        auc = 0.5
    return {
        "auc": auc,
        "loss": total_loss / max(n_batches, 1),
    }


def run_experiment(
    dataset: EEGGCNNDataset,
    task: EEGGCNNDiseaseDetection,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Dict[str, float]:
    """Run a single experiment: set_task, split, train, evaluate."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_dataset = dataset.set_task(task)
    train_ds, val_ds, test_ds = split_by_patient(
        sample_dataset, [0.7, 0.1, 0.2]
    )

    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    model = GCN(
        dataset=sample_dataset,
        embedding_dim=64,
        nhid=32,
        dropout=0.5,
        num_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc = 0.0
    best_state = None
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d/%d — train_loss=%.4f val_auc=%.4f",
            epoch + 1, epochs, train_loss, val_metrics["auc"],
        )
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    logger.info("Test AUC: %.4f", test_metrics["auc"])
    return test_metrics


# -------------------------------------------------------------------
# Ablation experiments
# -------------------------------------------------------------------

def ablation_adjacency(
    dataset: EEGGCNNDataset, **kwargs
) -> Dict[str, Dict[str, float]]:
    """Experiment 1: Adjacency type ablation."""
    results = {}
    for adj_type in ("combined", "spatial", "functional", "none"):
        logger.info("=== Adjacency type: %s ===", adj_type)
        task = EEGGCNNDiseaseDetection(adjacency_type=adj_type)
        results[adj_type] = run_experiment(dataset, task, **kwargs)
    return results


def ablation_frequency_bands(
    dataset: EEGGCNNDataset, **kwargs
) -> Dict[str, Dict[str, float]]:
    """Experiment 2: Frequency band ablation.

    Tests individual bands and progressive combinations.
    """
    band_names = list(DEFAULT_BANDS.keys())
    results = {}

    # Individual bands
    for name in band_names:
        logger.info("=== Single band: %s ===", name)
        task = EEGGCNNDiseaseDetection(
            bands={name: DEFAULT_BANDS[name]}
        )
        results[name] = run_experiment(dataset, task, **kwargs)

    # Progressive combinations: delta, delta+theta, delta+theta+alpha, ...
    for k in range(2, len(band_names) + 1):
        combo_names = band_names[:k]
        combo_key = "+".join(combo_names)
        logger.info("=== Band combination: %s ===", combo_key)
        combo_bands = {n: DEFAULT_BANDS[n] for n in combo_names}
        task = EEGGCNNDiseaseDetection(bands=combo_bands)
        results[combo_key] = run_experiment(dataset, task, **kwargs)

    return results


def ablation_connectivity(
    dataset: EEGGCNNDataset, **kwargs
) -> Dict[str, Dict[str, float]]:
    """Experiment 3: Connectivity measure ablation."""
    results = {}
    for measure in ("coherence", "wpli"):
        logger.info("=== Connectivity: %s ===", measure)
        task = EEGGCNNDiseaseDetection(
            adjacency_type="functional",
            connectivity_measure=measure,
        )
        results[measure] = run_experiment(dataset, task, **kwargs)
    return results


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EEG-GCNN ablation study"
    )
    parser.add_argument(
        "--root", type=str, required=True,
        help="Root directory of EEG-GCNN data (TUAB + LEMON)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "adjacency", "bands", "connectivity"],
        help="Which ablation experiment to run",
    )
    args = parser.parse_args()

    dataset = EEGGCNNDataset(root=args.root)
    dataset.stats()

    train_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    all_results = {}

    if args.experiment in ("all", "adjacency"):
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT 1: Adjacency Type Ablation")
        logger.info("=" * 60)
        all_results["adjacency"] = ablation_adjacency(dataset, **train_kwargs)

    if args.experiment in ("all", "bands"):
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT 2: Frequency Band Ablation")
        logger.info("=" * 60)
        all_results["bands"] = ablation_frequency_bands(
            dataset, **train_kwargs
        )

    if args.experiment in ("all", "connectivity"):
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT 3: Connectivity Measure Ablation")
        logger.info("=" * 60)
        all_results["connectivity"] = ablation_connectivity(
            dataset, **train_kwargs
        )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION RESULTS SUMMARY")
    logger.info("=" * 60)
    for exp_name, exp_results in all_results.items():
        logger.info("\n--- %s ---", exp_name)
        for config, metrics in exp_results.items():
            logger.info("  %-30s  AUC=%.4f", config, metrics["auc"])

    # Save results to JSON
    output_path = "eeg_gcnn_ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nResults saved to %s", output_path)


if __name__ == "__main__":
    main()
