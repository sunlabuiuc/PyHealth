"""Combined EEG-GCNN Pipeline — Dataset/Task + Model Integration.

Self-contained end-to-end pipeline combining both team contributions:
  - Dataset & Task (Option 1 — jburhan):
      EEG preprocessing, PSD feature extraction, graph adjacency matrices
  - Model (Option 2 — racoffey):
      EEGGraphConvNet (GCN, paper baseline) and EEGGATConvNet (GAT, novel)

This script runs standalone — no PyHealth installation required for demo
mode.  For real-data mode, PyHealth with the EEG-GCNN contribution must
be installed (pip install -e . from the PyHealth fork).

Paper:
    Wagh, N. & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
    Electroencephalogram-based Neurological Disease Diagnosis using a
    Domain-guided Graph Convolutional Neural Network. ML4H @ NeurIPS 2020.
    https://proceedings.mlr.press/v136/wagh20a.html

Usage (demo mode — synthetic data, no downloads needed):
    python eeg_gcnn_combined_pipeline.py --demo

Usage (demo with GAT model):
    python eeg_gcnn_combined_pipeline.py --demo --model gat

Usage (demo, single experiment):
    python eeg_gcnn_combined_pipeline.py --demo --experiment model

Usage (with real EEG data — requires PyHealth with EEG-GCNN contribution):
    python eeg_gcnn_combined_pipeline.py --root /path/to/eeg-gcnn-data

Dependencies (demo mode):
    torch, torch_geometric, numpy, scikit-learn
"""

import argparse
import json
import logging
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from torch_geometric.nn import (
    BatchNorm,
    GATConv,
    GCNConv,
    global_add_pool,
)
from torch_geometric.utils import dense_to_sparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===================================================================
# Constants (from EEG-GCNN paper, Section 4)
# ===================================================================

NUM_CHANNELS = 8
NUM_NODES = 8

DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "lower_beta": (12.0, 20.0),
    "higher_beta": (20.0, 30.0),
    "gamma": (30.0, 50.0),
}


# ===================================================================
# Models (Option 2 contribution — racoffey)
# ===================================================================


class EEGGraphConvNet(nn.Module):
    """Deep EEG-GCNN model using GCNConv layers.

    Follows the architecture from the paper: 4 GCN layers with
    leaky ReLU, global add pooling, and a 3-layer FC classifier.

    Args:
        num_node_features: Number of PSD bands per node. Default: 6.
        output_size: Number of output classes. Default: 1 (binary).
    """

    def __init__(self, num_node_features: int = 6, output_size: int = 1):
        super().__init__()

        self.conv1 = GCNConv(num_node_features, 16, improved=True,
                             cached=False, normalize=False)
        self.conv2 = GCNConv(16, 32, improved=True, cached=False,
                             normalize=False)
        self.conv3 = GCNConv(32, 64, improved=True, cached=False,
                             normalize=False)
        self.conv4 = GCNConv(64, 50, improved=True, cached=False,
                             normalize=False)
        self.conv4_bn = BatchNorm(50)

        self.fc1 = nn.Linear(50, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, output_size)

        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_normal_(fc.weight, gain=1)

    def forward(self, x, edge_index, edge_weight, batch):
        x = F.leaky_relu(self.conv1(x, edge_index, edge_weight))
        x = F.leaky_relu(self.conv2(x, edge_index, edge_weight))
        x = F.leaky_relu(self.conv3(x, edge_index, edge_weight))
        x = F.leaky_relu(self.conv4_bn(
            self.conv4(x, edge_index, edge_weight)))
        out = global_add_pool(x, batch=batch)
        out = F.leaky_relu(self.fc1(out), negative_slope=0.01)
        out = F.dropout(out, p=0.2, training=self.training)
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        return self.fc3(out)


class EEGGATConvNet(nn.Module):
    """EEG Graph Attention Network — novel model contribution.

    Replaces GCNConv with multi-head GATConv layers to learn
    attention-weighted edge importance, rather than relying on
    fixed adjacency weights.

    Args:
        num_node_features: Number of PSD bands per node. Default: 6.
        output_size: Number of output classes. Default: 1 (binary).
    """

    def __init__(self, num_node_features: int = 6, output_size: int = 1):
        super().__init__()

        self.conv1 = GATConv(num_node_features, 16, heads=4, concat=True,
                             negative_slope=0.2, dropout=0.0,
                             add_self_loops=False, edge_dim=1)
        self.conv2 = GATConv(64, 32, heads=4, concat=True,
                             negative_slope=0.2, dropout=0.0,
                             add_self_loops=False, edge_dim=1)
        self.conv3 = GATConv(128, 16, heads=4, concat=True,
                             negative_slope=0.2, dropout=0.0,
                             add_self_loops=False, edge_dim=1)
        self.conv4 = GATConv(64, 50, heads=1, concat=False,
                             negative_slope=0.2, dropout=0.0,
                             add_self_loops=False, edge_dim=1)
        self.conv4_bn = BatchNorm(50)

        self.fc1 = nn.Linear(50, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, output_size)

        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_normal_(fc.weight, gain=1)

    def forward(self, x, edge_index, edge_weight, batch):
        edge_attr = edge_weight.unsqueeze(-1) if edge_weight.dim() == 1 \
            else edge_weight
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr=edge_attr))
        x = F.leaky_relu(self.conv4_bn(
            self.conv4(x, edge_index, edge_attr=edge_attr)))
        out = global_add_pool(x, batch=batch)
        out = F.leaky_relu(self.fc1(out), negative_slope=0.01)
        out = F.dropout(out, p=0.2, training=self.training)
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        return self.fc3(out)


# ===================================================================
# Graph batching helper
# ===================================================================

def build_graph_batch(node_features_batch, adj_matrix_batch, device):
    """Convert batched dense tensors into a single PyG-style graph batch.

    Args:
        node_features_batch: (B, 8, n_bands) tensor
        adj_matrix_batch: (B, 8, 8) tensor
        device: torch device

    Returns:
        (x, edge_index, edge_weight, batch) tensors on device
    """
    all_edge_index = []
    all_edge_weight = []
    all_x = []
    batch_ids = []

    offset = 0
    for i in range(node_features_batch.shape[0]):
        x = node_features_batch[i].float()
        adj = adj_matrix_batch[i].float()
        ei, ew = dense_to_sparse(adj)
        all_edge_index.append(ei + offset)
        all_edge_weight.append(ew)
        all_x.append(x)
        batch_ids.extend([i] * x.shape[0])
        offset += x.shape[0]

    return (
        torch.cat(all_x, dim=0).to(device),
        torch.cat(all_edge_index, dim=1).to(device),
        torch.cat(all_edge_weight).to(device),
        torch.tensor(batch_ids, dtype=torch.long, device=device),
    )


# ===================================================================
# Lightweight dataset utilities (standalone, no PyHealth needed)
# ===================================================================

class DictDataset(Dataset):
    """Map-style dataset wrapping a list of sample dicts."""

    def __init__(self, samples: List[Dict]):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def collate_dict_batch(batch: List[Dict]) -> Dict:
    """Collate a list of sample dicts into a batched dict.

    Stacks tensors, collects strings into lists, and stacks ints/floats.
    """
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        vals = [s[k] for s in batch]
        if isinstance(vals[0], torch.Tensor):
            collated[k] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            collated[k] = torch.tensor(vals, dtype=torch.float32)
        else:
            collated[k] = vals
    return collated


def split_by_patient(
    samples: List[Dict], ratios: List[float], seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split samples into train/val/test by patient_id.

    Ensures all windows from the same patient stay in the same split.

    Args:
        samples: List of sample dicts with "patient_id" key.
        ratios: [train, val, test] fractions summing to 1.0.
        seed: Random seed for reproducibility.

    Returns:
        (train_samples, val_samples, test_samples)
    """
    rng = np.random.RandomState(seed)
    patient_ids = sorted(set(s["patient_id"] for s in samples))
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_pids = set(patient_ids[:n_train])
    val_pids = set(patient_ids[n_train:n_train + n_val])
    test_pids = set(patient_ids[n_train + n_val:])

    train = [s for s in samples if s["patient_id"] in train_pids]
    val = [s for s in samples if s["patient_id"] in val_pids]
    test = [s for s in samples if s["patient_id"] in test_pids]
    return train, val, test


def make_loader(
    samples: List[Dict], batch_size: int, shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader from a list of sample dicts."""
    return DataLoader(
        DictDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_dict_batch,
    )


# ===================================================================
# Synthetic demo data (Option 1 — same schema as real task output)
# ===================================================================

def generate_demo_samples(
    n_patients: int = 40,
    windows_per_patient: int = 5,
    n_bands: int = 6,
    adjacency_type: str = "combined",
    seed: int = 42,
) -> List[Dict]:
    """Generate synthetic samples matching real task output schema.

    Creates ``n_patients`` patients (half label-0, half label-1) each
    with ``windows_per_patient`` windows.  PSD features and adjacency
    matrices are random but reproducible.

    The output schema matches what the real EEGGCNNDiseaseDetection task
    produces:
        - node_features: (8, n_bands) float32 tensor
        - adj_matrix: (8, 8) float32 tensor
        - label: int (0 = patient-normal, 1 = healthy-control)
    """
    rng = np.random.RandomState(seed)
    samples = []
    for p in range(n_patients):
        pid = f"demo_{p:03d}"
        label = 0 if p < n_patients // 2 else 1
        for _ in range(windows_per_patient):
            psd = rng.randn(NUM_CHANNELS, n_bands).astype(np.float32)
            # Shift class-1 features so the model can learn separation
            if label == 1:
                psd += 0.5

            # Build adjacency based on type
            if adjacency_type == "none":
                adj = np.eye(NUM_CHANNELS, dtype=np.float32)
            else:
                adj = np.eye(NUM_CHANNELS, dtype=np.float32)
                off = rng.uniform(0.1, 0.5, (NUM_CHANNELS, NUM_CHANNELS))
                off = (off + off.T) / 2.0
                np.fill_diagonal(off, 0.0)
                adj = adj + off.astype(np.float32)

            samples.append({
                "patient_id": pid,
                "node_features": torch.FloatTensor(psd),
                "adj_matrix": torch.FloatTensor(adj),
                "label": label,
            })
    return samples


# ===================================================================
# Training / evaluation
# ===================================================================

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        node_features = batch["node_features"].to(device)
        adj_matrix = batch["adj_matrix"].to(device)
        labels = batch["label"].to(device).float()

        x, edge_index, edge_weight, batch_ids = build_graph_batch(
            node_features, adj_matrix, device
        )

        optimizer.zero_grad()
        logits = model(x, edge_index, edge_weight, batch_ids)
        loss = loss_fn(logits.squeeze(-1), labels.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Evaluate model, return AUC and loss."""
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        node_features = batch["node_features"].to(device)
        adj_matrix = batch["adj_matrix"].to(device)
        labels = batch["label"].to(device).float()

        x, edge_index, edge_weight, batch_ids = build_graph_batch(
            node_features, adj_matrix, device
        )

        logits = model(x, edge_index, edge_weight, batch_ids)
        loss = loss_fn(logits.squeeze(-1), labels.squeeze(-1))
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.squeeze(-1).cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = float(roc_auc_score(all_labels, all_probs))
    except (ValueError, IndexError):
        auc = 0.5
    return {
        "auc": auc,
        "loss": total_loss / max(n_batches, 1),
    }


def run_experiment(
    task_config: Dict[str, Any],
    model_type: str = "gcn",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    demo: bool = True,
    dataset: Any = None,
) -> Dict[str, float]:
    """Run a single experiment: generate data, split, train, evaluate.

    Args:
        task_config: Dict with keys like adjacency_type, bands,
            connectivity_measure — mirrors EEGGCNNDiseaseDetection init args.
        model_type: "gcn" or "gat".
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        demo: If True, use synthetic data. If False, use real data.
        dataset: PyHealth EEGGCNNDataset instance (only needed if demo=False).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bands = task_config.get("bands", DEFAULT_BANDS)
    n_bands = len(bands)
    adjacency_type = task_config.get("adjacency_type", "combined")

    if demo:
        samples = generate_demo_samples(
            n_bands=n_bands, adjacency_type=adjacency_type,
        )
    else:
        # Real data path — requires PyHealth with EEG-GCNN contribution
        from pyhealth.tasks import EEGGCNNDiseaseDetection
        task = EEGGCNNDiseaseDetection(**task_config)
        sample_dataset = dataset.set_task(task)
        samples = [sample_dataset[i] for i in range(len(sample_dataset))]

    train_samples, val_samples, test_samples = split_by_patient(
        samples, [0.7, 0.1, 0.2]
    )

    train_loader = make_loader(train_samples, batch_size, shuffle=True)
    val_loader = make_loader(val_samples, batch_size)
    test_loader = make_loader(test_samples, batch_size)

    if model_type == "gat":
        model = EEGGATConvNet(
            num_node_features=n_bands, output_size=1
        ).to(device)
    else:
        model = EEGGraphConvNet(
            num_node_features=n_bands, output_size=1
        ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auc = 0.0
    best_state = None
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        logger.info(
            "Epoch %d/%d — train_loss=%.4f val_auc=%.4f",
            epoch + 1, epochs, train_loss, val_metrics["auc"],
        )
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    logger.info("Test AUC: %.4f", test_metrics["auc"])
    return test_metrics


# ===================================================================
# Ablation experiments
# ===================================================================

def ablation_adjacency(**kwargs) -> Dict[str, Dict[str, float]]:
    """Experiment 1: Adjacency type ablation."""
    results = {}
    for adj_type in ("combined", "spatial", "functional", "none"):
        logger.info("=== Adjacency type: %s ===", adj_type)
        results[adj_type] = run_experiment(
            task_config={"adjacency_type": adj_type}, **kwargs)
    return results


def ablation_frequency_bands(**kwargs) -> Dict[str, Dict[str, float]]:
    """Experiment 2: Frequency band ablation.

    Tests individual bands and progressive combinations.
    """
    band_names = list(DEFAULT_BANDS.keys())
    results = {}

    # Individual bands
    for name in band_names:
        logger.info("=== Single band: %s ===", name)
        results[name] = run_experiment(
            task_config={"bands": {name: DEFAULT_BANDS[name]}}, **kwargs)

    # Progressive combinations
    for k in range(2, len(band_names) + 1):
        combo_names = band_names[:k]
        combo_key = "+".join(combo_names)
        logger.info("=== Band combination: %s ===", combo_key)
        combo_bands = {n: DEFAULT_BANDS[n] for n in combo_names}
        results[combo_key] = run_experiment(
            task_config={"bands": combo_bands}, **kwargs)

    return results


def ablation_connectivity(**kwargs) -> Dict[str, Dict[str, float]]:
    """Experiment 3: Connectivity measure ablation."""
    results = {}
    for measure in ("coherence", "wpli"):
        logger.info("=== Connectivity: %s ===", measure)
        results[measure] = run_experiment(
            task_config={
                "adjacency_type": "functional",
                "connectivity_measure": measure,
            }, **kwargs)
    return results


def ablation_model_comparison(**kwargs) -> Dict[str, Dict[str, float]]:
    """Experiment 4: Model comparison — GCN vs GAT (novel contribution)."""
    results = {}
    for model_type in ("gcn", "gat"):
        logger.info("=== Model: %s ===", model_type.upper())
        results[model_type] = run_experiment(
            task_config={}, model_type=model_type, **kwargs)
    return results


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EEG-GCNN Combined Pipeline — Dataset/Task + Model"
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="Root directory of EEG-GCNN data (TUAB + LEMON)",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic data (no downloads needed)",
    )
    parser.add_argument(
        "--model", type=str, default="gcn", choices=["gcn", "gat"],
        help="Model architecture: gcn (paper default) or gat (novel)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=["all", "adjacency", "bands", "connectivity", "model"],
        help="Which ablation experiment to run",
    )
    args = parser.parse_args()

    if not args.demo and args.root is None:
        parser.error("--root is required unless --demo is set")

    dataset = None
    if not args.demo:
        from pyhealth.datasets import EEGGCNNDataset
        dataset = EEGGCNNDataset(root=args.root)
        dataset.stats()

    if args.demo:
        logger.info(
            "Running in DEMO mode with synthetic data "
            "(results are illustrative, not meaningful)"
        )

    run_kwargs = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_type=args.model,
        demo=args.demo,
        dataset=dataset,
    )

    all_results = {}

    if args.experiment in ("all", "adjacency"):
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT 1: Adjacency Type Ablation")
        logger.info("=" * 60)
        all_results["adjacency"] = ablation_adjacency(**run_kwargs)

    if args.experiment in ("all", "bands"):
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT 2: Frequency Band Ablation")
        logger.info("=" * 60)
        all_results["bands"] = ablation_frequency_bands(**run_kwargs)

    if args.experiment in ("all", "connectivity"):
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT 3: Connectivity Measure Ablation")
        logger.info("=" * 60)
        all_results["connectivity"] = ablation_connectivity(**run_kwargs)

    if args.experiment in ("all", "model"):
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT 4: Model Comparison (GCN vs GAT)")
        logger.info("=" * 60)
        # Model comparison tests both; remove model_type from kwargs
        kw = dict(run_kwargs)
        kw.pop("model_type")
        all_results["model"] = ablation_model_comparison(**kw)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION RESULTS SUMMARY")
    logger.info("=" * 60)
    for exp_name, exp_results in all_results.items():
        logger.info("\n--- %s ---", exp_name)
        for config, metrics in exp_results.items():
            logger.info("  %-30s  AUC=%.4f", config, metrics["auc"])

    output_path = "eeg_gcnn_combined_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nResults saved to %s", output_path)


if __name__ == "__main__":
    main()
