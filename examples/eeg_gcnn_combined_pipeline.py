"""Combined EEG-GCNN Pipeline — Dataset/Task + Model Integration.

Self-contained end-to-end pipeline combining both team contributions:
  - Dataset & Task (Option 1 -- jburhan2):
      EEGGCNNRawDataset, EEGGCNNDiseaseDetection — PSD feature extraction,
      graph adjacency matrices, preprocessing pipeline
  - Model (Option 2 -- racoffey2):
      EEGGraphConvNet (GCN, paper baseline) and EEGGATConvNet (GAT, novel)

Model architectures, training hyperparameters, and evaluation protocol are
faithful to racoffey2's original implementation, which reproduces the paper.

Paper:
    Wagh, N. & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
    Electroencephalogram-based Neurological Disease Diagnosis using a
    Domain-guided Graph Convolutional Neural Network. ML4H @ NeurIPS 2020.
    https://proceedings.mlr.press/v136/wagh20a.html

Usage:
    # Demo mode -- synthetic data, no downloads needed (~1 min)
    python eeg_gcnn_combined_pipeline.py --demo

    # FigShare pre-computed features (225K windows, 1593 patients)
    python eeg_gcnn_combined_pipeline.py \\
        --figshare /path/to/figshare_upload_FINAL

    # Real EEG data via PyHealth dataset pipeline
    python eeg_gcnn_combined_pipeline.py --root /path/to/eeg-gcnn-data

    # Run specific experiments
    python eeg_gcnn_combined_pipeline.py --figshare /path --experiment model
    python eeg_gcnn_combined_pipeline.py --demo --experiment adjacency

Dependencies (demo/figshare mode):
    torch, torch_geometric, numpy, scikit-learn, joblib
"""

import argparse
import json
import logging
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from torch_geometric.nn import (
    BatchNorm,
    GATConv,
    GCNConv,
    global_add_pool,
)
from torch_geometric.utils import dense_to_sparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ===================================================================
# Constants (from EEG-GCNN paper, Section 4)
# ===================================================================

NUM_CHANNELS = 8
NUM_BANDS = 6

DEFAULT_BANDS = ["delta", "theta", "alpha", "lower_beta", "higher_beta", "gamma"]
BAND_RANGES: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "lower_beta": (12.0, 20.0),
    "higher_beta": (20.0, 30.0),
    "gamma": (30.0, 50.0),
}

# 8 bipolar channels (same order as PSD feature extraction)
BIPOLAR_CHANNELS = [
    "F7-F3", "F8-F4", "T7-C3", "T8-C4",
    "P7-P3", "P8-P4", "O1-P3", "O2-P4",
]

# Reference electrode labels for spatial adjacency (10-10 system midpoints)
# T3=T7, T4=T8 in 10-20 system; O1/O2 used where PO3/PO4 unavailable
REF_ELECTRODES = ["F5", "F6", "C5", "C6", "P5", "P6", "O1", "O2"]


# ===================================================================
# Models (Option 2 contribution -- racoffey2)
#
# Architectures match racoffey2's originals:
#   code_psd_deep_eeg_gcnn/EEGGraphConvNet.py
#   code_psd_deep_eeg_gcnn_GAT/EEGGATConvNet.py
# ===================================================================


class EEGGraphConvNet(nn.Module):
    """Deep EEG-GCNN model using GCNConv layers (paper baseline).

    Architecture: 4 GCN layers (6->16->32->64->50), BatchNorm on last
    conv layer, global add pooling, 3-layer FC classifier (50->30->20->2).

    Faithful to racoffey2's implementation which reproduces the paper.
    """

    def __init__(self, num_node_features: int = 6, num_classes: int = 2):
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
        self.fc3 = nn.Linear(20, num_classes)

        # Xavier initialization on FC layers (matches original)
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
    """EEG Graph Attention Network -- novel model contribution (racoffey2).

    Replaces GCNConv with multi-head GATConv layers. Uses concat=True
    so attention heads are concatenated (dims multiply by num_heads).

    Architecture: 4 GAT layers with heads=4 (concat=True for layers 1-3,
    heads=1 for layer 4), BatchNorm on last layer, same FC classifier.

    Faithful to racoffey2's code_psd_deep_eeg_gcnn_GAT/EEGGATConvNet.py.
    """

    def __init__(self, num_node_features: int = 6, num_classes: int = 2,
                 heads: int = 4):
        super().__init__()
        # concat=True: output dim = out_channels * heads
        # Layer 1: 6 -> 16*4 = 64
        self.conv1 = GATConv(num_node_features, 16, heads=heads, concat=True,
                             negative_slope=0.2, dropout=0.0,
                             add_self_loops=False, edge_dim=1)
        # Layer 2: 64 -> 32*4 = 128
        self.conv2 = GATConv(16 * heads, 32, heads=heads, concat=True,
                             negative_slope=0.2, dropout=0.0,
                             add_self_loops=False, edge_dim=1)
        # Layer 3: 128 -> 16*4 = 64
        self.conv3 = GATConv(32 * heads, 16, heads=heads, concat=True,
                             negative_slope=0.2, dropout=0.0,
                             add_self_loops=False, edge_dim=1)
        # Layer 4: 64 -> 50 (heads=1, no concat)
        self.conv4 = GATConv(16 * heads, 50, heads=1, concat=False,
                             negative_slope=0.2, dropout=0.0,
                             add_self_loops=False, edge_dim=1)
        self.conv4_bn = BatchNorm(50)

        self.fc1 = nn.Linear(50, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, num_classes)

        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_normal_(fc.weight, gain=1)

    def forward(self, x, edge_index, edge_weight, batch):
        # GAT uses edge_attr instead of edge_weight
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
    all_x, all_edge_index, all_edge_weight, batch_ids = [], [], [], []
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
# Data loading: FigShare pre-computed features
# ===================================================================


def load_electrode_positions(tsv_path: str) -> Dict[str, np.ndarray]:
    """Load 3D electrode positions from standard_1010.tsv."""
    positions = {}
    with open(tsv_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                positions[parts[0]] = np.array([
                    float(parts[1]), float(parts[2]), float(parts[3])
                ])
    return positions


def compute_geodesic_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Geodesic distance between two points on the unit sphere (arccos).

    Matches racoffey2's EEGGraphDataset.get_geodesic_distance().
    """
    dot = np.clip(np.dot(pos1, pos2), -1.0, 1.0)
    return math.acos(round(float(dot), 2))


def compute_spatial_distances(
    electrode_positions: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute pairwise geodesic distances between reference electrodes.

    Uses the same reference electrode mapping as racoffey2's
    EEGGraphDataset: midpoint approximated by a single 10-10 electrode.

    Returns:
        (64,) array of pairwise distances (8*8 fully connected)
    """
    positions = []
    for name in REF_ELECTRODES:
        if name not in electrode_positions:
            raise KeyError(f"Missing electrode position for {name}")
        positions.append(electrode_positions[name])

    distances = []
    for i in range(NUM_CHANNELS):
        for j in range(NUM_CHANNELS):
            distances.append(
                compute_geodesic_distance(positions[i], positions[j])
            )

    # Normalize to [0, 1] (matches racoffey2's normalization)
    a = np.array(distances, dtype=np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return a


def load_figshare_data(data_dir: str):
    """Load all FigShare data.

    Returns:
        X: (N, 48) PSD features -- 8 channels x 6 bands
        y: (N,) labels -- 0='diseased', 1='healthy'
        coh: (N, 64) spectral coherence values
        patient_ids: (N,) patient IDs as strings
        spatial_dists: (64,) normalized geodesic distances
    """
    import pandas as pd

    logger.info("Loading FigShare data from %s ...", data_dir)

    X = joblib.load(os.path.join(data_dir, "psd_features_data_X"))
    y_raw = joblib.load(os.path.join(data_dir, "labels_y"))
    coh = np.load(os.path.join(data_dir, "spec_coh_values.npy"))

    meta = pd.read_csv(
        os.path.join(data_dir, "master_metadata_index.csv"), low_memory=False
    )
    patient_ids = meta["patient_ID"].astype(str).values

    # Convert string labels to 0/1 (matches np.unique mapping:
    # 'diseased'=0, 'healthy'=1)
    label_mapping, y = np.unique(y_raw, return_inverse=True)
    logger.info("Label mapping: %s", dict(enumerate(label_mapping)))

    # L2 normalize features per sample (matches racoffey2's training_pipeline)
    X_norm = preprocessing.normalize(X.reshape(len(y), -1))
    X = X_norm.reshape(len(y), 48).astype(np.float32)

    # Compute spatial distances
    elec_pos = load_electrode_positions(
        os.path.join(data_dir, "standard_1010.tsv.txt")
    )
    spatial_dists = compute_spatial_distances(elec_pos)

    logger.info(
        "Loaded: %d windows, %d patients (diseased=%d, healthy=%d)",
        len(X), len(set(patient_ids)),
        (y == 0).sum(), (y == 1).sum(),
    )
    return X, y, coh, patient_ids, spatial_dists


def prepare_figshare_samples(
    X: np.ndarray,
    y: np.ndarray,
    coh: np.ndarray,
    patient_ids: np.ndarray,
    spatial_dists: np.ndarray,
    adjacency_type: str = "combined",
    band_indices: Optional[List[int]] = None,
) -> List[Dict]:
    """Convert raw FigShare arrays into sample dicts.

    Edge weights combine spatial distance + spectral coherence,
    matching racoffey2's EEGGraphDataset.get() method.

    Args:
        X: (N, 48) normalized PSD features
        y: (N,) integer labels (0=diseased, 1=healthy)
        coh: (N, 64) spectral coherence values
        patient_ids: (N,) patient IDs
        spatial_dists: (64,) normalized geodesic distances
        adjacency_type: "combined", "spatial", "functional", or "none"
        band_indices: Which band columns to keep (default: all 6)
    """
    if band_indices is None:
        band_indices = list(range(NUM_BANDS))
    n_bands = len(band_indices)

    samples = []
    for i in range(len(X)):
        # Reshape PSD: (48,) -> (8, 6) then select bands -> (8, n_bands)
        psd = X[i].reshape(NUM_CHANNELS, NUM_BANDS)[:, band_indices]

        # Build 8x8 adjacency from 64-element vectors
        if adjacency_type == "none":
            adj = np.eye(NUM_CHANNELS, dtype=np.float32)
        elif adjacency_type == "spatial":
            adj = spatial_dists.reshape(NUM_CHANNELS, NUM_CHANNELS).copy()
        elif adjacency_type == "functional":
            adj = coh[i].reshape(NUM_CHANNELS, NUM_CHANNELS).astype(np.float32)
        else:  # combined -- matches racoffey2: edge_weights = distances + coh
            adj = (spatial_dists + coh[i]).reshape(
                NUM_CHANNELS, NUM_CHANNELS
            ).astype(np.float32)

        samples.append({
            "patient_id": str(patient_ids[i]),
            "node_features": torch.FloatTensor(psd),
            "adj_matrix": torch.FloatTensor(adj),
            "label": int(y[i]),
        })
    return samples


# ===================================================================
# Dataset / DataLoader utilities
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
    """Collate sample dicts into a batched dict."""
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        vals = [s[k] for s in batch]
        if isinstance(vals[0], torch.Tensor):
            collated[k] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            collated[k] = torch.tensor(vals, dtype=torch.long)
        else:
            collated[k] = vals
    return collated


def split_by_patient(
    samples: List[Dict], test_size: float = 0.30, val_size: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split samples by patient_id into train/val/test.

    Matches racoffey2's splitting strategy:
      1. 70% train+val / 30% test (at patient level)
      2. 80% train / 20% val (within train+val)
    """
    from sklearn.model_selection import train_test_split

    patient_ids = sorted(set(s["patient_id"] for s in samples))

    train_val_pids, test_pids = train_test_split(
        patient_ids, test_size=test_size, random_state=seed
    )
    train_pids, val_pids = train_test_split(
        train_val_pids, test_size=val_size, random_state=seed
    )

    train_set = set(train_pids)
    val_set = set(val_pids)
    test_set = set(test_pids)

    train = [s for s in samples if s["patient_id"] in train_set]
    val = [s for s in samples if s["patient_id"] in val_set]
    test = [s for s in samples if s["patient_id"] in test_set]

    logger.info(
        "Split: train=%d (%d patients), val=%d (%d patients), "
        "test=%d (%d patients)",
        len(train), len(train_set), len(val), len(val_set),
        len(test), len(test_set),
    )
    return train, val, test


def make_loader(
    samples: List[Dict], batch_size: int, shuffle: bool = False,
    weighted_sampling: bool = False,
) -> DataLoader:
    """Create a DataLoader, optionally with weighted sampling for class balance.

    When weighted_sampling=True, uses WeightedRandomSampler to handle
    class imbalance (matches racoffey2's training pipeline).
    """
    dataset = DictDataset(samples)

    sampler = None
    if weighted_sampling:
        labels = np.array([s["label"] for s in samples])
        _, counts = np.unique(labels, return_counts=True)
        class_weights = 1.0 / counts
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(samples), replacement=True
        )
        shuffle = False  # mutually exclusive with sampler

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        sampler=sampler, collate_fn=collate_dict_batch,
    )


# ===================================================================
# Synthetic demo data
# ===================================================================


def generate_demo_samples(
    n_patients: int = 40,
    windows_per_patient: int = 5,
    n_bands: int = 6,
    adjacency_type: str = "combined",
    seed: int = 42,
) -> List[Dict]:
    """Generate synthetic samples matching real task output schema.

    Creates n_patients patients (half label-0, half label-1) each with
    windows_per_patient windows.
    """
    rng = np.random.RandomState(seed)
    samples = []
    for p in range(n_patients):
        pid = f"demo_{p:03d}"
        label = 0 if p < n_patients // 2 else 1
        for _ in range(windows_per_patient):
            psd = rng.randn(NUM_CHANNELS, n_bands).astype(np.float32)
            if label == 1:
                psd += 0.5

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
# Training / evaluation (matches racoffey2's training_pipeline.py)
# ===================================================================


def get_device():
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        nf = batch["node_features"].to(device)
        adj = batch["adj_matrix"].to(device)
        labels = batch["label"].to(device).long()

        x, ei, ew, bid = build_graph_batch(nf, adj, device)
        optimizer.zero_grad()
        logits = model(x, ei, ew, bid)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_patient_level(model, loader, loss_fn, device):
    """Evaluate model with patient-level aggregation and Youden's J.

    Matches racoffey2's collect_metrics() + get_patient_prediction():
      1. Collect window-level softmax probabilities
      2. Average probabilities per patient
      3. Select threshold via Youden's J statistic on patient-level ROC
      4. Compute patient-level AUC, precision, recall, F1, balanced accuracy
    """
    model.eval()
    all_probs, all_labels, all_pids = [], [], []
    total_loss, n = 0.0, 0

    for batch in loader:
        nf = batch["node_features"].to(device)
        adj = batch["adj_matrix"].to(device)
        labels = batch["label"].to(device).long()
        pids = batch["patient_id"]

        x, ei, ew, bid = build_graph_batch(nf, adj, device)
        logits = model(x, ei, ew, bid)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        n += 1

        # Softmax -> class probabilities (matches original: 2-class output)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_pids.extend(pids)

    all_probs = np.concatenate(all_probs)  # (N, 2)
    all_labels = np.array(all_labels)

    # --- Patient-level aggregation (matches get_patient_prediction) ---
    from collections import defaultdict
    patient_data = defaultdict(lambda: {"probs": [], "label": None})
    for i in range(len(all_labels)):
        pid = all_pids[i]
        patient_data[pid]["probs"].append(all_probs[i])
        patient_data[pid]["label"] = all_labels[i]

    patient_labels = []
    patient_probs = []
    for pid in sorted(patient_data.keys()):
        patient_labels.append(patient_data[pid]["label"])
        patient_probs.append(np.mean(patient_data[pid]["probs"], axis=0))

    patient_labels = np.array(patient_labels)
    patient_probs = np.array(patient_probs)  # (n_patients, 2)

    # Window-level AUC
    try:
        window_auc = float(roc_auc_score(all_labels, all_probs[:, 1]))
    except ValueError:
        window_auc = 0.5

    # Patient-level AUC
    try:
        patient_auc = float(roc_auc_score(patient_labels, patient_probs[:, 1]))
    except ValueError:
        patient_auc = 0.5

    # Youden's J for optimal threshold (matches original)
    try:
        fpr, tpr, thresholds = roc_curve(
            patient_labels, patient_probs[:, 1], pos_label=1
        )
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
    except (ValueError, IndexError):
        optimal_threshold = 0.5

    # Patient-level predictions using optimal threshold
    patient_preds = (patient_probs[:, 1] >= optimal_threshold).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = float(precision_score(
            patient_labels, patient_preds, pos_label=0, zero_division=0))
        recall_val = float(recall_score(
            patient_labels, patient_preds, pos_label=0, zero_division=0))
        f1 = float(f1_score(
            patient_labels, patient_preds, pos_label=0, zero_division=0))
        bal_acc = float(balanced_accuracy_score(patient_labels, patient_preds))

    return {
        "patient_auc": patient_auc,
        "window_auc": window_auc,
        "precision": precision,
        "recall": recall_val,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "threshold": float(optimal_threshold),
        "loss": total_loss / max(n, 1),
        "n_patients": len(patient_labels),
    }


def run_experiment(
    samples: List[Dict],
    model_type: str = "gcn",
    num_node_features: int = 6,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 0.01,
) -> Dict[str, float]:
    """Run a single experiment: split, train, evaluate.

    Training matches racoffey2's training_pipeline.py:
      - CrossEntropyLoss (2-class output)
      - SGD optimizer with lr=0.01
      - MultiStepLR scheduler (decay every 10 epochs)
      - WeightedRandomSampler for class imbalance
      - 30 epochs (configurable)
    """
    device = get_device()
    logger.info("Using device: %s", device)

    train_samples, val_samples, test_samples = split_by_patient(samples)
    train_loader = make_loader(
        train_samples, batch_size, weighted_sampling=True)
    val_loader = make_loader(val_samples, batch_size)
    test_loader = make_loader(test_samples, batch_size)

    if model_type == "gat":
        model = EEGGATConvNet(
            num_node_features=num_node_features, num_classes=2
        ).to(device)
    else:
        model = EEGGraphConvNet(
            num_node_features=num_node_features, num_classes=2
        ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[i * 10 for i in range(1, 26)], gamma=0.1
    )

    best_val_auc = 0.0
    best_state = None
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        val_metrics = evaluate_patient_level(
            model, val_loader, loss_fn, device)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "  Epoch %d/%d -- train_loss=%.4f "
                "val_patient_auc=%.4f val_bal_acc=%.4f",
                epoch + 1, epochs, train_loss,
                val_metrics["patient_auc"],
                val_metrics["balanced_accuracy"],
            )
        if val_metrics["patient_auc"] > best_val_auc:
            best_val_auc = val_metrics["patient_auc"]
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_patient_level(
        model, test_loader, loss_fn, device)
    logger.info(
        "  >> TEST: patient_AUC=%.4f  bal_acc=%.4f  F1=%.4f  "
        "precision=%.4f  recall=%.4f  (%d patients)",
        test_metrics["patient_auc"], test_metrics["balanced_accuracy"],
        test_metrics["f1"], test_metrics["precision"],
        test_metrics["recall"], test_metrics["n_patients"],
    )
    return test_metrics


# ===================================================================
# Ablation experiments
# ===================================================================


def ablation_adjacency(data_kwargs, **run_kwargs):
    """Experiment 1: Adjacency type ablation."""
    results = {}
    for adj_type in ("combined", "spatial", "functional", "none"):
        logger.info("=== Adjacency type: %s ===", adj_type)
        samples = data_kwargs["prepare_fn"](
            adjacency_type=adj_type, **data_kwargs["data_args"])
        results[adj_type] = run_experiment(samples, **run_kwargs)
    return results


def ablation_frequency_bands(data_kwargs, **run_kwargs):
    """Experiment 2: Frequency band ablation -- individual + progressive."""
    results = {}

    # Individual bands
    for i, band_name in enumerate(DEFAULT_BANDS):
        logger.info("=== Single band: %s ===", band_name)
        samples = data_kwargs["prepare_fn"](
            band_indices=[i], **data_kwargs["data_args"])
        results[band_name] = run_experiment(
            samples, num_node_features=1, **run_kwargs)

    # Progressive combinations
    for k in range(2, len(DEFAULT_BANDS) + 1):
        combo_key = "+".join(DEFAULT_BANDS[:k])
        logger.info("=== Band combination: %s ===", combo_key)
        samples = data_kwargs["prepare_fn"](
            band_indices=list(range(k)), **data_kwargs["data_args"])
        results[combo_key] = run_experiment(
            samples, num_node_features=k, **run_kwargs)

    return results


def ablation_connectivity(data_kwargs, **run_kwargs):
    """Experiment 3: Connectivity measure.

    FigShare data only contains spectral coherence. WPLI would require
    raw EDF files. We report coherence and spatial-only as baseline.
    """
    results = {}
    logger.info("=== Connectivity: coherence (combined = spatial + coh) ===")
    samples = data_kwargs["prepare_fn"](
        adjacency_type="combined", **data_kwargs["data_args"])
    results["coherence"] = run_experiment(samples, **run_kwargs)

    logger.info("=== Connectivity: spatial-only (no functional) ===")
    samples = data_kwargs["prepare_fn"](
        adjacency_type="spatial", **data_kwargs["data_args"])
    results["spatial_only"] = run_experiment(samples, **run_kwargs)
    return results


def ablation_model_comparison(data_kwargs, **run_kwargs):
    """Experiment 4: GCN vs GAT model comparison."""
    results = {}
    samples = data_kwargs["prepare_fn"](
        adjacency_type="combined", **data_kwargs["data_args"])
    for model_type in ("gcn", "gat"):
        logger.info("=== Model: %s ===", model_type.upper())
        results[model_type] = run_experiment(
            samples, model_type=model_type, **run_kwargs)
    return results


# ===================================================================
# Main
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="EEG-GCNN Combined Pipeline -- Dataset/Task + Model"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic data (no downloads needed)",
    )
    group.add_argument(
        "--figshare", type=str, metavar="DIR",
        help="Path to figshare_upload_FINAL directory",
    )
    group.add_argument(
        "--root", type=str,
        help="Root directory of raw EEG data (TUAB + LEMON)",
    )
    parser.add_argument(
        "--model", type=str, default="gcn", choices=["gcn", "gat"],
        help="Model architecture: gcn (paper baseline) or gat (novel)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=["all", "adjacency", "bands", "connectivity", "model"],
        help="Which ablation experiment to run",
    )
    args = parser.parse_args()

    # --- Build data preparation function based on mode ---
    if args.demo:
        logger.info(
            "DEMO MODE -- synthetic data (results are illustrative only)")

        def prepare_fn(adjacency_type="combined", band_indices=None, **kw):
            n_bands = len(band_indices) if band_indices else 6
            return generate_demo_samples(
                n_bands=n_bands, adjacency_type=adjacency_type)

        data_kwargs = {"prepare_fn": prepare_fn, "data_args": {}}

    elif args.figshare:
        X, y, coh, patient_ids, spatial_dists = load_figshare_data(
            args.figshare)

        def prepare_fn(adjacency_type="combined", band_indices=None,
                       X=X, y=y, coh=coh, patient_ids=patient_ids,
                       spatial_dists=spatial_dists):
            return prepare_figshare_samples(
                X, y, coh, patient_ids, spatial_dists,
                adjacency_type=adjacency_type, band_indices=band_indices,
            )

        data_kwargs = {"prepare_fn": prepare_fn, "data_args": {}}

    else:
        # Real data via PyHealth dataset pipeline
        from pyhealth.datasets import EEGGCNNRawDataset
        from pyhealth.tasks import EEGGCNNDiseaseDetection

        dataset = EEGGCNNRawDataset(root=args.root)
        dataset.stats()

        def prepare_fn(adjacency_type="combined", band_indices=None,
                       dataset=dataset):
            bands = BAND_RANGES
            if band_indices is not None:
                band_names = [DEFAULT_BANDS[i] for i in band_indices]
                bands = {n: BAND_RANGES[n] for n in band_names}
            task = EEGGCNNDiseaseDetection(
                adjacency_type=adjacency_type, bands=bands)
            sample_ds = dataset.set_task(task)
            return [sample_ds[i] for i in range(len(sample_ds))]

        data_kwargs = {"prepare_fn": prepare_fn, "data_args": {}}

    run_kwargs = dict(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    all_results = {}

    if args.experiment in ("all", "adjacency"):
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT 1: Adjacency Type Ablation")
        logger.info("=" * 60)
        all_results["adjacency"] = ablation_adjacency(
            data_kwargs, **run_kwargs)

    if args.experiment in ("all", "bands"):
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT 2: Frequency Band Ablation")
        logger.info("=" * 60)
        all_results["bands"] = ablation_frequency_bands(
            data_kwargs, **run_kwargs)

    if args.experiment in ("all", "connectivity"):
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT 3: Connectivity Measure Ablation")
        logger.info("=" * 60)
        all_results["connectivity"] = ablation_connectivity(
            data_kwargs, **run_kwargs)

    if args.experiment in ("all", "model"):
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT 4: Model Comparison (GCN vs GAT)")
        logger.info("=" * 60)
        kw = dict(run_kwargs)
        kw.pop("model_type")
        all_results["model"] = ablation_model_comparison(data_kwargs, **kw)

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for exp_name, exp_results in all_results.items():
        logger.info("\n--- %s ---", exp_name)
        for config, metrics in exp_results.items():
            logger.info(
                "  %-35s  AUC=%.4f  bal_acc=%.4f  F1=%.4f",
                config,
                metrics.get("patient_auc", 0),
                metrics.get("balanced_accuracy", 0),
                metrics.get("f1", 0),
            )

    output_path = "eeg_gcnn_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nResults saved to %s", output_path)


if __name__ == "__main__":
    main()
