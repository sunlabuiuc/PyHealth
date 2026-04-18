"""Pre-compute EEG graph features from raw TUAB/LEMON data and run GCN inference.

This script demonstrates the full raw-EEG → GCN pipeline:

    EEGGCNNRawDataset (raw EDF/BrainVision files)
        ↓  EEGGCNNDiseaseDetection task
    SampleDataset  (node_features: (8,6), adj_matrix: (8,8), label: int)
        ↓  EEGGraphConvNet
    Predictions  (AUC, loss)

It is intended to be run against the sample data shipped with the repo:

    pyhealth/eeg-gcnn-data/
        tuab/train/normal/01_tcp_ar/  ← 3 sample TUAB EDF files
        lemon/sub-010002/             ← 3 sample LEMON BrainVision recordings
        lemon/sub-010003/
        lemon/sub-010004/

Usage
-----
From the repo root:

    python examples/eeg_gcnn/pre_compute_gcnn.py

Or supply a different data root:

    python examples/eeg_gcnn/pre_compute_gcnn.py --root /path/to/eeg-gcnn-data

The script prints each processing stage, the shape of the produced tensors,
and performs a single forward pass through EEGGraphConvNet to verify
end-to-end compatibility.
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve repo root so the script runs from any working directory.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pyhealth"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from pyhealth.datasets.eeg_gcnn_raw import EEGGCNNRawDataset
from pyhealth.tasks.eeg_gcnn_nd_detection import EEGGCNNDiseaseDetection
from pyhealth.models.eeg_gcnn import EEGGraphConvNet
from pyhealth.datasets.sample_dataset import InMemorySampleDataset


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = str(REPO_ROOT / "pyhealth" / "eeg-gcnn-data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing TUAB and/or LEMON raw data.",
    )
    p.add_argument(
        "--adjacency",
        choices=["combined", "spatial", "functional", "none"],
        default="combined",
        help="Adjacency matrix type (default: combined).",
    )
    p.add_argument(
        "--subset",
        choices=["both", "tuab", "lemon"],
        default="both",
        help="Which data source to load (default: both).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for the forward pass (default: 32).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage 1 — Dataset: scan raw files and build metadata CSVs
# ---------------------------------------------------------------------------

def load_dataset(root: str, subset: str) -> EEGGCNNRawDataset:
    print(f"\n{'='*60}")
    print("  Stage 1 — Dataset: EEGGCNNRawDataset")
    print(f"  root   : {root}")
    print(f"  subset : {subset}")
    print(f"{'='*60}")

    dataset = EEGGCNNRawDataset(root=root, subset=subset)
    dataset.stats()
    return dataset


# ---------------------------------------------------------------------------
# Stage 2 — Task: extract PSD features + adjacency matrix per window
# ---------------------------------------------------------------------------

def apply_task(
    dataset: EEGGCNNRawDataset,
    adjacency_type: str,
) -> list:
    print(f"\n{'='*60}")
    print("  Stage 2 — Task: EEGGCNNDiseaseDetection")
    print(f"  adjacency_type : {adjacency_type}")
    print(f"{'='*60}")

    task = EEGGCNNDiseaseDetection(adjacency_type=adjacency_type)
    sample_dataset = dataset.set_task(task)

    n = len(sample_dataset)
    print(f"\n  Produced {n} windows from {len(dataset.unique_patient_ids)} patients")

    # Inspect the first sample to confirm output schema.
    s = sample_dataset[0]
    nf = s["node_features"]
    am = s["adj_matrix"]
    label = s["label"]

    print(f"\n  Sample[0] keys      : {list(s.keys())}")
    print(f"  node_features dtype : {nf.dtype}, shape: {nf.shape}")
    print(f"  adj_matrix dtype    : {am.dtype}, shape: {am.shape}")
    print(f"  label               : {label}  (0=diseased/TUAB, 1=healthy/LEMON)")
    print(f"\n  node_features[0]    : {nf[0].numpy()}")
    print(f"  adj_matrix[0]       : {am[0].numpy()}")

    return sample_dataset


# ---------------------------------------------------------------------------
# Stage 3 — Model: single forward pass through EEGGraphConvNet
# ---------------------------------------------------------------------------

def run_model(sample_dataset, batch_size: int) -> None:
    print(f"\n{'='*60}")
    print("  Stage 3 — Model: EEGGraphConvNet")
    print(f"{'='*60}")

    # Build a minimal InMemorySampleDataset for model initialisation.
    init_samples = [
        {
            "node_features": torch.randn(8, 6).float(),
            "adj_matrix": torch.eye(8).float(),
            "label": i % 2,
        }
        for i in range(4)
    ]
    dummy_ds = InMemorySampleDataset(
        samples=init_samples,
        input_schema={"node_features": "tensor", "adj_matrix": "tensor"},
        output_schema={"label": "binary"},
    )

    model = EEGGraphConvNet(dataset=dummy_ds, num_node_features=6)
    model.eval()

    # Collect one batch from the real sample_dataset.
    indices = list(range(min(batch_size, len(sample_dataset))))
    node_features_list = []
    adj_matrix_list = []
    labels_list = []

    for i in indices:
        s = sample_dataset[i]
        nf = s["node_features"]
        am = s["adj_matrix"]
        if isinstance(nf, np.ndarray):
            nf = torch.FloatTensor(nf)
        if isinstance(am, np.ndarray):
            am = torch.FloatTensor(am)
        node_features_list.append(nf)
        adj_matrix_list.append(am)
        labels_list.append(float(s["label"]))

    node_features_batch = torch.stack(node_features_list)  # (B, 8, 6)
    adj_matrix_batch = torch.stack(adj_matrix_list)        # (B, 8, 8)
    labels_batch = torch.FloatTensor(labels_list).unsqueeze(1)  # (B, 1)

    print(f"\n  Input  node_features : {node_features_batch.shape}")
    print(f"  Input  adj_matrix    : {adj_matrix_batch.shape}")
    print(f"  Input  labels        : {labels_batch.shape}")

    with torch.no_grad():
        out = model(
            node_features=node_features_batch,
            adj_matrix=adj_matrix_batch,
            label=labels_batch,
        )

    print(f"\n  Output keys    : {list(out.keys())}")
    print(f"  y_prob shape   : {out['y_prob'].shape}")
    print(f"  y_true shape   : {out['y_true'].shape}")
    print(f"  loss           : {out['loss'].item():.4f}")
    print(f"\n  First 5 predicted probabilities: {out['y_prob'][:5].squeeze().tolist()}")
    print(f"  First 5 true labels            : {out['y_true'][:5].squeeze().tolist()}")

    print(f"\n  ✓ GCN forward pass successful — tensors flow end-to-end from")
    print(f"    raw TUAB/LEMON EDF files to model predictions.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.root):
        print(f"ERROR: data root not found: {args.root}")
        print("Set --root to a directory containing TUAB and/or LEMON data.")
        sys.exit(1)

    print("\nEEG-GCNN Pre-compute Pipeline  (GCN)")
    print("=" * 60)
    print(f"  Paper  : Wagh & Varatharajah (2020), ML4H @ NeurIPS 2020")
    print(f"  Model  : EEGGraphConvNet (GCN)")
    print(f"  Data   : {args.root}")
    print(f"  Output : node_features (8,6)  +  adj_matrix (8,8)  +  label")

    dataset = load_dataset(args.root, args.subset)
    sample_dataset = apply_task(dataset, args.adjacency)
    run_model(sample_dataset, args.batch_size)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("  Outputs are ready for training — see training_pipeline_shallow_gcnn.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
