"""EEG-GCNN neurological disease classification with EEGGraphConvNet.

End-to-end example: load pre-computed EEG graph features, train a shallow
Graph Convolutional Network to distinguish neurological patients (TUAB)
from healthy controls (LEMON), and evaluate at the patient level.

Paper:
    Wagh, N., & Varatharajah, Y. (2020). EEG-GCNN: Augmenting
    Electroencephalogram-based Neurological Disease Diagnosis using a
    Domain-guided Graph Convolutional Neural Network.
    ML4H @ NeurIPS 2020. https://proceedings.mlr.press/v136/wagh20a.html

Dataset:
    FigShare pre-computed arrays (1,593 subjects, 225,334 windows).
    Download from:
    https://figshare.com/articles/dataset/13251509

    Place the five files in a local directory (e.g. ``data/eeg-gcnn/``):
        psd_features_data_X
        labels_y
        master_metadata_index.csv
        spec_coh_values.npy
        standard_1010.tsv.txt

    Alternatively, generate them from raw TUAB + LEMON EDF/BrainVision
    files using ``examples/eeg_gcnn/pre_compute.py``.

Usage:
    # Train on FigShare pre-computed data
    python examples/eeg_gcnn_classification_eeg_graph_conv_net.py \\
        --data_root /path/to/eeg-gcnn

    # Quick dev run with a small patient subset
    python examples/eeg_gcnn_classification_eeg_graph_conv_net.py \\
        --data_root /path/to/eeg-gcnn --max_patients 50 --num_epochs 5

    # Spatial-only adjacency (alpha=1.0), 2 folds
    python examples/eeg_gcnn_classification_eeg_graph_conv_net.py \\
        --data_root /path/to/eeg-gcnn --alpha 1.0 --num_folds 2

Authors:
    Jimmy Burhan  (jburhan2@illinois.edu) — Dataset & Task
    Robert Coffey (rc37@illinois.edu)    — Model & Training
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from pyhealth.datasets import EEGGCNNDataset, get_dataloader
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.models import EEGGraphConvNet
from pyhealth.tasks import EEGGCNNClassification


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="EEG-GCNN disease classification with GCN"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to directory containing the 5 FigShare arrays"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Adjacency mix: 1.0=spatial only, 0.0=functional only (default: 0.5)"
    )
    parser.add_argument("--num_folds",   type=int,   default=10)
    parser.add_argument("--num_epochs",  type=int,   default=100)
    parser.add_argument("--lr",          type=float, default=0.01)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument(
        "--max_patients", type=int, default=None,
        help="Limit number of patients (None = full dataset)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        output = model(
            node_features=batch["node_features"].to(device),
            adj_matrix=batch["adj_matrix"].to(device),
            label=batch["label"].to(device),
        )
        output["loss"].backward()
        optimizer.step()
        total_loss += output["loss"].item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return patient-level AUROC via mean-pooled window probabilities."""
    model.eval()
    pid_probs: dict = {}
    pid_labels: dict = {}

    for batch in loader:
        output = model(
            node_features=batch["node_features"].to(device),
            adj_matrix=batch["adj_matrix"].to(device),
            label=batch["label"].to(device),
        )
        probs  = output["y_prob"].cpu().numpy()
        labels = output["y_true"].cpu().numpy()
        pids   = batch["patient_id"]

        for pid, prob, label in zip(pids, probs, labels):
            pid_probs.setdefault(pid, []).append(float(prob))
            pid_labels[pid] = int(label)

    patient_probs  = np.array([np.mean(pid_probs[p])  for p in pid_probs])
    patient_labels = np.array([pid_labels[p]           for p in pid_probs])
    return roc_auc_score(patient_labels, patient_probs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load dataset
    print(f"\nLoading EEGGCNNDataset from: {args.data_root}")
    dataset = EEGGCNNDataset(root=args.data_root, alpha=args.alpha)

    # 2. Apply task
    sample_ds = dataset.set_task(EEGGCNNClassification())
    print(f"Total windows: {len(sample_ds)}")

    # 3. Optionally limit to a patient subset
    all_patients = list(dataset.patients.keys())
    if args.max_patients:
        all_patients = all_patients[:args.max_patients]
        sample_ds = sample_ds.filter(
            lambda s: s["patient_id"] in set(all_patients)
        )

    # 4. 10-fold cross-validation (patient-level split)
    patient_labels = np.array([
        dataset.patients[p].get_events("metadata")[0].label
        for p in all_patients
    ])
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    fold_aucs = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(all_patients, patient_labels)
    ):
        train_pids = {all_patients[i] for i in train_idx}
        test_pids  = {all_patients[i] for i in test_idx}

        train_split = sample_ds.filter(lambda s: s["patient_id"] in train_pids)
        test_split  = sample_ds.filter(lambda s: s["patient_id"] in test_pids)

        # Handle class imbalance (~7:1 TUAB:LEMON) with weighted sampling
        labels      = [s["label"] for s in train_split]
        class_counts = np.bincount(labels)
        weights     = 1.0 / class_counts[labels]
        sampler     = WeightedRandomSampler(weights, len(weights))

        train_loader = DataLoader(
            train_split, batch_size=args.batch_size, sampler=sampler
        )
        test_loader  = DataLoader(
            test_split, batch_size=args.batch_size, shuffle=False
        )

        # 5. Initialise model
        model = EEGGraphConvNet(
            dataset=dataset,
            feature_keys=["node_features", "adj_matrix"],
            label_key="label",
            mode="binary",
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # 6. Train
        for epoch in range(args.num_epochs):
            loss = train_epoch(model, train_loader, optimizer, device)
            if (epoch + 1) % 20 == 0:
                print(
                    f"  Fold {fold+1}/{args.num_folds} "
                    f"Epoch {epoch+1}/{args.num_epochs}  loss={loss:.4f}"
                )

        # 7. Evaluate
        auc = evaluate(model, test_loader, device)
        fold_aucs.append(auc)
        print(f"  Fold {fold+1} patient AUROC: {auc:.4f}")

    print(
        f"\nFinal patient AUROC: "
        f"{np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}"
    )


if __name__ == "__main__":
    main()
