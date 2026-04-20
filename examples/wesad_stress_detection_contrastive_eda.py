"""
Full pipeline example: Contrastive EDA pre-training and stress detection on WESAD.

Reproduces the core experiment from:
    Matton, K., Lewis, R., Guttag, J., & Picard, R. (2023).
    "Contrastive Learning of Electrodermal Activity Representations
    for Stress Detection." CHIL 2023.

This script demonstrates:
    1. Loading and windowing the WESAD dataset
    2. Contrastive pre-training of the EDA encoder
    3. Fine-tuning for binary stress detection
    4. Ablation: full vs. generic-only vs. EDA-specific augmentations

Usage (real data):
    python examples/wesad_stress_detection_contrastive_eda.py \
        --data_root /path/to/WESAD \
        --augmentation_group full \
        --pretrain_epochs 50 \
        --finetune_epochs 20

Usage (synthetic demo, no download required):
    python examples/wesad_stress_detection_contrastive_eda.py \
        --synthetic \
        --pretrain_epochs 2 \
        --finetune_epochs 2

Authors:
    Megan Saunders, Jennifer Miranda, Jesus Torres
    {meganas4, jm123, jesusst2}@illinois.edu
"""

import argparse
import logging
import os
import pickle
import tempfile
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from pyhealth.datasets import WESADDataset
from pyhealth.models import ContrastiveEDAModel
from pyhealth.tasks.stress_detection import StressDetectionDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# EDA sample rate of the Empatica E4 wrist device (Hz)
_EDA_SAMPLE_RATE = 4


def _make_synthetic_wesad(root: str, subjects: List[str], n_seconds: int = 120) -> None:
    """Creates a minimal synthetic WESAD directory for demo purposes.

    Generates one pickle file per subject containing random EDA and label
    arrays that match the real WESAD file format, allowing the full pipeline
    to run without downloading the actual dataset.

    Args:
        root: Directory in which to create per-subject subdirectories.
        subjects: Subject IDs to generate (e.g. ['S2', 'S3']).
        n_seconds: Duration of synthetic recording per subject in seconds.
    """
    n_eda = n_seconds * _EDA_SAMPLE_RATE
    n_chest = n_eda * 175  # ~700 Hz chest device rate
    for sid in subjects:
        eda = np.random.rand(n_eda, 1).astype(np.float32)
        labels = np.ones(n_chest, dtype=int)
        labels[n_chest // 3: 2 * n_chest // 3] = 2  # stress segment
        data = {"signal": {"wrist": {"EDA": eda}}, "label": labels}
        subject_dir = os.path.join(root, sid)
        os.makedirs(subject_dir, exist_ok=True)
        with open(os.path.join(subject_dir, f"{sid}.pkl"), "wb") as f:
            pickle.dump(data, f)
    logger.info(f"Synthetic WESAD data written to {root} for subjects {subjects}")


# LNSO folds matching the authors' dataset_splits/WESAD/ files
LNSO_FOLDS = [
    ["S2", "S3"],
    ["S4", "S5"],
    ["S6", "S7"],
    ["S8", "S9"],
    ["S10", "S11"],
]


def pretrain(
    model: ContrastiveEDAModel,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> List[float]:
    """Runs contrastive pre-training loop.

    Args:
        model: ContrastiveEDAModel in pretrain mode.
        train_loader: DataLoader yielding (eda, label) tuples.
        epochs: Number of training epochs.
        device: Torch device.
        lr: Learning rate.

    Returns:
        List of per-epoch training losses.
    """
    model.set_pretrain_mode()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.pretrain_step(x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(train_loader)
        losses.append(avg)
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Pretrain epoch {epoch+1}/{epochs} loss={avg:.4f}")

    return losses


def finetune_and_evaluate(
    model: ContrastiveEDAModel,
    train_ds: StressDetectionDataset,
    test_ds: StressDetectionDataset,
    label_fraction: float,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    batch_size: int = 64,
    freeze_encoder: bool = False,
) -> float:
    """Fine-tunes model on a fraction of labeled data and evaluates.

    Args:
        model: ContrastiveEDAModel with pre-trained encoder.
        train_ds: Training StressDetectionDataset.
        test_ds: Test StressDetectionDataset.
        label_fraction: Fraction of training labels to use (e.g. 0.01 = 1%).
        epochs: Number of fine-tuning epochs.
        device: Torch device.
        lr: Learning rate.
        batch_size: Batch size.
        freeze_encoder: Whether to freeze encoder during fine-tuning.

    Returns:
        Balanced accuracy on the test set.
    """
    # Subsample labeled training data
    n_labeled = max(1, int(len(train_ds) * label_fraction))
    indices = np.random.choice(len(train_ds), size=n_labeled, replace=False)
    labeled_samples = [train_ds.samples[i] for i in indices]
    labeled_ds = StressDetectionDataset(labeled_samples)

    train_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model.freeze_encoder = freeze_encoder
    model.set_finetune_mode(num_classes=2)
    model.to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss, _ = model.finetune_step(x, y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)["logit"]
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    return balanced_accuracy_score(all_labels, all_preds)


def run_fold(
    fold_idx: int,
    test_subjects: List[str],
    all_samples: List[Dict],
    augmentation_group: str,
    pretrain_epochs: int,
    finetune_epochs: int,
    label_fraction: float,
    device: torch.device,
    output_dir: str,
    window_size: int = 60,
) -> float:
    """Runs one LNSO fold: pretrain on train subjects, evaluate on test subjects.

    Args:
        fold_idx: Fold number for logging.
        test_subjects: Subject IDs held out for testing.
        all_samples: All windowed samples from WESADDataset.
        augmentation_group: Augmentation group name for ContrastiveEDAModel.
        pretrain_epochs: Number of contrastive pre-training epochs.
        finetune_epochs: Number of supervised fine-tuning epochs.
        label_fraction: Fraction of labeled training data to use.
        device: Torch device.
        output_dir: Directory to save encoder checkpoints.
        window_size: EDA window size in samples.

    Returns:
        Balanced accuracy for this fold.
    """
    logger.info(f"\nFold {fold_idx} | test subjects: {test_subjects}")

    full_task = StressDetectionDataset(all_samples)
    train_ds, test_ds = full_task.get_subject_splits(test_subjects)

    logger.info(f"  Train windows: {len(train_ds)} | Test windows: {len(test_ds)}")

    # Pretrain
    model = ContrastiveEDAModel(
        window_size=window_size,
        num_classes=2,
        augmentation_group=augmentation_group,
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    pretrain(model, train_loader, epochs=pretrain_epochs, device=device)

    # Save encoder checkpoint
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(
        output_dir, f"encoder_fold{fold_idx}_{augmentation_group}.pt"
    )
    model.save_encoder(ckpt_path)

    # Finetune and evaluate
    bal_acc = finetune_and_evaluate(
        model=model,
        train_ds=train_ds,
        test_ds=test_ds,
        label_fraction=label_fraction,
        epochs=finetune_epochs,
        device=device,
    )

    logger.info(f"  Fold {fold_idx} balanced accuracy: {bal_acc:.4f}")
    return bal_acc


def run_ablation(
    all_samples: List[Dict],
    pretrain_epochs: int,
    finetune_epochs: int,
    label_fraction: float,
    device: torch.device,
    output_dir: str,
    window_size: int = 60,
) -> None:
    """Runs ablation study comparing augmentation groups.

    Evaluates three conditions across all LNSO folds:
        - full: all augmentations (EDA-specific + generic)
        - generic_only: Gaussian noise, temporal cutout, amplitude scaling
        - eda_specific_only: tonic/phasic extraction, loose sensor artifact

    Args:
        all_samples: All windowed samples from WESADDataset.
        pretrain_epochs: Contrastive pre-training epochs.
        finetune_epochs: Fine-tuning epochs.
        label_fraction: Fraction of labeled training data.
        device: Torch device.
        output_dir: Output directory for checkpoints.
        window_size: EDA window size in samples.
    """
    groups = ["full", "generic_only", "eda_specific_only"]
    results = {g: [] for g in groups}

    for group in groups:
        logger.info(f"\n{'='*60}")
        logger.info(f"Augmentation group: {group}")
        logger.info(f"{'='*60}")
        for fold_idx, test_subjects in enumerate(LNSO_FOLDS):
            bal_acc = run_fold(
                fold_idx=fold_idx,
                test_subjects=test_subjects,
                all_samples=all_samples,
                augmentation_group=group,
                pretrain_epochs=pretrain_epochs,
                finetune_epochs=finetune_epochs,
                label_fraction=label_fraction,
                device=device,
                output_dir=output_dir,
                window_size=window_size,
            )
            results[group].append(bal_acc)

    # Print results table
    logger.info("\n" + "="*60)
    logger.info("ABLATION RESULTS")
    logger.info("="*60)
    logger.info(f"{'Augmentation Group':<25} {'Mean Bal Acc':>12} {'Std':>8}")
    logger.info("-"*60)
    for group in groups:
        scores = results[group]
        logger.info(
            f"{group:<25} {np.mean(scores):>12.4f} {np.std(scores):>8.4f}"
        )
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Contrastive EDA pre-training and stress detection on WESAD"
    )
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Path to WESAD dataset root directory. Omit when using --synthetic."
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic WESAD data and run a short demo (no download needed)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory to save encoder checkpoints"
    )
    parser.add_argument(
        "--augmentation_group", type=str, default="full",
        choices=["full", "generic_only", "eda_specific_only", "ablation"],
        help="Augmentation group to use. Use 'ablation' to run full ablation study."
    )
    parser.add_argument(
        "--pretrain_epochs", type=int, default=50,
        help="Number of contrastive pre-training epochs"
    )
    parser.add_argument(
        "--finetune_epochs", type=int, default=20,
        help="Number of supervised fine-tuning epochs"
    )
    parser.add_argument(
        "--label_fraction", type=float, default=0.01,
        help="Fraction of labeled training data to use (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--window_size", type=int, default=60,
        help="EDA window size in samples (default: 60 = 15 seconds at 4Hz)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    if not args.synthetic and args.data_root is None:
        parser.error("Provide --data_root <path> or pass --synthetic for a demo run.")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Optionally generate synthetic data for a quick demo run
    _tmp_dir = None
    if args.synthetic:
        _tmp_dir = tempfile.mkdtemp()
        demo_subjects = ["S2", "S3", "S4", "S5"]
        _make_synthetic_wesad(_tmp_dir, demo_subjects, n_seconds=120)
        data_root = _tmp_dir
        logger.info("Running in synthetic demo mode.")
    else:
        data_root = args.data_root

    # Load dataset
    logger.info(f"Loading WESAD from {data_root}")
    dataset = WESADDataset(
        root=data_root,
        window_size=args.window_size,
        step_size=10,
        label_map={1: 0, 2: 1},
    )
    logger.info(f"Total windows: {len(dataset)}")

    if args.augmentation_group == "ablation":
        run_ablation(
            all_samples=dataset.samples,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            label_fraction=args.label_fraction,
            device=device,
            output_dir=args.output_dir,
            window_size=args.window_size,
        )
    else:
        # Single augmentation group across all folds
        fold_scores = []
        for fold_idx, test_subjects in enumerate(LNSO_FOLDS):
            bal_acc = run_fold(
                fold_idx=fold_idx,
                test_subjects=test_subjects,
                all_samples=dataset.samples,
                augmentation_group=args.augmentation_group,
                pretrain_epochs=args.pretrain_epochs,
                finetune_epochs=args.finetune_epochs,
                label_fraction=args.label_fraction,
                device=device,
                output_dir=args.output_dir,
                window_size=args.window_size,
            )
            fold_scores.append(bal_acc)

        logger.info(f"\nMean balanced accuracy: {np.mean(fold_scores):.4f}")
        logger.info(f"Std: {np.std(fold_scores):.4f}")


if __name__ == "__main__":
    main()