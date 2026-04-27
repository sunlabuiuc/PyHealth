"""PH2 Melanoma Classifier — 5-fold CV (whole mode, val_strategy=none).

Replicates the PH2-trained classifiers from Jin et al. (CHIL 2025) for the
whole-image mode.  Results are used in:
  - Table 2: PH2-trained classifiers evaluated on PH2/ISIC/HAM10000
  - Table 4: PH2-trained classifiers evaluated on diffusion-augmented PH2

Setup
-----
  PH2 dataset root (--root):  ~/ph2/PH2-dataset-master/
    Expected contents: images/, PH2_simple_dataset.csv  (GitHub mirror layout)
    OR: PH2_Dataset_images/, PH2_dataset.xlsx           (original layout)
  Labels      : melanoma → 1 ; common_nevus / atypical_nevus → 0

Splits
------
  KFold(n_splits=5, shuffle=True, random_state=42) — identical to paper.
  Training uses the full train fold (no validation holdout).

Hyperparameters (paper-faithful)
---------------------------------
  Model   : ResNet-50, ImageNet pretrained (weights="DEFAULT")
  FC      : Linear(2048 → 1)
  Loss    : BCEWithLogitsLoss
  Optim   : Adam lr=1e-4
  Epochs  : 10
  Batch   : 32
  Input   : 224×224, ImageNet normalised

Outputs
-------
  Checkpoints: ~/ph2_checkpoints/whole/resnet50_fold{k}.pt  (k = 0..4)
  Per-fold AUROC printed to stdout.

Usage
-----
  pixi run -e base python examples/ph2_train_resnet50.py --root ~/ph2/PH2-dataset-master
  pixi run -e base python examples/ph2_train_resnet50.py --root ~/ph2/PH2-dataset-master --test 20
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torchvision import models

from pyhealth.datasets import PH2Dataset, create_sample_dataset, get_dataloader
from pyhealth.processors import DermoscopicImageProcessor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CKPT_DIR = Path(os.path.expanduser("~/ph2_checkpoints/whole"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def build_model():
    model = models.resnet50(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        imgs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)  # shape [B, 1], float32
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        probs = torch.sigmoid(model(batch["image"].to(DEVICE))).squeeze(1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch["label"].view(-1).numpy())
    return roc_auc_score(all_labels, all_probs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        default=os.path.expanduser("~/ph2/PH2-dataset-master"),
        help="Root directory of the PH2 dataset.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--test", type=int, default=0, help="Smoke-test: only use first N images")
    p.add_argument("--resume", action="store_true", help="Skip folds whose checkpoint exists")
    return p.parse_args()


def main():
    args = parse_args()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Ensure PH2 metadata is prepared, then build binary samples
    # ------------------------------------------------------------------
    PH2Dataset(root=args.root)  # validates layout and writes ph2_metadata_pyhealth.csv

    raw_samples = []
    meta = Path(args.root) / "ph2_metadata_pyhealth.csv"
    with open(meta) as f:
        for row in csv.DictReader(f):
            if row["path"] and row["diagnosis"]:
                raw_samples.append({
                    "image": row["path"],
                    "label": int(row["diagnosis"] == "melanoma"),
                })

    if args.test:
        raw_samples = raw_samples[: args.test]

    processor = DermoscopicImageProcessor(mode="whole")
    samples = create_sample_dataset(
        raw_samples,
        input_schema={"image": "image"},
        output_schema={"label": "binary"},
        input_processors={"image": processor},
    )

    indices = np.arange(len(samples))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
        ckpt_path = CKPT_DIR / f"resnet50_fold{fold}.pt"
        if args.resume and ckpt_path.exists():
            print(f"[Fold {fold}] checkpoint exists — skipping training")
            model = build_model()
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        else:
            print(f"\n{'='*50}\nFold {fold}\n{'='*50}")
            train_loader = get_dataloader(
                samples.subset(indices[train_idx]),
                batch_size=args.batch,
                shuffle=True,
            )

            model = build_model()
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            for epoch in range(args.epochs):
                loss = train_one_epoch(model, train_loader, criterion, optimizer)
                print(f"  Epoch {epoch+1:2d}/{args.epochs}  loss={loss:.4f}")

            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved → {ckpt_path}")

        test_loader = get_dataloader(
            samples.subset(indices[test_idx]),
            batch_size=args.batch,
            shuffle=False,
        )
        auroc = evaluate(model, test_loader)
        aurocs.append(auroc)
        print(f"  Fold {fold} test AUROC: {auroc:.4f}")

    mean, std = np.mean(aurocs), np.std(aurocs, ddof=1)
    print(f"\nPH2 whole — 5-fold AUROC: {mean:.4f} ±{std:.4f}")
    print("Per-fold:", [f"{a:.4f}" for a in aurocs])
    samples.close()


if __name__ == "__main__":
    main()
