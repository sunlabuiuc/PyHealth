"""PH2 Melanoma Classifier — 5-fold CV (whole mode, val_strategy=none).

Replicates the PH2-trained classifiers from Jin et al. (CHIL 2025) for the
whole-image mode.  Results are used in:
  - Table 2: PH2-trained classifiers evaluated on PH2/ISIC/HAM10000
  - Table 4: PH2-trained classifiers evaluated on diffusion-augmented PH2

Setup
-----
  PH2 images  : ~/ph2/PH2-dataset-master/images/{ID}.jpg   (200 images)
  Metadata    : ~/ph2/PH2-dataset-master/ph2_metadata_pyhealth.csv
                columns: image_id, path, diagnosis
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
  pixi run -e base python examples/ph2_train_resnet50.py
  pixi run -e base python examples/ph2_train_resnet50.py --test 20
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CKPT_DIR = Path(os.path.expanduser("~/ph2_checkpoints/whole"))
META_PATH = Path(os.path.expanduser("~/ph2/PH2-dataset-master/ph2_metadata_pyhealth.csv"))

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PH2WholeDataset(Dataset):
    def __init__(self, records, transform=None):
        """
        Args:
            records: list of (image_path, label) tuples.
            transform: torchvision transform to apply.
        """
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


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
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.unsqueeze(1).to(DEVICE)
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
    for imgs, labels in loader:
        probs = torch.sigmoid(model(imgs.to(DEVICE))).squeeze(1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    return roc_auc_score(all_labels, all_probs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--test", type=int, default=0, help="Smoke-test: only use first N images")
    p.add_argument("--resume", action="store_true", help="Skip folds whose checkpoint exists")
    return p.parse_args()


def main():
    args = parse_args()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    import csv
    records = []
    with open(META_PATH) as f:
        for row in csv.DictReader(f):
            label = 1 if row["diagnosis"] == "melanoma" else 0
            records.append((row["path"], label))

    if args.test:
        records = records[: args.test]

    records = np.array(records, dtype=object)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    aurocs = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(records)):
        ckpt_path = CKPT_DIR / f"resnet50_fold{fold}.pt"
        if args.resume and ckpt_path.exists():
            print(f"[Fold {fold}] checkpoint exists — skipping training")
            model = build_model()
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        else:
            print(f"\n{'='*50}\nFold {fold}\n{'='*50}")
            train_ds = PH2WholeDataset(records[train_idx].tolist(), TRANSFORM)
            train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)

            model = build_model()
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            for epoch in range(args.epochs):
                loss = train_one_epoch(model, train_loader, criterion, optimizer)
                print(f"  Epoch {epoch+1:2d}/{args.epochs}  loss={loss:.4f}")

            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved → {ckpt_path}")

        test_ds = PH2WholeDataset(records[test_idx].tolist(), TRANSFORM)
        test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4)
        auroc = evaluate(model, test_loader)
        aurocs.append(auroc)
        print(f"  Fold {fold} test AUROC: {auroc:.4f}")

    mean, std = np.mean(aurocs), np.std(aurocs, ddof=1)
    print(f"\nPH2 whole — 5-fold AUROC: {mean:.4f} ±{std:.4f}")
    print("Per-fold:", [f"{a:.4f}" for a in aurocs])


if __name__ == "__main__":
    main()
