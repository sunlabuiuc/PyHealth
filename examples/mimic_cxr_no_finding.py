"""MIMIC‑CXR · No‑Finding fine‑tuning demo
========================================
This standalone script doubles as a README for the new **MIMICCXRDataset**.
It shows how to load the dataset, build a simple DenseNet‑121 classifier,
train for a couple of epochs, and compute AUROC on the validation split.

Directory layout expected::

    ~/data/mimic_cxr/
        images/                # JPEGs from PhysioNet
            p10/p10012345/s12345678_0001_0.jpg
        metadata.csv           # path,label,sex,ethnicity,age,split

How to generate ``metadata.csv``:
  • see `examples/make_mimic_metadata.ipynb` (joins labels + demographics)
  • or adapt your own preprocessing pipeline.

Quick run
---------
Activate the PyHealth env, then:

>>> python examples/mimic_cxr_no_finding.py \\
        --root ~/data/mimic_cxr \\
        --epochs 2

This will print training loss each epoch and a one‑line AUROC summary.

Script arguments
----------------
```
--root          root directory of images/ + metadata.csv
--batch_size    default 16
--epochs        default 2 (feel free to raise)
--lr            learning rate (1e-4 default)
```

Note: this demo intentionally uses a small subset (val_fold 0 only) so it
runs on CPU in < 2 min.  For full reproduction pass ``--all`` to load every
fold and use GPU.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision.models import densenet121

from pyhealth.datasets import MIMICCXRDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="path to mimic_cxr root dir")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--all", action="store_true", help="load every fold")
    return p.parse_args()


def make_loader(root: Path, split: str, batch_size: int, all_folds: bool):
    ds = MIMICCXRDataset(root=root, task="No Finding", split=split)
    if not all_folds:
        # select only first 5k for quick demo
        idx = list(range(min(5000, len(ds))))
        ds = Subset(ds, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=4)


def main():
    args = parse_args()
    root = Path(args.root).expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(root, "train", args.batch_size, args.all)
    val_loader = make_loader(root, "val", args.batch_size, args.all)

    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, meta in train_loader:
            imgs = imgs.to(device)
            y = meta["label"].unsqueeze(1).to(device)
            logits = model(imgs)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * imgs.size(0)
        print(f"epoch {epoch} train loss {(running/len(train_loader.dataset)):.4f}")

    # ----- validation AUROC -----
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for imgs, meta in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs).cpu()
            ps.append(torch.sigmoid(logits))
            ys.append(meta["label"].unsqueeze(1))
    y_true = torch.cat(ys).numpy().ravel()
    y_prob = torch.cat(ps).numpy().ravel()
    print("val AUROC:", roc_auc_score(y_true, y_prob))


if __name__ == "__main__":
    main()
