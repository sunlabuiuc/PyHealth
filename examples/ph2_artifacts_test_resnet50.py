"""
PH2 Artifacts Evaluation — ISIC and PH2 ResNet-50 classifiers on clean + perturbed PH2.
========================================================================================

Replication of Table 4 from Jin et al. (CHIL 2025):
  Evaluates classifiers trained on PH2 or ISIC on DreamBooth-augmented PH2 images
  where a specific artifact is added to each image.

Paper artifacts: dark_corner, gel_bubble, ink, patches, ruler (hair not in paper).

Checkpoints
-----------
  ISIC : ~/isic2018_data/checkpoints/whole_sigma1.0_none/fold{1..5}.pt  (1-indexed)
  PH2  : ~/ph2_checkpoints/whole/resnet50_fold{0..4}.pt                 (0-indexed)

Augmented images
----------------
  ~/ph2_augmented/{artifact}/{image_id}.jpg
  Produced by ph2_artifacts_augment_sd.py

PH2 labels
----------
  melanoma → 1 ;  common_nevus / atypical_nevus → 0

Results (Table 4 replication — whole mode, mean AUROC ± std over 5 folds)
--------------------------------------------------------------------------
  Artifact       PH2-trained      Paper†    ISIC-trained     Paper†
  ─────────────────────────────────────────────────────────────────
  original       0.998±0.002      0.975     0.900±0.029      0.858
  dark_corner    0.992±0.004      0.978     0.847±0.083      0.816
  gel_bubble     0.996±0.003      0.973     0.892±0.045      0.841
  ink            0.994±0.007      0.959     0.905±0.029      0.788
  patches        0.995±0.005      0.976     0.909±0.037      0.848
  ruler          0.992±0.010      0.966     0.904±0.040      0.752
  hair (ours)    0.972±0.007      —         0.866±0.041      —

Usage
-----
  pixi run -e base python examples/ph2_artifacts_test_resnet50.py
  pixi run -e base python examples/ph2_artifacts_test_resnet50.py --source isic
  pixi run -e base python examples/ph2_artifacts_test_resnet50.py --source ph2
  pixi run -e base python examples/ph2_artifacts_test_resnet50.py \\
      --artifacts clean dark_corner ruler
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AUG_DIR   = Path(os.path.expanduser("~/ph2_augmented"))
META_PATH = AUG_DIR / "augmented_metadata.csv"

ISIC_CKPT_DIR = Path(os.path.expanduser("~/isic2018_data/checkpoints/whole_sigma1.0_none"))
PH2_CKPT_DIR  = Path(os.path.expanduser("~/ph2_checkpoints/whole"))

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARTIFACTS = ["clean", "dark_corner", "gel_bubble", "ink", "patches", "ruler", "hair"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PH2AugDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_resnet50():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(DEVICE)


def load_isic_checkpoint(fold_k: int) -> nn.Module:
    """Load ISIC whole_none checkpoint (1-indexed: fold1..fold5).

    PyHealth saves with ``model.`` prefix and ``_dummy_param`` key; strip both.
    """
    path = ISIC_CKPT_DIR / f"fold{fold_k + 1}.pt"
    raw = torch.load(path, map_location=DEVICE)
    state = {k.removeprefix("model."): v for k, v in raw.items() if k != "_dummy_param"}
    model = build_resnet50()
    model.load_state_dict(state)
    model.eval()
    return model


def load_ph2_checkpoint(fold_k: int) -> nn.Module:
    """Load PH2 whole checkpoint (0-indexed: fold0..fold4)."""
    path = PH2_CKPT_DIR / f"resnet50_fold{fold_k}.pt"
    model = build_resnet50()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_auroc(model: nn.Module, records) -> float:
    ds = PH2AugDataset(records, TRANSFORM)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    probs, labels = [], []
    for imgs, lbls in loader:
        p = torch.sigmoid(model(imgs.to(DEVICE))).squeeze(1).cpu().numpy()
        probs.extend(p)
        labels.extend(lbls.numpy())
    return roc_auc_score(labels, probs)


def evaluate_source(source: str, records_by_artifact: dict):
    """Evaluate all 5 folds of `source` (isic|ph2) across artifacts."""
    print(f"\n{'='*60}")
    print(f"Source: {source.upper()} classifiers → PH2 (whole mode)")
    print(f"{'='*60}")

    loader_fn = load_isic_checkpoint if source == "isic" else load_ph2_checkpoint

    results = {}
    for artifact, recs in records_by_artifact.items():
        fold_aurocs = []
        for k in range(5):
            model = loader_fn(k)
            auroc = compute_auroc(model, recs)
            fold_aurocs.append(auroc)
            del model
            torch.cuda.empty_cache()
        mean = np.mean(fold_aurocs)
        std  = np.std(fold_aurocs, ddof=1)
        results[artifact] = (mean, std, fold_aurocs)
        print(f"  {artifact:15s}  AUROC={mean:.3f} ±{std:.3f}   "
              f"folds={[f'{a:.3f}' for a in fold_aurocs]}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate ISIC/PH2 classifiers on clean + artifact-augmented PH2 (Table 4)")
    p.add_argument(
        "--source", choices=["isic", "ph2", "both"], default="both",
        help="Which trained classifiers to evaluate",
    )
    p.add_argument(
        "--artifacts", nargs="+", default=ARTIFACTS,
        help="Artifact conditions to evaluate",
    )
    p.add_argument(
        "--aug_dir", type=str, default=str(AUG_DIR),
        help="Directory containing augmented images and augmented_metadata.csv",
    )
    return p.parse_args()


def main():
    args = parse_args()
    meta_path = Path(args.aug_dir) / "augmented_metadata.csv"

    # Load metadata grouped by artifact
    records_by_artifact: dict[str, list] = {}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            artifact = row["artifact"]
            label = 1 if row["diagnosis"] == "melanoma" else 0
            records_by_artifact.setdefault(artifact, []).append((row["path"], label))

    artifact_subset = {a: records_by_artifact[a] for a in args.artifacts if a in records_by_artifact}

    all_results = {}
    if args.source in ("isic", "both"):
        if not any(ISIC_CKPT_DIR.glob("fold*.pt")):
            print(f"[WARN] No ISIC checkpoints found in {ISIC_CKPT_DIR}")
        else:
            all_results["isic"] = evaluate_source("isic", artifact_subset)

    if args.source in ("ph2", "both"):
        if not any(PH2_CKPT_DIR.glob("resnet50_fold*.pt")):
            print(f"[WARN] No PH2 checkpoints found in {PH2_CKPT_DIR}")
            print("       Run ph2_train_resnet50.py first.")
        else:
            all_results["ph2"] = evaluate_source("ph2", artifact_subset)

    if all_results:
        print(f"\n{'='*60}")
        print("Summary (mean AUROC)")
        print(f"{'Artifact':15s}  " + "  ".join(f"{s:>12}" for s in all_results))
        print("-" * 60)
        for artifact in args.artifacts:
            row_vals = []
            for src, res in all_results.items():
                if artifact in res:
                    m, s, _ = res[artifact]
                    row_vals.append(f"{m:.3f}±{s:.3f}")
                else:
                    row_vals.append("  —  ")
            print(f"{artifact:15s}  " + "  ".join(f"{v:>12}" for v in row_vals))


if __name__ == "__main__":
    main()

