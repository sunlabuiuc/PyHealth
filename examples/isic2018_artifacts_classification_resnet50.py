"""
ISIC 2018 — Binary melanoma classification under artifact modes.

Replicates image-mode experiments from:
    "A Study of Artifacts on Melanoma Classification under Diffusion-Based Perturbations"

Supports all 12 preprocessing modes (whole, lesion, background, bbox, bbox70,
bbox90, high_whole, low_whole, high_lesion, low_lesion, high_background,
low_background) via ``--mode``.

5-fold stratified cross-validation splits are generated from sample labels.

Expected directory structure under ``--root``::

    <root>/
        <annotations_csv>                       ← annotation file (--annotations_csv)
        <image_dir>/                            ← images          (--image_dir)
        <mask_dir>/                             ← masks           (--mask_dir)

Annotation file:
    https://github.com/alceubissoto/debiasing-skin/tree/main/artefacts-annotation

Images / masks (~9 GB total):
    https://challenge.isic-archive.com/data/#2018
    (or pass ``--download`` to fetch automatically)

Usage::

    python isic2018_artifacts_classification.py --root /path/to/isic2018_data
    python isic2018_artifacts_classification.py --root /path/to/isic2018_data --mode lesion
"""

import argparse
import logging
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

from pyhealth.datasets import ISIC2018ArtifactsDataset, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.processors.dermoscopic_image_processor import VALID_MODES
from pyhealth.tasks import ISIC2018ArtifactsBinaryClassification
from pyhealth.trainer import Trainer

parser = argparse.ArgumentParser(description="Train ISIC2018 artifact classifier")
parser.add_argument(
    "--root",
    type=str,
    required=True,
    help="Root directory containing the annotation CSV, images, and masks.",
)
parser.add_argument(
    "--image_dir",
    type=str,
    default="ISIC2018_Task1-2_Training_Input",
    help="Sub-directory (relative to root, or absolute path) for ISIC images.",
)
parser.add_argument(
    "--mask_dir",
    type=str,
    default="ISIC2018_Task1_Training_GroundTruth",
    help="Sub-directory (relative to root, or absolute path) for segmentation masks.",
)
parser.add_argument(
    "--annotations_csv",
    type=str,
    default="isic_bias.csv",
    help="Annotation CSV filename (relative to root, or absolute path).",
)
parser.add_argument(
    "--mode",
    type=str,
    default="whole",
    choices=VALID_MODES,
    help="Image preprocessing mode.",
)
parser.add_argument(
    "--model",
    type=str,
    default="resnet50",
    help="Torchvision model backbone (e.g. resnet50, vit_b_16).",
)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--n_splits", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--download", action="store_true", help="Auto-download data.")
args = parser.parse_args()

# Route PyHealth trainer logs to stdout so per-epoch metrics are visible.
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger("pyhealth.trainer").addHandler(_handler)
logging.getLogger("pyhealth.trainer").setLevel(logging.INFO)


# =============================================================================
# Example run & results
# =============================================================================
# Command:
#   python isic2018_artifacts_classification_resnet50.py --root /path/to/isic2018_data
#
# Parameters:
#   --mode        whole
#   --model       resnet50
#   --epochs      10
#   --batch_size  32
#   --lr          1e-4
#   --n_splits    5
#   --seed        42
#
# 5-fold stratified CV results (whole mode, ResNet-50, ImageNet pretrained):
#
#   Split 1  AUROC: 0.800  Accuracy: 0.844
#   Split 2  AUROC: 0.803  Accuracy: 0.829
#   Split 3  AUROC: 0.758  Accuracy: 0.788
#   Split 4  AUROC: 0.790  Accuracy: 0.807
#   Split 5  AUROC: 0.829  Accuracy: 0.840
#   ─────────────────────────────────────
#   Mean     AUROC: 0.796  Accuracy: 0.822
#
# Matches findings from:
#   "A Study of Artifacts on Melanoma Classification under
#    Diffusion-Based Perturbations"
# =============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Build dataset — all path resolution delegated to the loader
    # ------------------------------------------------------------------
    dataset = ISIC2018ArtifactsDataset(
        root=args.root,
        annotations_csv=args.annotations_csv,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        mode=args.mode,
        download=args.download,
    )
    dataset.stats()

    # ------------------------------------------------------------------
    # 2. Apply task → SampleDataset with binary labels
    # ------------------------------------------------------------------
    task = ISIC2018ArtifactsBinaryClassification()
    samples = dataset.set_task(task)

    # ------------------------------------------------------------------
    # 3. Generate stratified K-fold splits from sample labels
    # ------------------------------------------------------------------
    labels = np.array([samples[i]["label"] for i in range(len(samples))])
    indices = np.arange(len(labels))

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    output_dir = os.path.join(args.root, "checkpoints", args.mode)
    os.makedirs(output_dir, exist_ok=True)

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(indices, labels), start=1):
        print(f"\n{'='*60}")
        print(f"  Mode: {args.mode}  |  Split {fold}/{args.n_splits}")
        print(f"{'='*60}")

        # Use 10% of train_val as validation
        val_size = max(1, int(0.1 * len(train_val_idx)))
        rng = np.random.default_rng(args.seed + fold)
        rng.shuffle(train_val_idx)
        val_idx = train_val_idx[:val_size]
        train_idx = train_val_idx[val_size:]

        train_loader = get_dataloader(
            samples.subset(train_idx), batch_size=args.batch_size, shuffle=True
        )
        val_loader = get_dataloader(
            samples.subset(val_idx), batch_size=args.batch_size, shuffle=False
        )
        test_loader = get_dataloader(
            samples.subset(test_idx), batch_size=args.batch_size, shuffle=False
        )

        # --------------------------------------------------------------
        # 4. Fresh model per fold
        # --------------------------------------------------------------
        model = TorchvisionModel(
            dataset=samples,
            model_name=args.model,
            model_config={"weights": "DEFAULT"},
        )

        # --------------------------------------------------------------
        # 5. Train
        # --------------------------------------------------------------
        trainer = Trainer(
            model=model,
            metrics=["accuracy", "roc_auc"],
        )
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=args.epochs,
            optimizer_params={"lr": args.lr},
        )

        # --------------------------------------------------------------
        # 6. Evaluate
        # --------------------------------------------------------------
        scores = trainer.evaluate(test_loader)
        print(f"Mode: {args.mode}  Split {fold} test results: {scores}")

    samples.close()

