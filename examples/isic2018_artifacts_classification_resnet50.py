"""
ISIC 2018 — Binary Melanoma Classification Under Artifact Modes
================================================================

Replicates image-mode experiments from:
  "A Study of Artifacts on Melanoma Classification under Diffusion-Based
  Perturbations"

Supports 14 preprocessing modes via ``--mode``:
  whole, lesion, background, bbox, bbox70, bbox90,
  high_whole, high_lesion, high_background,
  low_whole, low_lesion, low_background,
  blur_bg, gray_whole

Dataset
-------
Images / masks (~9 GB):  https://challenge.isic-archive.com/data/#2018
Artifact annotations:    https://github.com/alceubissoto/debiasing-skin

Expected layout under ``--root``::

    <root>/
        isic_bias.csv                           (--annotations_csv)
        ISIC2018_Task1-2_Training_Input/        (--image_dir)
        ISIC2018_Task1_Training_GroundTruth/    (--mask_dir)

Pass ``--download`` to fetch automatically.

Usage
-----
    # Whole-image mode (default)
    python isic2018_artifacts_classification_resnet50.py --root /path/to/data

    # Specific mode
    python isic2018_artifacts_classification_resnet50.py --root /path/to/data --mode lesion

    # Sigma ablation
    python isic2018_artifacts_classification_resnet50.py --root /path/to/data --mode low_whole --sigma 2.0

Experimental Setup
------------------
All runs use ResNet-50 with ImageNet pretrained weights.

Common parameters:

    Model        : ResNet-50, ImageNet pretrained (weights="DEFAULT")
    Optimizer    : Adam, lr=1e-4, weight_decay=0.0
    Epochs       : 10
    Batch size   : 32
    CV           : 5-fold KFold (shuffle=True, random_state=42) — matches reference
    Sigma        : 1.0  [GaussianBlur for high_* / low_* modes]
    Filter backend: scipy.ndimage.gaussian_filter (reference-faithful)

Two validation strategies are supported via ``--val_strategy``:

    none (default)  Train on full train_val split, evaluate at last epoch.
                    Matches reference methodology. Use for replication.
    best            Hold out 10% val per fold, load best checkpoint by val
                    AUROC. Use for ablation / model selection.

All results are 5-fold CV on the ISIC 2018 *training* partition only
(no independent test set); metrics may overestimate generalization.

"""

import argparse
import logging
import os
import sys

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from pyhealth.datasets import ISIC2018ArtifactsDataset, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.processors import DermoscopicImageProcessor
from pyhealth.processors.dermoscopic_image_processor import VALID_MODES
from pyhealth.tasks import ISIC2018ArtifactsBinaryClassification
from pyhealth.trainer import Trainer

parser = argparse.ArgumentParser(
    description="Train ISIC2018 artifact classifier")
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
parser.add_argument(
    "--stratified",
    action="store_true",
    help="Use StratifiedKFold instead of KFold to preserve class balance per fold.")
parser.add_argument(
    "--sigma",
    type=float,
    default=1.0,
    help="Gaussian sigma for high_* / low_* filter modes (default: 1.0).")
parser.add_argument(
    "--high_grayscale",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="If True (default), apply high-pass filter in grayscale then stack to 3 channels "
         "(matches reference grayscale=True). "
         "Use --no-high-grayscale to apply HPF per RGB channel instead.")
parser.add_argument(
    "--val_strategy",
    type=str,
    default="none",
    choices=["none", "best"],
    help="Validation strategy. "
         "'none' (default): train on full train_val split, no val holdout, "
         "evaluate last epoch — matches the reference implementation. "
         "'best': hold out 10%% of train_val as validation and load the "
         "best-scoring checkpoint at the end (ablation).")
parser.add_argument(
    "--resume",
    action="store_true",
    help="Skip folds whose checkpoint already exists in the output directory.")
parser.add_argument(
    "--cache_only",
    action="store_true",
    help="Build the litdata sample cache and exit without training. "
         "Use this to pre-warm caches in parallel before training runs.")
parser.add_argument(
    "--download",
    action="store_true",
    help="Auto-download data.")
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="Number of worker processes for building the litdata cache (default: 8).")
args = parser.parse_args()

# Route PyHealth trainer logs to stdout so per-epoch metrics are visible.
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger("pyhealth.trainer").addHandler(_handler)
logging.getLogger("pyhealth.trainer").setLevel(logging.INFO)



if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Build dataset — all path resolution delegated to the loader
    # ------------------------------------------------------------------
    dataset = ISIC2018ArtifactsDataset(
        root=args.root,
        annotations_csv=args.annotations_csv,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        download=args.download,
        num_workers=args.num_workers,
    )
    dataset.stats()

    # ------------------------------------------------------------------
    # 2. Apply task → SampleDataset with binary labels
    # ------------------------------------------------------------------
    processor = DermoscopicImageProcessor(
        mode=args.mode,
        sigma=args.sigma,
        mask_dir=dataset.mask_dir,
        high_grayscale=args.high_grayscale,
    )
    task = ISIC2018ArtifactsBinaryClassification()
    samples = dataset.set_task(task, input_processors={"image": processor})

    if args.cache_only:
        print(f"Cache built for mode={args.mode} sigma={args.sigma}. Exiting (--cache_only).")
        samples.close()
        sys.exit(0)

    # ------------------------------------------------------------------
    # 3. Generate stratified K-fold splits from sample labels
    # ------------------------------------------------------------------
    labels = np.array([samples[i]["label"] for i in range(len(samples))])
    indices = np.arange(len(labels))

    splitter_cls = StratifiedKFold if args.stratified else KFold
    skf = splitter_cls(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed)

    color_tag = "" if args.high_grayscale else "_color"
    strat_tag = "_stratified" if args.stratified else ""
    output_dir = os.path.join(
        args.root, "checkpoints",
        f"{args.mode}_sigma{args.sigma}{color_tag}{strat_tag}_{args.val_strategy}")
    os.makedirs(output_dir, exist_ok=True)

    split_input = (indices, labels) if args.stratified else (indices,)
    for fold, (train_val_idx, test_idx) in enumerate(
            skf.split(*split_input), start=1):
        ckpt_path = os.path.join(output_dir, f"fold{fold}.pt")
        if args.resume and os.path.exists(ckpt_path):
            print(f"\nSkipping fold {fold}/{args.n_splits} — checkpoint exists: {ckpt_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Mode: {args.mode}  |  Split {fold}/{args.n_splits}"
              f"  |  val_strategy={args.val_strategy}")
        print(f"{'=' * 60}")

        if args.val_strategy == "best":
            # Hold out 10% of train_val for model selection
            val_size = max(1, int(0.1 * len(train_val_idx)))
            rng = np.random.default_rng(args.seed + fold)
            rng.shuffle(train_val_idx)
            val_idx = train_val_idx[:val_size]
            train_idx = train_val_idx[val_size:]
            val_loader = get_dataloader(
                samples.subset(val_idx), batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
            )
        else:
            # Reference-faithful: train on full train_val, no validation
            train_idx = train_val_idx
            val_loader = None

        train_loader = get_dataloader(
            samples.subset(train_idx), batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,
        )
        test_loader = get_dataloader(
            samples.subset(test_idx), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
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
            load_best_model_at_last=(args.val_strategy == "best"),
        )

        # --------------------------------------------------------------
        # 6. Evaluate
        # --------------------------------------------------------------
        scores = trainer.evaluate(test_loader)
        print(f"Mode: {args.mode}  Split {fold} test results: {scores}")

        # --------------------------------------------------------------
        # 7. Save checkpoint
        # --------------------------------------------------------------
        trainer.save_ckpt(ckpt_path)
        print(f"Checkpoint saved → {ckpt_path}")

    samples.close()
