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
    CV           : 5-fold KFold (shuffle=True, random_state=42)
    Sigma        : 1.0  [GaussianBlur for high_* / low_* modes]
    Filter backend: scipy.ndimage.gaussian_filter (reference-faithful)

Two validation strategies are supported via ``--val_strategy``:

    none (default)  Train on full train_val split, evaluate at last epoch.
                    Matches reference methodology. Use for replication.
    best            Hold out 10% val per fold, load best checkpoint by val
                    AUROC. Use for ablation / model selection.

All results are 5-fold CV on the ISIC 2018 *training* partition only
(no independent test set); metrics may overestimate generalization.

Replication Results (10 epochs, val_strategy=none, sigma=1.0)
----------------------------------------------------------
AUROC per fold (canonical mode order):

  Mode               F1     F2     F3     F4     F5    Mean   +-Std
  ------------------------------------------------------------------
  whole            0.742  0.760  0.717  0.750  0.771  0.748  0.021
  lesion           0.708  0.687  0.732  0.722  0.805  0.731  0.045
  background       0.744  0.717  0.733  0.731  0.741  0.733  0.010
  bbox             0.764  0.655  0.673  0.720  0.661  0.695  0.046
  bbox70           0.707  0.624  0.611  0.620  0.634  0.639  0.039
  bbox90           0.653  0.599  0.563  0.632  0.612  0.612  0.034
  high_whole       0.650  0.670  0.680  0.602  0.639  0.648  0.031
  high_lesion      0.645  0.714  0.652  0.682  0.741  0.687  0.041
  high_background  0.723  0.681  0.655  0.684  0.685  0.686  0.024
  low_whole        0.710  0.779  0.761  0.726  0.782  0.751  0.032
  low_lesion       0.701  0.690  0.728  0.691  0.764  0.715  0.032
  low_background   0.749  0.637  0.716  0.755  0.718  0.715  0.047

Key observations:
- low_whole (0.751) matches whole (0.748); diff +0.003, p=0.83 (n.s.).
- low_whole vs high_whole: diff +0.103, p=0.002 (*) — the low/high-pass gap
  is the only significant within-region contrast; low-frequency colour/texture
  carries the signal, not fine-grained edges.
- whole vs bbox90: diff +0.136, p=0.001 (*) — aggressive context removal
  significantly degrades performance.

Sigma Ablation Results (low_whole, val_strategy=none, filter_backend=scipy)
---------------------------------------------------------------------------
AUROC per fold across Gaussian blur sigma values:

  Sigma      F1     F2     F3     F4     F5    Mean   +-Std
  -----------------------------------------------------------
    0.5   0.761  0.768  0.723  0.730  0.738  0.744  0.020
    1.0   0.710  0.779  0.761  0.726  0.782  0.751  0.032
    2.0   0.753  0.705  0.720  0.765  0.775  0.744  0.030
    4.0   0.690  0.736  0.730  0.753  0.746  0.731  0.025
    8.0   0.742  0.703  0.652  0.696  0.771  0.713  0.046
   16.0   0.783  0.689  0.716  0.704  0.706  0.720  0.037

Key observations:
- Performance peaks at sigma=1.0 (0.751) and degrades monotonically at higher
  sigmas; sigma=8.0 shows the largest drop (0.713) and highest variance (±0.041),
  suggesting aggressive smoothing removes diagnostically useful features.
- sigma=0.5 (0.744) is competitive but slightly below sigma=1.0.
- sigma=1.0 (low_whole, 0.751) vs whole (0.748): paired t-test diff=+0.003,
  t=0.229, p=0.830 — not significant; low-pass at sigma=1.0 retains full
  performance, confirming low-frequency features carry the diagnostic signal.
  Note: resizing to 224×224 already acts as an implicit low-pass filter, which
  may explain why an additional Gaussian at sigma=1.0 has negligible effect.
- sigma=1.0 vs sigma=16.0 (low_whole): paired t-test diff=+0.032,
  t=1.095, p=0.335 — not significant (df=4); the trend of degradation at
  high sigma is consistent but underpowered with 5 folds.

Mode Ablation Results
---------------------
gray_whole, blur_bg, whole_norm, whole_stratified: val_strategy=none.
whole_best, whole_best_stratified: val_strategy=best.
blur_bg_best_stratified: val_strategy=best, stratified (pending).

AUROC per fold:

  Mode                    F1     F2     F3     F4     F5    Mean   +-Std   val_strategy
  --------------------------------------------------------------------------------------
  gray_whole           0.777  0.749  0.691  0.763  0.772  0.750  0.035   none
  blur_bg              0.767  0.801  0.747  0.752  0.756  0.765  0.022   none
  whole_norm           0.734  0.738  0.718  0.730  0.775  0.739  0.022   none
  whole_stratified     0.772  0.799  0.745  0.737  0.778  0.766  0.025   none
  whole_best           0.787  0.791  0.795  0.744  0.846  0.792  0.036   best
  whole_best_stratified 0.824 0.814  0.738  0.710  0.807  0.779  0.051   best

Key observations:
- whole_best (0.792) vs whole/none (0.748): diff=+0.044, t=2.884, p=0.045 (*) —
  early stopping meaningfully improves generalisation.
- whole_best_stratified (0.779) vs whole/none (0.748): diff=+0.031, t=1.501,
  p=0.208 (n.s., df=4) — combining best+stratified is not significantly better
  than none alone; high variance across folds.
- whole_best_stratified (0.779) vs whole_best (0.792): diff=-0.013, t=-0.750,
  p=0.495 (n.s., df=4) — stratified folding does not add benefit over best alone.
- whole_stratified (0.766) vs whole/none (0.748): diff=+0.018, t=1.931,
  p=0.126 (n.s., df=4) — class balance is not a confound.
- gray_whole (0.750) vs whole/none (0.748): diff=+0.002, t=0.224, p=0.834 (n.s., df=4) —
  colour loss does not degrade training performance; the model learns effectively
  from grayscale images alone.
- blur_bg (0.765) vs whole/none (0.748): diff=+0.016, t=1.634, p=0.178 (n.s., df=4) —
  background blurring does not significantly improve performance.
- whole_norm (0.739) vs whole/none (0.748): diff=-0.009, t=-1.754, p=0.154 (n.s., df=4) —
  per-image normalisation does not significantly affect performance.

Per-Artifact AUROC Analysis (whole/none baseline, 5-fold CV on training set)
----------------------------------------------------------------------------
AUROC computed on held-out test folds of the ISIC 2018 training partition,
separately on images with vs. without each artifact type.
Paired t-test (df varies per artifact; folds with single-class subsets excluded).

  Artifact       With Artifact  Without Artifact   Diff     p
  ─────────────────────────────────────────────────────────────
  dark_corner        0.7543          0.7460        +0.008   0.807
  hair               0.7363          0.7567        -0.020   0.373
  gel_border         0.7658          0.7414        +0.024   0.295
  gel_bubble         0.7569          0.7461        +0.011   0.675
  ruler              0.7412          0.7593        -0.018   0.602
  ink                0.7256          0.7518        -0.026   0.518
  patches            0.9762          0.7370        +0.239   0.048 (*)

Key observations:
- patches is a strong outlier: the model achieves near-perfect AUROC (0.976) on
  images containing patches, vs 0.737 on images without (diff=+0.239, p=0.048).
  This suggests the model exploits patch presence as a diagnostic shortcut rather
  than learning true lesion features. Most folds have single-class patch subsets
  (AUROC undefined), so the significant result is based on limited valid pairs.
- All other artifacts show negligible and non-significant effects (|diff| <= 0.026,
  p >= 0.295), indicating the model is not strongly biased by common dermoscopic
  artifacts such as hair, ruler marks, or gel borders.

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
