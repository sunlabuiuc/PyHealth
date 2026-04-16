"""
PH2 Evaluation — ISIC-trained ResNet-50 checkpoints
====================================================

Evaluates ISIC 2018 Phase checkpoints (trained on binary melanoma
classification) on the PH2 dermoscopy dataset using a zero-shot transfer
setup: no fine-tuning, same preprocessing mode, binary label mapping
(melanoma=1, common_nevus/atypical_nevus=0).

Only mask-free modes are supported (PH2 mirror has no segmentation masks):
    whole, high_whole, low_whole

All 5 fold checkpoints are evaluated on the full PH2 dataset (200 images)
and AUROC / accuracy are reported per fold and averaged.

Checkpoint layout (produced by isic2018_artifacts_classification_resnet50.py)::

    <ckpt_root>/
        whole_sigma1.0_none/fold1.pt ... fold5.pt
        high_whole_sigma1.0_none/fold1.pt ... fold5.pt
        low_whole_sigma1.0_none/fold1.pt ... fold5.pt

Results (ISIC Phase 1 checkpoints → PH2, zero-shot transfer)
-------------------------------------------------------------
Binary: melanoma=1, common_nevus/atypical_nevus=0 (200 images)

  AUROC per fold:
  Mode           F1     F2     F3     F4     F5    Mean  +-Std
  ─────────────────────────────────────────────────────────────
  whole        0.874  0.939  0.913  0.895  0.896  0.903  0.024
  low_whole    0.891  0.916  0.900  0.675  0.937  0.864  0.107
  high_whole   0.475  0.686  0.397  0.505  0.605  0.534  0.113

- whole (0.903): strong zero-shot transfer to PH2.
- low_whole (0.864): slightly lower with higher variance (fold 4: 0.675).
- high_whole (0.534): near-random — high-frequency features do not transfer.

Usage
-----
    pixi run -e base python examples/ph2_artifacts_eval_resnet50.py \\
        --ph2_root ~/ph2/PH2-dataset-master \\
        --ckpt_root ~/isic2018_data/checkpoints

    # Single mode
    pixi run -e base python examples/ph2_artifacts_eval_resnet50.py \\
        --ph2_root ~/ph2/PH2-dataset-master \\
        --ckpt_root ~/isic2018_data/checkpoints \\
        --modes whole
"""

import argparse
import logging
import os
import sys
from typing import Dict, List

import numpy as np

from pyhealth.datasets import PH2Dataset, get_dataloader
from pyhealth.models import TorchvisionModel
from pyhealth.processors import DermoscopicImageProcessor
from pyhealth.tasks import BaseTask
from pyhealth.trainer import Trainer

# Suppress noisy dataset init logs during eval
logging.getLogger("pyhealth").setLevel(logging.WARNING)

# Mask-free modes available for PH2 (no segmentation mask required)
SUPPORTED_MODES = ["whole", "high_whole", "low_whole"]


# ---------------------------------------------------------------------------
# Binary task: melanoma (1) vs non-melanoma (0)
# ---------------------------------------------------------------------------

class PH2BinaryMelanomaClassification(BaseTask):
    """Binary melanoma vs non-melanoma task for PH2.

    Maps the 3-class PH2 diagnosis to binary labels matching the ISIC
    training setup: melanoma=1, common_nevus/atypical_nevus=0.
    """

    task_name: str = "PH2BinaryMelanomaClassification"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __call__(self, patient):
        samples = []
        for event in patient.get_events(event_type="ph2"):
            diagnosis = event["diagnosis"]
            label = 1 if diagnosis == "melanoma" else 0
            samples.append({"image": event["path"], "label": label})
        return samples


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Evaluate ISIC-trained checkpoints on PH2 (zero-shot transfer)")
parser.add_argument(
    "--ph2_root",
    type=str,
    default=os.path.expanduser("~/ph2/PH2-dataset-master"),
    help="Root directory of the PH2 dataset (mirror format).",
)
parser.add_argument(
    "--ckpt_root",
    type=str,
    default=os.path.expanduser("~/isic2018_data/checkpoints"),
    help="Root directory containing per-mode checkpoint subdirectories.",
)
parser.add_argument(
    "--modes",
    nargs="+",
    default=SUPPORTED_MODES,
    choices=SUPPORTED_MODES,
    help="Modes to evaluate (default: all mask-free modes).",
)
parser.add_argument(
    "--sigma",
    type=float,
    default=1.0,
    help="Gaussian sigma used during ISIC training (default: 1.0).",
)
parser.add_argument(
    "--n_splits",
    type=int,
    default=5,
    help="Number of fold checkpoints to evaluate (default: 5).",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Build PH2 dataset and apply binary task
    # ------------------------------------------------------------------
    print(f"Loading PH2 dataset from {args.ph2_root} ...")
    dataset = PH2Dataset(root=args.ph2_root, num_workers=1)
    task = PH2BinaryMelanomaClassification()

    print(f"\nAUROC per fold:")
    print(f"{'Mode':<16}  {'F1':>6} {'F2':>6} {'F3':>6} {'F4':>6} {'F5':>6}  "
          f"{'Mean':>6} {'+-Std':>6}")
    print("─" * 70)

    all_results = {}

    for mode in args.modes:
        ckpt_dir = os.path.join(
            args.ckpt_root, f"{mode}_sigma{args.sigma}_none")
        if not os.path.isdir(ckpt_dir):
            print(f"{mode:<16}  [checkpoint dir not found: {ckpt_dir}]")
            continue

        # Build processed samples for this mode
        processor = DermoscopicImageProcessor(
            mode=mode,
            sigma=args.sigma,
            mask_dir="",            # no masks for PH2
            high_grayscale=True,
        )
        samples = dataset.set_task(task, input_processors={"image": processor})
        loader = get_dataloader(
            samples,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # One dummy model to get output schema; weights replaced per fold
        model = TorchvisionModel(
            dataset=samples,
            model_name="resnet50",
            model_config={"weights": None},   # random init; overwritten below
        )
        trainer = Trainer(model=model, metrics=["accuracy", "roc_auc"])

        fold_aucs = []
        fold_accs = []

        for fold in range(1, args.n_splits + 1):
            ckpt_path = os.path.join(ckpt_dir, f"fold{fold}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  [skip] {mode} fold{fold}: checkpoint not found")
                continue
            trainer.load_ckpt(ckpt_path)
            scores = trainer.evaluate(loader)
            fold_aucs.append(scores["roc_auc"])
            fold_accs.append(scores["accuracy"])

        samples.close()

        if fold_aucs:
            mean_auc = np.mean(fold_aucs)
            std_auc  = np.std(fold_aucs, ddof=1)
            fold_str = "  ".join(f"{v:.4f}" for v in fold_aucs)
            print(f"{mode:<16}  {fold_str}  {mean_auc:.4f} {std_auc:.4f}")
            all_results[mode] = {"aucs": fold_aucs, "mean": mean_auc, "std": std_auc}

    print("\nDone.")
    if all_results:
        print("\nSummary (AUROC mean ± std):")
        for mode, r in all_results.items():
            print(f"  {mode:<16}  {r['mean']:.4f} ± {r['std']:.4f}")
