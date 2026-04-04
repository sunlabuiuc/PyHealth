"""
Quick inference test: TFMTokenizer on TUAB using local weightfiles/.

Two weight setups (ask your PI which matches their training):

  1) Default (matches conformal example scripts):
     - tokenizer: weightfiles/tfm_tokenizer_last.pth  (multi-dataset tokenizer)
     - classifier: weightfiles/TFM_Tokenizer_multiple_finetuned_on_TUAB/.../best_model.pth

  2) PI benchmark TUAB-specific files (place in weightfiles/):
     - tokenizer: tfm_tokenizer_tuab.pth
     - classifier: tfm_encoder_best_model_tuab.pth
     Use:  --pi-tuab-weights

Split modes:
  - conformal (default): same test set as conformal runs (TUH eval via patient conformal split).
  - pi_benchmark: train/val ratio [0.875, 0.125] on train partition; test = TUH eval (same patients as official eval).

Usage:
    python examples/conformal_eeg/test_tfm_tuab_inference.py
    python examples/conformal_eeg/test_tfm_tuab_inference.py --pi-tuab-weights
    python examples/conformal_eeg/test_tfm_tuab_inference.py \\
        --tuab-pi-weights-dir /shared/eng/conformal_eeg --split pi_benchmark
    python examples/conformal_eeg/test_tfm_tuab_inference.py --tokenizer-weights PATH --classifier-weights PATH
"""

import argparse
import os
import time

import torch

from pyhealth.datasets import (
    TUABDataset,
    get_dataloader,
    split_by_patient_conformal_tuh,
    split_by_patient_tuh,
)
from pyhealth.models import TFMTokenizer
from pyhealth.tasks import EEGAbnormalTUAB
from pyhealth.trainer import Trainer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTFILES = os.path.join(REPO_ROOT, "weightfiles")
DEFAULT_TOKENIZER = os.path.join(WEIGHTFILES, "tfm_tokenizer_last.pth")
CLASSIFIER_WEIGHTS_DIR = os.path.join(
    WEIGHTFILES, "TFM_Tokenizer_multiple_finetuned_on_TUAB"
)
PI_TOKENIZER = os.path.join(WEIGHTFILES, "tfm_tokenizer_tuab.pth")
PI_CLASSIFIER = os.path.join(WEIGHTFILES, "tfm_encoder_best_model_tuab.pth")


def main():
    parser = argparse.ArgumentParser(description="TFM TUAB inference sanity check")
    parser.add_argument(
        "--root",
        type=str,
        default="/srv/local/data/TUH/tuh_eeg_abnormal/v3.0.0/edf",
        help="Path to TUAB edf/ directory.",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Which fine-tuned classifier folder _1.._5 (only if not using --classifier-weights).",
    )
    parser.add_argument(
        "--pi-tuab-weights",
        action="store_true",
        help="Use PI TUAB-specific files under weightfiles/: tfm_tokenizer_tuab.pth, "
        "tfm_encoder_best_model_tuab.pth",
    )
    parser.add_argument(
        "--tuab-pi-weights-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory containing PI's TUAB TFM files (e.g. /shared/eng/conformal_eeg). "
        "Loads tfm_tokenizer_tuab.pth + tfm_encoder_best_model_tuab.pth from there. "
        "Overrides --pi-tuab-weights and default weightfiles paths unless "
        "--tokenizer-weights / --classifier-weights are set explicitly.",
    )
    parser.add_argument(
        "--tokenizer-weights",
        type=str,
        default=None,
        help="Override tokenizer checkpoint path.",
    )
    parser.add_argument(
        "--classifier-weights",
        type=str,
        default=None,
        help="Override classifier checkpoint path (single .pth file).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["conformal", "pi_benchmark"],
        default="conformal",
        help="conformal: same as EEG conformal scripts; pi_benchmark: 0.875/0.125 train/val on train partition.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="RNG seed for patient shuffle (pi_benchmark and conformal).",
    )
    args = parser.parse_args()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    if args.tuab_pi_weights_dir is not None:
        d = os.path.expanduser(args.tuab_pi_weights_dir)
        tok = os.path.join(d, "tfm_tokenizer_tuab.pth")
        cls_path = os.path.join(d, "tfm_encoder_best_model_tuab.pth")
    elif args.pi_tuab_weights:
        tok = PI_TOKENIZER
        cls_path = PI_CLASSIFIER
    else:
        tok = DEFAULT_TOKENIZER
        cls_path = os.path.join(
            CLASSIFIER_WEIGHTS_DIR,
            f"TFM_Tokenizer_multiple_finetuned_on_TUAB_{args.seed}",
            "best_model.pth",
        )

    if args.tokenizer_weights is not None:
        tok = args.tokenizer_weights
    if args.classifier_weights is not None:
        cls_path = args.classifier_weights

    print(f"Device:             {device}")
    print(f"TUAB root:          {args.root}")
    print(f"Split mode:         {args.split}")
    print(f"Tokenizer weights:  {tok}")
    print(f"Classifier weights: {cls_path}")

    t0 = time.time()
    base_dataset = TUABDataset(root=args.root, subset="both")
    print(f"Dataset loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    sample_dataset = base_dataset.set_task(
        EEGAbnormalTUAB(
            resample_rate=200,
            normalization="95th_percentile",
            compute_stft=True,
        ),
        num_workers=16,
    )
    print(f"Task set in {time.time() - t0:.1f}s  |  total samples: {len(sample_dataset)}")

    if args.split == "conformal":
        _, _, _, test_ds = split_by_patient_conformal_tuh(
            dataset=sample_dataset,
            ratios=[0.6, 0.2, 0.2],
            seed=args.split_seed,
        )
    else:
        _, _, test_ds = split_by_patient_tuh(
            sample_dataset,
            [0.875, 0.125],
            seed=args.split_seed,
        )

    test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)
    print(f"Test set size: {len(test_ds)}")

    model = TFMTokenizer(dataset=sample_dataset).to(device)
    model.load_pretrained_weights(
        tokenizer_checkpoint_path=tok,
        classifier_checkpoint_path=cls_path,
    )

    trainer = Trainer(
        model=model,
        device=device,
        metrics=[
            "accuracy",
            "balanced_accuracy",
            "f1_weighted",
            "f1_macro",
            "roc_auc_weighted_ovr",
        ],
        enable_logging=False,
    )
    t0 = time.time()
    results = trainer.evaluate(test_loader)
    print(f"\nEval time: {time.time() - t0:.1f}s")
    print("\n=== Test Results ===")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
