"""
Quick inference test: TFMTokenizer on TUEV using local weightfiles/.

Mirrors the PI's benchmark script but uses the weightfiles/ paths already
present in this repo.  No training — pure inference to verify weights and
normalization are correct.

Usage:
    python examples/conformal_eeg/test_tfm_tuev_inference.py
    python examples/conformal_eeg/test_tfm_tuev_inference.py --gpu_id 1
    python examples/conformal_eeg/test_tfm_tuev_inference.py --seed 2  # use _2/best_model.pth
"""

import argparse
import os
import time

import torch

from pyhealth.datasets import TUEVDataset, get_dataloader, split_by_patient_conformal_tuh
from pyhealth.models import TFMTokenizer
from pyhealth.tasks import EEGEventsTUEV
from pyhealth.trainer import Trainer

TUEV_ROOT = "/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf/"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOKENIZER_WEIGHTS  = os.path.join(REPO_ROOT, "weightfiles", "tfm_tokenizer_last.pth")
CLASSIFIER_WEIGHTS_DIR = os.path.join(
    REPO_ROOT, "weightfiles", "TFM_Tokenizer_multiple_finetuned_on_TUEV"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--seed", type=int, default=1, choices=[1, 2, 3, 4, 5],
        help="Which fine-tuned classifier to use (1-5)."
    )
    args = parser.parse_args()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    classifier_weights = os.path.join(
        CLASSIFIER_WEIGHTS_DIR,
        f"TFM_Tokenizer_multiple_finetuned_on_TUEV_{args.seed}",
        "best_model.pth",
    )

    print(f"Device:             {device}")
    print(f"Tokenizer weights:  {TOKENIZER_WEIGHTS}")
    print(f"Classifier weights: {classifier_weights}")

    # ------------------------------------------------------------------ #
    # STEP 1: Load dataset
    # ------------------------------------------------------------------ #
    t0 = time.time()
    base_dataset = TUEVDataset(root=TUEV_ROOT, subset="both")
    print(f"Dataset loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ #
    # STEP 2: Set task  — normalization="95th_percentile" matches training
    # ------------------------------------------------------------------ #
    t0 = time.time()
    sample_dataset = base_dataset.set_task(
        EEGEventsTUEV(
            resample_rate=200,
            normalization="95th_percentile",
            compute_stft=True,
        )
    )
    print(f"Task set in {time.time() - t0:.1f}s  |  total samples: {len(sample_dataset)}")

    # ------------------------------------------------------------------ #
    # STEP 3: Extract fixed test set (TUH eval partition)
    # ------------------------------------------------------------------ #
    _, _, _, test_ds = split_by_patient_conformal_tuh(
        dataset=sample_dataset,
        ratios=[0.6, 0.2, 0.2],
        seed=42,
    )
    test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)
    print(f"Test set size: {len(test_ds)}")

    # ------------------------------------------------------------------ #
    # STEP 4: Load TFMTokenizer with pre-trained weights (no training)
    # ------------------------------------------------------------------ #
    model = TFMTokenizer(dataset=sample_dataset).to(device)
    model.load_pretrained_weights(
        tokenizer_checkpoint_path=TOKENIZER_WEIGHTS,
        classifier_checkpoint_path=classifier_weights,
    )

    # ------------------------------------------------------------------ #
    # STEP 5: Evaluate
    # ------------------------------------------------------------------ #
    trainer = Trainer(
        model=model,
        device=device,
        metrics=["accuracy", "f1_weighted", "f1_macro"],
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
