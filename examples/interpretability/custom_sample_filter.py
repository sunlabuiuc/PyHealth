"""Evaluate all interpretability methods on StageNet + MIMIC-IV dataset using comprehensiveness 
and sufficiency metrics.

This example demonstrates:
1. Loading a pre-trained StageNet model with processors and MIMIC-IV dataset
2. Computing attributions with various interpretability methods
3. Evaluating attribution faithfulness with Comprehensiveness & Sufficiency for each method
4. Presenting results in a summary table
"""

import datetime
import argparse

import torch
from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.interpret.methods import *
from pyhealth.metrics.interpretability import evaluate_attribution
from pyhealth.metrics.interpretability.utils import SampleClass
from pyhealth.models import Transformer
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.trainer import Trainer
from pyhealth.datasets.utils import load_processors
from pathlib import Path
import pandas as pd

# python -u examples/interpretability/custom_sample_filter.py  --pos_threshold 0.5 --neg_threshold 0.1 --device cuda:2
def main():
    parser = argparse.ArgumentParser(
        description="Comma separated list of interpretability methods to evaluate"
    )
    parser.add_argument(
        "--pos_threshold",
        type=float,
        default=None,
        help="Positive threshold for interpretability evaluation (default: 0.5).",
    )
    parser.add_argument(
        "--neg_threshold",
        type=float,
        default=None,
        help="Negative threshold for interpretability evaluation (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for evaluation (default: cuda:0)",
    )
    args = parser.parse_args()
    """Main execution function."""
    print("=" * 70)
    print("Interpretability Metrics Example: Transformer + MIMIC-IV")
    print("=" * 70)
    
    now = datetime.datetime.now()
    print(f"Start Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # Set path
    CACHE_DIR = Path("/home/yongdaf2/interpret/cache/mp_mimic4")
    CKPTS_DIR = Path("/shared/eng/pyhealth_dka/ckpts/mp_transformer_mimic4")
    OUTPUT_DIR = Path("/home/yongdaf2/interpret/output/mp_transformer_mimic4")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CKPTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nUsing cache dir: {CACHE_DIR}")
    print(f"Using checkpoints dir: {CKPTS_DIR}")
    print(f"Using output dir: {OUTPUT_DIR}")

    # Set device
    device = args.device
    print(f"\nUsing device: {device}")

    # Load MIMIC-IV dataset
    print("\n Loading MIMIC-IV dataset...")
    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
        cache_dir=str(CACHE_DIR),
        num_workers=16,
    )

    # Apply mortality prediction task
    if not (CKPTS_DIR / "input_processors.pkl").exists():
        raise FileNotFoundError(f"Input processors not found in {CKPTS_DIR}. ")
    if not (CKPTS_DIR / "output_processors.pkl").exists():
        raise FileNotFoundError(f"Output processors not found in {CKPTS_DIR}. ")
    input_processors, output_processors = load_processors(str(CKPTS_DIR))
    print("✓ Loaded input and output processors from checkpoint directory.")
    
    sample_dataset = base_dataset.set_task(
        MortalityPredictionStageNetMIMIC4(),
        num_workers=16,
        input_processors=input_processors,
        output_processors=output_processors,
    )
    print(f"✓ Loaded {len(sample_dataset)} samples")

    # Split dataset and get test loader
    _, _, test_dataset = split_by_patient(sample_dataset, [0.9, 0.09, 0.01], seed=233)
    test_loader = get_dataloader(test_dataset, batch_size=16, shuffle=False)
    print(f"✓ Test set: {len(test_dataset)} samples")

    # Initialize and load pre-trained model
    print("\n Loading pre-trained Transformer model...")
    model = Transformer(
        dataset=sample_dataset,
        embedding_dim=128,
        heads=4,
        dropout=0.3,
        num_layers=3,
    )

    trainer = Trainer(model=model, device=device)
    trainer.load_ckpt(str(CKPTS_DIR / "best.ckpt"))
    model = model.to(device)
    model.eval()
    print(f"✓ Loaded checkpoint: {CKPTS_DIR / 'best.ckpt'}")
    print(f"✓ Model moved to {device}")

    pos_threshold = args.pos_threshold
    neg_threshold = args.neg_threshold
    def sample_filter_fn(
        y_probs: torch.Tensor,
        classifier_type: str,
    ) -> torch.Tensor:
        """
        Custom sample filter function that classifies samples based on
        positive and negative probability thresholds.

        negative samples: 0 < y_probs < neg_threshold
        ignored samples: neg_threshold <= y_probs < pos_threshold
        positive samples: y_probs >= pos_threshold
        """
        nonlocal pos_threshold, neg_threshold
        batch_size = y_probs.shape[0]
        result = torch.full(
            (batch_size,),
            SampleClass.POSITIVE,
            dtype=torch.long,
            device=y_probs.device,
        )
        if classifier_type in ("binary", "multilabel"):
            if pos_threshold is not None:
                result[y_probs < pos_threshold] = SampleClass.IGNORE
            if neg_threshold is not None:
                result[y_probs < neg_threshold] = SampleClass.NEGATIVE
        return result

    interpreter = IntegratedGradients(model, use_embeddings=True)
    print(f"\nEvaluating using Integrated Gradients...")

    # Option 1: Functional API (simple one-off evaluation)
    print("\nEvaluating with Functional API on full dataset...")
    print("Using: evaluate_attribution(model, dataloader, method, ...)")

    results_functional = evaluate_attribution(
        model,
        test_loader,
        interpreter,
        metrics=["comprehensiveness", "sufficiency"],
        percentages=[25, 50, 99],
        sample_filter=sample_filter_fn,
    )

    print("\n" + "=" * 70)
    print("Dataset-Wide Results (Functional API)")
    print("=" * 70)
    comp = results_functional["comprehensiveness"]
    suff = results_functional["sufficiency"]
    print(f"\nComprehensiveness: {comp:.4f}")
    print(f"Sufficiency:       {suff:.4f}")
    
    print("")
    print("=" * 70)
    print("Summary of Results for All Methods")
    print({"Method": "Integrated Gradients", "Comprehensiveness": comp, "Sufficiency": suff})
    
    end = datetime.datetime.now()
    print(f"End Time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration: {end - now}")

if __name__ == "__main__":
    main()
