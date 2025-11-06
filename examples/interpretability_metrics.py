"""Minimal example of interpretability metrics with StageNet on MIMIC-IV.

This example demonstrates:
1. Loading a pre-trained StageNet model
2. Computing attributions with Integrated Gradients
3. Evaluating attribution faithfulness with Comprehensiveness & Sufficiency
4. Using both the functional API and class-based API for evaluation
"""

import torch

from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient
from pyhealth.interpret.methods import IntegratedGradients
from pyhealth.metrics.interpretability import Evaluator, evaluate_approach
from pyhealth.models import StageNet
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.trainer import Trainer


def main():
    """Main execution function."""
    print("=" * 70)
    print("Interpretability Metrics Example: StageNet + MIMIC-IV")
    print("=" * 70)

    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load MIMIC-IV dataset
    print("\n[1/6] Loading MIMIC-IV dataset...")
    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
    )

    # Apply mortality prediction task
    sample_dataset = base_dataset.set_task(
        MortalityPredictionStageNetMIMIC4(),
        num_workers=4,
        cache_dir="../../mimic4_stagenet_cache",
    )
    print(f"✓ Loaded {len(sample_dataset)} samples")

    # Split dataset and get test loader
    _, _, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])
    test_loader = get_dataloader(test_dataset, batch_size=8, shuffle=False)
    print(f"✓ Test set: {len(test_dataset)} samples")

    # Initialize and load pre-trained model
    print("\n[2/6] Loading pre-trained StageNet model...")
    model = StageNet(
        dataset=sample_dataset,
        embedding_dim=128,
        chunk_size=128,
        levels=3,
        dropout=0.3,
    )

    checkpoint_path = (
        "/home/johnwu3/projects/PyHealth_Branch_Testing/PyHealth/output/"
        "20251028-191219/best.ckpt"
    )
    trainer = Trainer(model=model, device=device)
    trainer.load_ckpt(checkpoint_path)
    model = model.to(device)
    model.eval()
    print(f"✓ Loaded checkpoint: {checkpoint_path}")
    print(f"✓ Model moved to {device}")

    # Initialize attribution method
    print("\n[3/6] Initializing Integrated Gradients...")
    ig = IntegratedGradients(model, use_embeddings=True)
    print("✓ Integrated Gradients initialized")

    # Option 1: Functional API (simple one-off evaluation)
    print("\n[4/6] Evaluating with Functional API...")
    print("-" * 70)
    print("Using: evaluate_approach(model, dataloader, method, ...)")

    results_functional = evaluate_approach(
        model,
        test_loader,
        ig,
        metrics=["comprehensiveness", "sufficiency"],
        percentages=[10, 20, 50],
    )

    print("\n" + "=" * 70)
    print("Dataset-Wide Results (Functional API)")
    print("=" * 70)
    comp = results_functional["comprehensiveness"]
    suff = results_functional["sufficiency"]
    print(f"\nComprehensiveness: {comp:.4f}")
    print(f"Sufficiency:       {suff:.4f}")

    # Option 2: Class-based API (for multiple evaluations or advanced use)
    print("\n[5/6] Evaluating with Class-based API...")
    print("-" * 70)
    print("Using: Evaluator(model, ...).evaluate_approach(dataloader, method)")
    print("(Recommended for comparing multiple methods)")

    evaluator = Evaluator(model, percentages=[10, 20, 50])
    results_class = evaluator.evaluate_approach(
        test_loader, ig, metrics=["comprehensiveness", "sufficiency"]
    )

    print("\n" + "=" * 70)
    print("Dataset-Wide Results (Class-based API)")
    print("=" * 70)
    comp = results_class["comprehensiveness"]
    suff = results_class["sufficiency"]
    print(f"\nComprehensiveness: {comp:.4f}")
    print(f"Sufficiency:       {suff:.4f}")
    print("\nNote: Both APIs produce identical results!")

    # Get a single batch for detailed analysis
    print("\n[6/6] Detailed analysis on a single batch...")
    print("-" * 70)
    batch = next(iter(test_loader))

    # Move batch to device
    batch_on_device = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_on_device[key] = value.to(device)
        elif isinstance(value, tuple) and len(value) >= 2:
            # Handle (time, values) tuples
            time_part = value[0]
            if time_part is not None and isinstance(time_part, torch.Tensor):
                time_part = time_part.to(device)

            values_part = value[1]
            if isinstance(values_part, torch.Tensor):
                values_part = values_part.to(device)

            batch_on_device[key] = (time_part, values_part) + value[2:]
        else:
            batch_on_device[key] = value

    batch_size = len(batch_on_device["patient_id"])
    print(f"Analyzing batch of {batch_size} samples...")

    # Compute attributions
    attributions = ig.attribute(**batch_on_device, steps=10)
    print("✓ Computed attributions")

    # Get detailed breakdown with debug info
    print("\n*** COMPREHENSIVENESS DEBUG ***")
    comp_detailed = evaluator.metrics["comprehensiveness"].compute_detailed(
        batch_on_device, attributions, debug=True
    )

    print("\n*** SUFFICIENCY DEBUG ***")
    suff_detailed = evaluator.metrics["sufficiency"].compute_detailed(
        batch_on_device, attributions, debug=True
    )

    print("\n" + "=" * 70)
    print("Summary by Percentage (Single Batch)")
    print("=" * 70)

    print("\nComprehensiveness by percentage:")
    for pct in [10, 20, 50]:
        scores = comp_detailed[pct]
        valid_scores = scores[~torch.isnan(scores)]
        if len(valid_scores) > 0:
            print(
                f"  {pct:3d}%: mean={valid_scores.mean():.4f}, "
                f"std={valid_scores.std():.4f} "
                f"(n={len(valid_scores)})"
            )

    print("\nSufficiency by percentage:")
    for pct in [10, 20, 50]:
        scores = suff_detailed[pct]
        valid_scores = scores[~torch.isnan(scores)]
        if len(valid_scores) > 0:
            print(
                f"  {pct:3d}%: mean={valid_scores.mean():.4f}, "
                f"std={valid_scores.std():.4f} "
                f"(n={len(valid_scores)})"
            )

    # Interpretation guide
    print("\n" + "=" * 70)
    print("Interpretation Guide")
    print("=" * 70)
    print(
        """
Comprehensiveness: Measures how much removing important features hurts
                   the prediction. Higher scores = more faithful.
                   Typical range: 0.3-0.5 for good attributions

Sufficiency:       Measures how much keeping only important features
                   hurts the prediction. Lower scores = more faithful.
                   Typical range: 0.0-0.2 for good attributions

Good attributions should have:
  • High comprehensiveness (important features are necessary)
  • Low sufficiency (important features are sufficient)

Two API Options:
  1. Functional API (simple, one-off evaluations):
     results = evaluate_approach(model, dataloader, method, ...)
     
  2. Class-based API (efficient for multiple comparisons):
     evaluator = Evaluator(model, percentages=[10, 20, 50])
     ig_results = evaluator.evaluate_approach(dataloader, ig)
     chefer_results = evaluator.evaluate_approach(dataloader, chefer)
    """
    )


if __name__ == "__main__":
    main()
