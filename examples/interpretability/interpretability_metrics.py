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
from pyhealth.metrics.interpretability import Evaluator, evaluate_attribution
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
    test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False)
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

    # Simple forward pass on one batch with debug output
    print("\n[4/6] Running single batch with debug output...")
    print("-" * 70)

    # Get one batch
    test_batch = next(iter(test_loader))

    # Move batch to device (handles both tensors and tuples)
    test_batch_device = {}
    for key, value in test_batch.items():
        if isinstance(value, torch.Tensor):
            test_batch_device[key] = value.to(device)
        elif isinstance(value, tuple):
            # StageNet format: (time, values) or similar tuples
            test_batch_device[key] = tuple(
                v.to(device) if isinstance(v, torch.Tensor) else v for v in value
            )
        else:
            # Keep non-tensor values as-is (labels, metadata, etc.)
            test_batch_device[key] = value

    # Get batch size from first tensor found
    batch_size_val = None
    for value in test_batch_device.values():
        if isinstance(value, torch.Tensor):
            batch_size_val = value.shape[0]
            break
        elif isinstance(value, tuple):
            for v in value:
                if isinstance(v, torch.Tensor):
                    batch_size_val = v.shape[0]
                    break
            if batch_size_val is not None:
                break

    print(f"Batch size: {batch_size_val}")

    # Compute attributions for this batch
    print("\nComputing attributions with Integrated Gradients...")
    attributions = ig.attribute(**test_batch_device, steps=10)
    print(f"✓ Attributions computed for {len(attributions)} feature types")

    # Initialize evaluator
    evaluator = Evaluator(model, percentages=[1, 99])

    # Compute metrics for single batch
    print("\nComputing metrics on single batch...")
    comp_metric = evaluator.metrics["comprehensiveness"]
    comp_scores, comp_mask = comp_metric.compute(test_batch_device, attributions)

    suff_metric = evaluator.metrics["sufficiency"]
    suff_scores, suff_mask = suff_metric.compute(test_batch_device, attributions)

    print("\n" + "=" * 70)
    print("Single Batch Results Summary")
    print("=" * 70)
    if comp_mask.sum() > 0:
        valid_comp = comp_scores[comp_mask]
        print(f"Comprehensiveness (valid samples): {valid_comp.mean():.4f}")
        print(f"  Valid samples: {comp_mask.sum()}/{len(comp_mask)}")
    else:
        print("Comprehensiveness: No valid samples")

    if suff_mask.sum() > 0:
        valid_suff = suff_scores[suff_mask]
        print(f"Sufficiency (valid samples): {valid_suff.mean():.4f}")
        print(f"  Valid samples: {suff_mask.sum()}/{len(suff_mask)}")
    else:
        print("Sufficiency: No valid samples")

    # Option 1: Functional API (simple one-off evaluation)
    print("\n[5/6] Evaluating with Functional API on full dataset...")
    print("-" * 70)
    print("Using: evaluate_attribution(model, dataloader, method, ...)")

    results_functional = evaluate_attribution(
        model,
        test_loader,
        ig,
        metrics=["comprehensiveness", "sufficiency"],
        percentages=[25, 50, 99],
    )

    print("\n" + "=" * 70)
    print("Dataset-Wide Results (Functional API)")
    print("=" * 70)
    comp = results_functional["comprehensiveness"]
    suff = results_functional["sufficiency"]
    print(f"\nComprehensiveness: {comp:.4f}")
    print(f"Sufficiency:       {suff:.4f}")


if __name__ == "__main__":
    main()
