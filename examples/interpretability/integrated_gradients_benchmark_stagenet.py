"""
Example of using StageNet for mortality prediction on MIMIC-IV with Integrated Gradients.

This example demonstrates:
1. Loading MIMIC-IV data
2. Loading existing processors
3. Applying the MortalityPredictionStageNetMIMIC4 task
4. Loading a pre-trained StageNet model
5. Benchmarking model performance on test set
6. Computing Integrated Gradients attributions for test samples

Processor Caching:
    The script loads existing processors from:
    ../../output/processors/stagenet_mortality_mimic4/
"""

from pathlib import Path

import torch

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    load_processors,
    save_processors,
    split_by_patient,
)
from pyhealth.interpret.methods import IntegratedGradients
from pyhealth.metrics.interpretability import evaluate_attribution
from pyhealth.models import StageNet
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.trainer import Trainer

# Configuration
CHECKPOINT_PATH = (
    "/home/johnwu3/projects/PyHealth_Branch_Testing/PyHealth/output/"
    "20260131-184735/best.ckpt"
)
PROCESSOR_DIR = "../output/processors/stagenet_mortality_mimic4"


def main():
    """Main execution function for StageNet mortality prediction with IG."""

    # STEP 1: Load MIMIC-IV base dataset
    print("Loading MIMIC-IV dataset...")
    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
        num_workers=8,
        cache_dir="/shared/eng/pyhealth/ig",
    )

    # STEP 2: Check for existing processors and load/create accordingly
    processor_dir_path = Path(PROCESSOR_DIR)
    input_procs_file = processor_dir_path / "input_processors.pkl"
    output_procs_file = processor_dir_path / "output_processors.pkl"

    input_processors = None
    output_processors = None

    if input_procs_file.exists() and output_procs_file.exists():
        # Load existing processors
        print(f"\n{'='*60}")
        print("LOADING EXISTING PROCESSORS")
        print(f"{'='*60}")
        input_processors, output_processors = load_processors(PROCESSOR_DIR)
        print(f"✓ Using pre-fitted processors from {PROCESSOR_DIR}")
    else:
        # Will create new processors
        print(f"\n{'='*60}")
        print("NO EXISTING PROCESSORS FOUND")
        print(f"{'='*60}")
        print(f"Will create and save new processors to {PROCESSOR_DIR}")

    # STEP 3: Apply StageNet mortality prediction task
    print("Applying MortalityPredictionStageNetMIMIC4 task...")
    sample_dataset = base_dataset.set_task(
        MortalityPredictionStageNetMIMIC4(),
        num_workers=8,
        input_processors=input_processors,
        output_processors=output_processors,
    )

    print(f"Total samples: {len(sample_dataset)}")

    # Save processors if they were newly created
    if input_processors is None and output_processors is None:
        print(f"\n{'='*60}")
        print("SAVING NEWLY CREATED PROCESSORS")
        print(f"{'='*60}")
        save_processors(sample_dataset, PROCESSOR_DIR)
        print(f"✓ Processors saved to {PROCESSOR_DIR}")

    # STEP 4: Split dataset
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=256, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=256, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=256, shuffle=False)

    # STEP 5: Initialize and train/load model
    print("\nInitializing StageNet model...")
    model = StageNet(
        dataset=sample_dataset,
        embedding_dim=128,
        chunk_size=128,
        levels=3,
        dropout=0.3,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {num_params} parameters")

    trainer = Trainer(
        model=model,
        device="cuda:0",
        metrics=["pr_auc", "roc_auc", "accuracy", "f1"],
    )

    # Check if checkpoint exists before loading
    checkpoint_path_obj = Path(CHECKPOINT_PATH)
    if checkpoint_path_obj.exists():
        print(f"\n{'='*60}")
        print("LOADING EXISTING CHECKPOINT")
        print(f"{'='*60}")
        print(f"Path: {CHECKPOINT_PATH}")
        trainer.load_ckpt(CHECKPOINT_PATH)
        print("✓ Checkpoint loaded successfully")
    else:
        print(f"\n{'='*60}")
        print("TRAINING NEW MODEL")
        print(f"{'='*60}")
        print(f"Checkpoint not found at: {CHECKPOINT_PATH}")
        print("Training a new model from scratch...")

        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=25,
            monitor="pr_auc",
            optimizer_params={"lr": 1e-5},
        )
        print("\n✓ Training completed")

    # STEP 6: Benchmark model on test set
    print("\n" + "=" * 70)
    print("BENCHMARKING MODEL ON TEST SET")
    print("=" * 70)

    test_metrics = trainer.evaluate(test_loader)

    print("\nTest Set Performance:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    # STEP 7: Compute Integrated Gradients Faithfulness Metrics
    print("\n" + "=" * 70)
    print("COMPUTING INTEGRATED GRADIENTS FAITHFULNESS METRICS")
    print("=" * 70)

    # Initialize Integrated Gradients with 5 steps for faster computation
    ig = IntegratedGradients(model, use_embeddings=True, steps=5)
    print("✓ Integrated Gradients initialized (steps=5)")

    # Compute sufficiency and comprehensiveness on test set
    print("\nEvaluating Integrated Gradients on test set...")
    print("This computes sufficiency and comprehensiveness metrics.")

    # Use the functional API to evaluate attribution faithfulness
    results = evaluate_attribution(
        model,
        test_loader,
        ig,
        metrics=["comprehensiveness", "sufficiency"],
        percentages=[25, 50, 99],
    )

    # Print results
    print("\n" + "=" * 70)
    print("FAITHFULNESS METRICS RESULTS")
    print("=" * 70)

    print("\nMetrics (averaged over test set):")
    print(f"  Comprehensiveness: {results['comprehensiveness']:.6f}")
    print(f"  Sufficiency:       {results['sufficiency']:.6f}")

    print("\nInterpretation:")
    print("  • Comprehensiveness: How much does removing the top features")
    print("                       change the model's prediction?")
    print("                       (Higher is better)")
    print("  • Sufficiency: How much does keeping only the top features")
    print("                 maintain the model's prediction?")
    print("                 (Lower is better)")


if __name__ == "__main__":
    main()
