"""
Example of training a Transformer-based synthetic EHR generator on MIMIC-III data.

This example demonstrates the complete workflow for training a generative model
that can create synthetic patient histories:

1. Loading MIMIC-III data
2. Applying the SyntheticEHRGenerationMIMIC3 task
3. Training a TransformerEHRGenerator model
4. Generating synthetic patient histories
5. Converting synthetic data to different formats

Usage:
    python synthetic_ehr_mimic3_transformer.py \\
        --mimic_root /path/to/mimic3 \\
        --output_dir /path/to/output \\
        --epochs 50 \\
        --batch_size 32
"""

import os
import argparse
import pandas as pd
import torch

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.tasks import SyntheticEHRGenerationMIMIC3
from pyhealth.models import TransformerEHRGenerator
from pyhealth.trainer import Trainer
from pyhealth.utils.synthetic_ehr_utils import (
    nested_codes_to_sequences,
    sequences_to_tabular,
)


def main(args):
    """Main training and generation pipeline."""

    print("\n" + "=" * 80)
    print("STEP 1: Load MIMIC-III Dataset")
    print("=" * 80)

    # Load MIMIC-III base dataset
    base_dataset = MIMIC3Dataset(
        root=args.mimic_root,
        tables=["DIAGNOSES_ICD"],  # Only need diagnosis codes
        code_mapping={"ICD9CM": "CCSCM"} if args.use_ccs else None,
        num_workers=args.num_workers,
    )

    print(f"\nDataset loaded:")
    print(f"  Total patients: {len(base_dataset.patient_to_index)}")
    print(f"  Total admissions: {sum(len(p.get_events('admissions')) for p in base_dataset)}")

    print("\n" + "=" * 80)
    print("STEP 2: Apply Synthetic EHR Generation Task")
    print("=" * 80)

    # Create task for synthetic generation
    task = SyntheticEHRGenerationMIMIC3(
        min_visits=args.min_visits, max_visits=args.max_visits
    )

    # Generate samples
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)

    print(f"\nTask applied:")
    print(f"  Total samples: {len(sample_dataset)}")
    print(f"  Input schema: {sample_dataset.input_schema}")
    print(f"  Output schema: {sample_dataset.output_schema}")

    # Inspect a sample
    sample = sample_dataset[0]
    print(f"\nSample structure:")
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"  Visit codes shape: {sample['visit_codes'].shape}")
    print(f"  Number of visits: {sample['visit_codes'].shape[0]}")
    print(f"  Max codes per visit: {sample['visit_codes'].shape[1]}")

    print("\n" + "=" * 80)
    print("STEP 3: Split Dataset")
    print("=" * 80)

    # Split by patient (important to prevent data leakage)
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [args.train_ratio, args.val_ratio, 1 - args.train_ratio - args.val_ratio]
    )

    print(f"\nDataset split:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    print("\n" + "=" * 80)
    print("STEP 4: Initialize TransformerEHRGenerator Model")
    print("=" * 80)

    # Create the generative model
    model = TransformerEHRGenerator(
        dataset=sample_dataset,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized:")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Vocabulary size: {model.vocab_size}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Num heads: {args.num_heads}")

    print("\n" + "=" * 80)
    print("STEP 5: Train the Model")
    print("=" * 80)

    # Create trainer
    trainer = Trainer(
        model=model,
        device=args.device,
        output_path=args.output_dir,
        exp_name="transformer_ehr_generator",
    )

    # Train
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="loss",  # Monitor validation loss
        monitor_criterion="min",
        optimizer_params={"lr": args.learning_rate, "weight_decay": args.weight_decay},
    )

    print("\n" + "=" * 80)
    print("STEP 6: Evaluate on Test Set")
    print("=" * 80)

    # Evaluate
    test_results = trainer.evaluate(test_loader)
    print("\nTest Results:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 80)
    print("STEP 7: Generate Synthetic Patient Histories")
    print("=" * 80)

    print(f"\nGenerating {args.num_synthetic_samples} synthetic patients...")

    # Generate synthetic samples
    model.eval()
    with torch.no_grad():
        synthetic_nested_codes = model.generate(
            num_samples=args.num_synthetic_samples,
            max_visits=args.max_visits,
            max_codes_per_visit=args.max_codes_per_visit,
            max_length=args.max_seq_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    print(f"Generated {len(synthetic_nested_codes)} synthetic patients")

    # Convert to different formats
    print("\nConverting to different formats...")

    # 1. Convert to text sequences
    synthetic_sequences = nested_codes_to_sequences(synthetic_nested_codes)

    # 2. Convert to tabular format
    synthetic_df = sequences_to_tabular(synthetic_sequences)

    # Display statistics
    print(f"\nSynthetic data statistics:")
    print(f"  Total patients: {len(synthetic_nested_codes)}")
    print(f"  Total visits: {synthetic_df['HADM_ID'].nunique()}")
    print(f"  Total codes: {len(synthetic_df)}")
    print(f"  Avg visits per patient: {len(synthetic_df) / len(synthetic_nested_codes):.2f}")
    print(f"  Unique codes: {synthetic_df['ICD9_CODE'].nunique()}")

    # Save synthetic data
    print("\n" + "=" * 80)
    print("STEP 8: Save Synthetic Data")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save as CSV
    synthetic_csv_path = os.path.join(
        args.output_dir, "synthetic_ehr_transformer.csv"
    )
    synthetic_df.to_csv(synthetic_csv_path, index=False)
    print(f"\nSaved synthetic data to: {synthetic_csv_path}")

    # Save sequences as text file
    synthetic_seq_path = os.path.join(
        args.output_dir, "synthetic_sequences_transformer.txt"
    )
    with open(synthetic_seq_path, "w") as f:
        for seq in synthetic_sequences:
            f.write(seq + "\n")
    print(f"Saved synthetic sequences to: {synthetic_seq_path}")

    # Display sample synthetic patient
    print("\n" + "=" * 80)
    print("Sample Synthetic Patient History")
    print("=" * 80)
    print(f"\nPatient 0:")
    for visit_idx, visit_codes in enumerate(synthetic_nested_codes[0]):
        print(f"  Visit {visit_idx + 1}: {visit_codes}")

    print("\n" + "=" * 80)
    print("COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Transformer-based synthetic EHR generator on MIMIC-III"
    )

    # Dataset arguments
    parser.add_argument(
        "--mimic_root",
        type=str,
        required=True,
        help="Path to MIMIC-III data directory",
    )
    parser.add_argument(
        "--use_ccs",
        action="store_true",
        help="Map ICD9 codes to CCS categories",
    )
    parser.add_argument(
        "--min_visits", type=int, default=2, help="Minimum visits per patient"
    )
    parser.add_argument(
        "--max_visits",
        type=int,
        default=None,
        help="Maximum visits per patient (None = no limit)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )

    # Model arguments
    parser.add_argument(
        "--embedding_dim", type=int, default=256, help="Embedding dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=1024,
        help="Feedforward network dimension",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    # Generation arguments
    parser.add_argument(
        "--num_synthetic_samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--max_codes_per_visit",
        type=int,
        default=20,
        help="Maximum codes per visit during generation",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-k sampling (0 = disabled)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Nucleus sampling threshold"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./synthetic_ehr_output",
        help="Output directory for model and synthetic data",
    )

    args = parser.parse_args()

    # Run main pipeline
    main(args)
