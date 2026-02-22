"""
Synthetic EHR Generation Baselines using PyHealth

This script demonstrates how to use PyHealth's infrastructure with various
baseline generative models for synthetic EHR data.

Supported models:
- GReaT: Tabular data generation using language models
- CTGAN: Conditional GAN for tabular data
- TVAE: Variational Autoencoder for tabular data
- TransformerBaseline: Custom transformer for sequential EHR

Usage:
    # Using GReaT model
    python synthetic_ehr_baselines.py \\
        --mimic_root /path/to/mimic3 \\
        --train_patients /path/to/train_ids.txt \\
        --test_patients /path/to/test_ids.txt \\
        --output_dir /path/to/output \\
        --mode great

    # Using CTGAN model
    python synthetic_ehr_baselines.py \\
        --mimic_root /path/to/mimic3 \\
        --train_patients /path/to/train_ids.txt \\
        --test_patients /path/to/test_ids.txt \\
        --output_dir /path/to/output \\
        --mode ctgan

    # Using PyHealth TransformerEHRGenerator
    python synthetic_ehr_baselines.py \\
        --mimic_root /path/to/mimic3 \\
        --train_patients /path/to/train_ids.txt \\
        --test_patients /path/to/test_ids.txt \\
        --output_dir /path/to/output \\
        --mode transformer_baseline
"""

import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm, trange

from pyhealth.utils.synthetic_ehr_utils import (
    process_mimic_for_generation,
    tabular_to_sequences,
    sequences_to_tabular,
)


def train_great_model(train_flattened, args):
    """Train GReaT model on flattened EHR data."""
    try:
        import be_great
    except ImportError:
        raise ImportError(
            "be_great is not installed. Install with: pip install be-great"
        )

    print("\n=== Training GReaT Model ===")
    model = be_great.GReaT(
        llm=args.great_llm,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dataloader_num_workers=args.num_workers,
        fp16=torch.cuda.is_available(),
    )

    model.fit(train_flattened)

    # Save model
    save_path = os.path.join(args.output_dir, "great")
    os.makedirs(save_path, exist_ok=True)
    model.save(save_path)

    # Generate synthetic data
    print(f"\n=== Generating {args.num_synthetic_samples} synthetic samples ===")
    synthetic_data = model.sample(n_samples=args.num_synthetic_samples)

    # Save
    synthetic_data.to_csv(
        os.path.join(save_path, "great_synthetic_flattened_ehr.csv"), index=False
    )

    print(f"Saved synthetic data to {save_path}")
    return synthetic_data


def train_ctgan_model(train_flattened, args):
    """Train CTGAN model on flattened EHR data."""
    try:
        from sdv.metadata import Metadata
        from sdv.single_table import CTGANSynthesizer
    except ImportError:
        raise ImportError("sdv is not installed. Install with: pip install sdv")

    print("\n=== Training CTGAN Model ===")

    # Auto-detect metadata
    metadata = Metadata.detect_from_dataframe(data=train_flattened)

    # Set all columns as numerical
    for column in train_flattened.columns:
        metadata.update_column(column_name=column, sdtype="numerical")

    # Initialize and train
    synthesizer = CTGANSynthesizer(
        metadata, epochs=args.epochs, batch_size=args.batch_size
    )
    synthesizer.fit(train_flattened)

    # Save model
    save_path = os.path.join(args.output_dir, "ctgan")
    os.makedirs(save_path, exist_ok=True)
    synthesizer.save(filepath=os.path.join(save_path, "synthesizer.pkl"))

    # Generate synthetic data
    print(f"\n=== Generating {args.num_synthetic_samples} synthetic samples ===")
    synthetic_data = synthesizer.sample(num_rows=args.num_synthetic_samples)

    # Save
    synthetic_data.to_csv(
        os.path.join(save_path, "ctgan_synthetic_flattened_ehr.csv"), index=False
    )

    print(f"Saved synthetic data to {save_path}")
    return synthetic_data


def train_tvae_model(train_flattened, args):
    """Train TVAE model on flattened EHR data."""
    try:
        from sdv.metadata import Metadata
        from sdv.single_table import TVAESynthesizer
    except ImportError:
        raise ImportError("sdv is not installed. Install with: pip install sdv")

    print("\n=== Training TVAE Model ===")

    # Auto-detect metadata
    metadata = Metadata.detect_from_dataframe(data=train_flattened)

    # Set all columns as numerical
    for column in train_flattened.columns:
        metadata.update_column(column_name=column, sdtype="numerical")

    # Initialize and train
    synthesizer = TVAESynthesizer(
        metadata, epochs=args.epochs, batch_size=args.batch_size
    )
    synthesizer.fit(train_flattened)

    # Save model
    save_path = os.path.join(args.output_dir, "tvae")
    os.makedirs(save_path, exist_ok=True)
    synthesizer.save(filepath=os.path.join(save_path, "synthesizer.pkl"))

    # Generate synthetic data
    print(f"\n=== Generating {args.num_synthetic_samples} synthetic samples ===")
    synthetic_data = synthesizer.sample(num_rows=args.num_synthetic_samples)

    # Save
    synthetic_data.to_csv(
        os.path.join(save_path, "tvae_synthetic_flattened_ehr.csv"), index=False
    )

    print(f"Saved synthetic data to {save_path}")
    return synthetic_data


def train_transformer_baseline(train_ehr, args):
    """Train PyHealth TransformerEHRGenerator on sequential EHR data."""
    from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
    from pyhealth.tasks import SyntheticEHRGenerationMIMIC3
    from pyhealth.models import TransformerEHRGenerator
    from pyhealth.trainer import Trainer

    print("\n=== Training Transformer Baseline with PyHealth ===")

    # Load MIMIC-III dataset
    print("Loading MIMIC-III dataset...")
    base_dataset = MIMIC3Dataset(
        root=args.mimic_root, tables=["DIAGNOSES_ICD"], num_workers=args.num_workers
    )

    # Apply task
    print("Applying SyntheticEHRGenerationMIMIC3 task...")
    task = SyntheticEHRGenerationMIMIC3(min_visits=2)
    sample_dataset = base_dataset.set_task(task, num_workers=args.num_workers)

    # Split dataset
    train_dataset, val_dataset, _ = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])

    # Create dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    print("Initializing TransformerEHRGenerator...")
    model = TransformerEHRGenerator(
        dataset=sample_dataset,
        embedding_dim=256,
        num_heads=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_length=512,
    )

    # Train
    print("Training model...")
    trainer = Trainer(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        output_path=args.output_dir,
        exp_name="transformer_baseline",
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        monitor="loss",
        monitor_criterion="min",
        optimizer_params={"lr": 1e-4},
    )

    # Generate synthetic data
    print(f"\n=== Generating {args.num_synthetic_samples} synthetic samples ===")
    model.eval()
    with torch.no_grad():
        synthetic_nested_codes = model.generate(
            num_samples=args.num_synthetic_samples,
            max_visits=10,
            max_codes_per_visit=20,
            max_length=512,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
        )

    # Convert to sequences and tabular
    from pyhealth.utils.synthetic_ehr_utils import (
        nested_codes_to_sequences,
        sequences_to_tabular,
    )

    synthetic_sequences = nested_codes_to_sequences(synthetic_nested_codes)
    synthetic_df = sequences_to_tabular(synthetic_sequences)

    # Save
    save_path = os.path.join(args.output_dir, "transformer_baseline")
    os.makedirs(save_path, exist_ok=True)

    synthetic_df.to_csv(
        os.path.join(save_path, "transformer_baseline_synthetic_ehr.csv"), index=False
    )

    print(f"Saved synthetic data to {save_path}")
    return synthetic_df


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline models for synthetic EHR generation"
    )

    # Data paths
    parser.add_argument(
        "--mimic_root",
        type=str,
        required=True,
        help="Path to MIMIC data directory",
    )
    parser.add_argument(
        "--train_patients",
        type=str,
        default=None,
        help="Path to train patient IDs file",
    )
    parser.add_argument(
        "--test_patients",
        type=str,
        default=None,
        help="Path to test patient IDs file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./synthetic_data",
        help="Output directory for synthetic data",
    )

    # Model selection
    parser.add_argument(
        "--mode",
        type=str,
        default="transformer_baseline",
        choices=["great", "ctgan", "tvae", "transformer_baseline"],
        help="Baseline model to use",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers"
    )
    parser.add_argument(
        "--num_synthetic_samples",
        type=int,
        default=10000,
        help="Number of synthetic samples to generate",
    )

    # Model-specific parameters
    parser.add_argument(
        "--great_llm",
        type=str,
        default="tabularisai/Qwen3-0.3B-distil",
        help="Language model for GReaT",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process MIMIC data
    print("=" * 80)
    print("Processing MIMIC Data")
    print("=" * 80)

    if args.mode == "transformer_baseline":
        # For transformer baseline, we process data through PyHealth
        # Dataset will be loaded in the training function
        print("Will load data through PyHealth dataset...")
        train_transformer_baseline(None, args)

    else:
        # For tabular models, we need flattened representation
        print("Processing MIMIC data for tabular models...")
        data = process_mimic_for_generation(
            args.mimic_root,
            args.train_patients,
            args.test_patients,
        )

        train_flattened = data["train_flattened"]
        print(f"Train flattened shape: {train_flattened.shape}")

        # Train selected model
        print("\n" + "=" * 80)
        print(f"Training {args.mode.upper()} Model")
        print("=" * 80)

        if args.mode == "great":
            synthetic_data = train_great_model(train_flattened, args)
        elif args.mode == "ctgan":
            synthetic_data = train_ctgan_model(train_flattened, args)
        elif args.mode == "tvae":
            synthetic_data = train_tvae_model(train_flattened, args)

        print("\n" + "=" * 80)
        print("Synthetic Data Statistics")
        print("=" * 80)
        print(f"Shape: {synthetic_data.shape}")
        print(f"Columns: {len(synthetic_data.columns)}")
        print(f"\nFirst few rows:")
        print(synthetic_data.head())

    print("\n" + "=" * 80)
    print("COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
