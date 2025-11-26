"""PromptEHR: Training and Generation Example on MIMIC-III

This example demonstrates the complete PromptEHR pipeline:
1. Load MIMIC-III patient records
2. Train PromptEHR model for synthetic EHR generation
3. Generate synthetic patients with realistic visit structures
4. Evaluate generation quality

References:
    - Paper: "PromptEHR: Conditional Electronic Health Records Generation with Prompt Learning"
    - pehr_scratch implementation: /u/jalenj4/pehr_scratch/
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import BartConfig, get_linear_schedule_with_warmup

# PyHealth imports
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.models import PromptEHR
from pyhealth.trainer import Trainer
from pyhealth.datasets.promptehr_dataset import (
    create_promptehr_tokenizer,
    PromptEHRDataset,
    load_mimic_data
)
from pyhealth.datasets.promptehr_collator import EHRDataCollator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_promptehr(
    mimic3_root: str,
    output_dir: str = "./promptehr_outputs",
    num_patients: int = 46520,  # Full MIMIC-III dataset
    batch_size: int = 16,
    num_epochs: int = 20,
    learning_rate: float = 1e-5,
    warmup_steps: int = 1000,
    val_split: float = 0.2,
    device: str = "cuda",
    checkpoint_path: str = None
):
    """Train PromptEHR model on MIMIC-III dataset.

    Args:
        mimic3_root: Path to MIMIC-III data directory containing:
            - PATIENTS.csv
            - ADMISSIONS.csv
            - DIAGNOSES_ICD.csv
        output_dir: Directory to save outputs (checkpoints, logs)
        num_patients: Number of patients to load (default: full dataset)
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: AdamW learning rate
        warmup_steps: Linear warmup steps for scheduler
        val_split: Validation split ratio
        device: Device to use ('cuda' or 'cpu')
        checkpoint_path: Path to resume from checkpoint (optional)

    Returns:
        Trained PromptEHR model
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("PromptEHR Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"MIMIC-III root: {mimic3_root}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")

    # Step 1: Load MIMIC-III patient records
    logger.info("\n" + "=" * 80)
    logger.info("Loading MIMIC-III Patient Records")
    logger.info("=" * 80)

    patients_path = os.path.join(mimic3_root, "PATIENTS.csv")
    admissions_path = os.path.join(mimic3_root, "ADMISSIONS.csv")
    diagnoses_path = os.path.join(mimic3_root, "DIAGNOSES_ICD.csv")

    patient_records, diagnosis_codes = load_mimic_data(
        patients_path=patients_path,
        admissions_path=admissions_path,
        diagnoses_path=diagnoses_path,
        num_patients=num_patients,
        logger=logger
    )

    logger.info(f"Loaded {len(patient_records)} patients")
    logger.info(f"Vocabulary size: {len(diagnosis_codes)} diagnosis codes")

    # Step 2: Create tokenizer
    logger.info("\n" + "=" * 80)
    logger.info("Creating Tokenizer")
    logger.info("=" * 80)

    tokenizer = create_promptehr_tokenizer(diagnosis_codes)
    vocab_size = tokenizer.get_vocabulary_size()
    logger.info(f"Tokenizer vocabulary size: {vocab_size}")
    logger.info(f"  Special tokens: 7")
    logger.info(f"  Diagnosis codes: {len(diagnosis_codes)}")
    logger.info(f"  Code offset: 7")

    # Step 3: Create dataset
    logger.info("\n" + "=" * 80)
    logger.info("Creating Dataset")
    logger.info("=" * 80)

    dataset = PromptEHRDataset(patient_records, tokenizer, logger)
    logger.info(f"Dataset size: {len(dataset)} patients")

    # Train/validation split
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logger.info(f"Train size: {train_size}, Validation size: {val_size}")

    # Create data collator
    collator = EHRDataCollator(
        tokenizer=tokenizer,
        max_seq_length=512,
        logger=logger
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )

    logger.info(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # Step 4: Initialize model
    logger.info("\n" + "=" * 80)
    logger.info("Initializing PromptEHR Model")
    logger.info("=" * 80)

    model = PromptEHR(
        dataset=None,  # Generative model, no discriminative task
        n_num_features=1,  # Age
        cat_cardinalities=[2],  # Gender (M/F)
        d_hidden=128,
        prompt_length=1,
        bart_config_name="facebook/bart-base",
        _custom_vocab_size=vocab_size  # Custom vocab size for MIMIC-III
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Step 5: Configure trainer
    logger.info("\n" + "=" * 80)
    logger.info("Configuring Trainer")
    logger.info("=" * 80)

    trainer = Trainer(
        model=model,
        checkpoint_path=checkpoint_path,
        metrics=["loss"],
        device=device,
        enable_logging=True,
        output_path=str(output_dir)
    )

    # Step 6: Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=num_epochs,
        optimizer_params={"lr": learning_rate, "weight_decay": 0.01},
        monitor="loss"
    )

    # Step 7: Save final model
    final_checkpoint = checkpoint_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.bart_model.state_dict(),  # Save BART model state
        'tokenizer': tokenizer,
        'diagnosis_codes': diagnosis_codes,
        'config': {
            'dataset': None,
            'n_num_features': 1,
            'cat_cardinalities': [2],
            'd_hidden': 128,
            'prompt_length': 1,
            'bart_config_name': "facebook/bart-base",
            '_custom_vocab_size': vocab_size
        }
    }, final_checkpoint)
    logger.info(f"\nFinal model saved to: {final_checkpoint}")

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)

    return model, tokenizer


def generate_synthetic_patients(
    model: PromptEHR,
    tokenizer,
    patient_records: List,
    num_patients: int = 100,
    temperature: float = 0.7,
    device: str = "cuda"
):
    """Generate synthetic patients using trained PromptEHR model.

    Args:
        model: Trained PromptEHR model
        tokenizer: PromptEHR tokenizer
        patient_records: Real patient records (for structure sampling)
        num_patients: Number of synthetic patients to generate
        temperature: Sampling temperature
        device: Device to use

    Returns:
        List of generated patient dictionaries
    """
    from pyhealth.datasets.promptehr_dataset import VisitStructureSampler
    from pyhealth.models.promptehr import generate_patient_with_structure_constraints

    logger.info("\n" + "=" * 80)
    logger.info(f"Generating {num_patients} Synthetic Patients")
    logger.info("=" * 80)

    # Initialize visit structure sampler
    structure_sampler = VisitStructureSampler(patient_records, seed=42)
    logger.info(f"Structure sampler: {structure_sampler}")

    # Set model to eval mode
    model.eval()
    model.to(device)

    # Generate patients
    generated_patients = []
    for i in range(num_patients):
        if (i + 1) % 20 == 0:
            logger.info(f"Generated {i + 1}/{num_patients} patients...")

        # Sample realistic visit structure
        target_structure = structure_sampler.sample_structure()

        # Generate patient
        result = generate_patient_with_structure_constraints(
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_structure=target_structure,
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            max_codes_per_visit=25
        )

        # Store result
        demo = result['demographics']
        generated_patients.append({
            'patient_id': f"SYNTH_{i+1:04d}",
            'age': demo['age'],
            'sex': 'M' if demo['sex'] == 0 else 'F',
            'num_visits': result['num_visits'],
            'visits': result['generated_visits']
        })

    logger.info(f"\nGeneration complete: {num_patients} patients created")

    # Display statistics
    total_visits = sum(p['num_visits'] for p in generated_patients)
    total_codes = sum(len(code) for p in generated_patients for visit in p['visits'] for code in visit)
    unique_codes = len(set(code for p in generated_patients for visit in p['visits'] for code in visit))

    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total patients: {num_patients}")
    logger.info(f"  Total visits: {total_visits}")
    logger.info(f"  Total diagnosis codes: {total_codes}")
    logger.info(f"  Unique codes: {unique_codes}")
    logger.info(f"  Average visits/patient: {total_visits/num_patients:.2f}")
    logger.info(f"  Average codes/patient: {total_codes/num_patients:.1f}")

    return generated_patients


def save_synthetic_dataset(
    patients: List[Dict],
    output_path: str,
    format: str = "csv"
):
    """Save generated patients to file.

    Args:
        patients: List of patient dictionaries
        output_path: Path to save file
        format: Output format ('csv' or 'json')
    """
    import csv
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['patient_id', 'age', 'sex', 'num_visits', 'visit_num', 'diagnosis_codes'])

            for patient in patients:
                for visit_idx, visit_codes in enumerate(patient['visits']):
                    codes_str = ';'.join(visit_codes)
                    writer.writerow([
                        patient['patient_id'],
                        f"{patient['age']:.1f}",
                        patient['sex'],
                        patient['num_visits'],
                        visit_idx + 1,
                        codes_str
                    ])

        logger.info(f"Saved {len(patients)} patients to {output_path} (CSV format)")

    elif format == "json":
        with open(output_path, 'w') as f:
            json.dump(patients, f, indent=2)

        logger.info(f"Saved {len(patients)} patients to {output_path} (JSON format)")


def main():
    """Main entry point for PromptEHR training and generation."""
    import argparse

    parser = argparse.ArgumentParser(description="PromptEHR Training and Generation")
    parser.add_argument("--mimic3_root", type=str, required=True,
                        help="Path to MIMIC-III data directory")
    parser.add_argument("--output_dir", type=str, default="./promptehr_outputs",
                        help="Output directory for checkpoints and results")
    parser.add_argument("--num_patients", type=int, default=46520,
                        help="Number of patients to load for training")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--generate_only", action="store_true",
                        help="Skip training, only generate (requires --checkpoint)")
    parser.add_argument("--num_synthetic", type=int, default=100,
                        help="Number of synthetic patients to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for generation")

    args = parser.parse_args()

    # Training
    if not args.generate_only:
        model, tokenizer = train_promptehr(
            mimic3_root=args.mimic3_root,
            output_dir=args.output_dir,
            num_patients=args.num_patients,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            checkpoint_path=args.checkpoint
        )
    else:
        # Load from checkpoint
        if args.checkpoint is None:
            raise ValueError("--checkpoint required when using --generate_only")

        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        tokenizer = checkpoint['tokenizer']

        model = PromptEHR(**checkpoint['config'])
        model.bart_model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
        model.eval()

    # Load patient records for structure sampling
    patients_path = os.path.join(args.mimic3_root, "PATIENTS.csv")
    admissions_path = os.path.join(args.mimic3_root, "ADMISSIONS.csv")
    diagnoses_path = os.path.join(args.mimic3_root, "DIAGNOSES_ICD.csv")

    patient_records, _ = load_mimic_data(
        patients_path=patients_path,
        admissions_path=admissions_path,
        diagnoses_path=diagnoses_path,
        num_patients=args.num_patients,
        logger=logger
    )

    # Generation
    generated_patients = generate_synthetic_patients(
        model=model,
        tokenizer=tokenizer,
        patient_records=patient_records,
        num_patients=args.num_synthetic,
        temperature=args.temperature,
        device=args.device
    )

    # Save results
    output_csv = Path(args.output_dir) / f"synthetic_patients_{args.num_synthetic}.csv"
    save_synthetic_dataset(generated_patients, output_csv, format="csv")

    logger.info("\n" + "=" * 80)
    logger.info("PromptEHR Pipeline Complete!")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Synthetic dataset: {output_csv}")


if __name__ == "__main__":
    main()
