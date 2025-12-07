#!/usr/bin/env python3
"""Quick local generation test for PromptEHR (CPU-only).

This script demonstrates how to:
1. Load a trained PromptEHR checkpoint
2. Generate synthetic patients on CPU (no GPU required)
3. Display results in human-readable format

Usage:
    python3 examples/promptehr_generate_local.py
"""

import sys
sys.path.insert(0, '/u/jalenj4/final/PyHealth')

import torch
import logging
from pathlib import Path

# PyHealth imports
from pyhealth.models import PromptEHR
from pyhealth.datasets.promptehr_dataset import load_mimic_data
from pyhealth.models.promptehr import (
    VisitStructureSampler,
    generate_patient_with_structure_constraints
)


def main():
    """Generate 10 synthetic patients locally on CPU."""

    # Setup
    device = torch.device("cpu")  # Force CPU (no GPU required)
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise, only show warnings/errors
        format='%(message)s'
    )
    logger = logging.getLogger(__name__)

    print("\n" + "="*80)
    print("PromptEHR Local Generation Test (CPU mode)")
    print("="*80)

    # Load checkpoint
    print("\n[1/4] Loading trained checkpoint...")
    checkpoint_path = "./promptehr_outputs/checkpoints/final_model.pt"

    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please ensure training has completed and checkpoint exists.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = checkpoint['tokenizer']

    # Add convenience properties and methods if not present
    # (for compatibility with old checkpoints saved before these were added)
    if not hasattr(tokenizer, 'bos_token_id'):
        tokenizer.pad_token_id = tokenizer.vocabulary("<pad>")  # ID 0
        tokenizer.bos_token_id = tokenizer.vocabulary("<s>")     # ID 1
        tokenizer.eos_token_id = tokenizer.vocabulary("</s>")    # ID 2
        tokenizer.code_offset = 7  # First diagnosis code ID (after 7 special tokens)
    if not hasattr(tokenizer, 'convert_tokens_to_ids'):
        # Add method alias: pehr_scratch API uses convert_tokens_to_ids(token) → int
        def convert_tokens_to_ids(token: str) -> int:
            return tokenizer.convert_tokens_to_indices([token])[0]
        tokenizer.convert_tokens_to_ids = convert_tokens_to_ids
    if not hasattr(tokenizer, 'vocab'):
        # Add vocab object for idx2code and code2idx mappings
        class VocabCompat:
            def __init__(self, tok):
                self.idx2code = tok.vocabulary.idx2token
                self.code2idx = tok.vocabulary.token2idx
            def __len__(self):
                return len(self.idx2code)
        tokenizer.vocab = VocabCompat(tokenizer)

    # Rebuild model
    print("[2/4] Rebuilding model from checkpoint...")
    config = checkpoint['config']
    model = PromptEHR(**config)
    model.bart_model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"      Model vocabulary size: {config['_custom_vocab_size']}")
    print(f"      Hidden dimension: {config['d_hidden']}")
    print(f"      Prompt length: {config['prompt_length']}")

    # Load MIMIC data for structure sampling
    print("[3/4] Loading MIMIC-III data for structure sampling...")
    print("      (Loading 1000 patients for realistic visit distributions)")

    patient_records, _ = load_mimic_data(
        patients_path="/u/jalenj4/pehr_scratch/data_files/PATIENTS.csv",
        admissions_path="/u/jalenj4/pehr_scratch/data_files/ADMISSIONS.csv",
        diagnoses_path="/u/jalenj4/pehr_scratch/data_files/DIAGNOSES_ICD.csv",
        num_patients=1000,
        logger=logger
    )

    # Initialize structure sampler
    structure_sampler = VisitStructureSampler(patient_records, seed=42)
    print(f"      {structure_sampler}")

    # Generate synthetic patients
    n_patients = 10
    print(f"\n[4/4] Generating {n_patients} synthetic patients...")
    print("      (This will take ~10-15 seconds)")
    print()

    print("="*80)
    print("SYNTHETIC PATIENTS")
    print("="*80)
    print()

    for i in range(n_patients):
        # Sample realistic visit structure
        target_structure = structure_sampler.sample_structure()

        # Generate patient
        result = generate_patient_with_structure_constraints(
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_structure=target_structure,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            max_codes_per_visit=25
        )

        # Display patient
        demo = result['demographics']
        print(f"Patient {i+1}:")
        print(f"  Age: {demo['age']} years")
        print(f"  Sex: {'Male' if demo['sex'] == 0 else 'Female'}")
        print(f"  Number of visits: {result['num_visits']}")
        print(f"  Diagnosis codes:")

        for visit_idx, codes in enumerate(result['generated_visits'], 1):
            if codes:
                print(f"    Visit {visit_idx}: {', '.join(codes)}")
            else:
                print(f"    Visit {visit_idx}: (no diagnoses)")
        print()

    print("="*80)
    print("Generation complete!")
    print("="*80)
    print()
    print(f"Successfully generated {n_patients} synthetic patients on CPU.")
    print()


if __name__ == "__main__":
    main()
