"""
Example of using StageNet for DKA (Diabetic Ketoacidosis) prediction on MIMIC-IV.

This example demonstrates:
1. Loading MIMIC-IV data with relevant tables for DKA prediction
2. Applying the DKAPredictionMIMIC4 task
3. Creating a SampleDataset with StageNet processors
4. Training a StageNet model for DKA prediction
5. Testing with synthetic hold-out set (unseen codes, varying lengths)

Target Population:
    - Patients with Type 1 Diabetes Mellitus (T1DM)
    - Predicts DKA occurrence within 90 days of T1DM diagnosis
"""

import os
import random
import numpy as np
import torch

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
    SampleDataset,
)
from pyhealth.datasets.utils import save_processors, load_processors
from pyhealth.models import StageNet
from pyhealth.tasks import DKAPredictionMIMIC4
from pyhealth.trainer import Trainer


def generate_holdout_set(
    sample_dataset: SampleDataset, num_samples: int = 10, seed: int = 42
) -> SampleDataset:
    """Generate synthetic hold-out set with unseen codes and varying lengths.

    This function creates synthetic samples to test the processor's ability to:
    1. Handle completely unseen tokens (mapped to <unk>)
    2. Handle sequence lengths larger than training but within padding

    Args:
        sample_dataset: Original SampleDataset with fitted processors
        num_samples: Number of synthetic samples to generate
        seed: Random seed for reproducibility

    Returns:
        SampleDataset with synthetic samples using fitted processors
    """
    random.seed(seed)
    np.random.seed(seed)

    # Get the fitted processors
    diagnoses_processor = sample_dataset.input_processors["diagnoses"]

    # Get max nested length from diagnoses processor
    max_diag_len = diagnoses_processor._max_nested_len
    # Handle both old and new processor versions
    padding = getattr(diagnoses_processor, "_padding", 0)

    print("\n=== Hold-out Set Generation ===")
    print(f"Diagnoses max nested length: {max_diag_len}")
    print(f"Padding: {padding}")
    print(f"Observed max (without padding): {max_diag_len - padding}")

    synthetic_samples = []

    # DKA-relevant ICD codes for synthetic samples
    synthetic_t1dm_codes = [
        "E1010", "E1011", "E1065", "E1100", "E1122",
        "25001", "25003", "25011", "25013",
    ]
    synthetic_other_codes = [
        "I10", "N179", "J189", "K219", "R509", "I509",
    ]

    for i in range(num_samples):
        # Generate random number of visits (1-5)
        num_visits = random.randint(1, 5)

        # Generate diagnosis codes
        diagnoses_list = []
        diagnoses_times_list = []

        for visit_idx in range(num_visits):
            # Generate sequence length
            observed_max = max_diag_len - padding
            seq_len = random.randint(max(1, observed_max - 2), max(observed_max, 3))

            # Mix of known and unseen codes
            visit_codes = []
            for j in range(seq_len):
                if random.random() < 0.3:
                    # 30% chance of T1DM-related code
                    visit_codes.append(random.choice(synthetic_t1dm_codes))
                elif random.random() < 0.5:
                    # 35% chance of other known code
                    visit_codes.append(random.choice(synthetic_other_codes))
                else:
                    # 35% chance of unseen code
                    visit_codes.append(f"NEWCODE_{i}_{visit_idx}_{j}")

            diagnoses_list.append(visit_codes)

            # Generate time intervals (hours from previous visit)
            if visit_idx == 0:
                diagnoses_times_list.append(0.0)
            else:
                diagnoses_times_list.append(random.uniform(24.0, 720.0))

        # Generate lab data (6-dimensional vectors for DKA task)
        # Categories: glucose, bicarbonate, anion_gap, potassium, sodium, chloride
        num_lab_timestamps = random.randint(3, 10)
        lab_values_list = []
        lab_times_list = []

        for ts_idx in range(num_lab_timestamps):
            # Generate 6D vector with realistic ranges
            lab_vector = []
            # glucose (mg/dL)
            lab_vector.append(
                random.uniform(70.0, 500.0) if random.random() < 0.9 else None
            )
            # bicarbonate (mEq/L)
            lab_vector.append(
                random.uniform(10.0, 30.0) if random.random() < 0.85 else None
            )
            # anion_gap (mEq/L)
            lab_vector.append(
                random.uniform(8.0, 30.0) if random.random() < 0.7 else None
            )
            # potassium (mEq/L)
            lab_vector.append(
                random.uniform(3.0, 6.5) if random.random() < 0.9 else None
            )
            # sodium (mEq/L)
            lab_vector.append(
                random.uniform(130.0, 150.0) if random.random() < 0.9 else None
            )
            # chloride (mEq/L)
            lab_vector.append(
                random.uniform(95.0, 110.0) if random.random() < 0.85 else None
            )

            lab_values_list.append(lab_vector)
            lab_times_list.append(random.uniform(0.0, 48.0))

        # Create sample in the expected format (before processing)
        synthetic_sample = {
            "patient_id": f"HOLDOUT_PATIENT_{i}",
            "record_id": f"HOLDOUT_PATIENT_{i}",
            "diagnoses": (diagnoses_times_list, diagnoses_list),
            "labs": (lab_times_list, lab_values_list),
            "label": random.randint(0, 1),
        }

        synthetic_samples.append(synthetic_sample)

    # Create a new SampleDataset with the FITTED processors
    holdout_dataset = SampleDataset(
        samples=synthetic_samples,
        input_schema=sample_dataset.input_schema,
        output_schema=sample_dataset.output_schema,
        dataset_name=f"{sample_dataset.dataset_name}_holdout",
        task_name=sample_dataset.task_name,
        input_processors=sample_dataset.input_processors,
        output_processors=sample_dataset.output_processors,
    )

    print(f"Generated {len(holdout_dataset)} synthetic samples")
    sample_seq_lens = [len(s["diagnoses"][1]) for s in synthetic_samples[:3]]
    print(f"Sample diagnosis sequence lengths: {sample_seq_lens}")
    sample_codes_per_visit = [
        [len(visit) for visit in s["diagnoses"][1]] for s in synthetic_samples[:3]
    ]
    print(f"Sample codes per visit: {sample_codes_per_visit}")

    return holdout_dataset


def main():
    """Main function to run DKA prediction pipeline."""
    
    # Configuration
    MIMIC4_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.2/"
    PROCESSOR_DIR = "../../output/processors/stagenet_dka_mimic4"
    CACHE_DIR = "../../mimic4_dka_stagenet_cache"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("DKA PREDICTION WITH STAGENET ON MIMIC-IV")
    print("=" * 60)
    
    # STEP 1: Load MIMIC-IV base dataset
    print("\n=== Step 1: Loading MIMIC-IV Dataset ===")
    base_dataset = MIMIC4Dataset(
        ehr_root=MIMIC4_ROOT,
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
        # dev=True,  # Uncomment for faster development iteration
    )
    
    print(f"Dataset loaded with {len(base_dataset.patients)} patients")
    
    # STEP 2: Apply DKA prediction task
    print("\n=== Step 2: Applying DKA Prediction Task ===")
    
    # Create task with 90-day DKA window and padding for unseen sequences
    dka_task = DKAPredictionMIMIC4(dka_window_days=90, padding=20)
    
    print(f"Task: {dka_task.task_name}")
    print(f"DKA window: {dka_task.dka_window_days} days")
    print(f"Input schema: {list(dka_task.input_schema.keys())}")
    print(f"Output schema: {list(dka_task.output_schema.keys())}")
    
    # Check for pre-fitted processors
    if os.path.exists(os.path.join(PROCESSOR_DIR, "input_processors.pkl")):
        print("\nLoading pre-fitted processors...")
        input_processors, output_processors = load_processors(PROCESSOR_DIR)
        
        sample_dataset = base_dataset.set_task(
            dka_task,
            num_workers=4,
            cache_dir=CACHE_DIR,
            input_processors=input_processors,
            output_processors=output_processors,
        )
    else:
        print("\nFitting new processors...")
        sample_dataset = base_dataset.set_task(
            dka_task,
            num_workers=4,
            cache_dir=CACHE_DIR,
        )
        
        # Save processors for future runs
        print("Saving processors...")
        os.makedirs(PROCESSOR_DIR, exist_ok=True)
        save_processors(sample_dataset, PROCESSOR_DIR)
    
    print(f"\nTotal samples: {len(sample_dataset)}")
    
    # Count label distribution
    label_counts = {0: 0, 1: 0}
    for sample in sample_dataset.samples:
        label_counts[sample["label"]] += 1
    
    print(f"Label distribution:")
    print(f"  No DKA (0): {label_counts[0]} ({100*label_counts[0]/len(sample_dataset):.1f}%)")
    print(f"  Has DKA (1): {label_counts[1]} ({100*label_counts[1]/len(sample_dataset):.1f}%)")
    
    # Inspect a sample
    sample = sample_dataset.samples[0]
    print("\nSample structure:")
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"  Diagnoses: {len(sample['diagnoses'][1])} admission(s)")
    print(f"  Labs: {len(sample['labs'][0])} timestep(s)")
    print(f"  Label: {sample['label']}")
    
    # STEP 3: Split dataset
    print("\n=== Step 3: Splitting Dataset ===")
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=256, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=256, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=256, shuffle=False)
    
    # STEP 4: Initialize StageNet model
    print("\n=== Step 4: Initializing StageNet Model ===")
    model = StageNet(
        dataset=sample_dataset,
        embedding_dim=128,
        chunk_size=128,
        levels=3,
        dropout=0.3,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # STEP 5: Train the model
    print("\n=== Step 5: Training Model ===")
    trainer = Trainer(
        model=model,
        device=DEVICE,
        metrics=["pr_auc", "roc_auc", "accuracy", "f1"],
    )
    
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=20,
        monitor="roc_auc",
        optimizer_params={"lr": 1e-4},
    )
    
    # STEP 6: Evaluate on test set
    print("\n=== Step 6: Evaluation ===")
    results = trainer.evaluate(test_loader)
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    # STEP 7: Inspect model predictions
    print("\n=== Step 7: Sample Predictions ===")
    sample_batch = next(iter(test_loader))
    with torch.no_grad():
        output = model(**sample_batch)
    
    print(f"Predicted probabilities: {output['y_prob'][:5]}")
    print(f"True labels: {output['y_true'][:5]}")
    
    # STEP 8: Test with synthetic hold-out set
    print("\n" + "=" * 60)
    print("TESTING PROCESSOR ROBUSTNESS WITH SYNTHETIC HOLD-OUT SET")
    print("=" * 60)
    
    holdout_dataset = generate_holdout_set(
        sample_dataset=sample_dataset, num_samples=50, seed=42
    )
    
    holdout_loader = get_dataloader(holdout_dataset, batch_size=16, shuffle=False)
    
    # Inspect processed samples
    print("\n=== Inspecting Processed Hold-out Samples ===")
    holdout_batch = next(iter(holdout_loader))
    
    print(f"Batch size: {len(holdout_batch['patient_id'])}")
    print(f"Diagnoses tensor shape: {holdout_batch['diagnoses'][1].shape}")
    
    # Check for unknown tokens
    diagnoses_processor = sample_dataset.input_processors["diagnoses"]
    unk_token_idx = diagnoses_processor.code_vocab["<unk>"]
    pad_token_idx = diagnoses_processor.code_vocab["<pad>"]
    
    print(f"\n<unk> token index: {unk_token_idx}")
    print(f"<pad> token index: {pad_token_idx}")
    
    # Count unknown and padding tokens
    diag_values = holdout_batch["diagnoses"][1]
    num_unk = (diag_values == unk_token_idx).sum().item()
    num_pad = (diag_values == pad_token_idx).sum().item()
    total_tokens = diag_values.numel()
    
    print("\nToken statistics in hold-out batch:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Unknown tokens: {num_unk} ({100*num_unk/total_tokens:.1f}%)")
    print(f"  Padding tokens: {num_pad} ({100*num_pad/total_tokens:.1f}%)")
    
    # Run model inference on hold-out set
    print("\n=== Model Inference on Hold-out Set ===")
    with torch.no_grad():
        holdout_output = model(**holdout_batch)
    
    print(f"Predictions shape: {holdout_output['y_prob'].shape}")
    print(f"Sample predictions: {holdout_output['y_prob'][:5]}")
    
    print("\n" + "=" * 60)
    print("HOLD-OUT SET TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    # STEP 9: Processor information
    print("\n=== Processor Information ===")
    print(f"Processors saved at: {PROCESSOR_DIR}")
    print(f"\nDiagnoses Processor:")
    print(f"  Vocabulary size: {diagnoses_processor.size()}")
    print(f"  Max nested length: {diagnoses_processor._max_nested_len}")
    
    labs_processor = sample_dataset.input_processors["labs"]
    print(f"\nLabs Processor:")
    print(f"  Feature dimension: {labs_processor.size}")
    
    return results


if __name__ == "__main__":
    main()

