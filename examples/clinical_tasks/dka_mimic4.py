"""
Example of using StageNet for DKA (Diabetic Ketoacidosis) prediction on MIMIC-IV.

This example demonstrates:
1. Loading MIMIC-IV data with relevant tables for DKA prediction
2. Applying the DKAPredictionMIMIC4 task (general population)
3. Creating a SampleDataset with StageNet processors
4. Training a StageNet model for DKA prediction

Target Population:
    - ALL patients in MIMIC-IV (no diabetes filtering)
    - Much larger patient pool with more negative samples
    - Label: 1 if patient has ANY DKA diagnosis, 0 otherwise

Note: For T1DM-specific DKA prediction, see t1dka_mimic4.py
"""

import os
import torch

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.datasets.utils import save_processors, load_processors
from pyhealth.models import StageNet
from pyhealth.tasks import DKAPredictionMIMIC4
from pyhealth.trainer import Trainer


def main():
    """Main function to run DKA prediction pipeline on general population."""
    
    # Configuration
    MIMIC4_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.2/"
    DATASET_CACHE_DIR = "/shared/rsaas/pyhealth/cache/mimic4_dataset"
    TASK_CACHE_DIR = "/shared/rsaas/pyhealth/cache/mimic4_dka_general_stagenet"
    PROCESSOR_DIR = "/shared/rsaas/pyhealth/processors/stagenet_dka_general_mimic4"
    DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("DKA PREDICTION (GENERAL POPULATION) WITH STAGENET ON MIMIC-IV")
    print("=" * 60)
    
    # STEP 1: Load MIMIC-IV base dataset
    print("\n=== Step 1: Loading MIMIC-IV Dataset ===")
    base_dataset = MIMIC4Dataset(
        ehr_root=MIMIC4_ROOT,
        ehr_tables=[
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
        cache_dir=DATASET_CACHE_DIR,
        # dev=True,  # Uncomment for faster development iteration
    )
    
    print("Dataset initialized, proceeding to task processing...")
    
    # STEP 2: Apply DKA prediction task (general population)
    print("\n=== Step 2: Applying DKA Prediction Task (General Population) ===")
    
    # Create task with padding for unseen sequences
    # No T1DM filtering - includes ALL patients
    dka_task = DKAPredictionMIMIC4(padding=10)
    
    print(f"Task: {dka_task.task_name}")
    print(f"Input schema: {list(dka_task.input_schema.keys())}")
    print(f"Output schema: {list(dka_task.output_schema.keys())}")
    print("Note: This includes ALL patients (not just diabetics)")
    
    # Check for pre-fitted processors
    if os.path.exists(os.path.join(PROCESSOR_DIR, "input_processors.pkl")):
        print("\nLoading pre-fitted processors...")
        input_processors, output_processors = load_processors(PROCESSOR_DIR)
        
        sample_dataset = base_dataset.set_task(
            dka_task,
            num_workers=4,
            cache_dir=TASK_CACHE_DIR,
            input_processors=input_processors,
            output_processors=output_processors,
        )
    else:
        print("\nFitting new processors...")
        sample_dataset = base_dataset.set_task(
            dka_task,
            num_workers=4,
            cache_dir=TASK_CACHE_DIR,
        )
        
        # Save processors for future runs
        print("Saving processors...")
        os.makedirs(PROCESSOR_DIR, exist_ok=True)
        save_processors(sample_dataset, PROCESSOR_DIR)
    
    print(f"\nTotal samples: {len(sample_dataset)}")
    
    # Count label distribution
    label_counts = {0: 0, 1: 0}
    for sample in sample_dataset:
        label_counts[int(sample["label"].item())] += 1
    
    print(f"Label distribution:")
    print(f"  No DKA (0): {label_counts[0]} ({100*label_counts[0]/len(sample_dataset):.1f}%)")
    print(f"  Has DKA (1): {label_counts[1]} ({100*label_counts[1]/len(sample_dataset):.1f}%)")
    
    # Inspect a sample
    sample = sample_dataset[0]
    print("\nSample structure:")
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"  ICD codes (diagnoses + procedures): {sample['icd_codes'][1].shape} (visits x codes)")
    print(f"  Labs: {sample['labs'][0].shape} (timesteps x features)")
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
        epochs=50,
        monitor="roc_auc",
        optimizer_params={"lr": 1e-5},
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
    
    print("\n" + "=" * 60)
    print("DKA PREDICTION (GENERAL POPULATION) TRAINING COMPLETED!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
