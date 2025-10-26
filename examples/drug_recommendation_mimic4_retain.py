"""
Example of using RETAIN for drug recommendation on MIMIC-IV.

This example demonstrates:
1. Loading MIMIC-IV data
2. Applying the DrugRecommendationMIMIC4 task
3. Creating a SampleDataset with nested sequence processors
4. Training a RETAIN model
"""

import torch

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import RETAIN
from pyhealth.tasks import DrugRecommendationMIMIC4
from pyhealth.trainer import Trainer

# STEP 1: Load MIMIC-IV base dataset
base_dataset = MIMIC4Dataset(
    ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
    ehr_tables=[
        "patients",
        "admissions",
        "diagnoses_icd",
        "procedures_icd",
        "prescriptions",
    ],
)

# STEP 2: Apply drug recommendation task
sample_dataset = base_dataset.set_task(
    DrugRecommendationMIMIC4(),
    num_workers=4,
    cache_dir="../../mimic4_drug_rec_cache",
)

print(f"Total samples: {len(sample_dataset)}")
print(f"Input schema: {sample_dataset.input_schema}")
print(f"Output schema: {sample_dataset.output_schema}")

# Inspect a sample
sample = sample_dataset.samples[0]
print("\nSample structure:")
print(f"  Patient ID: {sample['patient_id']}")
print(f"  Visit ID: {sample['visit_id']}")
print(f"  Conditions (history): {len(sample['conditions'])} visits")
print(f"  Procedures (history): {len(sample['procedures'])} visits")
print(f"  Drugs history: {len(sample['drugs_hist'])} visits")
print(f"  Target drugs: {len(sample['drugs'])} drugs")
print(f"\n  First visit conditions: {sample['conditions'][0][:5]}...")
print(f"  Target drugs sample: {sample['drugs'][:5]}...")

# STEP 3: Split dataset
train_dataset, val_dataset, test_dataset = split_by_patient(
    sample_dataset, [0.8, 0.1, 0.1]
)

print("\nDataset split:")
print(f"  Train: {len(train_dataset)} samples")
print(f"  Validation: {len(val_dataset)} samples")
print(f"  Test: {len(test_dataset)} samples")

# Create dataloaders
train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

# STEP 4: Initialize RETAIN model
model = RETAIN(
    dataset=sample_dataset,
    embedding_dim=128,
    dropout=0.5,
)

num_params = sum(p.numel() for p in model.parameters())
print(f"\nModel initialized with {num_params:,} parameters")
print(f"Feature keys: {model.feature_keys}")
print(f"Label key: {model.label_key}")

# STEP 5: Train the model
trainer = Trainer(
    model=model,
    device="cuda:4",  # or "cpu"
    metrics=["pr_auc_samples", "f1_samples", "jaccard_samples"],
)

print("\nStarting training...")
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=50,
    monitor="pr_auc_samples",
    optimizer_params={"lr": 1e-3},
)

# STEP 6: Evaluate on test set
print("\nEvaluating on test set...")
results = trainer.evaluate(test_loader)
print("\nTest Results:")
for metric, value in results.items():
    print(f"  {metric}: {value:.4f}")

# STEP 7: Inspect model predictions
print("\nSample predictions:")
sample_batch = next(iter(test_loader))

with torch.no_grad():
    output = model(**sample_batch)

print(f"  Batch size: {output['y_prob'].shape[0]}")
print(f"  Number of drug classes: {output['y_prob'].shape[1]}")
print("  Predicted probabilities (first 5 drugs of first patient):")
print(f"    {output['y_prob'][0, :5].cpu().numpy()}")
print("  True labels (first 5 drugs of first patient):")
print(f"    {output['y_true'][0, :5].cpu().numpy()}")
