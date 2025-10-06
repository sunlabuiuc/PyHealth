"""
Example of using StageNet for mortality prediction on MIMIC-IV.

This example demonstrates:
1. Loading MIMIC-IV data
2. Applying the MortalityPredictionStageNetMIMIC4 task
3. Creating a SampleDataset with StageNet processors
4. Training a StageNet model
"""

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import StageNet
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.trainer import Trainer
import torch

# STEP 1: Load MIMIC-IV base dataset
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

# STEP 2: Apply StageNet mortality prediction task
sample_dataset = base_dataset.set_task(
    MortalityPredictionStageNetMIMIC4(),
    num_workers=4,
    cache_dir="../../mimic4_stagenet_cache",
)

print(f"Total samples: {len(sample_dataset)}")
print(f"Input schema: {sample_dataset.input_schema}")
print(f"Output schema: {sample_dataset.output_schema}")

# Inspect a sample
sample = sample_dataset.samples[0]
print("\nSample structure:")
print(f"  Patient ID: {sample['patient_id']}")
print(f"ICD Codes: {sample['icd_codes']}")
print(f"  Labs shape: {len(sample['labs'].value)} timesteps")
print(f"  Mortality: {sample['mortality']}")

# STEP 3: Split dataset
train_dataset, val_dataset, test_dataset = split_by_patient(
    sample_dataset, [0.8, 0.1, 0.1]
)

# Create dataloaders
train_loader = get_dataloader(train_dataset, batch_size=256, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=256, shuffle=False)
test_loader = get_dataloader(test_dataset, batch_size=256, shuffle=False)

# STEP 4: Initialize StageNet model
model = StageNet(
    dataset=sample_dataset,
    embedding_dim=128,
    chunk_size=128,
    levels=3,
    dropout=0.3,
)

num_params = sum(p.numel() for p in model.parameters())
print(f"\nModel initialized with {num_params} parameters")

# STEP 5: Train the model
trainer = Trainer(
    model=model,
    device="cuda:2",  # or "cpu"
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
results = trainer.evaluate(test_loader)
print("\nTest Results:")
for metric, value in results.items():
    print(f"  {metric}: {value:.4f}")

# STEP 7: Inspect model predictions
sample_batch = next(iter(test_loader))
with torch.no_grad():
    output = model(**sample_batch)

print("\nSample predictions:")
print(f"  Predicted probabilities: {output['y_prob'][:5]}")
print(f"  True labels: {output['y_true'][:5]}")
