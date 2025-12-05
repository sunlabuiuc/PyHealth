"""
Example: Lab Test Outcome Prediction with LabTOP

This example shows how to use LabTOP model for predicting
lab test values on MIMIC-IV dataset.

Requirements:
- MIMIC-IV dataset access
- PyHealth installed
"""

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.models import LabTOP
from pyhealth.trainer import Trainer

# Step 1: Load dataset
dataset = MIMIC4Dataset(
    root="/path/to/mimic4",
    tables=["labevents", "patients", "admissions"],
)

# Step 2: Create train/val/test splits
train_dataset, val_dataset, test_dataset = dataset.split(ratios=[0.7, 0.15, 0.15])

# Step 3: Initialize model
model = LabTOP(
    dataset=dataset,
    feature_keys=["demographics", "lab_history"],
    label_key="lab_value",
    mode="regression",
    embedding_dim=768,
)

# Step 4: Train
trainer = Trainer(model=model, device="cuda")
trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=10,
    batch_size=32,
)

# Step 5: Evaluate
metrics = trainer.evaluate(test_dataset)
print(f"Test MAE: {metrics['mae']:.4f}")