"""
Ablation Study: Readmission Prediction on eICU with RNN

This example demonstrates how to use the modernized eICUDataset with the
ReadmissionPredictionEICU task class for predicting ICU readmission.

It conducts an ablation study comparing:
1. Univariate: Conditions only
2. Multi-modal: Conditions + Procedures + Drugs

Features:
- Uses the new BaseDataset-based eICUDataset with YAML configuration
- Uses the new ReadmissionPredictionEICU BaseTask class
- Demonstrates the standardized PyHealth workflow
"""

import tempfile
from pyhealth.datasets import eICUDataset, split_by_patient, get_dataloader
from pyhealth.models import RNN
from pyhealth.tasks import ReadmissionPredictionEICU
from pyhealth.trainer import Trainer

def run_ablation(train_ds, val_ds, test_ds, feature_keys, name):
    print(f"\n>>> Running Ablation: {name} (Features: {feature_keys})")
    
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

    model = RNN(dataset=train_ds, feature_keys=feature_keys)
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=3,
        monitor="roc_auc",
    )
    return trainer.evaluate(test_loader)

if __name__ == "__main__":
    # STEP 1: Load dataset
    base_dataset = eICUDataset(
        root="https://storage.googleapis.com/pyhealth/eicu-demo/",
        tables=["diagnosis", "medication", "physicalexam"],
        cache_dir=tempfile.TemporaryDirectory().name,
        dev=True,
    )

    # STEP 2: Set task
    task = ReadmissionPredictionEICU()
    sample_dataset = base_dataset.set_task(task)

    # STEP 3: Split
    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])

    # STEP 4: Run Ablations
    res_1 = run_ablation(train_ds, val_ds, test_ds, ["conditions"], "Conditions-Only")
    res_2 = run_ablation(train_ds, val_ds, test_ds, ["conditions", "procedures", "drugs"], "Full Multi-modal")

    # STEP 5: Compare
    print(f"Univariate ROC-AUC: {res_1['roc_auc']:.4f}")
    print(f"Multi-modal ROC-AUC: {res_2['roc_auc']:.4f}")