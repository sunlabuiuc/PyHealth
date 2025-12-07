"""
Name: Hyunsoo Lee
NetId: hyunsoo2
Description: This example demonstrates how to use the SHy model for next-visit diagnosis
prediction on the MIMIC-III dataset. Update the data path as necessary.
"""

import torch
import os
from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import SHy
from pyhealth.tasks import DiagnosisPredictionMIMIC3
from pyhealth.trainer import Trainer

# Import the task
from pyhealth.tasks.diagnosis_prediction import DiagnosisPredictionMIMIC3

if __name__ == "__main__":
    # STEP 1: Load data
    # Data can be downloaded from https://physionet.org/content/mimiciii/1.4/
    root = "data/mimic3_demo"
    
    if not os.path.exists(root):
        print(f"Directory {root} not found.")
        print("Please download MIMIC-III files to this folder or update the path accordingly.")
        os.makedirs(root, exist_ok=True)

    base_dataset = MIMIC3Dataset(
        root=root,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=False,
    )
    base_dataset.stats()

    # STEP 2: Set task
    # DiagnosisPredictionMIMIC3 predicts the next visit's diagnoses
    # based on the current and past visit sequences
    sample_dataset = base_dataset.set_task(DiagnosisPredictionMIMIC3())

    # STEP 3: Split dataset
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=8, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=8, shuffle=False)

    # STEP 4: Initialize model
    # SHy learns temporal phenotypes via hypergraph neural networks and
    # hypergraph structure learning (HSL)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SHy(
        dataset=sample_dataset,
        feature_keys=["conditions"],
        label_key="label",
        mode="multilabel",
        embedding_dim=32,
        hgnn_dim=64,
        num_temporal_phenotypes=3,
        dropout=0.5,
        hgnn_layers=2,
    )

    # STEP 5: Train model
    trainer = Trainer(
        model=model,
        metrics=["pr_auc_samples", "f1_samples"],
        device=device,
    )

    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=2,
        monitor="pr_auc_samples",
    )

    # STEP 6: Evaluate on test set
    print("Evaluating on test set...")
    results = trainer.evaluate(test_dataloader)
    print(results)