import torch
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import MortalityPredictionMIMIC3
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import MedFuse
from pyhealth.trainer import Trainer

def run_ablation(hidden_dim):
    print(f"\n{'='*20}")
    print(f"RUNNING ABLATION: hidden_dim={hidden_dim}")
    print(f"{'='*20}")

    # 1. Load Dataset
    dataset = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=False, # Use the full synthetic dataset to ensure we get mortality cases
    )

    # 2. Set Task
    task = MortalityPredictionMIMIC3()
    samples = dataset.set_task(task)

    # 3. Split by Patient (Standard Tutorial Way)
    train_dataset, val_dataset, test_dataset = split_by_patient(
        samples, ratios=[0.7, 0.15, 0.15] # Giving more data to Val/Test
    )

    # 4. Create Dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # 5. Model
    model = MedFuse(dataset=samples, hidden_dim=hidden_dim)

    # 6. Trainer
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=3,
        monitor="roc_auc",
    )

    # 7. Evaluate
    result = trainer.evaluate(test_loader)
    print(f"Final AUROC for hidden_dim {hidden_dim}: {result['roc_auc']:.4f}")
    return result['roc_auc']

if __name__ == "__main__":
    auc_small = run_ablation(64)
    auc_large = run_ablation(256)
    
    print("\n--- ABLATION SUMMARY ---")
    print(f"Hidden Dim 64:  AUROC = {auc_small:.4f}")
    print(f"Hidden Dim 256: AUROC = {auc_large:.4f}")