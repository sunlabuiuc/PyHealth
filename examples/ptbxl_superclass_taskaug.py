"""
Example script demonstrating TaskAug on the PTB-XL dataset.
Paper: Data Augmentation for Electrocardiograms (Raghu et al., 2022)
"""
import torch
from pyhealth.datasets import PTBXLDataset
from pyhealth.models import TaskAug
from pyhealth.trainer import Trainer
from torch.utils.data import DataLoader
from pyhealth.tasks import PTBXLAllDiagnosticClassification

def main():
    dataset = PTBXLDataset(
        root="tests/data/ptbxl_synthetic",
        cache_dir="tests/data/my_clean_cache"
    ) 
    
    samples = dataset.set_task(
        PTBXLAllDiagnosticClassification(scp_statements_path="tests/data/ptbxl_synthetic/scp_statements.csv")
    )

    train_dataset = samples[:len(samples)//2]
    val_dataset = samples[len(samples)//2:]

    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)

    model_taskaug = TaskAug(
        dataset=samples,
        feature_keys=["signal"],
        label_key="labels",
        mode="multilabel",
        num_stages=2
    )

    trainer = Trainer(model=model_taskaug, metrics=["roc_auc_samples", "pr_auc_samples"])
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=1, 
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},
    )
    print("TaskAug example completed successfully!")

if __name__ == '__main__':
    main()