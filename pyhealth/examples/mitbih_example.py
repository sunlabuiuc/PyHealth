import os
import torch
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.trainer import Trainer

from mitbih_dataset import MITBIHArrhythmiaDataset


# 1. Define Task Function
def mitbih_classification_fn(sample):
    """Convert MIT-BIH Sample → PyHealth-compatible sample."""
    return {
        "patient_id": sample.patient_id,
        "visit_id": sample.visit_id,
        "signals": sample.data["signal"],   # (1,187)
        "label": sample.label,
    }


# 2. Create Custom CNN Model
class ECGCNN(BaseModel):
    """Simple CNN for arrhythmia classification."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 46, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5),  # 5 arrhythmia classes
        )

    def forward(self, batch):
        x = batch["signals"]   # shape (B,1,187)
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        logits = self.fc(feat)
        return logits


# 3. Main Example
def main():
    root = "./data"  # user supplies Kaggle CSVs here

    # Load dataset (train + test)
    train_raw = MITBIHArrhythmiaDataset(root, split="train")
    test_raw = MITBIHArrhythmiaDataset(root, split="test")

    # Wrap with PyHealth SampleDataset
    train_dataset = SampleDataset(train_raw, task=mitbih_classification_fn)
    test_dataset = SampleDataset(test_raw, task=mitbih_classification_fn)

    # Build model
    model = ECGCNN(
        dataset=train_dataset,
        feature_keys=["signals"],
        label_key="label",
        mode="multiclass",
    )

    # Trainer
    trainer = Trainer(model=model)

    # Train
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        epochs=5,
        batch_size=64,
        learning_rate=1e-3,
    )

    # Evaluate
    metrics = trainer.evaluate(test_dataset)
    print("Final Test Metrics:", metrics)


if __name__ == "__main__":
    main()
