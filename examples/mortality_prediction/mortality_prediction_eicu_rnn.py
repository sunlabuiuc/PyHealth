"""
Mortality Prediction on eICU with RNN

This example demonstrates how to use the modernized eICUDataset with the 
MortalityPredictionEICU task class for in-hospital mortality prediction
using an RNN model.

Features:
- Uses the new BaseDataset-based eICUDataset with YAML configuration
- Uses the MortalityPredictionEICU BaseTask class
- Demonstrates the standardized PyHealth workflow
"""

import tempfile

from pyhealth.datasets import eICUDataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import RNN
from pyhealth.tasks import MortalityPredictionEICU
from pyhealth.trainer import Trainer


if __name__ == "__main__":
    # Use tempfile to automate cleanup
    cache_dir = tempfile.TemporaryDirectory()

    # STEP 1: Load dataset
    # Replace with your eICU dataset path
    base_dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "physicalexam"],
        cache_dir=cache_dir.name
    )
    base_dataset.stats()

    # STEP 2: Set task using MortalityPredictionEICU
    # By default, patients under 18 are excluded
    task = MortalityPredictionEICU()
    sample_dataset = base_dataset.set_task(task)
    sample_dataset.stats()

    # STEP 3: Split and create dataloaders
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # STEP 4: Define model
    model = RNN(
        dataset=sample_dataset,
        feature_keys=["conditions", "procedures", "drugs"],
        label_key="mortality",
        mode="binary",
    )

    # STEP 5: Train
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=20,
        monitor="roc_auc",
    )

    # STEP 6: Evaluate
    print(trainer.evaluate(test_dataloader))

    # Cleanup
    sample_dataset.close()





