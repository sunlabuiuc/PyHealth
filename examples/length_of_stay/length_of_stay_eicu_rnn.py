"""
Length of Stay Prediction on eICU with RNN

This example demonstrates how to use the modernized eICUDataset with the
LengthOfStayPredictioneICU task class for predicting ICU length of stay
using an RNN model.

Length of stay is categorized into 10 classes:
- 0: < 1 day
- 1: 1 day
- 2: 2 days
- 3: 3 days
- 4: 4 days
- 5: 5 days
- 6: 6 days
- 7: 7 days
- 8: 1-2 weeks
- 9: > 2 weeks

Features:
- Uses the new BaseDataset-based eICUDataset with YAML configuration
- Uses the LengthOfStayPredictioneICU BaseTask class
- Demonstrates the standardized PyHealth workflow
"""

import tempfile

from pyhealth.datasets import eICUDataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import RNN
from pyhealth.tasks import LengthOfStayPredictioneICU
from pyhealth.trainer import Trainer


if __name__ == "__main__":
    # STEP 1: Load dataset
    # Replace with your eICU dataset path
    base_dataset = eICUDataset(
        root="https://storage.googleapis.com/pyhealth/eicu-demo/",
        tables=["diagnosis", "medication", "physicalexam"],
        num_workers=4,
        cache_dir=tempfile.TemporaryDirectory().name,
    )
    base_dataset.stats()

    # STEP 2: Set task using LengthOfStayPredictioneICU
    task = LengthOfStayPredictioneICU()
    sample_dataset = base_dataset.set_task(task, num_workers=16)

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
    )

    # STEP 5: Train
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=1,
        monitor="accuracy",
    )

    # STEP 6: Evaluate
    print(trainer.evaluate(test_dataloader))
