import tempfile

from pyhealth.datasets import OMOPDataset, get_dataloader, split_by_patient
from pyhealth.models import RNN
from pyhealth.tasks import ReadmissionPredictionOMOP
from pyhealth.trainer import Trainer


if __name__ == "__main__":
    # STEP 1: Load dataset
    base_dataset = OMOPDataset(
        root="https://physionet.org/files/mimic-iv-demo-omop/0.9/1_omop_data_csv",
        tables=[
            "person",
            "visit_occurrence",
            "condition_occurrence",
            "procedure_occurrence",
            "drug_exposure",
        ],
        cache_dir=tempfile.TemporaryDirectory().name,
        dev=True,
    )
    base_dataset.stats()

    # STEP 2: Set task
    task = ReadmissionPredictionOMOP()
    sample_dataset = base_dataset.set_task(task)

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
        monitor="roc_auc",
    )

    # STEP 6: Evaluate
    trainer.evaluate(test_dataloader)
