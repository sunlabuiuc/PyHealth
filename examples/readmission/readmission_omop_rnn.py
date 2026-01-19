import tempfile

from pyhealth.datasets import OMOPDataset, get_dataloader, split_by_patient
from pyhealth.models import RNN
from pyhealth.tasks import ReadmissionPredictionOMOP
from pyhealth.trainer import Trainer

# Since PyHealth uses multiprocessing, it is best practice to use a main guard.
if __name__ == '__main__':
    # Use tempfile to automate cleanup
    cache_dir = tempfile.TemporaryDirectory()

    base_dataset = OMOPDataset(
        root="https://physionet.org/files/mimic-iv-demo-omop/0.9/1_omop_data_csv",
        tables=["person", "visit_occurrence", "condition_occurrence", "procedure_occurrence", "drug_exposure"],
        cache_dir=cache_dir.name
    )
    base_dataset.stats()

    sample_dataset = base_dataset.set_task(ReadmissionPredictionOMOP())

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    model = RNN(
        dataset=sample_dataset,
    )

    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=1,
        monitor="roc_auc",
    )

    trainer.evaluate(test_dataloader)

    sample_dataset.close()
