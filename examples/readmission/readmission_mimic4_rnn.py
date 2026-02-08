import tempfile

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import RNN
from pyhealth.tasks import ReadmissionPredictionMIMIC4
from pyhealth.trainer import Trainer

# Since PyHealth uses multiprocessing, it is best practice to use a main guard.
if __name__ == "__main__":
    # Use tempfile to automate cleanup
    cache_dir = tempfile.TemporaryDirectory()

    base_dataset = MIMIC4Dataset(
        ehr_root="https://physionet.org/files/mimic-iv-demo/2.2/",
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir=cache_dir.name,
    )
    base_dataset.stats()

    sample_dataset = base_dataset.set_task(ReadmissionPredictionMIMIC4())

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
