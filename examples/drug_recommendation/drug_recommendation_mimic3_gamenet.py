import tempfile

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import GAMENet
from pyhealth.tasks import DrugRecommendationMIMIC3
from pyhealth.trainer import Trainer

if __name__ == "__main__":
    # STEP 1: load data
    base_dataset = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        cache_dir=tempfile.TemporaryDirectory().name,
        dev=True,
    )
    base_dataset.stats()

    # STEP 2: set task
    task = DrugRecommendationMIMIC3()
    sample_dataset = base_dataset.set_task(task)

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # STEP 3: define model
    model = GAMENet(
        sample_dataset,
    )

    # STEP 4: define trainer
    trainer = Trainer(
        model=model,
        metrics=["jaccard_samples", "f1_samples", "pr_auc_samples", "ddi"],
    )

    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=1,
        monitor="pr_auc_samples",
    )

    # STEP 5: evaluate
    print(trainer.evaluate(test_dataloader))
