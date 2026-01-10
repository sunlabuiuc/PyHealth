from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import Agent
from pyhealth.tasks import mortality_prediction_mimic3_fn
from pyhealth.trainer import Trainer

if __name__ == "__main__":
    # STEP 1: load data
    base_dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"ICD9CM": "CCSCM", "ICD9PROC": "CCSPROC", "NDC": "ATC"},
        dev=False,
        refresh_cache=False,
    )
    base_dataset.stat()

    # STEP 2: set task
    sample_dataset = base_dataset.set_task(mortality_prediction_mimic3_fn)
    sample_dataset.stat()

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=256, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=256, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=256, shuffle=False)

    # STEP 3: define model
    model = Agent(
        dataset=sample_dataset,
        feature_keys=["conditions", "procedures"],
        label_key="label",
        mode="binary",
        embedding_dim=32,
        hidden_dim=32,
    )

    # STEP 4: define trainer
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=50,
        monitor="roc_auc",
    )

    # STEP 5: evaluate
    print(trainer.evaluate(test_dataloader))
