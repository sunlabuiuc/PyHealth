# import pyhealth
import pyhealth

# import mimic4 dataset and drug recommendaton task
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import drug_recommendation_mimic4_fn

# import dataloader related functions
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets import split_by_patient, get_dataloader

# import gamenet model
from pyhealth.models import GAMENet

# import trainer
from pyhealth.trainer import Trainer

_DEV = False
_EPOCHS = 20
_LR = 1e-3
_DECAY_WEIGHT = 1e-5


def prepare_drug_task_data():
    mimicvi = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        dev=_DEV,
        refresh_cache=False,
    )

    print("stat")
    mimicvi.stat()
    print("info")
    mimicvi.info()

    mimic4_sample = mimicvi.set_task(drug_recommendation_mimic4_fn)
    print(mimic4_sample[0])

    return mimic4_sample


def get_dataloaders(mimic4_sample):
    train_dataset, val_dataset, test_dataset = split_by_patient(
        mimic4_sample, [0.8, 0.1, 0.1]
    )
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


def train_gamenet(mimic4_sample, train_loader, val_loader):
    # gamenet = GAMENet(mimicvi)
    gamenet = GAMENet(mimic4_sample)

    # print(gamenet.generate_ddi_adj())
    trainer = Trainer(
        model=gamenet,
        # metrics = ["jaccard_weighted", "pr_auc_micro", "pr_auc_macro"],
        # metrics = ["jaccard", "pr_auc_micro", "pr_auc_macro"],
        metrics=[
            "jaccard_samples",
            "accuracy",
            "hamming_loss",
            "precision_samples",
            "recall_samples",
            "pr_auc_samples",
            "f1_samples",
        ],
        device="cuda",
        exp_name="drug_recommendation",
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=_EPOCHS,
        # monitor = "jaccard_weighted",
        # monitor = "pr_auc_macro",
        # monitor = "jaccard_samples",
        monitor="accuracy",
        monitor_criterion="max",
        weight_decay=_DECAY_WEIGHT,
        optimizer_params={"lr": _LR},
    )

    return gamenet, trainer


def evaluate_gamenet(trainer, test_loader):
    result = trainer.evaluate(test_loader)
    print(result)
    return result


if __name__ == "__main__":
    data = prepare_drug_task_data()
    train_loader, val_loader, test_loader = get_dataloaders(data)

    model, trainer = train_gamenet(data, train_loader, val_loader)

    result = evaluate_gamenet(trainer, test_loader)
