"""Mortality prediction on synthetic MIMIC-III with TCN.

Trains a temporal convolutional network over the visit sequence.
Task: MortalityPredictionMIMIC3, which predicts whether the patient dies in the
next hospital visit from the diagnosis, procedure and drug codes of the current
visit. The script trains for one epoch with dev=True, so it only shows the
pipeline end to end. The small dev test split may contain no deaths, in which
case ROC-AUC prints nan. For real results, set dev=False and train for more
epochs.

Data: the public synthetic MIMIC-III
(https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III). It is
downloaded on the first run and needs no credentials. To use real MIMIC-III
v1.4 (credentialed access, https://physionet.org/content/mimiciii/1.4/), set
``root`` to your local copy.

Usage:
    python examples/mortality_prediction/mortality_mimic3_tcn.py

Runs on a CPU in a few minutes, most of which is the download.
"""

import tempfile

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import TCN
from pyhealth.tasks import MortalityPredictionMIMIC3
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
    task = MortalityPredictionMIMIC3()
    sample_dataset = base_dataset.set_task(task)

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=256, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=256, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=256, shuffle=False)

    # STEP 3: define model
    model = TCN(
        dataset=sample_dataset,
        embedding_dim=32,
    )

    # STEP 4: define trainer
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=1,
        monitor="roc_auc",
    )

    # STEP 5: evaluate
    print(trainer.evaluate(test_dataloader))
