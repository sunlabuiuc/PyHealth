"""Readmission prediction on MIMIC-IV with an RNN.

Loads diagnoses, procedures and prescriptions, applies
``ReadmissionPredictionMIMIC4`` (binary readmission label per admission),
and trains/evaluates an ``RNN`` for one epoch.

Data: by default ``ehr_root`` points at the open-access MIMIC-IV demo on
PhysioNet (https://physionet.org/content/mimic-iv-demo/2.2/, downloaded on
the fly, no credentials). For full MIMIC-IV (credentialed) set ``ehr_root``
to your local ``mimiciv/2.2`` folder (the one containing ``hosp/``) and set
``dev=False``.

Run: ``python examples/readmission/readmission_mimic4_rnn.py`` (CPU).
Offline: set ``ehr_root`` to ``test-resources/core/mimic4demo`` and
``dev=False``.
"""
import tempfile

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import RNN
from pyhealth.tasks import ReadmissionPredictionMIMIC4
from pyhealth.trainer import Trainer


if __name__ == "__main__":
    # STEP 1: Load dataset
    base_dataset = MIMIC4Dataset(
        ehr_root="https://physionet.org/files/mimic-iv-demo/2.2/",
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir=tempfile.TemporaryDirectory().name,
        dev=True,
    )
    base_dataset.stats()

    # STEP 2: Set task
    task = ReadmissionPredictionMIMIC4()
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
