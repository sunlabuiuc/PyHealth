"""Sleep staging on Sleep-EDF with SparcNet (ContraWR is a drop-in swap).

Loads the Sleep-EDF Expanded cassette recordings with ``SleepEDFDataset``,
cuts them into 30-second epochs labeled W/N1/N2/N3/N4/REM with the default
``SleepStagingSleepEDF`` task, splits by patient, and trains ``SparcNet``.
Despite the file name, the model is SparcNet; replace it with
``ContraWR(dataset=sleep_staging_ds)`` to train ContraWR instead.

Data: Sleep-EDF Database Expanded v1.0.0, open access, no credentials
(https://physionet.org/content/sleep-edfx/1.0.0/). Set ``SLEEPEDF_ROOT`` to the
folder that holds ``SC-subjects.xls`` and ``sleep-cassette/``.

Run: ``python examples/eeg/sleep_staging/sleep_staging_sleepEDF_contrawr.py``.
Reading all 153 cassette recordings takes a while and a few GB of RAM; point
the root at a copy with only a few subjects for a quick try.
"""

from pyhealth.datasets import SleepEDFDataset, get_dataloader, split_by_patient
from pyhealth.models import SparcNet
from pyhealth.tasks import SleepStagingSleepEDF
from pyhealth.trainer import Trainer

SLEEPEDF_ROOT = "/path/to/sleep-edf-database-expanded-1.0.0"

# set_task spawns worker processes, so the script needs a main guard.
if __name__ == "__main__":
    # step 1: load signal data
    dataset = SleepEDFDataset(root=SLEEPEDF_ROOT, subset="cassette")

    # step 2: set task
    sleep_staging_ds = dataset.set_task(SleepStagingSleepEDF())
    print(f"Total task samples: {len(sleep_staging_ds)}")

    # split dataset
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sleep_staging_ds, [0.6, 0.2, 0.2]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)
    print(
        "loader size: train/val/test",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    # STEP 3: define model
    model = SparcNet(dataset=sleep_staging_ds)

    # STEP 4: define trainer
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=5,
        monitor="accuracy",
    )

    # STEP 5: evaluate
    print(trainer.evaluate(test_dataloader))
