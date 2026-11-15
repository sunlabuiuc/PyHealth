"""Sleep staging on SHHS with SparcNet (ContraWR is a drop-in swap).

Loads the Sleep Heart Health Study polysomnography release with
``SHHSDataset``, cuts it into 30-second epochs labeled W/N1/N2/N3/N4/REM with
``SleepStagingSHHS``, splits by patient and trains ``SparcNet``. Despite the
file name, the model is SparcNet; replace it with
``ContraWR(dataset=sample_dataset)`` to train ContraWR instead.

Data: SHHS visits 1 and 2, restricted. An NSRR account and an approved data
access request are required, and the data are not redistributable
(https://sleepdata.org/datasets/shhs). After approval, download with
``gem install nsrr`` then ``nsrr download shhs/polysomnography``, and set
``SHHS_ROOT`` to the resulting ``polysomnography`` folder, which holds
``edfs/`` and ``annotations-events-profusion/``.

Run: ``python examples/eeg/sleep_staging/sleep_staging_shhs_contrawr.py``.
The full release is hundreds of GB, so the script passes ``dev=True`` and
defaults to visit 1 only; drop ``dev`` for a real run.
"""

from pathlib import Path

from pyhealth.datasets import SHHSDataset, get_dataloader, split_by_patient
from pyhealth.models import SparcNet
from pyhealth.tasks import SleepStagingSHHS
from pyhealth.trainer import Trainer

#: The ``polysomnography`` folder of the NSRR SHHS release.
SHHS_ROOT = "/path/to/shhs/polysomnography"
#: Where to cache the tokenized samples and the fitted processors.
CACHE_DIR = Path.home() / ".cache" / "pyhealth" / "shhs_demo"


def main() -> None:
    # STEP 1: build the metadata tables and load them.
    # ``dev=True`` limits the event table to the first 1000 participants.
    base_dataset = SHHSDataset(
        root=SHHS_ROOT,
        visits=("shhs1",),
        cache_dir=CACHE_DIR,
        dev=True,
    )
    base_dataset.stats()

    # STEP 2: apply the sleep-staging task.
    # 30 s epochs, the length the reading centre scored. Epochs labelled as
    # movement are dropped, so the label space is 0-5.
    task = SleepStagingSHHS(epoch_sec=30)
    print(f"Task: {task.task_name}")
    print(f"Input schema: {list(task.input_schema)}")
    print(f"Output schema: {list(task.output_schema)}")

    sample_dataset = base_dataset.set_task(task, num_workers=4)
    print(f"Total epoch samples: {len(sample_dataset)}")

    sample = sample_dataset[0]
    print(f"  patient_id: {sample['patient_id']}")
    print(f"  record_id: {sample['record_id']}")
    print(f"  signal: {tuple(sample['signal'].shape)} (channels x samples)")
    print(f"  label: {int(sample['label'].item())}")

    # STEP 3: split by participant, so both nights of a person never straddle
    # the train/test boundary.
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    print(
        "samples train/val/test",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
    train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # STEP 4: define the model.
    model = SparcNet(dataset=sample_dataset)

    # STEP 5: train.
    trainer = Trainer(model=model, metrics=["accuracy", "f1_weighted"])
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=5,
        monitor="accuracy",
    )

    # STEP 6: evaluate.
    print(trainer.evaluate(test_loader))


if __name__ == "__main__":
    main()
