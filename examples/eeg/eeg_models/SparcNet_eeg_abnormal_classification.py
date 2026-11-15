"""SparcNet for abnormal EEG detection on TUAB.

Loads the TUH Abnormal EEG Corpus (TUAB) with ``TUABDataset``, cuts each
recording into 10-second, 16-channel bipolar windows with ``EEGAbnormalTUAB``
(normal vs abnormal; the task uses a 2-class "multiclass" label so the
conformal methods in ``pyhealth.calib`` apply), and trains ``SparcNet``.

Data: TUAB v3.0.0 from the Temple University EEG Corpus
(https://isip.piconepress.com/projects/nedc/html/tuh_eeg/). Access is free but
requires registration. Set ``TUAB_ROOT`` below to the ``edf/`` folder that
holds ``train/`` and ``eval/``.

Run: ``python examples/eeg/eeg_models/SparcNet_eeg_abnormal_classification.py``.
``dev=True`` loads a small subset of patients. Reading and filtering the EDF
files is the slow step (hours for the full corpus); a GPU is used if available.
"""

from pyhealth.datasets import TUABDataset, get_dataloader, split_by_visit
from pyhealth.models import SparcNet
from pyhealth.tasks import EEGAbnormalTUAB
from pyhealth.trainer import Trainer

TUAB_ROOT = "/path/to/tuh_eeg_abnormal/v3.0.0/edf/"

# set_task spawns worker processes, so the script needs a main guard.
if __name__ == "__main__":
    # step 1: load signal data
    dataset = TUABDataset(root=TUAB_ROOT, dev=True)
    dataset.stats()
    # step 2: set task
    TUAB_ds = dataset.set_task(EEGAbnormalTUAB(
        resample_rate=200,
        bandpass_filter=(0.1, 75.0),
        notch_filter=50.0,
        compute_stft=False,  # SparcNet takes one input: the raw signal
    ))
    print(f"Total task samples: {len(TUAB_ds)}")
    print(f"Input schema: {TUAB_ds.input_schema}")
    print(f"Output schema: {TUAB_ds.output_schema}")

    # Inspect a sample
    sample = TUAB_ds[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Signal shape: {sample['signal'].shape}")
    print(f"Label: {sample['label']}")

    # split dataset
    train_dataset, val_dataset, test_dataset = split_by_visit(
        TUAB_ds, [0.6, 0.2, 0.2]
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
    model = SparcNet(dataset=TUAB_ds)

    # STEP 4: define trainer
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=10,
        monitor="accuracy",
        optimizer_params={"lr": 1e-3},
    )

    # STEP 5: evaluate
    print(trainer.evaluate(test_dataloader))
