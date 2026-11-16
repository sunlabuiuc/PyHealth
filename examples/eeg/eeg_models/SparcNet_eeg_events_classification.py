"""SparcNet for EEG event classification on TUEV.

Loads the TUH EEG Events Corpus (TUEV) with ``TUEVDataset``, extracts one
5-second, 16-channel bipolar window per annotated event with ``EEGEventsTUEV``
(6 classes: SPSW, GPED, PLED, EYEM, ARTF, BCKG), and trains ``SparcNet``.

Data: TUEV v2.0.x from the Temple University EEG Corpus
(https://isip.piconepress.com/projects/nedc/html/tuh_eeg/). Access is free but
requires registration. Set ``TUEV_ROOT`` below to the ``edf/`` folder that
holds ``train/`` and ``eval/``.

Run: ``python examples/eeg/eeg_models/SparcNet_eeg_events_classification.py``.
``dev=True`` loads a small subset of patients. Reading and filtering the EDF
files is the slow step; a GPU is used if available.
"""

from pyhealth.datasets import TUEVDataset, get_dataloader, split_by_visit
from pyhealth.models import SparcNet
from pyhealth.tasks import EEGEventsTUEV
from pyhealth.trainer import Trainer

TUEV_ROOT = "/path/to/tuh_eeg_events/v2.0.0/edf/"

# set_task spawns worker processes, so the script needs a main guard.
if __name__ == "__main__":
    # step 1: load signal data
    dataset = TUEVDataset(root=TUEV_ROOT, dev=True)
    dataset.stats()
    # step 2: set task
    TUEV_ds = dataset.set_task(EEGEventsTUEV(
        resample_rate=200,    # Resample rate
        bandpass_filter=(0.1, 75.0),    # Bandpass filter
        notch_filter=50.0,    # Notch filter
        compute_stft=False,  # SparcNet takes one input: the raw signal
    ))

    print(f"Total task samples: {len(TUEV_ds)}")
    print(f"Input schema: {TUEV_ds.input_schema}")
    print(f"Output schema: {TUEV_ds.output_schema}")

    # Inspect a sample
    sample = TUEV_ds[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Signal shape: {sample['signal'].shape}")
    print(f"Label: {sample['label']}")


    # split dataset
    train_dataset, val_dataset, test_dataset = split_by_visit(
        TUEV_ds, [0.6, 0.2, 0.2]
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
    model = SparcNet(dataset=TUEV_ds)

    # STEP 4: define trainer
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=10,
        optimizer_params={"lr": 1e-3},
    )

    # STEP 5: evaluate
    print(trainer.evaluate(test_dataloader))
