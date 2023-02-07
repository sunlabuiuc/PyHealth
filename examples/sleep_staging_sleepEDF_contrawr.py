from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.trainer import Trainer
from pyhealth.datasets import SleepEDFCassetteDataset
from pyhealth.tasks import sleep_staging_sleepedf_cassette_fn

# step 1: load signal data
dataset = SleepEDFCassetteDataset(
    root="/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/sleep-cassette",
    dev=True,
    refresh_cache=True,
)

# step 2: set task
sleep_staging_ds = dataset.set_task(sleep_staging_sleepedf_cassette_fn)
sleep_staging_ds.stat()

# step 3: split dataset
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

print(next(iter(train_dataloader)))
