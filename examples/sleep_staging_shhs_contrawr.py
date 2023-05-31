from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.trainer import Trainer
from pyhealth.datasets import SHHSDataset
from pyhealth.tasks import sleep_staging_shhs_fn
from pyhealth.models import ContraWR, SparcNet

# step 1: load signal data
dataset = SHHSDataset(
    root="/srv/local/data/SHHS/polysomnography",
    dev=True,
    refresh_cache=False,
)

# step 2: set task
sleep_staging_ds = dataset.set_task(sleep_staging_shhs_fn)
sleep_staging_ds.stat()

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
model = SparcNet(
    dataset=sleep_staging_ds,
    feature_keys=["signal"],
    label_key="label",
    mode="multiclass",
)

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
