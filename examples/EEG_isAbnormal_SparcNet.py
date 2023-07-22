from pyhealth.datasets import split_by_visit, get_dataloader
from pyhealth.trainer import Trainer
from pyhealth.datasets import TUABDataset
from pyhealth.tasks import EEG_isAbnormal_fn
from pyhealth.models import SparcNet

# step 1: load signal data
dataset = TUABDataset(root="/srv/local/data/TUH/tuh_eeg_abnormal/v3.0.0/edf/", 
                            dev=True,
                            refresh_cache=True, 
                            )

# step 2: set task
TUAB_ds = dataset.set_task(EEG_isAbnormal_fn)
TUAB_ds.stat()

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
model = SparcNet(
    dataset=TUAB_ds,
    feature_keys=["signal"],
    label_key="label",
    mode="binary",
)

# STEP 4: define trainer
trainer = Trainer(model=model, device="cuda:4")
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=10,
    monitor="pr_auc",
    optimizer_params={"lr": 1e-3},
)

# STEP 5: evaluate
print(trainer.evaluate(test_dataloader))
