from pyhealth.datasets import split_by_visit, get_dataloader
from pyhealth.trainer import Trainer
from pyhealth.datasets import TUEVDataset
from pyhealth.tasks import EEGEventsTUEV
from pyhealth.models import SparcNet

# step 1: load signal data
dataset = TUEVDataset(root="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf/", 
                            dev=True,
                            refresh_cache=True, 
                            )
print(dataset.stats())
# step 2: set task
TUEV_ds = dataset.set_task(EEGEventsTUEV(
    resample_rate=200,    # Resample rate
    bandpass_filter=(0.1, 75.0),    # Bandpass filter
    notch_filter=50.0,    # Notch filter
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
model = SparcNet(
    dataset=TUEV_ds,
    feature_keys=["signal"],
    label_key="label",
    mode="multiclass",
)

# STEP 4: define trainer
trainer = Trainer(model=model, device="cuda:4")
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=10,
    optimizer_params={"lr": 1e-3},
)

# STEP 5: evaluate
print(trainer.evaluate(test_dataloader))
