from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.trainer import Trainer
from pyhealth.datasets import SHHSDataset
from pyhealth.tasks import sleep_staging_shhs_fn
from pyhealth.models import ContraWR

# step 1: load signal data with specific channels
dataset = SHHSDataset(
    root="/srv/local/data/SHHS/polysomnography",
    dev=True,
    refresh_cache=False,
    channels=['EEG1', 'EEG2']  # Specifically select EEG channels
)

# step 2: set task with custom parameters
sleep_staging_ds = dataset.set_task(
    sleep_staging_shhs_fn,
    task_kwargs={"epoch_seconds": 20}  # Modified epoch length
)
sleep_staging_ds.stat()

# split dataset with different ratio
train_dataset, val_dataset, test_dataset = split_by_patient(
    sleep_staging_ds, [0.7, 0.1, 0.2]  # Modified split ratio
)

# Use different batch sizes for different splits
train_dataloader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=128, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=128, shuffle=False)

print(
    "loader size: train/val/test",
    len(train_dataset),
    len(val_dataset),
    len(test_dataset),
)

# STEP 3: define model with attention mechanism
model = ContraWR(
    dataset=sleep_staging_ds,
    feature_keys=["signal"],
    label_key="label",
    mode="multiclass",
    hidden_dim=128,
    num_layers=3,
    dropout=0.2,
    use_attention=True  # Enable attention mechanism
)

# STEP 4: define trainer with custom settings
trainer = Trainer(
    model=model,
    metrics=["accuracy", "f1_macro", "cohen_kappa"],  # Additional metrics
    device="cuda:0"
)

# Custom learning rate scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=20,
    monitor="f1_macro",  # Monitor different metric
    optimizer_params={
        "lr": 0.001,
        "weight_decay": 1e-5
    },
    scheduler_cls=CosineAnnealingLR,
    scheduler_params={
        "T_max": 20
    }
)

# STEP 5: evaluate with detailed metrics
results = trainer.evaluate(test_dataloader)
print("\nTest Results:")
print("-------------")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Optional: Save model predictions for analysis
predictions = trainer.predict(test_dataloader)