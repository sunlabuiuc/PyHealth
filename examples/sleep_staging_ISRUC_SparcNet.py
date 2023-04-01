from pyhealth.calib import calibration
from pyhealth.datasets import ISRUCDataset, get_dataloader, split_by_patient
from pyhealth.models import ContraWR, SparcNet
from pyhealth.tasks import sleep_staging_isruc_fn
from pyhealth.trainer import Trainer, get_metrics_fn

# step 1: load signal data
dataset = ISRUCDataset(
    root="/srv/local/data/trash/",
    dev=True,
    refresh_cache=False,
    # download=True,
)

print(dataset.stat())

# step 2: set task
sleep_staging_ds = dataset.set_task(sleep_staging_isruc_fn)
sleep_staging_ds.stat()
print(sleep_staging_ds.samples[0])

# split dataset
train_dataset, val_dataset, test_dataset = split_by_patient(
    sleep_staging_ds, [0.34, 0.33, 0.33]
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
metrics = ['accuracy', 'f1_macro', 'f1_micro'] + ['cwECEt_adapt']
y_true_all, y_prob_all = trainer.inference(test_dataloader)[:2]
print(get_metrics_fn(model.mode)(y_true_all, y_prob_all, metrics=metrics))

# STEP 6: calibrate the model
cal_model = calibration.KCal(model, debug=True, dim=32)
cal_model.calibrate(
    cal_dataset=val_dataset,
    # Uncomment the following line if you want to re-train the embeddings
    # train_dataset=train_dataset,
)
y_true_all, y_prob_all = Trainer(model=cal_model).inference(test_dataloader)[:2]
print(get_metrics_fn(cal_model.mode)(y_true_all, y_prob_all, metrics=metrics))