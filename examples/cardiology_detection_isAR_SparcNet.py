from pyhealth.datasets import split_by_visit, get_dataloader
from pyhealth.trainer import Trainer
from pyhealth.datasets import CardiologyDataset
from pyhealth.tasks import cardiology_isAR_fn
from pyhealth.models import ContraWR, SparcNet

# step 1: load signal data
dataset = CardiologyDataset(root="/srv/local/data/physionet.org/files/challenge-2020/1.0.2/training", 
                            chosen_dataset=[1,1,1,1,1,1], 
                            refresh_cache=False, 
                            dev=True)

# step 2: set task
cardiology_ds = dataset.set_task(cardiology_isAR_fn)
cardiology_ds.stat()

# split dataset
train_dataset, val_dataset, test_dataset = split_by_visit(
    cardiology_ds, [0.6, 0.2, 0.2]
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
    dataset=cardiology_ds,
    feature_keys=["signal"],
    label_key="label",
    mode="binary",
)

# STEP 4: define trainer
trainer = Trainer(model=model, device="cuda:4")
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=5,
    monitor="pr_auc",
)

# STEP 5: evaluate
print(trainer.evaluate(test_dataloader))
