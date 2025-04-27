from pyhealth.datasets import TUSZBinaryDataset
from pyhealth.tasks import EEGIsSeizure
from pyhealth.datasets import split_by_sample
from pyhealth.datasets import get_dataloader
from pyhealth.models import RNN
from pyhealth.trainer import Trainer

root = "c:/dlh/v2.0.3/edf/train"
dataset = TUSZBinaryDataset(
    root=root,
)

dataset.stats()

task = EEGIsSeizure()
samples = dataset.set_task(task)

print(len(samples))

train_dataset, val_dataset, test_dataset = split_by_sample(
    dataset=samples,
    ratios=[0.7, 0.1, 0.2]
)

train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

model = RNN(
    dataset=samples,
)
trainer = Trainer(
    model=model,
    metrics=["roc_auc"]
)

print(trainer.evaluate(test_dataloader))

trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=10,
    monitor="roc_auc",  # Monitor roc_auc specifically
    optimizer_params={"lr": 1e-4}  # Using learning rate of 1e-4
)

print(trainer.evaluate(val_dataloader))
