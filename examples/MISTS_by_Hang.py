import sys
sys.path.append('./PyHealth')
import pyhealth
from pyhealth.datasets import MIMIC3Dataset

mimic3_ds = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/mimiciii-demo/1.4/",
        tables=["labevents"],
        dev=True,
)

from pyhealth.tasks import mimic3_48_ihm, MIMIC3_48_IHM

samples = mimic3_ds.set_task(MIMIC3_48_IHM())
from pyhealth.datasets import split_by_sample, get_dataloader

# data split
train_dataset, val_dataset, test_dataset = split_by_sample(samples, [0.8, 0.1, 0.1])

# create dataloaders (they are <torch.data.DataLoader> object)
train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)
from pyhealth.models import RNN

model = RNN(
    dataset=samples,
    # look up what are available for "feature_keys" and "label_keys" in dataset.samples[0]
    feature_keys=["discretized_feature"],
    label_key="mortality",
    mode="multiclass",
)

from pyhealth.trainer import Trainer

trainer = Trainer(
    model=model,
    metrics=["accuracy", "f1_weighted"], # the metrics that we want to log
    )

trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=20,
    monitor="accuracy",
    monitor_criterion="max",
)

# option 2: use our pyhealth.metrics to evaluate
from pyhealth.metrics.multiclass import multiclass_metrics_fn

y_true, y_prob, loss = trainer.inference(test_loader)
multiclass_metrics_fn(
    y_true,
    y_prob,
    metrics=["f1_weighted", "f1_micro", "cohen_kappa"]
)