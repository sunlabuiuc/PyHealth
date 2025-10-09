from pyhealth.datasets import MIMIC3Dataset, split_by_patient, get_dataloader
from pyhealth.models import ClinicalBERTWrapper
from pyhealth.tasks import HurtfulWordsBiasTask
from pyhealth.trainer import Trainer

# STEP 1: load MIMIC-III
base = MIMIC3Dataset(
    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    tables=["NOTEEVENTS", "PATIENTS"],
    dev=False,
    refresh_cache=False
)

# STEP 2: set our bias task
bias_task = HurtfulWordsBiasTask(positive_group="female", negative_group="male")
task_dataset = base.set_task(bias_task)
task_dataset.stat()

# STEP 3: train/test split & dataloaders
train_ds, val_ds, test_ds = split_by_patient(task_dataset, [0.8, 0.1, 0.1])
train_dl = get_dataloader(train_ds, batch_size=16, shuffle=True)
val_dl   = get_dataloader(val_ds,   batch_size=16, shuffle=False)
test_dl  = get_dataloader(test_ds,  batch_size=16, shuffle=False)

# STEP 4: wrap a ClinicalBERT model
model = ClinicalBERTWrapper(
    pretrained_model_name="emilyalsentzer/Bio_ClinicalBERT",
    device="cuda"
)

# STEP 5: train/calibrate if needed
trainer = Trainer(model=model, task=bias_task)
trainer.train(train_dl, val_dl, epochs=1, monitor=None)

# STEP 6: evaluate log-bias and precision_gap
metrics = ["log_bias", "precision_gap"]
results = trainer.evaluate(test_dl, metrics=metrics)
print("Fairness results:", results)
