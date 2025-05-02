from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import RNN
from pyhealth.tasks.length_of_stay_prediction import (
    LengthOfStayGreaterThanXPredictionMIMIC3,
)
from pyhealth.trainer import Trainer

# STEP 1: load data
base_dataset = MIMIC3Dataset(
    root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    dev=True,
)
base_dataset.stats()

# STEP 2: set task
sample_dataset = base_dataset.set_task(
    LengthOfStayGreaterThanXPredictionMIMIC3(threshold=3)
)

train_dataset, val_dataset, test_dataset = split_by_patient(
    sample_dataset, [0.8, 0.1, 0.1]
)
train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

# STEP 3: define model
model = RNN(
    dataset=sample_dataset,
)

# STEP 4: define trainer
trainer = Trainer(model=model)
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=5,
)

# STEP 5: evaluate
print(trainer.evaluate(test_dataloader))
