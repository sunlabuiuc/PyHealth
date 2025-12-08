from pyhealth.datasets import MIMIC4EHRDataset, get_dataloader, split_by_patient
from pyhealth.models import RNN
from pyhealth.tasks import LengthOfStayPredictionMIMIC4
from pyhealth.trainer import Trainer

# STEP 1: load data
base_dataset = MIMIC4EHRDataset(
    root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
    tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    code_mapping={"NDC": "ATC"},
    dev=False,
)
base_dataset.stat()

# STEP 2: set task
sample_dataset = base_dataset.set_task(LengthOfStayPredictionMIMIC4())
sample_dataset.stat()

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
trainer = Trainer(
    model=model,
    metrics=["accuracy", "f1_weighted", "f1_macro", "f1_micro"],
)
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=50,
    monitor="accuracy",
)

# STEP 5: evaluate
results = trainer.evaluate(test_dataloader)
print("\nTest Results:")
for metric, value in results.items():
    print(f"  {metric}: {value:.4f}")
