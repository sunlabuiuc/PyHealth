from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import Transformer
from pyhealth.tasks import drug_recommendation_mimic3_fn
from pyhealth.trainer import Trainer

# STEP 1: load data
dataset = MIMIC3Dataset(
    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
)
dataset.stat()

# STEP 2: set task
dataset.set_task(drug_recommendation_mimic3_fn)
dataset.stat()

train_dataset, val_dataset, test_dataset = split_by_patient(dataset, [0.8, 0.1, 0.1])
train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

# STEP 3: define model
model = Transformer(
    dataset=dataset,
    feature_keys=["conditions", "procedures"],
    label_key="drugs",
    mode="multilabel",
    operation_level="visit",
)

# STEP 4: define trainer
trainer = Trainer(model=model)
trainer.train(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    epochs=50,
    monitor="pr_auc_samples",
)

# STEP 5: evaluate
trainer.evaluate(test_dataloader)
