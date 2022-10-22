from torch.utils.data import DataLoader
import sys

sys.path.append("/home/chaoqiy2/github/PyHealth-OMOP")

from pyhealth.datasets import MIMIC3Dataset, eICUDataset, MIMIC4Dataset, OMOPDataset
from pyhealth.models import Transformer, RNN, RETAIN, MICRON, GAMENet, SafeDrug
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.tasks import (
    drug_recommendation_mimic3_fn,
    drug_recommendation_eicu_fn,
    drug_recommendation_mimic4_fn,
    drug_recommendation_omop_fn,
)
from pyhealth.utils import collate_fn_dict
from pyhealth.trainer import Trainer
from pyhealth.evaluator import evaluate
from pyhealth.metrics import *

###############
data = "mimic3"
################

# STEP 1 & 2: load data and set task

if data == "mimic3":
    mimic3dataset = MIMIC3Dataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=True,
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )
    mimic3dataset.stat()
    mimic3dataset.set_task(drug_recommendation_mimic3_fn)
    mimic3dataset.stat()
    dataset = mimic3dataset

elif data == "eicu":
    eicudataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "physicalExam"],
        dev=True,
        refresh_cache=False,
    )
    eicudataset.stat()
    eicudataset.set_task(task_fn=drug_recommendation_eicu_fn)
    eicudataset.stat()
    dataset = eicudataset

elif data == "mimic4":
    mimic4dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )
    mimic4dataset.stat()
    mimic4dataset.set_task(task_fn=drug_recommendation_mimic4_fn)
    mimic4dataset.stat()
    dataset = mimic4dataset

elif data == "omop":
    omopdataset = OMOPDataset(
        root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
        tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
        dev=True,
        refresh_cache=False,
    )
    omopdataset.stat()
    omopdataset.set_task(task_fn=drug_recommendation_omop_fn)
    omopdataset.stat()
    dataset = omopdataset

# data split
train_dataset, val_dataset, test_dataset = split_by_patient(dataset, [0.8, 0.1, 0.1])
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_dict
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_dict
)

# STEP 3: define model
device = "cuda:0"

model = SafeDrug(
    dataset=dataset,
    tables=["conditions", "procedures"],
    target="label",
    mode="multilabel",
)

model.to(device)

# STEP 4: define trainer
trainer = Trainer(enable_logging=True, output_path="../output", device=device)
trainer.fit(
    model,
    train_loader=train_loader,
    epochs=50,
    val_loader=val_loader,
    val_metric=pr_auc_multilabel,
)

# STEP 5: evaluate
model = trainer.load_best_model(model)
y_gt, y_prob, y_pred = evaluate(model, test_loader, device)

print(y_gt, y_prob, y_pred)

jaccard = jaccard_multilabel(y_gt, y_pred)
accuracy = accuracy_multilabel(y_gt, y_pred)
f1 = f1_multilabel(y_gt, y_pred)
prauc = pr_auc_multilabel(y_gt, y_prob)

# print metric name and score
print("jaccard: ", jaccard)
print("accuracy: ", accuracy)
print("f1: ", f1)
print("prauc: ", prauc)
