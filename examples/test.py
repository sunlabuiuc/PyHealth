import sys

sys.path.append("/home/chaoqiy2/github/PyHealth")

import numpy as np

N_pat, N_vis = 50, 10
condition_space = [f"cond-{i}" for i in range(100)]
procedure_space = [f"prod-{i}" for i in range(100)]

samples = []
for pat_i in range(N_pat):
    conditions = []
    procedures = []
    for visit_j in range(N_vis):
        patient_id = f"patient-{pat_i}"
        visit_id = f"visit-{visit_j}"
        # how many conditions to simulate
        N_cond = np.random.randint(3, 6)
        conditions.append(
            np.random.choice(condition_space, N_cond, replace=False).tolist()
        )
        # how many procedures to simulate
        procedures.append(np.random.random(5).tolist())
        # which binary label
        label = int(np.random.random() > 0.5)

        sample = {
            "patient_id": patient_id,
            "visit_id": visit_id,
            "conditions": conditions.copy(),
            "procedures": procedures.copy(),
            "label": label,
        }
        samples.append(sample)

# load into the dataset
from pyhealth.datasets import SampleDataset

dataset = SampleDataset(samples)

"""#### **get the train/val/test dataset**"""

from pyhealth.datasets.splitter import split_by_patient
from torch.utils.data import DataLoader
from pyhealth.datasets.utils import collate_fn_dict

# data split
train_dataset, val_dataset, test_dataset = split_by_patient(dataset, [0.8, 0.1, 0.1])

# create dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_dict
)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_dict
)

"""### **Step 2: select a ML model**
> In this tutorial, we use Transformer as the example.
"""

from pyhealth.models import Transformer

device = "cuda:0"
model = Transformer(
    dataset=dataset,
    feature_keys=["conditions", "procedures"],
    label_key="label",
    mode="binary",
    operation_level="visit",
)
model.to(device)

from pyhealth.trainer import Trainer

# use our Trainer to train the model

trainer = Trainer(model=model)
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=5,
    monitor="roc_auc",  # which metric do you want to monitor for selecting the best model, check https://pyhealth.readthedocs.io/en/latest/api/metrics/pyhealth.metrics.multiclass.html
)
