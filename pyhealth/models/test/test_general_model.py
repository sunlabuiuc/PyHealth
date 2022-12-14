from torch.utils.data import DataLoader

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets.utils import collate_fn_dict

# from pyhealth.models import CNN as Model
# from pyhealth.models import RNN as Model
from pyhealth.models import RETAIN as Model
# from pyhealth.models import Transformer as Model


def task_event(patient):
    samples = []
    for visit in patient:
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        mortality_label = int(visit.discharge_status)
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "list_label": drugs,
                "value_label": mortality_label,
            }
        )
    return samples


def task_visit(patient):
    samples = []
    for visit in patient:
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        mortality_label = int(visit.discharge_status)
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "list_label": drugs,
                "value_label": mortality_label,
            }
        )
    return samples


base_dataset = MIMIC3Dataset(
    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    dev=True,
    refresh_cache=False,
)

# event level + binary
sample_dataset = base_dataset.set_task(task_event)
dataloader = DataLoader(
    sample_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
)
model = Model(
    dataset=sample_dataset,
    feature_keys=["conditions", "procedures"],
    label_key="value_label",
    mode="binary",
)
model.to("cuda")
batch = iter(dataloader).next()
output = model(**batch)
print(output["loss"])

# visit level + binary
sample_dataset = base_dataset.set_task(task_visit)
dataloader = DataLoader(
    sample_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
)
model = Model(
    dataset=sample_dataset,
    feature_keys=["conditions", "procedures"],
    label_key="value_label",
    mode="binary",
)
model.to("cuda")
batch = iter(dataloader).next()
output = model(**batch)
print(output["loss"])

# event level + multiclass
sample_dataset = base_dataset.set_task(task_event)
dataloader = DataLoader(
    sample_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
)
model = Model(
    dataset=sample_dataset,
    feature_keys=["conditions", "procedures"],
    label_key="value_label",
    mode="multiclass",
)
model.to("cuda")
batch = iter(dataloader).next()
output = model(**batch)
print(output["loss"])

# visit level + multiclass
sample_dataset = base_dataset.set_task(task_visit)
dataloader = DataLoader(
    sample_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
)
model = Model(
    dataset=sample_dataset,
    feature_keys=["conditions", "procedures"],
    label_key="value_label",
    mode="multiclass",
)
model.to("cuda")
batch = iter(dataloader).next()
output = model(**batch)
print(output["loss"])

# event level + multilabel
sample_dataset = base_dataset.set_task(task_event)
dataloader = DataLoader(
    sample_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
)
model = Model(
    dataset=sample_dataset,
    feature_keys=["conditions", "procedures"],
    label_key="list_label",
    mode="multilabel",
)
model.to("cuda")
batch = iter(dataloader).next()
output = model(**batch)
print(output["loss"])

# visit level + multilabel
sample_dataset = base_dataset.set_task(task_visit)
dataloader = DataLoader(
    sample_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
)
model = Model(
    dataset=sample_dataset,
    feature_keys=["conditions", "procedures"],
    label_key="list_label",
    mode="multilabel",
)
model.to("cuda")
batch = iter(dataloader).next()
output = model(**batch)
print(output["loss"])
