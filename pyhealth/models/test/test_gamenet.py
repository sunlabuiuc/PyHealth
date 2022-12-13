from torch.utils.data import DataLoader

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.models import GAMENet
from pyhealth.tasks import drug_recommendation_mimic3_fn
from pyhealth.datasets.utils import collate_fn_dict

base_dataset = MIMIC3Dataset(
    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    dev=True,
    code_mapping={"NDC": "ATC"},
    refresh_cache=False,
)

# visit level + multilabel
sample_dataset = base_dataset.set_task(drug_recommendation_mimic3_fn)
dataloader = DataLoader(
    sample_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn_dict
)
model = GAMENet(
    dataset=sample_dataset,
)
model.to("cuda")
batch = iter(dataloader).next()
output = model(**batch)
print(output["loss"])
