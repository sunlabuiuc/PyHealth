# %% Loading MIMIC-IV dataset
from pyhealth.datasets import MIMIC4EHRDataset

dataset = MIMIC4EHRDataset(
    root="/home/logic/physionet.org/files/mimic-iv-demo/2.2/",
    tables=[
        "patients",
        "admissions",
        "diagnoses_icd",
        "procedures_icd",
        "labevents",
    ],
)

# %% Setting StageNet Mortality Prediction Task
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.datasets import get_dataloader, load_processors, save_processors, split_by_patient

input_processors, output_processors = load_processors("../resources/")

sample_dataset = dataset.set_task(
    MortalityPredictionStageNetMIMIC4(), 
    cache_dir="~/.cache/pyhealth/mimic4_stagenet_mortality",
    input_processors=input_processors,
    output_processors=output_processors,
)
print(f"Total samples: {len(sample_dataset)}")

# %% Loading Pretrained StageNet Model
import torch
from pyhealth.models import StageNet

model = StageNet(
    dataset=sample_dataset,
    embedding_dim=128,
    chunk_size=128,
    levels=3,
    dropout=0.3,
)
model.load_state_dict(torch.load('../resources/best.ckpt', map_location='cuda:0'))
print(model)

# %%
