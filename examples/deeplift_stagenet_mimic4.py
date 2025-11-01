# %% Loading MIMIC-IV dataset
from pyhealth.datasets import MIMIC4EHRDataset

dataset = MIMIC4EHRDataset(
    root="/home/logic/physionet.org/files/mimiciv/3.1/",
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

task = MortalityPredictionStageNetMIMIC4()
sample_dataset = dataset.set_task(task, cache_dir="~/.cache/pyhealth/mimic4_stagenet_mortality")

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
model.load_state_dict(torch.load('../ckpt/best.ckpt', map_location='cuda:0'))
