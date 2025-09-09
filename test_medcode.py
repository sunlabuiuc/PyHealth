from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import MIMIC3ICD9Coding

root = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III"
dataset = MIMIC3Dataset(
    root=root,
    dataset_name="mimic3",
    tables=["diagnoses_icd", "procedures_icd", "noteevents"],
)

mimic3_coding = MIMIC3ICD9Coding()
samples = dataset.set_task(mimic3_coding)
