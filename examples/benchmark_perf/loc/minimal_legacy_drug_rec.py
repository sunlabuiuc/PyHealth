from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import drug_recommendation_mimic4_fn
base_dataset = MIMIC4Dataset(root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
                             tables=["diagnoses_icd", "procedures_icd", "prescriptions"], dev=False,
                             code_mapping={"NDC": "ATC"}, refresh_cache=True)
sample_dataset = base_dataset.set_task(task_fn=drug_recommendation_mimic4_fn)
print(f"Samples: {len(sample_dataset.samples)}")
