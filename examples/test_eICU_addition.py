from pyhealth.datasets import eICUDataset
from pyhealth.tasks import mortality_prediction_eicu_fn, mortality_prediction_eicu_fn2

base_dataset = eICUDataset(
    root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
    tables=["diagnosis", "admissionDx", "treatment"],
    dev=False,
    refresh_cache=False,
)
sample_dataset = base_dataset.set_task(task_fn=mortality_prediction_eicu_fn2)
sample_dataset.stat()
print(sample_dataset.available_keys)

# base_dataset = eICUDataset(
#     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
#     tables=["diagnosis", "admissionDx", "treatment"],
#     dev=True,
#     refresh_cache=False,
# )
# sample_dataset = base_dataset.set_task(task_fn=mortality_prediction_eicu_fn2)
# sample_dataset.stat()
# print(sample_dataset.available_keys)
