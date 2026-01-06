from pyhealth.datasets import eICUDataset
from pyhealth.tasks import MortalityPredictionEICU, MortalityPredictionEICU2

if __name__ == "__main__":
    base_dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "admissionDx", "treatment"],
        dev=False,
        refresh_cache=False,
    )
    task = MortalityPredictionEICU2()
    sample_dataset = base_dataset.set_task(task=task)
    sample_dataset.stat()
    print(sample_dataset.available_keys)

# base_dataset = eICUDataset(
#     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
#     tables=["diagnosis", "admissionDx", "treatment"],
#     dev=True,
#     refresh_cache=False,
# )
# task = MortalityPredictionEICU2()
# sample_dataset = base_dataset.set_task(task=task)
# sample_dataset.stat()
# print(sample_dataset.available_keys)
