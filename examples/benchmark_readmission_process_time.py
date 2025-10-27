import time
from pyhealth.datasets import MIMIC4Dataset

start_time = time.perf_counter()
root = "/srv/local/data/MIMIC-IV/2.0"
dataset = MIMIC4Dataset(
    ehr_root=root,
    ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
)
end_time = time.perf_counter()
read_time = end_time - start_time
print(f"Read time: {read_time:.2f} seconds")


import time
from pyhealth.tasks import Readmission30DaysMIMIC4

start_time = time.perf_counter()
task = Readmission30DaysMIMIC4()
sample_dataset = dataset.set_task(task, num_workers=16)
end_time = time.perf_counter()
set_task_time = end_time - start_time
print(f"Set task time: {set_task_time:.2f} seconds")
