# %%
import psutil, os, time, threading
PEAK_MEM_USAGE = 0
SELF_PROC = psutil.Process(os.getpid())

def track_mem():
    global PEAK_MEM_USAGE
    while True:
        m = SELF_PROC.memory_info().rss
        if m > PEAK_MEM_USAGE:
            PEAK_MEM_USAGE = m
        time.sleep(0.1)

threading.Thread(target=track_mem, daemon=True).start()
print(f"[MEM] start={PEAK_MEM_USAGE / (1024**3)} GB")

# %%
from pyhealth.datasets import MIMIC4Dataset
DATASET_DIR = "/home/logic/physionet.org/files/mimiciv/3.1"
dataset = MIMIC4Dataset(
    ehr_root=DATASET_DIR,
    ehr_tables=[
        "patients",
        "admissions",
        "diagnoses_icd",
        "procedures_icd",
        "prescriptions",
        "labevents",
    ],
)
print(f"[MEM] __init__={PEAK_MEM_USAGE / (1024**3):.3f} GB")
# %%
