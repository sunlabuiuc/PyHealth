from pyhealth.datasets import MIMIC4EHRDataset
from pyhealth.tasks import RemainingLOSMIMIC4
from pyhealth.datasets import  SampleDataset    
import os
import numpy as np

MIMIC_ROOT = r"D:\cs598\mimic-iv"
CACHE_PATH = r"D:\cs598\.cache_dir"

def inspect():
    mimic4 = MIMIC4EHRDataset( 
            root=MIMIC_ROOT,
            tables=["diagnoses_icd", "labevents", "procedures_icd", "prescriptions", "chartevents"],
            dev=True, cache_dir=CACHE_PATH
        )
    
    mimic4.stats()

    # STEP 2: set task
    sample_dataset = mimic4.set_task(RemainingLOSMIMIC4())
    
    first_sample:SampleDataset = sample_dataset[0]
    print("Sample keys:", first_sample.keys())
    for (key, value) in first_sample.items():
        if key in ["timeseries"]:
            print(f"Sample '{key}' shape: {value.shape}")
        else:
            print(f"Sample '{key}' value: {value}")

if __name__ == "__main__":
    inspect()
