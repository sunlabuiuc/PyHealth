from datetime import datetime
from typing import Any, Dict, List, Optional
import os

# PyHealth Packages
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.ehr_foundational_model_mimic4 import EHRFoundationalModelMIMIC4
from pyhealth.tasks.base_task import BaseTask

# Load MIMIC4 Files
# There's probably better ways dealing with this on the cluster, but working locally for now 
# (see: https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/multimodal_mimic4_minimal.py)

PYHEALTH_REPO_ROOT = #'/Users/wpang/Desktop/PyHealth'

EHR_ROOT = os.path.join(PYHEALTH_REPO_ROOT, "srv/local/data/physionet.org/files/mimiciv/2.2")
NOTE_ROOT = os.path.join(PYHEALTH_REPO_ROOT, "srv/local/data/physionet.org/files/mimic-iv-note/2.2")
CXR_ROOT = os.path.join(PYHEALTH_REPO_ROOT,"srv/local/data/physionet.org/files/mimic-cxr-jpg/2.0.0")
CACHE_DIR = os.path.join(PYHEALTH_REPO_ROOT,"srv/local/data/wp/pyhealth_cache")

if __name__ == "__main__":

    dataset = MIMIC4Dataset(
            ehr_root=EHR_ROOT,
            note_root=NOTE_ROOT,
            ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
            note_tables=["discharge", "radiology"],
            cache_dir=CACHE_DIR,
            num_workers=16,
            dev=True
        )
    
    # Apply multimodal task
    task = EHRFoundationalModelMIMIC4() 
    samples = dataset.set_task(task, cache_dir=f"{CACHE_DIR}/task", num_workers=8)

    # Get and print sample
    sample = samples[0]
    print(sample)