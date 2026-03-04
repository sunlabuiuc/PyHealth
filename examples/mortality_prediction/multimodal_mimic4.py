from datetime import datetime
from typing import Any, Dict, List, Optional
import os

# PyHealth Packages
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.multimodal_mimic4 import ClinicalNotesMIMIC4
from pyhealth.tasks.base_task import BaseTask

# Load MIMIC4 Files
# There's probably better ways dealing with this on the cluster, but working locally for now 
# (see: https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/multimodal_mimic4_minimal.py)

TASK = "ClinicalNotesMIMIC4" # The idea here is that we want additive tasks so we can evaluate the value in adding more modalities

PYHEALTH_REPO_ROOT = '/Users/wpang/Desktop/PyHealth'

EHR_ROOT = os.path.join(PYHEALTH_REPO_ROOT, "local_data/local/data/physionet.org/files/mimiciv/2.2")
NOTE_ROOT = os.path.join(PYHEALTH_REPO_ROOT, "local_data/local/data/physionet.org/files/mimic-iv-note/2.2")
CXR_ROOT = os.path.join(PYHEALTH_REPO_ROOT,"local_data/local/data/physionet.org/files/mimic-cxr-jpg/2.0.0")
CACHE_DIR = os.path.join(PYHEALTH_REPO_ROOT,"local_data/local/data/wp/pyhealth_cache")

if __name__ == "__main__":

    if TASK == "ClinicalNotesMIMIC4": # A bit janky setup at the moment and open to iteration, but conveys the point for now
        dataset = MIMIC4Dataset(
                ehr_root=EHR_ROOT,
                note_root=NOTE_ROOT,
                ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
                note_tables=["discharge", "radiology"],
                cache_dir=CACHE_DIR,
                num_workers=8,
                dev=True
            )
        
        # Apply multimodal task
        task = ClinicalNotesMIMIC4() 
        samples = dataset.set_task(task)

        # Get and print sample
        sample = samples[0]
        print(sample)