from datetime import datetime
from typing import Any, Dict, List, Optional
import os

# PyHealth Packages
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.multimodal_mimic4 import ClinicalNotesMIMIC4, ClinicalNotesICDLabsMIMIC4, ClinicalNotesICDLabsCXRMIMIC4
from pyhealth.tasks.base_task import BaseTask

# Load MIMIC4 Files
# There's probably better ways dealing with this on the cluster, but working locally for now 
# (see: https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/multimodal_mimic4_minimal.py)

TASK = "ClinicalNotesICDLabsCXRMIMIC4" # The idea here is that we want additive tasks so we can evaluate the value in adding more modalities
DEV_MODE = True
ENVIRONMENT = "Local" # Either 'Local' or 'Cluster'

if ENVIRONMENT == "Local":
    pyhealth_repo_root = '/Users/wpang/Desktop/PyHealth'

    ehr_root = os.path.join(pyhealth_repo_root, "local_data/local/data/physionet.org/files/mimiciv/2.2")
    note_root = os.path.join(pyhealth_repo_root, "local_data/local/data/physionet.org/files/mimic-iv-note/2.2")
    cxr_root = os.path.join(pyhealth_repo_root,"local_data/local/data/physionet.org/files/mimic-cxr-jpg/2.0.0")
    cache_dir = os.path.join(pyhealth_repo_root,"local_data/local/data/wp/pyhealth_cache")
elif ENVIRONMENT == "Cluster":

    ehr_root = "/projects/illinois/eng/cs/jimeng/physionet.org/files/mimiciv/2.2"
    note_root = "/projects/illinois/eng/cs/jimeng/physionet.org/files/mimic-note"
    cxr_root = "/projects/illinois/eng/cs/jimeng/physionet.org/files/mimic-cxr-jpg/2.1.0"
    cache_dir = "/u/wp14/pyhealth_cache"


if __name__ == "__main__":

    if TASK == "ClinicalNotesMIMIC4": # A bit janky setup at the moment and open to iteration, but conveys the point for now
        dataset = MIMIC4Dataset(
                ehr_root=ehr_root,
                note_root=note_root,
                ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
                note_tables=["discharge", "radiology"],
                cache_dir=cache_dir,
                num_workers=8,
                dev=DEV_MODE
            )
        
        # Apply multimodal task
        task = ClinicalNotesMIMIC4() 
        samples = dataset.set_task(task)

        # Get and print sample
        sample = samples[0]
        print(sample)
    
    elif TASK == 'ClinicalNotesICDLabsMIMIC4':
        dataset = MIMIC4Dataset(
                ehr_root=ehr_root,
                note_root=note_root,
                ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
                note_tables=["discharge", "radiology"],
                cache_dir=cache_dir,
                num_workers=8,
                dev=DEV_MODE
            )
        
        # Apply multimodal task
        task = ClinicalNotesICDLabsMIMIC4() 
        samples = dataset.set_task(task)

        # Get and print sample
        sample = samples[0]
        print(sample)

    elif TASK == 'ClinicalNotesICDLabsCXRMIMIC4':
        dataset = MIMIC4Dataset(
                ehr_root=ehr_root,
                note_root=note_root,
                cxr_root=cxr_root,
                ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
                note_tables=["discharge", "radiology"],
                cxr_tables=["metadata", "negbio"],
                cache_dir=cache_dir,
                num_workers=8,
                dev=DEV_MODE
            )
        
        # Apply multimodal task
        task = ClinicalNotesICDLabsCXRMIMIC4() 
        samples = dataset.set_task(task)

        # Get and print sample
        sample = samples[0]
        print(sample)