import os

# PyHealth Packages
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.multimodal_mimic4 import (
    ICDLabsMIMIC4,
    NotesLabsMIMIC4,
    LabsMIMIC4,
    CXRMIMIC4,
)

# Load MIMIC4 Files
# There's probably better ways dealing with this on the cluster, but working locally for now
# (see: https://github.com/sunlabuiuc/PyHealth/blob/master/examples/mortality_prediction/multimodal_mimic4_minimal.py)

TASK = "NotesLabsMIMIC4"  # Options: ICDLabsMIMIC4, NotesLabsMIMIC4, LabsMIMIC4, CXRMIMIC4  # Each task isolates a different modality subset so we can evaluate the value of adding more modalities
DEV_MODE = True
ENVIRONMENT = "CampusCluster"  # Either 'Local' or 'CampusCluster' or "SunLabCluster"
NETID = "wp14" # For personal cache

if ENVIRONMENT == "Local":
    pyhealth_repo_root = "/Users/wpang/Desktop/PyHealth"

    ehr_root = os.path.join(
        pyhealth_repo_root, "local_data/local/data/physionet.org/files/mimiciv/2.2"
    )
    note_root = os.path.join(
        pyhealth_repo_root,
        "local_data/local/data/physionet.org/files/mimic-iv-note/2.2",
    )
    cxr_root = os.path.join(
        pyhealth_repo_root,
        "llocal_data/local/data/physionet.org/files/mimic-cxr-jpg/2.0.0",
    )
    cache_dir = os.path.join(
        pyhealth_repo_root, "local_data/local/data/wp/pyhealth_cache"
    )
elif ENVIRONMENT == "CampusCluster":

    ehr_root = "/projects/illinois/eng/cs/jimeng/physionet.org/files/mimiciv/2.2"
    note_root = "/projects/illinois/eng/cs/jimeng/physionet.org/files/mimic-note"
    cxr_root = None # Please fill this in
    cache_dir = f"/u/{NETID}/pyhealth_cache"
elif ENVIRONMENT == "SunLabCluster":

    ehr_root = "/shared/rsaas/physionet.org/files/mimiciv/2.2"
    note_root = "/shared/rsaas/physionet.org/files/mimic-note"
    cxr_root = None # Please fill this in
    cache_dir = f"/home/{NETID}/pyhealth_cache"


if __name__ == "__main__":

    if TASK == "ICDLabsMIMIC4":
        dataset = MIMIC4Dataset(
            ehr_root=ehr_root,
            ehr_tables=[
                "diagnoses_icd",
                "procedures_icd",
                "labevents",
                "prescriptions",
            ],
            cache_dir=cache_dir,
            num_workers=8,
            dev=DEV_MODE,
        )

        # Apply multimodal task
        task = ICDLabsMIMIC4()
        samples = dataset.set_task(task)

        # Get and print sample
        sample = samples[0]
        print(sample)

    elif TASK == "NotesLabsMIMIC4":
        dataset = MIMIC4Dataset(
            ehr_root=ehr_root,
            note_root=note_root,
            ehr_tables=[
                "diagnoses_icd",
                "procedures_icd",
                "prescriptions",
                "labevents",
            ],
            note_tables=["discharge", "radiology"],
            cache_dir=cache_dir,
            num_workers=8,
            dev=DEV_MODE,
        )

        # Apply multimodal task
        task = NotesLabsMIMIC4()
        samples = dataset.set_task(task)

        # Get and print sample
        sample = samples[0]
        print(sample)

    elif TASK == "LabsMIMIC4":
        dataset = MIMIC4Dataset(
            ehr_root=ehr_root,
            ehr_tables=["labevents"],
            cache_dir=cache_dir,
            num_workers=8,
            dev=DEV_MODE,
        )

        # Apply multimodal task
        task = LabsMIMIC4()
        samples = dataset.set_task(task)

        # Get and print sample
        sample = samples[0]
        print(sample)

    elif TASK == "CXRMIMIC4":
        dataset = MIMIC4Dataset(
            ehr_root=ehr_root,
            cxr_root=cxr_root,
            cxr_variant="sunlab",
            cxr_tables=["metadata", "negbio"],
            cache_dir=cache_dir,
            num_workers=8,
            dev=DEV_MODE,
        )

        # Apply multimodal task
        task = CXRMIMIC4()
        samples = dataset.set_task(task)

        # Get and print sample
        sample = samples[0]
        print(sample)
