"""Minimal multimodal MIMIC-IV loading: EHR, clinical notes and chest X-rays.

Loads MIMIC-IV EHR tables together with MIMIC-IV-Note (discharge and
radiology notes) and MIMIC-CXR (image metadata and NegBio findings) into one
MIMIC4Dataset. It then applies MultimodalMortalityPredictionMIMIC4 (one sample
per patient, labelled with death in a later admission) and prints the first
sample. No model is trained. For a fuller walkthrough, see
multimodal_mimic4_demo.py.

Data (all credentialed, from PhysioNet): MIMIC-IV v2.2
(https://physionet.org/content/mimiciv/2.2/), MIMIC-IV-Note v2.2
(https://physionet.org/content/mimic-iv-note/2.2/) and MIMIC-CXR-JPG v2.0.0
(https://physionet.org/content/mimic-cxr-jpg/2.0.0/), plus the MIMIC-CXR
NegBio labels. Set EHR_ROOT, NOTE_ROOT, CXR_ROOT and CACHE_DIR below. The task
needs all three sources.

Usage:
    python examples/mortality_prediction/multimodal_mimic4_minimal.py
"""

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import MultimodalMortalityPredictionMIMIC4

# Paths: point these at your copies
EHR_ROOT = "/path/to/mimiciv/2.2"
NOTE_ROOT = "/path/to/mimic-iv-note/2.2"
CXR_ROOT = "/path/to/mimic-cxr-jpg"
CACHE_DIR = "/path/to/cache"


if __name__ == "__main__":
    # Load multimodal dataset
    dataset = MIMIC4Dataset(
        ehr_root=EHR_ROOT,
        ehr_tables=["patients", "admissions", "diagnoses_icd",
                    "procedures_icd", "prescriptions", "labevents"],
        note_root=NOTE_ROOT,
        note_tables=["discharge", "radiology"],
        cxr_root=CXR_ROOT,
        cxr_tables=["metadata", "negbio"],
        cache_dir=CACHE_DIR,
        num_workers=8
    )

    # Apply multimodal task with X-rays available by prediction time
    task = MultimodalMortalityPredictionMIMIC4()
    samples = dataset.set_task(task, num_workers=8)

    # Get and print sample
    sample = samples[0]
    print(sample)

