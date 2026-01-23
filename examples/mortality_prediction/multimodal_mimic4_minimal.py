"""Minimal PyHealth Multimodal MIMIC-IV Demo - Explore all data modalities."""

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import MultimodalMortalityPredictionMIMIC4

# Paths
EHR_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.2"
NOTE_ROOT = "/srv/local/data/MIMIC-IV/2.0/"
CXR_ROOT = "/srv/local/data/MIMIC-CXR"
CACHE_DIR = "/srv/local/data/REDACTED_USER/pyhealth_cache"


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

    # Apply multimodal task
    task = MultimodalMortalityPredictionMIMIC4()
    samples = dataset.set_task(task, cache_dir=f"{CACHE_DIR}/task", num_workers=8)

    # Get and print sample
    sample = samples[0]
    print(sample)


