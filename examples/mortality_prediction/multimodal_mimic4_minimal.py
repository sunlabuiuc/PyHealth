"""Minimal PyHealth Multimodal MIMIC-IV Demo - Explore all data modalities."""

import os
from pathlib import Path

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import MultimodalMortalityPredictionMIMIC4

# Paths
EHR_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.2"
NOTE_ROOT = "/srv/local/data/MIMIC-IV/2.0/"
CXR_ROOT = "/srv/local/data/MIMIC-CXR"
CACHE_DIR = "/tmp/pyhealth_mm_minimal/"

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
)

# Apply multimodal task
task = MultimodalMortalityPredictionMIMIC4(cxr_root=CXR_ROOT)
samples = dataset.set_task(task, cache_dir=f"{CACHE_DIR}/task")

# Get sample
sample = samples.samples[0]

# EHR Codes
conditions = sample["conditions"]
procedures = sample["procedures"]
drugs = sample["drugs"]

# Clinical Notes
discharge = sample.get("discharge", "")
radiology = sample.get("radiology", "")

# Lab Events (time-series)
labs = sample.get("labs")  # (times, values) tuple

# X-Ray Data
xray_findings = sample.get("xrays_negbio", [])
image_path = sample.get("image")
if image_path:
    full_path = os.path.join(CXR_ROOT, image_path)
    if not os.path.exists(full_path):
        dicom_id = Path(image_path).stem
        full_path = os.path.join(CXR_ROOT, "images", f"{dicom_id}.jpg")

# Label
mortality = sample.get("mortality")
