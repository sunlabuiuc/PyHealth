"""Demo for catheter-associated infection prediction tasks on MIMIC-IV.

This script demonstrates how to:
1. Load MIMIC-IV EHR tables from a fixed dataset root
2. Apply either nested-sequence or StageNet catheter infection task
3. Inspect generated samples and label distribution
"""

from collections import Counter

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import (
    CatheterAssociatedInfectionPredictionMIMIC4,
    CatheterAssociatedInfectionPredictionStageNetMIMIC4,
)

EHR_ROOT = "/shared/rsaas/physionet.org/files/mimiciv/2.2"
CACHE_DIR = "/shared/eng/pyhealth_agent/cache"


def summarize_sample_dataset(sample_dataset, label_key: str = "label") -> None:
    """Print quick cohort summary and one sample preview."""
    print(f"Total samples: {len(sample_dataset)}")
    print(f"Input schema: {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")

    if len(sample_dataset) == 0:
        return

    labels = []
    for sample in sample_dataset:
        value = sample[label_key]
        if hasattr(value, "item"):
            labels.append(int(value.item()))
        else:
            labels.append(int(value))
    counts = Counter(labels)

    print("Label distribution:")
    print(f"  Negative (0): {counts.get(0, 0)}")
    print(f"  Positive (1): {counts.get(1, 0)}")

    first = sample_dataset[0]
    print("\nFirst sample keys:", list(first.keys()))
    print(f"Patient ID: {first['patient_id']}")

    if "conditions" in first:
        print(f"Condition visits: {len(first['conditions'])}")
        print(f"Procedure visits: {len(first['procedures'])}")
        print(f"Drug visits: {len(first['drugs'])}")
        print(f"Lab visits: {len(first['labs'])}")
    if "icd_codes" in first:
        print(f"ICD time points: {len(first['icd_codes'][0])}")
        print(f"ICD visit count: {len(first['icd_codes'][1])}")
        print(f"Lab time points: {len(first['labs'][0])}")


if __name__ == "__main__":
    base_dataset = MIMIC4Dataset(
        ehr_root=EHR_ROOT,
        cache_dir=CACHE_DIR,
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "prescriptions",
            "labevents",
        ],
    )

    print("\n=== Nested Sequence Variant ===")
    nested_task = CatheterAssociatedInfectionPredictionMIMIC4()
    nested_samples = base_dataset.set_task(nested_task, num_workers=4)
    summarize_sample_dataset(nested_samples)

    print("\n=== StageNet Variant ===")
    stagenet_task = CatheterAssociatedInfectionPredictionStageNetMIMIC4(padding=0)
    stagenet_samples = base_dataset.set_task(stagenet_task, num_workers=4)
    summarize_sample_dataset(stagenet_samples)
