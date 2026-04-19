"""
Simple MedLingo base demo for PyHealth.

This script demonstrates the contributed MedLingo dataset and the original
AbbreviationExpansionMedLingo task without any ablations or model training.

It shows:
1. Loading MedLingoDataset
2. Inspecting one patient and its events
3. Running the existing AbbreviationExpansionMedLingo task directly
4. Building a SampleDataset with dataset.set_task(task)

This is useful for a project demo/video because it shows the exact base
dataset + task functionality before any task-variation ablations.

This script demonstrates the base MedLingo dataset and task functionality only.
Quantitative model comparison is provided separately in 
medlingo_task_variation_ablation.py.
"""

import os
import tempfile
from pprint import pprint

from pyhealth.datasets.medlingo_dataset import MedLingoDataset
from pyhealth.tasks.medlingo_task import AbbreviationExpansionMedLingo


DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "test-resources",
    "MedLingo",
)


def main():
    questions_csv = os.path.join(DATA_ROOT, "questions.csv")
    if not os.path.isfile(questions_csv):
        raise FileNotFoundError(
            f"Could not find questions.csv under: {os.path.normpath(DATA_ROOT)}"
        )

    # Use a fresh cache dir so the demo is self-contained
    cache_dir = tempfile.mkdtemp(prefix="medlingo_base_demo_cache_")

    print("Loading MedLingoDataset...")
    dataset = MedLingoDataset(
        root=os.path.normpath(DATA_ROOT),
        cache_dir=cache_dir,
    )

    print(f"Loaded dataset with {len(dataset.unique_patient_ids)} patient IDs")
    print(f"First 5 patient IDs: {dataset.unique_patient_ids[:5]}")

    # ------------------------------------------------------------------
    # 1. Inspect one patient directly from the BaseDataset
    # ------------------------------------------------------------------
    first_patient_id = dataset.unique_patient_ids[0]
    patient = dataset.get_patient(first_patient_id)

    print("\nInspecting one patient from BaseDataset")
    print("--------------------------------------")
    print(f"patient_id: {patient.patient_id}")

    events = patient.get_events("questions")
    print(f"number of 'questions' events for this patient: {len(events)}")

    if events:
        first_event = events[0]
        print("\nFirst raw event fields:")
        print(f"question: {getattr(first_event, 'question', '')}")
        print(f"answer:   {getattr(first_event, 'answer', '')}")

    # ------------------------------------------------------------------
    # 2. Run the existing task directly on one patient
    # ------------------------------------------------------------------
    task = AbbreviationExpansionMedLingo()

    print("\nRunning AbbreviationExpansionMedLingo directly on one patient")
    print("-------------------------------------------------------------")
    task_samples = task(patient)
    print(f"number of task samples from this patient: {len(task_samples)}")

    if task_samples:
        print("\nFirst task sample:")
        pprint(task_samples[0])

    # ------------------------------------------------------------------
    # 3. Build the full SampleDataset through PyHealth
    # ------------------------------------------------------------------
    print("\nApplying dataset.set_task(task)...")
    sample_dataset = dataset.set_task(task)

    print(f"SampleDataset size: {len(sample_dataset)}")

    if len(sample_dataset) > 0:
        first_processed = sample_dataset[0]
        print("\nFirst processed sample keys:")
        print(list(first_processed.keys()))

        print("\nFirst processed sample preview:")
        preview = {}
        for key, value in first_processed.items():
            # keep preview readable for video/demo purposes
            if key in {"patient_id", "visit_id"}:
                preview[key] = value
            else:
                preview[key] = str(value)[:200]
        pprint(preview)

    print(
        "\nDone.\n"
        "This demo shows the base MedLingo dataset loading path and the original\n"
        "AbbreviationExpansionMedLingo task, both directly and through set_task()."
    )


if __name__ == "__main__":
    main()