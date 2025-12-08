"""Example: 30-day readmission prediction on MIMIC-IV using PyHealth.

This script demonstrates how to load the MIMIC-IV dataset, apply the
`Readmission30DaysMIMIC4` task, split the sample dataset by patients,
and inspect a batch of features.

The example uses `dev=True` to keep processing lightweight. Disable `dev`
for full dataset training.

Usage:
    python run_mimic4_readmission.py
"""

from __future__ import annotations

import torch
from pyhealth.datasets import MIMIC4Dataset, split_by_patient, get_dataloader
from pyhealth.tasks import Readmission30DaysMIMIC4


def main() -> None:
    """Runs the 30-day readmission prediction example.

    Loads the MIMIC-IV dataset, generates a sample dataset using the
    built-in readmission task, splits the dataset into train/val/test sets,
    and prints the structure of a single training batch.
    """
    data_dir = "/srv/local/data/physionet.org/mimiciv/2.2"

    dataset = MIMIC4Dataset(
        ehr_root=data_dir,
        ehr_tables=[
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "prescriptions",
        ],
        dev=True,
    )

    print(f"Loaded patients: {len(dataset.unique_patient_ids)}")

    task = Readmission30DaysMIMIC4()

    sample_dataset = dataset.set_task(task)
    print(f"Generated readmission samples: {len(sample_dataset)}")

    if len(sample_dataset) == 0:
        raise RuntimeError(
            "No samples generated — verify table paths, dataset version, "
            "and that dev mode is not too restrictive."
        )

    train_ds, val_ds, test_ds = split_by_patient(
        sample_dataset, [0.7, 0.1, 0.2]
    )

    print(f"Train: {len(train_ds)}")
    print(f"Val: {len(val_ds)}")
    print(f"Test: {len(test_ds)}")

    train_loader = get_dataloader(
        train_ds,
        batch_size=16,
        shuffle=True,
    )

    batch = next(iter(train_loader))

    print("Batch keys:", batch.keys())
    print("Conditions shape:", batch["conditions"].shape)
    print("Procedures shape:", batch["procedures"].shape)
    print("Drugs shape:", batch["drugs"].shape)


if __name__ == "__main__":
    main()
