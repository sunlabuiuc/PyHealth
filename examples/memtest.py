# %%
from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import StageNet
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.trainer import Trainer
import torch
import dask.config
from dask.distributed import Client, LocalCluster

if __name__ == "__main__":
    dask.config.set({"temporary-directory": "/mnt/tmpfs/"})
    with LocalCluster(
        n_workers=16,
        threads_per_worker=1,
        memory_limit="8GB",
    ) as cluster:
        with Client(cluster) as client:
            # STEP 1: Load MIMIC-IV base dataset
            base_dataset = MIMIC4Dataset(
                ehr_root="/home/logic/physionet.org/files/mimiciv/3.1",
                ehr_tables=[
                    "patients",
                    "admissions",
                    "diagnoses_icd",
                    "procedures_icd",
                    "labevents",
                ],
                dev=True
            )
            base_dataset._merged_cache()

    print(f"Patients: {base_dataset.unique_patient_ids[:10]}, ...")

    # STEP 2: Apply StageNet mortality prediction task
    sample_dataset = base_dataset.set_task(
        MortalityPredictionStageNetMIMIC4(),
        num_workers=4,
    )
    print(f"Total samples: {len(sample_dataset)}")
    print(f"Input schema: {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")

    # Inspect a sample
    sample = next(iter(sample_dataset))
    print("\nSample structure:")
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"ICD Codes: {sample['icd_codes']}")
    print(f"  Labs shape: {len(sample['labs'][0])} timesteps")
    print(f"  Mortality: {sample['mortality']}")

    # Create dataloaders
    train_loader = get_dataloader(sample_dataset, batch_size=256, shuffle=True)

    # STEP 4: Initialize StageNet model
    model = StageNet(
        dataset=sample_dataset,
        embedding_dim=128,
        chunk_size=128,
        levels=3,
        dropout=0.3,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized with {num_params} parameters")

    # STEP 5: Train the model
    trainer = Trainer(
        model=model,
        device="cuda:5",  # or "cpu"
        metrics=["pr_auc", "roc_auc", "accuracy", "f1"],
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=train_loader,
        epochs=50,
        monitor="roc_auc",
        optimizer_params={"lr": 1e-5},
    )
