"""
Example of using StageNet for mortality prediction on MIMIC-IV.

This example demonstrates:
1. Loading MIMIC-IV data
2. Applying the MortalityPredictionStageNetMIMIC4 task
3. Creating a SampleDataset with StageNet processors
4. Training a StageNet model
5. Testing with synthetic hold-out set (unseen codes, varying lengths)
"""

import os
from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.datasets.utils import save_processors, load_processors
from pyhealth.models import StageNet
from pyhealth.tasks import LengthOfStayStageNetMIMIC4
from pyhealth.trainer import Trainer
import torch

if __name__ == "__main__":
    # STEP 1: Load MIMIC-IV base dataset
    base_dataset = MIMIC4Dataset(
        ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
        num_workers=16,
    )

    # STEP 2: Apply StageNet length of stay prediction task with padding
    #
    # Processor Saving/Loading:
    # - Processors are saved after the first run to avoid refitting
    # - On subsequent runs, pre-fitted processors are loaded from disk
    # - This ensures consistent encoding and saves computation time
    # - Processors include vocabulary mappings and sequence length statistics
    processor_dir = "/home/yongdaf2/los_sn/processors"
    cache_dir = "/home/yongdaf2/los_sn/cache"

    if os.path.exists(os.path.join(processor_dir, "input_processors.pkl")):
        print("\n=== Loading Pre-fitted Processors ===")
        input_processors, output_processors = load_processors(processor_dir)

        sample_dataset = base_dataset.set_task(
            LengthOfStayStageNetMIMIC4(padding=20),
            num_workers=16,
            cache_dir=cache_dir,
            input_processors=input_processors,
            output_processors=output_processors,
        )
    else:
        print("\n=== Fitting New Processors ===")
        sample_dataset = base_dataset.set_task(
            LengthOfStayStageNetMIMIC4(padding=20),
            num_workers=16,
            cache_dir=cache_dir,
        )

        # Save processors for future runs
        print("\n=== Saving Processors ===")
        save_processors(sample_dataset, processor_dir)

    print(f"Total samples: {len(sample_dataset)}")
    print(f"Input schema: {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")

    # Inspect a sample
    sample = sample_dataset[0]
    print("\nSample structure:")
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"ICD Codes: {sample['icd_codes']}")
    print(f"  Labs shape: {len(sample['labs'][0])} timesteps")
    print(f"  Length of Stay: {sample['los']}")

    # STEP 3: Split dataset
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    # Create dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

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
        device="cuda:6",  # or "cpu"
        metrics=["accuracy", "f1_weighted", "f1_macro", "f1_micro"],
        output_path="/home/yongdaf2/los_sn/output"
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=30,
        monitor="roc_auc",
        optimizer_params={"lr": 1e-5},
    )

    # STEP 6: Evaluate on test set
    results = trainer.evaluate(test_loader)
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # STEP 7: Inspect model predictions
    sample_batch = next(iter(test_loader))
    with torch.no_grad():
        output = model(**sample_batch)

    print("\nSample predictions:")
    print(f"  Predicted probabilities: {output['y_prob'][:5]}")
    print(f"  True labels: {output['y_true'][:5]}")
