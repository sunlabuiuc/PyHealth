"""Length-of-stay prediction on MIMIC-IV with StageNet.

This example demonstrates:
1. Loading MIMIC-IV (patients, admissions, diagnoses, procedures, labs)
2. Applying the LengthOfStayStageNetMIMIC4 task (one patient-level sample,
   10-class LOS category of the most recent admission; ICD codes and lab
   vectors with time intervals)
3. Saving the fitted processors and reloading them on later runs
4. Training and evaluating a StageNet model

Note: to prevent leakage, LengthOfStayStageNetMIMIC4 excludes the
diagnosis/procedure codes of the target admission (the one whose LOS is
the label) since they're only known at-or-after its own discharge, and
caps that admission's labs to the first TARGET_ADMISSION_INPUT_WINDOW_HOURS
(default 48) hours after admission instead of through discharge. Earlier,
already-resolved admissions are unaffected.

Data: MIMIC-IV 2.2 from PhysioNet (https://physionet.org/content/mimiciv/,
credentialed). Set MIMIC4_ROOT below to the folder that contains ``hosp/``,
and PROCESSOR_DIR / OUTPUT_DIR to writable folders.

Run: ``python examples/length_of_stay/length_of_stay_mimic4_stagenet.py``.
The full dataset with 30 epochs needs a GPU and hours. Without credentials,
set MIMIC4_ROOT to ``test-resources/core/mimic4demo`` and epochs=1 (only 5
samples; use split ratios like [0.4, 0.3, 0.3] so no split is empty).
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

MIMIC4_ROOT = "/path/to/mimiciv/2.2"
PROCESSOR_DIR = "/path/to/los_sn/processors"
OUTPUT_DIR = "/path/to/los_sn/output"

if __name__ == "__main__":
    # STEP 1: Load MIMIC-IV base dataset
    base_dataset = MIMIC4Dataset(
        ehr_root=MIMIC4_ROOT,
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
    processor_dir = PROCESSOR_DIR

    if os.path.exists(os.path.join(processor_dir, "input_processors.pkl")):
        print("\n=== Loading Pre-fitted Processors ===")
        input_processors, output_processors = load_processors(processor_dir)

        sample_dataset = base_dataset.set_task(
            LengthOfStayStageNetMIMIC4(padding=20),
            num_workers=16,
            input_processors=input_processors,
            output_processors=output_processors,
        )
    else:
        print("\n=== Fitting New Processors ===")
        sample_dataset = base_dataset.set_task(
            LengthOfStayStageNetMIMIC4(padding=20),
            num_workers=16,
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
        device="cuda" if torch.cuda.is_available() else "cpu",
        metrics=["accuracy", "f1_weighted", "f1_macro", "f1_micro"],
        output_path=OUTPUT_DIR,
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=30,
        monitor="accuracy",
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
