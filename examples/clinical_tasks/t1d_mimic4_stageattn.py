"""StageAttentionNet for DKA prediction for type 1 diabetes patients on MIMIC-IV.

Builds ``T1DDKAPredictionMIMIC4`` samples (T1DM patients only; label 1 if DKA
occurs within 90 days of the T1DM diagnosis) from diagnoses, procedures and lab
events, trains a StageAttentionNet model, evaluates it on a patient-level test
split, and saves the fitted processors so later runs reuse the same encoding.

Data: MIMIC-IV v2.2 hosp tables (credentialed PhysioNet access:
https://physionet.org/content/mimiciv/2.2/).
Run: set ``ehr_root``, ``processor_dir`` and ``output_path`` below, then
    python examples/clinical_tasks/t1d_mimic4_stageattn.py
Full MIMIC-IV with 30 epochs needs a GPU and a few hours. To try it without
credentials, set ``ehr_root`` to ``test-resources/core/mimic4demo``, but the
demo has only 2 T1DM patients, so the 80/10/10 split leaves validation empty
and training stops there; train and evaluate on all samples to check the
pipeline.
"""

import os
from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.datasets.utils import save_processors, load_processors
from pyhealth.models import StageAttentionNet
from pyhealth.tasks import T1DDKAPredictionMIMIC4
from pyhealth.trainer import Trainer
import torch

if __name__ == "__main__":
    # STEP 1: Load MIMIC-IV base dataset
    base_dataset = MIMIC4Dataset(
        ehr_root="/path/to/mimiciv/2.2/",
        ehr_tables=[
            "patients",
            "admissions",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
        ],
        num_workers=16,
    )

    # STEP 2: Apply StageNet DKA prediction task with padding
    #
    # Processor Saving/Loading:
    # - Processors are saved after the first run to avoid refitting
    # - On subsequent runs, pre-fitted processors are loaded from disk
    # - This ensures consistent encoding and saves computation time
    # - Processors include vocabulary mappings and sequence length statistics
    processor_dir = "/path/to/t1d_sa/processors"

    if os.path.exists(os.path.join(processor_dir, "input_processors.pkl")):
        print("\n=== Loading Pre-fitted Processors ===")
        input_processors, output_processors = load_processors(processor_dir)

        sample_dataset = base_dataset.set_task(
            T1DDKAPredictionMIMIC4(padding=20),
            num_workers=16,
            input_processors=input_processors,
            output_processors=output_processors,
        )
    else:
        print("\n=== Fitting New Processors ===")
        sample_dataset = base_dataset.set_task(
            T1DDKAPredictionMIMIC4(padding=20),
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
    print(f"  T1D DKA: {sample['label']}")

    # STEP 3: Split dataset
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )

    # Create dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

    # STEP 4: Initialize StageNet model
    model = StageAttentionNet(
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
        metrics=["pr_auc", "roc_auc", "accuracy", "f1"],
        output_path="/path/to/t1d_sa/output"
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
