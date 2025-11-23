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
import random
import numpy as np
from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient,
    SampleDataset,
)
from pyhealth.datasets.utils import save_processors, load_processors
from pyhealth.models import StageNet
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.trainer import Trainer
import torch


def generate_holdout_set(
    sample_dataset: SampleDataset, num_samples: int = 10, seed: int = 42
) -> SampleDataset:
    """Generate synthetic hold-out set with unseen codes and varying lengths.

    This function creates synthetic samples to test the processor's ability to:
    1. Handle completely unseen tokens (mapped to <unk>)
    2. Handle sequence lengths larger than training but within padding

    Args:
        sample_dataset: Original SampleDataset with fitted processors
        num_samples: Number of synthetic samples to generate
        seed: Random seed for reproducibility

    Returns:
        SampleDataset with synthetic samples using fitted processors
    """
    random.seed(seed)
    np.random.seed(seed)

    # Get the fitted processors
    icd_processor = sample_dataset.input_processors["icd_codes"]

    # Get max nested length from ICD processor
    max_icd_len = icd_processor._max_nested_len
    # Handle both old and new processor versions
    padding = getattr(icd_processor, "_padding", 0)

    print("\n=== Hold-out Set Generation ===")
    print(f"Processor attributes: {dir(icd_processor)}")
    print(f"Has _padding attribute: {hasattr(icd_processor, '_padding')}")
    print(f"ICD max nested length: {max_icd_len}")
    print(f"Padding (via getattr): {padding}")
    if hasattr(icd_processor, "_padding"):
        print(f"Padding (direct access): {icd_processor._padding}")
    print(f"Observed max (without padding): {max_icd_len - padding}")

    synthetic_samples = []

    for i in range(num_samples):
        # Generate random number of visits (1-5)
        num_visits = random.randint(1, 5)

        # Generate ICD codes with unseen tokens
        icd_codes_list = []
        icd_times_list = []

        for visit_idx in range(num_visits):
            # Generate sequence length between observed_max and max_icd_len
            # This tests the padding capacity
            observed_max = max_icd_len - padding
            seq_len = random.randint(max(1, observed_max - 2), max_icd_len - 1)

            # Generate unseen codes
            visit_codes = [f"NEWCODE_{i}_{visit_idx}_{j}" for j in range(seq_len)]
            icd_codes_list.append(visit_codes)

            # Generate time intervals (hours from previous visit)
            if visit_idx == 0:
                icd_times_list.append(0.0)
            else:
                icd_times_list.append(random.uniform(24.0, 720.0))

        # Generate lab data (10-dimensional vectors)
        num_lab_timestamps = random.randint(5, 15)
        lab_values_list = []
        lab_times_list = []

        for ts_idx in range(num_lab_timestamps):
            # Generate 10D vector with some random values and some None
            lab_vector = []
            for dim in range(10):
                if random.random() < 0.8:  # 80% chance of value
                    lab_vector.append(random.uniform(50.0, 150.0))
                else:
                    lab_vector.append(None)

            lab_values_list.append(lab_vector)
            lab_times_list.append(random.uniform(0.0, 48.0))

        # Create sample in the expected format (before processing)
        synthetic_sample = {
            "patient_id": f"HOLDOUT_PATIENT_{i}",
            "icd_codes": (icd_times_list, icd_codes_list),
            "labs": (lab_times_list, lab_values_list),
            "mortality": random.randint(0, 1),
        }

        synthetic_samples.append(synthetic_sample)

    # Create a new SampleDataset with the FITTED processors
    holdout_dataset = SampleDataset(
        samples=synthetic_samples,
        input_schema=sample_dataset.input_schema,
        output_schema=sample_dataset.output_schema,
        dataset_name=f"{sample_dataset.dataset_name}_holdout",
        task_name=sample_dataset.task_name,
        input_processors=sample_dataset.input_processors,
        output_processors=sample_dataset.output_processors,
    )

    print(f"Generated {len(holdout_dataset)} synthetic samples")
    sample_seq_lens = [len(s["icd_codes"][1]) for s in synthetic_samples[:3]]
    print(f"Sample ICD sequence lengths: {sample_seq_lens}")
    sample_codes_per_visit = [
        [len(visit) for visit in s["icd_codes"][1]] for s in synthetic_samples[:3]
    ]
    print(f"Sample codes per visit: {sample_codes_per_visit}")

    return holdout_dataset


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
    # dev=True,
)

# STEP 2: Apply StageNet mortality prediction task with padding
#
# Processor Saving/Loading:
# - Processors are saved after the first run to avoid refitting
# - On subsequent runs, pre-fitted processors are loaded from disk
# - This ensures consistent encoding and saves computation time
# - Processors include vocabulary mappings and sequence length statistics
processor_dir = "../../output/processors/stagenet_mortality_mimic4"
cache_dir = "../../mimic4_stagenet_cache_v3"

if os.path.exists(os.path.join(processor_dir, "input_processors.pkl")):
    print("\n=== Loading Pre-fitted Processors ===")
    input_processors, output_processors = load_processors(processor_dir)

    sample_dataset = base_dataset.set_task(
        MortalityPredictionStageNetMIMIC4(padding=20),
        num_workers=1,
        cache_dir=cache_dir,
        input_processors=input_processors,
        output_processors=output_processors,
    )
else:
    print("\n=== Fitting New Processors ===")
    sample_dataset = base_dataset.set_task(
        MortalityPredictionStageNetMIMIC4(padding=20),
        num_workers=1,
        cache_dir=cache_dir,
    )

    # Save processors for future runs
    print("\n=== Saving Processors ===")
    save_processors(sample_dataset, processor_dir)

print(f"Total samples: {len(sample_dataset)}")
print(f"Input schema: {sample_dataset.input_schema}")
print(f"Output schema: {sample_dataset.output_schema}")

# Inspect a sample
sample = sample_dataset.samples[0]
print("\nSample structure:")
print(f"  Patient ID: {sample['patient_id']}")
print(f"ICD Codes: {sample['icd_codes']}")
print(f"  Labs shape: {len(sample['labs'][0])} timesteps")
print(f"  Mortality: {sample['mortality']}")

# STEP 3: Split dataset
train_dataset, val_dataset, test_dataset = split_by_patient(
    sample_dataset, [0.8, 0.1, 0.1]
)

# Create dataloaders
train_loader = get_dataloader(train_dataset, batch_size=256, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=256, shuffle=False)
test_loader = get_dataloader(test_dataset, batch_size=256, shuffle=False)

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
    device="cpu",  # or "cpu"
    metrics=["pr_auc", "roc_auc", "accuracy", "f1"],
)

trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=20,
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

# STEP 8: Test with synthetic hold-out set (unseen codes, varying lengths)
print("\n" + "=" * 60)
print("TESTING PROCESSOR ROBUSTNESS WITH SYNTHETIC HOLD-OUT SET")
print("=" * 60)

# Generate hold-out set with fitted processors
holdout_dataset = generate_holdout_set(
    sample_dataset=sample_dataset, num_samples=50, seed=42
)

# Create dataloader for hold-out set
holdout_loader = get_dataloader(holdout_dataset, batch_size=16, shuffle=False)

# Inspect processed samples
print("\n=== Inspecting Processed Hold-out Samples ===")
holdout_batch = next(iter(holdout_loader))

print(f"Batch size: {len(holdout_batch['patient_id'])}")
print(f"ICD codes tensor shape: {holdout_batch['icd_codes'][1].shape}")
print("ICD codes sample (first patient):")
print(f"  Time: {holdout_batch['icd_codes'][0][0][:5]}")
print(f"  Values (indices): {holdout_batch['icd_codes'][1][0][:3]}")

# Check for unknown tokens
icd_processor = sample_dataset.input_processors["icd_codes"]
unk_token_idx = icd_processor.code_vocab["<unk>"]
pad_token_idx = icd_processor.code_vocab["<pad>"]

print(f"\n<unk> token index: {unk_token_idx}")
print(f"<pad> token index: {pad_token_idx}")

# Count unknown and padding tokens in batch
icd_values = holdout_batch["icd_codes"][1]
num_unk = (icd_values == unk_token_idx).sum().item()
num_pad = (icd_values == pad_token_idx).sum().item()
total_tokens = icd_values.numel()

print("\nToken statistics in hold-out batch:")
print(f"  Total tokens: {total_tokens}")
print(f"  Unknown tokens: {num_unk} ({100*num_unk/total_tokens:.1f}%)")
print(f"  Padding tokens: {num_pad} ({100*num_pad/total_tokens:.1f}%)")

# Run model inference on hold-out set
print("\n=== Model Inference on Hold-out Set ===")
with torch.no_grad():
    holdout_output = model(**holdout_batch)

print(f"Predictions shape: {holdout_output['y_prob'].shape}")
print(f"Sample predictions: {holdout_output['y_prob'][:5]}")
print(f"True labels: {holdout_output['y_true'][:5]}")

print("\n" + "=" * 60)
print("HOLD-OUT SET TEST COMPLETED SUCCESSFULLY!")
print("Processors handled unseen codes and varying lengths correctly.")
print("=" * 60)

# STEP 9: Inspect saved processors
print("\n" + "=" * 60)
print("PROCESSOR INFORMATION")
print("=" * 60)
print(f"\nProcessors saved at: {processor_dir}")
print("\nICD Codes Processor:")
print(f"  {icd_processor}")
print(f"  Vocabulary size: {icd_processor.size()}")
print(f"  <unk> token index: {icd_processor.code_vocab['<unk>']}")
print(f"  <pad> token index: {icd_processor.code_vocab['<pad>']}")
print(f"  Max nested length: {icd_processor._max_nested_len}")
print(f"  Padding capacity: {getattr(icd_processor, '_padding', 0)}")

labs_processor = sample_dataset.input_processors["labs"]
print("\nLabs Processor:")
print(f"  {labs_processor}")
print(f"  Feature dimension: {labs_processor.size}")

print("\nTo reuse these processors in future runs:")
print("  1. Keep the processor_dir path the same")
print("  2. The script will automatically load them on next run")
print("  3. This ensures consistent encoding across experiments")
