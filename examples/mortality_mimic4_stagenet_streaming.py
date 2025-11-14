"""
Example of using StageNet for mortality prediction on MIMIC-IV with STREAMING MODE.

This example demonstrates the new streaming mode for memory-efficient training:
1. Loading MIMIC-IV data in streaming mode (stream=True)
2. Applying the MortalityPredictionStageNetMIMIC4 task
3. Creating an IterableSampleDataset with disk-backed storage
4. Training a StageNet model with streaming data

Key differences from non-streaming mode:
- stream=True: Data loaded from disk on-demand
- IterableSampleDataset: Samples not stored in memory
- No random shuffling (sequential iteration only)
- Much lower memory footprint
- Ideal for large datasets (>100k samples)

Note: IterableDataset is fully compatible with PyTorch DataLoader and Trainer!
"""

from pyhealth.datasets import (
    MIMIC4Dataset,
    get_dataloader,
    split_by_patient_stream,
)
from pyhealth.models import StageNet
from pyhealth.tasks import MortalityPredictionStageNetMIMIC4
from pyhealth.trainer import Trainer
import torch
import psutil
import os


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def print_memory_stats(stage_name):
    """Print current memory usage statistics."""
    mem_mb = get_memory_usage()
    print(f"[Memory] {stage_name}: {mem_mb:.1f} MB")
    return mem_mb


# Track memory at start
initial_memory = print_memory_stats("Initial")

# STEP 1: Load MIMIC-IV base dataset in STREAMING MODE
print("=" * 70)
print("STREAMING MODE: Memory-Efficient Dataset Processing")
print("=" * 70)

base_dataset = MIMIC4Dataset(
    ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
    ehr_tables=[
        "patients",
        "admissions",
        "diagnoses_icd",
        "procedures_icd",
        "labevents",
    ],
    stream=True,  # ⭐ Enable streaming mode for memory efficiency
    cache_dir="../../mimic4_streaming_cache",  # Disk-backed cache
    dev=True,  # Set to True for quick testing with limited patients
    dev_max_patients=1000,  # Only used if dev=True
)

print(f"Dataset mode: STREAMING (disk-backed)")
print(f"Cache directory: {base_dataset.cache_dir}")
print_memory_stats("After loading base dataset")

# STEP 2: Apply StageNet mortality prediction task in streaming mode
print("\n" + "=" * 70)
print("Applying Task with Streaming Sample Generation")
print("=" * 70)

sample_dataset = base_dataset.set_task(
    MortalityPredictionStageNetMIMIC4(),
    batch_size=256,  # ⭐ Process patients in batches for I/O efficiency
    cache_dir="../../mimic4_stagenet_streaming_cache",
)

print(f"Dataset type: {type(sample_dataset).__name__}")
print(f"Total samples: {len(sample_dataset)}")
print(f"Input schema: {sample_dataset.input_schema}")
print(f"Output schema: {sample_dataset.output_schema}")
print_memory_stats("After applying task (samples on disk)")

# Inspect a sample (IterableDataset requires iteration)
print("\nSample structure (from first sample):")
for i, sample in enumerate(sample_dataset):
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"  ICD Codes: {sample['icd_codes']}")
    print(f"  Labs shape: {len(sample['labs'][0])} timesteps")
    print(f"  Mortality: {sample['mortality']}")
    if i == 0:  # Only show first sample
        break

# STEP 3: Create train/val/test splits
# Note: For IterableDataset, we need to manually split the data
# We'll use patient IDs to create separate dataset instances
print("\n" + "=" * 70)
print("Creating Train/Val/Test Splits")
print("=" * 70)

# For simplicity in this example, we'll use the full dataset for training
# In production, you'd want to split patient IDs and filter accordingly
# or use the split_by_patient function with appropriate modifications

# Create dataloaders
# ⭐ Note: shuffle=False for IterableDataset (no random access)
train_loader = get_dataloader(sample_dataset, batch_size=256, shuffle=False)

print(f"Train loader created (batch_size=256)")
print(f"Note: IterableDataset uses sequential iteration (no shuffling)")
print_memory_stats("After creating dataloader")

# For validation, we'd need a separate filtered dataset
# For this demo, we'll skip validation and just show training

# STEP 4: Initialize StageNet model
print("\n" + "=" * 70)
print("Initializing StageNet Model")
print("=" * 70)

model = StageNet(
    dataset=sample_dataset,
    embedding_dim=128,
    chunk_size=128,
    levels=3,
    dropout=0.3,
)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model initialized with {num_params:,} parameters")
print_memory_stats("After model initialization")

# STEP 5: Train the model with streaming data
print("\n" + "=" * 70)
print("Training with Streaming Data")
print("=" * 70)

trainer = Trainer(
    model=model,
    device="cuda:5",  # or "cpu"
    metrics=["pr_auc", "roc_auc", "accuracy", "f1"],
)

# ⭐ Training works the same way with IterableDataset!
# PyTorch's DataLoader handles IterableDataset automatically
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=None,  # Would need separate validation dataset
    epochs=10,  # Fewer epochs for demo
    monitor="roc_auc",
    optimizer_params={"lr": 1e-5},
)

print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
final_memory = print_memory_stats("After training")
print(f"Total memory increase: {final_memory - initial_memory:.1f} MB")

# STEP 6: Evaluate on test set (would need separate test dataset)
print("\nFor full train/val/test workflow with streaming mode:")
print("1. Split patient IDs into train/val/test sets")
print("2. Create 3 separate IterableSampleDataset instances")
print("3. Filter each dataset to its respective patient IDs")
print("4. Create DataLoaders for each split")
print("5. Train with train_loader, validate with val_loader, test with test_loader")

# STEP 7: Show memory benefits
print("\n" + "=" * 70)
print("Memory Benefits of Streaming Mode")
print("=" * 70)
print("✓ Samples stored on disk (not in RAM)")
print("✓ Only active batch loaded in memory")
print("✓ Memory usage independent of dataset size")
print("✓ Ideal for datasets >100k samples")
print("✓ Enables training on massive datasets with limited RAM")
print(f"\nMemory Usage Summary:")
print(f"  Initial: {initial_memory:.1f} MB")
print(f"  Final:   {final_memory:.1f} MB")
print(f"  Increase: {final_memory - initial_memory:.1f} MB")
print(
    f"  Note: Most memory used by model parameters "
    f"({num_params * 4 / 1024 / 1024:.1f} MB)"
)

# STEP 8: Inspect model predictions
print("\n" + "=" * 70)
print("Sample Predictions")
print("=" * 70)

sample_batch = next(iter(train_loader))
with torch.no_grad():
    output = model(**sample_batch)

print(f"Predicted probabilities (first 5): {output['y_prob'][:5]}")
print(f"True labels (first 5): {output['y_true'][:5]}")

print("\n" + "=" * 70)
print("Streaming Mode Demo Complete!")
print("=" * 70)
