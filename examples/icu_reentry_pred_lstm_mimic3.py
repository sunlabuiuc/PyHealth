"""
PyHealth LSTM Example: ICU Re-entry Prediction with Clinical Features
=====================================================================
This script demonstrates how to:
  1. Generate synthetic ICU stay data (mimicking MIMIC-III structure)
  2. Apply the ICUReEntryClassification task using from_arrays()
  3. Build a SampleDataset with create_sample_dataset()
  4. Train an LSTM model for 7-day ICU re-entry prediction
  5. Evaluate on a held-out test set

The ICU re-entry task (Nestor et al. 2019) predicts whether a patient will
return to the ICU within 7 days of their current ICU episode's end, using
the first 24 hours of hourly vitals and labs as input.

In a real workflow, the (24, 65) vitals arrays come from MIMIC_Extract's
all_hourly_data.h5 after applying clinical aggregation to map LEVEL2 columns
to the 65 expert-defined feature categories.

Requirements:
    pip install pyhealth torch

References:
    - Nestor et al. 2019: https://proceedings.mlr.press/v106/nestor19a.html
    - PyHealth docs: https://pyhealth.readthedocs.io
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import random
import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, split_by_patient, get_dataloader
from pyhealth.tasks.mimic3_icu_reentry import ICUReEntryClassification
from pyhealth.models import RNN
from pyhealth.trainer import Trainer

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. Generate Synthetic ICU Data ──────────────────────────────────────────
# Simulate 200 ICU stays across 100 patients, each with a (24, 65) vitals
# tensor representing 24 hours of hourly measurements across 65 clinical
# feature categories (clinical aggregation from Nestor et al. 2019).
#
# For real MIMIC-III data, replace this section with:
#   dataset = MIMIC3Dataset(root="/path/to/mimic3", tables=[...])
#   followed by the MIMICExtract pipeline + apply_clinical_aggregation().

print("=" * 60)
print("Step 1 – Generating Synthetic ICU Stay Data")
print("=" * 60)

N_PATIENTS = 100
N_STAYS    = 200
N_HOURS    = 24
N_FEATURES = 65   # 65-category clinical aggregation (Nestor et al. 2019)

rng = np.random.default_rng(SEED)

# (N_STAYS, 24, 65) float32 vitals tensors
features = rng.standard_normal((N_STAYS, N_HOURS, N_FEATURES)).astype(np.float32)

# Patient assignments (multi-stay patients are common in MIMIC-III)
patient_ids = rng.integers(1001, 1001 + N_PATIENTS, size=N_STAYS).tolist()
stay_ids    = list(range(200_001, 200_001 + N_STAYS))

# Synthetic 7-day re-entry labels (~25% positive rate)
labels = rng.choice([0, 1], size=N_STAYS, p=[0.75, 0.25]).tolist()

n_pos = sum(labels)
print(f"  Stays            : {N_STAYS}")
print(f"  Unique patients  : {len(set(patient_ids))}")
print(f"  Positives        : {n_pos}  ({100 * n_pos / N_STAYS:.0f}%)")
print(f"  Feature shape    : ({N_HOURS}, {N_FEATURES})")

# ─── 2. Apply ICU Re-entry Task ───────────────────────────────────────────────
# ICUReEntryClassification.from_arrays() converts numpy arrays into PyHealth
# sample dicts.  Each dict contains:
#   - patient_id (str), visit_id (str)
#   - vitals_labs: np.ndarray of shape (24, 65)
#   - reentry_7day: int (0 or 1)

print("\n" + "=" * 60)
print("Step 2 – Applying ICU Re-entry Task")
print("=" * 60)

task = ICUReEntryClassification(feature_set="clinical")
samples = task.from_arrays(
    features    = features,
    labels      = np.array(labels),
    stay_ids    = stay_ids,
    patient_ids = patient_ids,
)
print(f"  Total samples  : {len(samples)}")
print(f"  Sample keys    : {list(samples[0].keys())}")
print(f"  vitals_labs    : shape {samples[0]['vitals_labs'].shape}")
print(f"  reentry_7day   : {samples[0]['reentry_7day']}")

# ─── 3. Build SampleDataset ───────────────────────────────────────────────────
# create_sample_dataset() fits feature processors (TensorProcessor for
# vitals_labs, BinaryProcessor for reentry_7day) and returns an
# InMemorySampleDataset ready for model training.

print("\n" + "=" * 60)
print("Step 3 – Creating SampleDataset")
print("=" * 60)

dataset = create_sample_dataset(
    samples       = samples,
    input_schema  = task.input_schema,    # {"vitals_labs": "tensor"}
    output_schema = task.output_schema,   # {"reentry_7day": "binary"}
    dataset_name  = "mimic3_icu_reentry_synthetic",
    task_name     = task.task_name,
)
print(f"  Dataset size   : {len(dataset)}")
print(f"  Input schema   : {dataset.input_schema}")
print(f"  Output schema  : {dataset.output_schema}")

# ─── 4. Train / Val / Test Split ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4 – Splitting Dataset (70 / 10 / 20)")
print("=" * 60)

train_ds, val_ds, test_ds = split_by_patient(
    dataset, [0.70, 0.10, 0.20], seed=SEED
)
print(f"  Train : {len(train_ds)} | Val : {len(val_ds)} | Test : {len(test_ds)}")

train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
val_loader   = get_dataloader(val_ds,   batch_size=32, shuffle=False)
test_loader  = get_dataloader(test_ds,  batch_size=32, shuffle=False)

# ─── 5. Build LSTM Model ─────────────────────────────────────────────────────
# RNN derives feature_keys and label_key from the dataset schemas.
# For vitals_labs of shape (24, 65), EmbeddingModel adds a Linear(65 →
# embedding_dim) projection before the LSTM sees (B, 24, embedding_dim).

print("\n" + "=" * 60)
print("Step 5 – Building LSTM Model")
print("=" * 60)

model = RNN(
    dataset       = dataset,
    embedding_dim = 128,
    hidden_dim    = 256,
    rnn_type      = "LSTM",
    num_layers    = 2,
    dropout       = 0.3,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Feature        : vitals_labs  ({N_HOURS}h × {N_FEATURES} features)")
print(f"  Embedding dim  : 128")
print(f"  Hidden dim     : 256")
print(f"  Layers         : 2")
print(f"  Total params   : {total_params:,}")

# ─── 6. Train with PyHealth Trainer ───────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6 – Training LSTM")
print("=" * 60)

trainer = Trainer(
    model   = model,
    metrics = ["pr_auc", "roc_auc", "f1"],
)

trainer.train(
    train_dataloader = train_loader,
    val_dataloader   = val_loader,
    epochs           = 10,
    optimizer_params = {"lr": 1e-3},
    monitor          = "roc_auc",
)

# ─── 7. Evaluate on Test Set ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 7 – Evaluating on Test Set")
print("=" * 60)

results = trainer.evaluate(test_loader)
print(f"  ROC-AUC : {results.get('roc_auc', 'N/A'):.4f}")
print(f"  PR-AUC  : {results.get('pr_auc',  'N/A'):.4f}")
print(f"  F1      : {results.get('f1',      'N/A'):.4f}")

# ─── 8. Single-Sample Inference Demo ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 8 – Single-Sample Inference Demo")
print("=" * 60)

model.eval()
sample_batch = next(iter(test_loader))
with torch.no_grad():
    output = model(**sample_batch)

y_prob  = output["y_prob"].squeeze(-1)
y_true  = output["y_true"].squeeze(-1)
for i in range(min(3, len(y_prob))):
    prob  = y_prob[i].item()
    truth = int(y_true[i].item())
    print(
        f"  Sample {i+1}: P(re-entry within 7d) = {prob:.3f}  |  "
        f"True = {truth}  |  "
        f"Pred = {'HIGH risk' if prob > 0.5 else 'LOW  risk'}"
    )

print("\nDone!")
