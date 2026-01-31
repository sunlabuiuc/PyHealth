"""
Conformal Prediction for COVID-19 Chest X-Ray Classification.

This example demonstrates:
1. Training a ResNet-18 model on COVID-19 CXR dataset
2. Conventional conformal prediction using LABEL
3. Covariate shift adaptive conformal prediction using CovariateLabel
4. Comparison of coverage and efficiency between the two methods
"""

import numpy as np
import torch

from pyhealth.calib.predictionset import LABEL
from pyhealth.calib.predictionset.covariate import CovariateLabel
from pyhealth.calib.utils import extract_embeddings
from pyhealth.datasets import (
    COVID19CXRDataset,
    get_dataloader,
    split_by_sample_conformal,
)
from pyhealth.models import TorchvisionModel
from pyhealth.trainer import Trainer, get_metrics_fn

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# STEP 1: Load and prepare dataset
# ============================================================================
print("=" * 80)
print("STEP 1: Loading COVID-19 CXR Dataset")
print("=" * 80)

root = "/home/johnwu3/projects/PyHealth_Branch_Testing/datasets/COVID-19_Radiography_Dataset"
base_dataset = COVID19CXRDataset(root)
sample_dataset = base_dataset.set_task(cache_dir="../../covid19cxr_cache")

print(f"Total samples: {len(sample_dataset)}")
print(f"Task mode: {sample_dataset.output_schema}")

# Split into train/val/cal/test
# For conformal prediction, we need a separate calibration set
train_data, val_data, cal_data, test_data = split_by_sample_conformal(
    dataset=sample_dataset, ratios=[0.6, 0.1, 0.15, 0.15]
)

print(f"Train: {len(train_data)}")
print(f"Val: {len(val_data)}")
print(f"Cal: {len(cal_data)} (for conformal calibration)")
print(f"Test: {len(test_data)}")

# Create data loaders
train_loader = get_dataloader(train_data, batch_size=32, shuffle=True)
val_loader = get_dataloader(val_data, batch_size=32, shuffle=False)
cal_loader = get_dataloader(cal_data, batch_size=32, shuffle=False)
test_loader = get_dataloader(test_data, batch_size=32, shuffle=False)

# ============================================================================
# STEP 2: Train ResNet-18 model
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Training ResNet-18 Model")
print("=" * 80)

# Initialize ResNet-18 with pretrained weights
resnet = TorchvisionModel(
    dataset=sample_dataset,
    model_name="resnet18",
    model_config={"weights": "DEFAULT"},
)

# Train the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
trainer = Trainer(model=resnet, device=device)

print(f"Training on device: {device}")
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=5,
    monitor="accuracy",
)

print("✓ Model training completed")

# Evaluate base model on test set
print("\nBase model performance on test set:")
y_true_base, y_prob_base, loss_base = trainer.inference(test_loader)
base_metrics = get_metrics_fn("multiclass")(
    y_true_base, y_prob_base, metrics=["accuracy", "f1_weighted"]
)
for metric, value in base_metrics.items():
    print(f"  {metric}: {value:.4f}")

# ============================================================================
# STEP 3: Conventional Conformal Prediction with LABEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Conventional Conformal Prediction (LABEL)")
print("=" * 80)

# Target miscoverage rate of 10% (90% coverage)
alpha = 0.1
print(f"Target miscoverage rate: {alpha} (90% coverage)")

# Create LABEL predictor
label_predictor = LABEL(model=resnet, alpha=alpha)

# Calibrate on calibration set
print("Calibrating LABEL predictor...")
label_predictor.calibrate(cal_dataset=cal_data)

# Evaluate on test set
print("Evaluating LABEL predictor on test set...")
y_true_label, y_prob_label, _, extra_label = Trainer(model=label_predictor).inference(
    test_loader, additional_outputs=["y_predset"]
)

label_metrics = get_metrics_fn("multiclass")(
    y_true_label,
    y_prob_label,
    metrics=["accuracy", "miscoverage_ps"],
    y_predset=extra_label["y_predset"],
)

# Calculate average set size
predset_label = (
    torch.tensor(extra_label["y_predset"])
    if isinstance(extra_label["y_predset"], np.ndarray)
    else extra_label["y_predset"]
)
avg_set_size_label = predset_label.float().sum(dim=1).mean().item()

# Extract scalar values from metrics (handle both scalar and array returns)
miscoverage_label = label_metrics["miscoverage_ps"]
if isinstance(miscoverage_label, np.ndarray):
    miscoverage_label = float(
        miscoverage_label.item()
        if miscoverage_label.size == 1
        else miscoverage_label.mean()
    )
else:
    miscoverage_label = float(miscoverage_label)

print("\nLABEL Results:")
print(f"  Accuracy: {label_metrics['accuracy']:.4f}")
print(f"  Empirical miscoverage: {miscoverage_label:.4f}")
print(f"  Average set size: {avg_set_size_label:.2f}")
print(f"  Target miscoverage: {alpha:.2f}")

# ============================================================================
# STEP 4: Covariate Shift Adaptive Conformal Prediction
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Covariate Shift Adaptive Conformal Prediction")
print("=" * 80)

# Extract embeddings from the model
# For TorchvisionModel, we extract features from avgpool layer (before fc)
print("Extracting embeddings from calibration set...")
cal_embeddings = extract_embeddings(resnet, cal_data, batch_size=32, device=device)
print(f"  Cal embeddings shape: {cal_embeddings.shape}")

print("Extracting embeddings from test set...")
test_embeddings = extract_embeddings(resnet, test_data, batch_size=32, device=device)
print(f"  Test embeddings shape: {test_embeddings.shape}")

# Create CovariateLabel predictor
print("\nCreating CovariateLabel predictor...")
covariate_predictor = CovariateLabel(model=resnet, alpha=alpha)

# Calibrate with embeddings (KDEs will be fitted automatically)
print("Calibrating CovariateLabel predictor...")
print("  - Fitting KDEs for covariate shift correction...")
covariate_predictor.calibrate(
    cal_dataset=cal_data, cal_embeddings=cal_embeddings, test_embeddings=test_embeddings
)
print("✓ Calibration completed")

# Evaluate on test set
print("Evaluating CovariateLabel predictor on test set...")
y_true_cov, y_prob_cov, _, extra_cov = Trainer(model=covariate_predictor).inference(
    test_loader, additional_outputs=["y_predset"]
)

cov_metrics = get_metrics_fn("multiclass")(
    y_true_cov,
    y_prob_cov,
    metrics=["accuracy", "miscoverage_ps"],
    y_predset=extra_cov["y_predset"],
)

# Calculate average set size
predset_cov = (
    torch.tensor(extra_cov["y_predset"])
    if isinstance(extra_cov["y_predset"], np.ndarray)
    else extra_cov["y_predset"]
)
avg_set_size_cov = predset_cov.float().sum(dim=1).mean().item()

# Extract scalar values from metrics (handle both scalar and array returns)
miscoverage_cov = cov_metrics["miscoverage_ps"]
if isinstance(miscoverage_cov, np.ndarray):
    miscoverage_cov = float(
        miscoverage_cov.item() if miscoverage_cov.size == 1 else miscoverage_cov.mean()
    )
else:
    miscoverage_cov = float(miscoverage_cov)

print("\nCovariateLabel Results:")
print(f"  Accuracy: {cov_metrics['accuracy']:.4f}")
print(f"  Empirical miscoverage: {miscoverage_cov:.4f}")
print(f"  Average set size: {avg_set_size_cov:.2f}")
print(f"  Target miscoverage: {alpha:.2f}")

# ============================================================================
# STEP 5: Compare Methods
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Comparison of Methods")
print("=" * 80)

print(f"\nTarget: {1-alpha:.0%} coverage (max {alpha:.0%} miscoverage)")
print("\n{:<40} {:<15} {:<15}".format("Metric", "LABEL", "CovariateLabel"))
print("-" * 70)

# Coverage (1 - miscoverage)
label_coverage = 1 - miscoverage_label
cov_coverage = 1 - miscoverage_cov
print(
    "{:<40} {:<15.2%} {:<15.2%}".format(
        "Empirical Coverage", label_coverage, cov_coverage
    )
)

# Miscoverage
print(
    "{:<40} {:<15.4f} {:<15.4f}".format(
        "Empirical Miscoverage",
        miscoverage_label,
        miscoverage_cov,
    )
)

# Average set size (smaller is better for efficiency)
print(
    "{:<40} {:<15.2f} {:<15.2f}".format(
        "Average Set Size", avg_set_size_label, avg_set_size_cov
    )
)

# Efficiency (inverse of average set size)
efficiency_label = 1.0 / avg_set_size_label
efficiency_cov = 1.0 / avg_set_size_cov
print(
    "{:<40} {:<15.4f} {:<15.4f}".format(
        "Efficiency (1/avg_set_size)", efficiency_label, efficiency_cov
    )
)

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

print("\nKey Observations:")
print("1. Both methods achieve near-target coverage guarantees")
print("2. LABEL: Standard conformal prediction")
print("3. CovariateLabel: Adapts to distribution shift between cal and test")
print("\nWhen to use CovariateLabel:")
print("  - When test distribution differs from calibration distribution")
print("  - When you have access to test embeddings/features")
print("  - When you want more robust coverage under distribution shift")
print("\nWhen to use LABEL:")
print("  - When cal and test distributions are similar (exchangeable)")
print("  - Simpler method, no need to fit KDEs")
print("  - Computationally more efficient")

# ============================================================================
# STEP 6: Visualize Prediction Sets
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: Example Predictions")
print("=" * 80)

# Show first 5 test examples
n_examples = 5
print(f"\nShowing first {n_examples} test examples:")
print("-" * 80)

for i in range(min(n_examples, len(y_true_label))):
    true_class = int(y_true_label[i])

    # LABEL prediction set
    if isinstance(predset_label, np.ndarray):
        label_set = np.where(predset_label[i])[0]
    else:
        label_set = torch.where(predset_label[i])[0].cpu().numpy()

    # CovariateLabel prediction set
    if isinstance(predset_cov, np.ndarray):
        cov_set = np.where(predset_cov[i])[0]
    else:
        cov_set = torch.where(predset_cov[i])[0].cpu().numpy()

    print(f"\nExample {i+1}:")
    print(f"  True class: {true_class}")
    print(f"  LABEL set: {label_set.tolist()} (size: {len(label_set)})")
    print(f"  CovariateLabel set: {cov_set.tolist()} (size: {len(cov_set)})")
    print(f"  Correct in LABEL? {true_class in label_set}")
    print(f"  Correct in CovariateLabel? {true_class in cov_set}")

print("\n" + "=" * 80)
print("Example completed successfully!")
print("=" * 80)
