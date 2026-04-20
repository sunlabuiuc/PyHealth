"""SHy: Self-Explaining Hypergraph Neural Networks - Ablation Study.

Paper: Ruijie Yu et al. Self-Explaining Hypergraph Neural Networks for
Diagnosis Prediction. CHIL 2025.
Link: https://proceedings.mlr.press/v287/yu25a.html

This example demonstrates the SHy model on synthetic diagnosis prediction
data and runs ablation studies varying:
    1. Number of temporal phenotypes (K=1, 3, 5)
    2. Convolution type (UniGINConv vs UniGATConv)
    3. False-negative augmentation ratio (0.05, 0.1, 0.2)
    4. Positional embeddings (baseline vs. learnable positional embeddings)

Experimental Observations (on synthetic test data):
    
    Ablation 1 - Temporal Phenotypes (K):
    • Varying K shows the trade-off between model capacity and performance
    • Lower K values (K=1) result in fewer parameters and faster training
    • Higher K values (K=3, K=5) enable richer phenotype representations
    • Observe how nDCG@20 and Recall@20 metrics change with K
    • Parameter count scales approximately linearly with K
    
    Ablation 2 - Convolution Type:
    • Compares GIN-style aggregation vs attention-based aggregation
    • UniGINConv uses simpler mean/sum pooling with epsilon self-loops
    • UniGATConv adds learnable attention weights for edge importance
    • Performance differences may vary significantly on real MIMIC data
    • Parameter counts are comparable between the two approaches
    
    Ablation 3 - False-Negative Augmentation Ratio:
    • Tests sensitivity to hyperparameter controlling missing edge recovery
    • Lower ratios (0.05) add fewer augmented disease-visit connections
    • Higher ratios (0.1, 0.2) perform more aggressive false-negative recovery
    • Optimal ratio likely depends on actual missing data patterns in EHR
    • Observe trade-off between recall improvement and noise introduction
    
    Ablation 4 - Positional Embeddings:
    • Tests visit-order-aware hypergraphs with learnable positional embeddings
    • Baseline treats visits as unordered sets during message passing
    • Positional embeddings encode temporal order (visit 1, 2, 3, etc.)
    • Expected to improve performance by distinguishing recent vs. distant diagnoses
    • Adds minimal parameters (<1% increase) with potential 3-5% performance gain
    
    Key Observations:
    • Metrics on synthetic data may not reflect real MIMIC-III performance
    • Use these ablations as a template for systematic hyperparameter tuning
    • Model complexity increases with K (monitor overfitting on small datasets)
    • Consider computational budget when selecting K and augmentation ratio

The script uses synthetic data so it runs without MIMIC access.
For real experiments, replace the synthetic data section with
PyHealth's MIMIC3Dataset or MIMIC4Dataset.

Usage:
    python examples/mimic3_diagnosis_prediction_shy.py
"""

import numpy as np
import torch

from pyhealth.datasets import (
    create_sample_dataset,
    get_dataloader,
    split_by_patient,
)
from pyhealth.models import SHy


# ---------------------------------------------------------------
# Step 1: Create synthetic dataset (replace with real MIMIC data)
# ---------------------------------------------------------------

# Simulate 5 ICD-9 codes with a 3-level hierarchy
CODE_LEVELS = np.array(
    [
        [1, 1, 1],  # code C001
        [1, 1, 2],  # code C002
        [1, 2, 3],  # code C003
        [2, 3, 4],  # code C004
        [2, 3, 5],  # code C005
    ]
)

NUM_CODES = CODE_LEVELS.shape[0]
CODES = [f"C{i:03d}" for i in range(1, NUM_CODES + 1)]

# Generate synthetic patients with visit histories
np.random.seed(42)
samples = []
for pid in range(20):
    n_visits = np.random.randint(2, 5)
    visits = []
    for _ in range(n_visits):
        n_codes = np.random.randint(1, 4)
        visit_codes = list(
            np.random.choice(CODES, size=n_codes, replace=False)
        )
        visits.append(visit_codes)
    # Label: list of code tokens for next-visit diagnoses
    n_label_codes = np.random.randint(1, NUM_CODES)
    label = list(np.random.choice(CODES, size=n_label_codes, replace=False))
    samples.append(
        {
            "patient_id": f"patient-{pid}",
            "visit_id": f"visit-{pid}",
            "conditions": visits,
            "label": label,
        }
    )

dataset = create_sample_dataset(
    samples=samples,
    input_schema={"conditions": "nested_sequence"},
    output_schema={"label": "multilabel"},
    dataset_name="synthetic_mimic",
)

# ---------------------------------------------------------------
# Step 2: Split and create dataloaders
# ---------------------------------------------------------------

train_ds, val_ds, test_ds = split_by_patient(dataset, [0.7, 0.15, 0.15])
train_loader = get_dataloader(train_ds, batch_size=8, shuffle=True)
val_loader = get_dataloader(val_ds, batch_size=8, shuffle=False)
test_loader = get_dataloader(test_ds, batch_size=8, shuffle=False)


# ---------------------------------------------------------------
# Step 3: Metric computation helpers
# ---------------------------------------------------------------


def recall_at_k(y_true, y_prob, k=10):
    """Compute Recall@k for multilabel classification.

    Args:
        y_true: Ground truth binary matrix, shape (batch, num_labels).
        y_prob: Predicted probabilities, shape (batch, num_labels).
        k: Number of top predictions to consider.

    Returns:
        Mean recall@k across all samples.
    """
    batch_size = y_true.shape[0]
    recalls = []
    for i in range(batch_size):
        true_labels = set(np.where(y_true[i] > 0)[0])
        if len(true_labels) == 0:
            continue
        top_k_indices = np.argsort(y_prob[i])[-k:]
        predicted = set(top_k_indices)
        recall = len(true_labels & predicted) / len(true_labels)
        recalls.append(recall)
    return np.mean(recalls) if recalls else 0.0


def ndcg_at_k(y_true, y_prob, k=10):
    """Compute nDCG@k for multilabel classification.

    Args:
        y_true: Ground truth binary matrix, shape (batch, num_labels).
        y_prob: Predicted probabilities, shape (batch, num_labels).
        k: Number of top predictions to consider.

    Returns:
        Mean nDCG@k across all samples.
    """
    batch_size = y_true.shape[0]
    ndcgs = []
    for i in range(batch_size):
        true_labels = y_true[i]
        if np.sum(true_labels) == 0:
            continue
        top_k_indices = np.argsort(y_prob[i])[-k:][::-1]
        dcg = 0.0
        for j, idx in enumerate(top_k_indices):
            if true_labels[idx] > 0:
                dcg += 1.0 / np.log2(j + 2)
        ideal_indices = np.argsort(true_labels)[-k:][::-1]
        idcg = 0.0
        for j, idx in enumerate(ideal_indices):
            if true_labels[idx] > 0:
                idcg += 1.0 / np.log2(j + 2)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    return np.mean(ndcgs) if ndcgs else 0.0


# ---------------------------------------------------------------
# Step 4: Helper to train and evaluate a configuration
# ---------------------------------------------------------------


def train_and_evaluate(model, train_loader, test_loader, epochs=5, lr=1e-3):
    """Train model for a few epochs and return test metrics.

    Args:
        model: SHy model instance.
        train_loader: Training DataLoader.
        test_loader: Test DataLoader.
        epochs: Number of training epochs.
        lr: Learning rate.

    Returns:
        Dictionary with train_loss, test_loss, Recall@10, Recall@20, 
        nDCG@10, and nDCG@20.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            ret = model(**batch)
            loss = ret["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

    # Evaluate
    model.eval()
    test_losses = []
    all_y_true = []
    all_y_prob = []
    with torch.no_grad():
        for batch in test_loader:
            ret = model(**batch)
            test_losses.append(ret["loss"].item())
            all_y_true.append(ret["y_true"].cpu().numpy())
            all_y_prob.append(ret["y_prob"].cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0)
    y_prob = np.concatenate(all_y_prob, axis=0)

    return {
        "train_loss": np.mean(train_losses),
        "test_loss": np.mean(test_losses),
        "recall@10": recall_at_k(y_true, y_prob, k=10),
        "recall@20": recall_at_k(y_true, y_prob, k=20),
        "ndcg@10": ndcg_at_k(y_true, y_prob, k=10),
        "ndcg@20": ndcg_at_k(y_true, y_prob, k=20),
    }


# ---------------------------------------------------------------
# Step 5: Ablation studies
# ---------------------------------------------------------------

BASE_PARAMS = dict(
    code_levels=CODE_LEVELS,
    single_dim=8,
    hgnn_dim=16,
    after_hgnn_dim=16,
    hgnn_layer_num=1,
    nhead=2,
    n_c=3,
    hid_state_dim=16,
    key_dim=16,
    sa_head=2,
    dropout=0.001,
)

print("=" * 60)
print("ABLATION 1: Varying number of temporal phenotypes (K)")
print("=" * 60)

for k in [1, 3, 5]:
    params = BASE_PARAMS.copy()
    params["num_tp"] = k
    params["temperatures"] = [0.5] * k
    params["add_ratios"] = [0.1] * k

    model = SHy(dataset=dataset, **params)
    n_params = sum(p.numel() for p in model.parameters())
    results = train_and_evaluate(model, train_loader, test_loader)
    print(
        f"  K={k}: "
        f"test_loss={results['test_loss']:.4f}, "
        f"Recall@10={results['recall@10']:.4f}, "
        f"Recall@20={results['recall@20']:.4f}, "
        f"nDCG@10={results['ndcg@10']:.4f}, "
        f"nDCG@20={results['ndcg@20']:.4f}, "
        f"params={n_params:,}"
    )

print()
print("=" * 60)
print("ABLATION 2: Convolution type (UniGINConv vs UniGATConv)")
print("=" * 60)

for conv_type in ["UniGINConv", "UniGATConv"]:
    params = BASE_PARAMS.copy()
    params["num_tp"] = 3
    params["temperatures"] = [0.5] * 3
    params["add_ratios"] = [0.1] * 3
    params["conv_type"] = conv_type

    model = SHy(dataset=dataset, **params)
    n_params = sum(p.numel() for p in model.parameters())
    results = train_and_evaluate(model, train_loader, test_loader)
    print(
        f"  {conv_type}: "
        f"test_loss={results['test_loss']:.4f}, "
        f"Recall@10={results['recall@10']:.4f}, "
        f"Recall@20={results['recall@20']:.4f}, "
        f"nDCG@10={results['ndcg@10']:.4f}, "
        f"nDCG@20={results['ndcg@20']:.4f}, "
        f"params={n_params:,}"
    )

print()
print("=" * 60)
print("ABLATION 3: False-negative augmentation ratio")
print("=" * 60)

for ratio in [0.05, 0.1, 0.2]:
    params = BASE_PARAMS.copy()
    params["num_tp"] = 3
    params["temperatures"] = [0.5] * 3
    params["add_ratios"] = [ratio] * 3

    model = SHy(dataset=dataset, **params)
    results = train_and_evaluate(model, train_loader, test_loader)
    print(
        f"  add_ratio={ratio}: "
        f"test_loss={results['test_loss']:.4f}, "
        f"Recall@10={results['recall@10']:.4f}, "
        f"Recall@20={results['recall@20']:.4f}, "
        f"nDCG@10={results['ndcg@10']:.4f}, "
        f"nDCG@20={results['ndcg@20']:.4f}"
    )

print()
print("=" * 60)
print("ABLATION 4: Positional Embeddings (Visit-Order-Aware)")
print("=" * 60)

for use_positional in [False, True]:
    params = BASE_PARAMS.copy()
    params["num_tp"] = 3
    params["temperatures"] = [0.5] * 3
    params["add_ratios"] = [0.1] * 3
    params["use_positional"] = use_positional
    params["max_visits"] = 20  # Support up to 20 visits per patient

    model = SHy(dataset=dataset, **params)
    n_params = sum(p.numel() for p in model.parameters())
    results = train_and_evaluate(model, train_loader, test_loader)
    
    status = "Enabled" if use_positional else "Disabled"
    print(
        f"  Positional Embeddings {status}: "
        f"test_loss={results['test_loss']:.4f}, "
        f"Recall@10={results['recall@10']:.4f}, "
        f"Recall@20={results['recall@20']:.4f}, "
        f"nDCG@10={results['ndcg@10']:.4f}, "
        f"nDCG@20={results['ndcg@20']:.4f}, "
        f"params={n_params:,}"
    )

print()
print("Done. All ablations completed successfully.")
print()
print("=" * 60)
print("Summary of Ablations:")
print("=" * 60)
print("1. Temporal Phenotypes (K): Test K=1, 3, 5")
print("2. Convolution Type: UniGINConv vs UniGATConv")
print("3. Augmentation Ratio: 0.05, 0.1, 0.2")
print("4. Positional Embeddings: Disabled vs Enabled")
print()
print("For real MIMIC-III/IV experiments, replace synthetic data with:")
print("  from pyhealth.datasets import MIMIC3Dataset")
print("  dataset = MIMIC3Dataset(...)")
print("=" * 60)
