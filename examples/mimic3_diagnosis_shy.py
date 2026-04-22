"""
Ablation Study: SHy on MIMIC-III Diagnosis Prediction

Goal:
    Evaluate how SHy performance changes under different:
        - number of hypergraph layers
        - hidden dimension
        - number of temporal phenotypes

Setup:
    - Dataset: MIMIC-III
    - Task: Diagnosis prediction (multilabel classification)
    - Model: SHy
    - Metrics:
        - PR-AUC (sample averaged)
        - ROC-AUC
        - Recall@5
        - Precision@5

Output:
Using epochs=50, optimizer_params={"lr": 1e-3}, mimic-iii data, dev=False
+--------------+----------+-----------+------------+---------------+--------+
| Config       |   PR-AUC |   ROC-AUC |   Recall@5 |   Precision@5 |   Loss |
+==============+==========+===========+============+===============+========+
| layers_0     |   0.1433 |    0.5781 |     0.0919 |           0.2 | 0.2878 |
+--------------+----------+-----------+------------+---------------+--------+
| layers_2     |   0.1512 |    0.6135 |     0.0919 |           0.2 | 0.2759 |
+--------------+----------+-----------+------------+---------------+--------+
| layers_5     |   0.1535 |    0.5756 |     0.0919 |           0.2 | 0.2884 |
+--------------+----------+-----------+------------+---------------+--------+
| layers_7     |   0.1616 |    0.5935 |     0.0919 |           0.2 | 0.2752 |
+--------------+----------+-----------+------------+---------------+--------+
| hdim_64      |   0.155  |    0.6157 |     0.0919 |           0.2 | 0.3248 |
+--------------+----------+-----------+------------+---------------+--------+
| hdim_128     |   0.1514 |    0.5651 |     0.0294 |           0.1 | 0.2965 |
+--------------+----------+-----------+------------+---------------+--------+
| hdim_256     |   0.1453 |    0.5359 |     0.0294 |           0.1 | 0.3369 |
+--------------+----------+-----------+------------+---------------+--------+
| hdim_512     |   0.1425 |    0.5379 |     0.0919 |           0.2 | 0.384  |
+--------------+----------+-----------+------------+---------------+--------+
| phenotypes_2 |   0.1631 |    0.5742 |     0.0919 |           0.2 | 0.2897 |
+--------------+----------+-----------+------------+---------------+--------+
| phenotypes_3 |   0.1516 |    0.6    |     0.0919 |           0.2 | 0.2889 |
+--------------+----------+-----------+------------+---------------+--------+
| phenotypes_5 |   0.1648 |    0.5816 |     0.0919 |           0.2 | 0.2891 |
+--------------+----------+-----------+------------+---------------+--------+
| phenotypes_7 |   0.1515 |    0.5801 |     0.0919 |           0.2 | 0.2863 |
+--------------+----------+-----------+------------+---------------+--------+

Key Observations:
    - Effect of hidden dimension:
        Best ROC-AUC is at hdim_64 (0.6157). Increasing dimension (128 → 512)
        degrades both PR-AUC (0.155 → 0.1425) and ROC-AUC (0.6157 → 0.5379),
        while loss increases. The best is hdim_64. Very small hdim_128 and
        hdim_256 also hurt recall/precision.
    - Effect of number of layers:
        Adding layers (2,5,7) slightly improves PR-AUC (0.1433 → 0.1616) and
        reduces loss (0.2878 → 0.2752), but ROC-AUC fluctuates. layers_7 is best
        overall among layer variants.
    - Effect of phenotype count:
        phenotypes_5 gives best PR-AUC and competitive
        loss, but phenotypes_2 and phenotypes_3 also perform similarly.
        No clear monotonic trend: phenotypes_7 drops in PR-AUC.
    - Recall@5 and Precision@5 are mostly flat:
        Nearly all configs: Recall@5 ≈ 0.0919, Precision@5 = 0.2 except hdim_128 and 256
        drop performance. The ranking performance is not very sensitive to most hyperparameters.
        The model is underfitting.
    - Overall my model performance is weak:
        All metrics (especially PR-AUC < 0.17) indicate the model struggles with the task.
        The constant top5 retrieval scores for most runs suggest the model may be
        defaulting to trivial or identical predictions.
"""

import torch
import random
import numpy as np
from tabulate import tabulate

from pyhealth.models import SHy
from pyhealth.trainer import Trainer
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks.diagnosis_prediction_mimic3 import DiagnosisPredictionMIMIC3
from pyhealth.datasets import get_dataloader, split_by_patient

SEED = 598
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def recall_at_k(y_prob, y_true, k=5):
    """
    Compute Recall@K for multi-label predictions.

    Recall@K measures how many of the true positive labels are captured
    within the top-K predicted labels for each sample.

    Args:
        y_prob (torch.Tensor): Predicted probabilities or scores with shape.
        y_true (torch.Tensor): Ground truth binary labels with shape,
                                where 1 indicates a true label.
        k (int, optional): Number of top predictions to consider. Default is 5.

    Returns:
        float: Average Recall@K across all samples that have at least one true label.
    """
    topk = torch.topk(y_prob, k, dim=1).indices

    recalls = []
    for i in range(y_true.shape[0]):
        true_labels = torch.where(y_true[i] == 1)[0]
        pred_labels = topk[i]

        if len(true_labels) == 0:
            continue

        hit = len(set(pred_labels.tolist()) & set(true_labels.tolist()))
        recalls.append(hit / len(true_labels))

    return sum(recalls) / len(recalls)

def precision_at_k(y_prob, y_true, k=5):
    """
    Compute Precision@K for multi-label predictions.

    Precision@K measures how many of the top-K predicted labels are actually correct
    for each sample.

    Args:
        y_prob (torch.Tensor): Predicted probabilities or scores with shape.
        y_true (torch.Tensor): Ground truth binary labels with shape,
                                where 1 indicates a true label.
        k (int, optional): Number of top predictions to consider. Default is 5.

    Returns:
        float: Average Precision@K across all samples.
    """
    topk = torch.topk(y_prob, k, dim=1).indices

    precisions = []
    for i in range(y_true.shape[0]):
        true_labels = torch.where(y_true[i] == 1)[0]
        pred_labels = topk[i]

        if len(pred_labels) == 0:
            continue

        hit = len(set(pred_labels.tolist()) & set(true_labels.tolist()))
        precisions.append(hit / k)

    return sum(precisions) / len(precisions)

def run_model(sample_dataset, train_loader, val_loader, test_loader, **kwargs):
    """Train SHy with given hyperparameters and evaluate."""

    # Default parameters
    default_args = {
        "embedding_dim": 512,
        "hidden_dim": 128,
        "num_layers": 2,
        "num_phenotypes": 3,
    }
    # Override with kwargs passed in
    default_args.update(kwargs)

    model = SHy(dataset=sample_dataset, **default_args)

    trainer = Trainer(
        model=model,
        metrics=["pr_auc_samples", "roc_auc_samples"],
        enable_logging=False,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=50,
        optimizer_params={"lr": 1e-3},
        monitor="pr_auc_samples",
        monitor_criterion="max",
    )

    res = trainer.evaluate(test_loader)
    all_probs = []
    all_labels = []

    for batch in test_loader:
        out = model(**batch)
        all_probs.append(out["y_prob"])
        all_labels.append(out["y_true"])

    y_prob = torch.cat(all_probs, dim=0)
    y_true = torch.cat(all_labels, dim=0)

    res.update({
        "Recall@5": recall_at_k(y_prob, y_true, k=5),
        "Precision@5": precision_at_k(y_prob, y_true, k=5)
    })
    return res


def main():
    """
    Ablation Study / Example Usage
    """
    # Load dataset
    dataset = MIMIC3Dataset(
        root="/path/to/mimic-iii/1.4",    # path to your local MIMIC-III CSV files
        tables=["DIAGNOSES_ICD"],
        # dev=True,
    )

    # Define task and apply
    task = DiagnosisPredictionMIMIC3()
    dataset = dataset.set_task(task)

    # Split data
    train_ds, val_ds, test_ds = split_by_patient(dataset, [0.7, 0.2, 0.1], seed=SEED)

    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

    results = {}

    # ------------------------------------------------------------
    # Ablation 1: number of hypergraph layers (num_layers)
    # ------------------------------------------------------------
    for n in [0, 2, 5, 7]:
        cfg = {"num_layers": n, "hidden_dim": 128, "num_phenotypes": 3}
        res = run_model(
            dataset, train_loader, val_loader, test_loader, **cfg
        )
        results[f"layers_{n}"] = res

    # ------------------------------------------------------------
    # Ablation 2: hidden dimension
    # ------------------------------------------------------------
    for hdim in [64, 128, 256, 512]:
        cfg = {"num_layers": 1, "hidden_dim": hdim, "num_phenotypes": 3}
        res = run_model(
            dataset, train_loader, val_loader, test_loader, **cfg
        )
        results[f"hdim_{hdim}"] = res

    # ------------------------------------------------------------
    # Ablation 3: number of temporal phenotypes (num_phenotypes)
    # ------------------------------------------------------------
    for k in [2, 3, 5, 7]:
        cfg = {"num_layers": 1, "hidden_dim": 128, "num_phenotypes": k}
        res = run_model(
            dataset, train_loader, val_loader, test_loader, **cfg
        )
        results[f"phenotypes_{k}"] = res

    # Print final result table
    table_data = []
    for name, metrics in results.items():
        table_data.append([
            name,
            f"{metrics['pr_auc_samples']:.4f}",
            f"{metrics['roc_auc_samples']:.4f}",
            f"{metrics['Recall@5']:.4f}",
            f"{metrics['Precision@5']:.4f}",
            f"{metrics['loss']:.4f}",
        ])

    headers = ["Config", "PR-AUC", "ROC-AUC", "Recall@5", "Precision@5", "Loss"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()