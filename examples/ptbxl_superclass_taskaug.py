"""Example ablation study demonstrating TaskAug across configurations on PTB-XL.

Paper: Data Augmentation for Electrocardiograms (Raghu et al., 2022).

Experimental Setup:
    Uses the local synthetic PTB-XL dataset and cache directory to run fast,
    reproducible ablation experiments without requiring external downloads.

Configurations Compared:
    1. num_stages=1 with PTBXLSuperclassClassification (lr=1e-3).
    2. num_stages=2 with PTBXLAllDiagnosticClassification (lr=1e-3).

Findings:
    Both configurations execute end-to-end (dataset loading, task sample
    generation, model instantiation, training, and evaluation) and report
    training/validation loss and evaluation metrics for side-by-side comparison.
"""

import torch
from torch.utils.data import DataLoader

from pyhealth.datasets import PTBXLDataset
from pyhealth.models import TaskAug
from pyhealth.tasks import (
    PTBXLAllDiagnosticClassification,
    PTBXLSuperclassClassification,
)
from pyhealth.trainer import Trainer


def run_experiment(
    dataset_root: str,
    cache_dir: str,
    task,
    num_stages: int,
    learning_rate: float,
    epochs: int = 1,
) -> dict[str, float]:
    """Runs a single TaskAug configuration and returns summary metrics."""
    dataset = PTBXLDataset(
        root=dataset_root,
        cache_dir=cache_dir,
    )
    samples = dataset.set_task(task)

    split_idx = max(1, len(samples) // 2)
    train_dataset = samples[:split_idx]
    val_dataset = samples[split_idx:] if len(samples) > split_idx else samples

    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)

    model = TaskAug(
        dataset=samples,
        feature_keys=["signal"],
        label_key="labels",
        mode="multilabel",
        num_stages=num_stages,
    )

    trainer = Trainer(model=model, metrics=["roc_auc_samples", "pr_auc_samples"])
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": learning_rate},
    )

    eval_outputs = trainer.evaluate(val_loader)
    return {
        "num_stages": float(num_stages),
        "num_classes": float(len(samples[0]["labels"])),
        "val_loss": float(eval_outputs.get("loss", 0.0)),
        "roc_auc_samples": float(eval_outputs.get("roc_auc_samples", 0.0)),
        "pr_auc_samples": float(eval_outputs.get("pr_auc_samples", 0.0)),
    }


def main() -> None:
    """Executes the ablation study across task and stage configurations."""
    dataset_root = "tests/data/ptbxl_synthetic"
    cache_dir = "tests/data/my_clean_cache"
    scp_path = "tests/data/ptbxl_synthetic/scp_statements.csv"

    ablation_configs = [
        (
            "superclass_1stage",
            PTBXLSuperclassClassification(scp_statements_path=scp_path),
            1,
            1e-3,
        ),
        (
            "diagnostic_71_2stage",
            PTBXLAllDiagnosticClassification(scp_statements_path=scp_path),
            2,
            1e-3,
        ),
    ]

    results = {}
    for name, task, stages, lr in ablation_configs:
        results[name] = run_experiment(
            dataset_root=dataset_root,
            cache_dir=cache_dir,
            task=task,
            num_stages=stages,
            learning_rate=lr,
            epochs=1,
        )

    print("\nAblation Comparison Summary:")
    for name, metrics in results.items():
        print(
            f"{name}: stages={int(metrics['num_stages'])}, "
            f"classes={int(metrics['num_classes'])}, "
            f"val_loss={metrics['val_loss']:.4f}, "
            f"roc_auc_samples={metrics['roc_auc_samples']:.4f}, "
            f"pr_auc_samples={metrics['pr_auc_samples']:.4f}"
        )

    print("TaskAug ablation study completed successfully!")


if __name__ == "__main__":
    main()
