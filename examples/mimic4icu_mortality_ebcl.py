"""Demo ablation study for the EBCL model on synthetic PyHealth data.

This script creates a small synthetic event-centered dataset using
``create_sample_dataset`` and compares several EBCL configurations on a simple
binary mortality-prediction-style task.

Why this qualifies as an ablation study:
- It varies hyperparameters not ablated in the original EBCL paper:
  learning rate, hidden dimension, dropout, and contrastive weight.
- It reports performance comparison across configurations.
- It is fully runnable with synthetic/demo data only.

Experimental setup:
- Task: binary classification ("mortality prediction"-style demo task)
- Data: synthetic pre-event and post-event paired features
- Model: EBCL
- Metrics: validation loss and test accuracy
- Comparison:
    1. baseline
    2. lower learning rate
    3. larger hidden dimension
    4. higher dropout
    5. no contrastive objective

How to run:
    python examples/mimic4icu_mortality_ebcl.py

Expected findings:
- Different learning rates may affect optimization stability.
- Larger hidden dimensions may improve representation capacity.
- Higher dropout may regularize the model but can also hurt on tiny data.
- contrastive_weight=0.0 removes the EBCL contrastive objective, which often
  lowers performance on paired event data.

  However, due to the fact that we didn't get the real dataset 
  and it isn't run with acutal data, the results are for demonstration purposes
  and may not reflect real-world performance differences.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from statistics import mean

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import EBCL


@dataclass
class AblationConfig:
    """Configuration for one ablation run.

    Attributes:
        name: Human-readable experiment name.
        learning_rate: Adam learning rate.
        hidden_dim: Hidden dimension used in EBCL.
        dropout: Dropout used in EBCL.
        contrastive_weight: Weight for contrastive loss.
    """

    name: str
    learning_rate: float
    hidden_dim: int
    dropout: float
    contrastive_weight: float


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_demo_samples(
    num_samples: int = 120,
    seed: int = 42,
) -> list[dict]:
    """Builds synthetic paired event-centered samples.

    The label is correlated with both pre-event and post-event values so the
    model has a learnable signal.

    Args:
        num_samples: Number of samples to generate.
        seed: Random seed.

    Returns:
        A list of dictionaries compatible with create_sample_dataset().
    """
    rng = random.Random(seed)
    samples: list[dict] = []

    for i in range(num_samples):
        label = rng.randint(0, 1)

        if label == 1:
            conditions = ["HF", "CKD", "HTN"]
            labs = [
                3.0 + rng.random(),
                6.0 + rng.random(),
                8.0 + rng.random(),
            ]
            post_conditions = ["HF", "ICU", "VENT"]
            post_labs = [
                4.0 + rng.random(),
                7.0 + rng.random(),
                9.0 + rng.random(),
            ]
        else:
            conditions = ["WELL", "FOLLOWUP"]
            labs = [
                0.3 + rng.random(),
                1.0 + rng.random(),
                1.8 + rng.random(),
            ]
            post_conditions = ["WELL", "DISCHARGE"]
            post_labs = [
                0.4 + rng.random(),
                1.1 + rng.random(),
                1.7 + rng.random(),
            ]

        samples.append(
            {
                "patient_id": f"patient-{i}",
                "visit_id": f"visit-{i}",
                "conditions": conditions,
                "labs": labs,
                "post_conditions": post_conditions,
                "post_labs": post_labs,
                "label": label,
            }
        )

    return samples


def build_dataset(num_samples: int = 120, seed: int = 42):
    """Creates a synthetic PyHealth SampleDataset.

    Args:
        num_samples: Number of samples to generate.
        seed: Random seed.

    Returns:
        A PyHealth SampleDataset.
    """
    samples = build_demo_samples(num_samples=num_samples, seed=seed)
    dataset = create_sample_dataset(
        samples=samples,
        input_schema={
            "conditions": "sequence",
            "labs": "tensor",
            "post_conditions": "sequence",
            "post_labs": "tensor",
        },
        output_schema={"label": "binary"},
        dataset_name="demo_mortality_prediction_ebcl",
    )
    return dataset


def split_samples(
    samples: list[dict],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Splits samples into train/val/test lists.

    Args:
        samples: Full sample list.
        train_ratio: Train split ratio.
        val_ratio: Validation split ratio.

    Returns:
        Train, validation, and test sample lists.
    """
    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_samples = samples[:n_train]
    val_samples = samples[n_train : n_train + n_val]
    test_samples = samples[n_train + n_val :]
    return train_samples, val_samples, test_samples


def build_split_datasets(
    num_samples: int = 120,
    seed: int = 42,
):
    """Builds train/val/test datasets from synthetic samples.

    Args:
        num_samples: Number of synthetic samples.
        seed: Random seed.

    Returns:
        Tuple of train, validation, and test datasets.
    """
    samples = build_demo_samples(num_samples=num_samples, seed=seed)
    rng = random.Random(seed)
    rng.shuffle(samples)

    train_samples, val_samples, test_samples = split_samples(samples)

    schema = {
        "conditions": "sequence",
        "labs": "tensor",
        "post_conditions": "sequence",
        "post_labs": "tensor",
    }
    output_schema = {"label": "binary"}

    train_dataset = create_sample_dataset(
        samples=train_samples,
        input_schema=schema,
        output_schema=output_schema,
        dataset_name="demo_ebcl_train",
    )
    val_dataset = create_sample_dataset(
        samples=val_samples,
        input_schema=schema,
        output_schema=output_schema,
        dataset_name="demo_ebcl_val",
    )
    test_dataset = create_sample_dataset(
        samples=test_samples,
        input_schema=schema,
        output_schema=output_schema,
        dataset_name="demo_ebcl_test",
    )
    return train_dataset, val_dataset, test_dataset


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Moves tensor values in a batch to the target device.

    Args:
        batch: Batch dictionary from PyHealth dataloader.
        device: Torch device.

    Returns:
        Batch dictionary with tensors moved to device.
    """
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, tuple):
            moved[key] = tuple(
                item.to(device) if isinstance(item, torch.Tensor) else item
                for item in value
            )
        else:
            moved[key] = value
    return moved


def binary_accuracy(y_prob: torch.Tensor, y_true: torch.Tensor) -> float:
    """Computes binary accuracy.

    Args:
        y_prob: Predicted probabilities.
        y_true: True binary labels.

    Returns:
        Accuracy as a float.
    """
    preds = (y_prob >= 0.5).long()
    y_true = y_true.view_as(preds).long()
    return (preds == y_true).float().mean().item()


def evaluate_model(
    model: EBCL,
    dataloader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluates a model on a dataloader.

    Args:
        model: EBCL model.
        dataloader: Validation or test dataloader.
        device: Torch device.

    Returns:
        Mean loss and mean accuracy.
    """
    model.eval()
    losses: list[float] = []
    accuracies: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            losses.append(outputs["loss"].item())
            accuracies.append(
                binary_accuracy(outputs["y_prob"], outputs["y_true"])
            )

    return mean(losses), mean(accuracies)


def train_one_config(
    train_dataset,
    val_dataset,
    test_dataset,
    config: AblationConfig,
    epochs: int = 10,
    batch_size: int = 16,
    seed: int = 42,
) -> dict:
    """Trains and evaluates one EBCL configuration.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        config: Ablation configuration.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        seed: Random seed.

    Returns:
        Dictionary of metrics for this configuration.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EBCL(
        dataset=train_dataset,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        contrastive_weight=config.contrastive_weight,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_loss, val_acc = evaluate_model(model, val_loader, device)
        print(
            f"[{config.name}] "
            f"epoch={epoch + 1:02d} "
            f"train_loss={mean(train_losses):.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate_model(model, test_loader, device)

    return {
        "config": config.name,
        "learning_rate": config.learning_rate,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "contrastive_weight": config.contrastive_weight,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def print_results_table(results: list[dict]) -> None:
    """Prints a compact comparison table.

    Args:
        results: List of result dictionaries.
    """
    print("\n=== EBCL Ablation Results ===")
    header = (
        "config".ljust(18)
        + "lr".ljust(12)
        + "hidden".ljust(10)
        + "dropout".ljust(10)
        + "c_weight".ljust(12)
        + "val_loss".ljust(12)
        + "test_acc"
    )
    print(header)
    print("-" * len(header))

    for row in results:
        print(
            str(row["config"]).ljust(18)
            + f'{row["learning_rate"]:.0e}'.ljust(12)
            + str(row["hidden_dim"]).ljust(10)
            + f'{row["dropout"]:.2f}'.ljust(10)
            + f'{row["contrastive_weight"]:.1f}'.ljust(12)
            + f'{row["best_val_loss"]:.4f}'.ljust(12)
            + f'{row["test_acc"]:.4f}'
        )


def main() -> None:
    """Runs the synthetic EBCL ablation study."""
    set_seed(42)
    train_dataset, val_dataset, test_dataset = build_split_datasets(
        num_samples=120,
        seed=42,
    )

    configs = [
        AblationConfig(
            name="baseline",
            learning_rate=1e-3,
            hidden_dim=32,
            dropout=0.10,
            contrastive_weight=1.0,
        ),
        AblationConfig(
            name="low_lr",
            learning_rate=3e-4,
            hidden_dim=32,
            dropout=0.10,
            contrastive_weight=1.0,
        ),
        AblationConfig(
            name="bigger_hidden",
            learning_rate=1e-3,
            hidden_dim=64,
            dropout=0.10,
            contrastive_weight=1.0,
        ),
        AblationConfig(
            name="high_dropout",
            learning_rate=1e-3,
            hidden_dim=32,
            dropout=0.30,
            contrastive_weight=1.0,
        ),
        AblationConfig(
            name="no_contrastive",
            learning_rate=1e-3,
            hidden_dim=32,
            dropout=0.10,
            contrastive_weight=0.0,
        ),
    ]

    results = []
    for config in configs:
        result = train_one_config(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            config=config,
            epochs=10,
            batch_size=16,
            seed=42,
        )
        results.append(result)

    print_results_table(results)

    best = max(results, key=lambda row: row["test_acc"])
    print("\nBest configuration:")
    print(best)


if __name__ == "__main__":
    main()