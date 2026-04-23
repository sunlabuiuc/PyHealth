import random
from typing import Dict

import numpy as np
import torch
from pyhealth.datasets import eICUDataset, get_dataloader, split_by_sample
from pyhealth.models import MLP, RNN
from pyhealth.trainer import Trainer
from pyhealth.tasks.temporal_mortality import TemporalMortalityPredictionEICU

DATA_ROOT = r"../dataset/eicu-collaborative-research-database-demo-2.0.1"


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sample_dataset(root: str):
    """Loads the eICU demo dataset and applies the custom temporal task."""
    dataset = eICUDataset(
        root=root,
        tables=["diagnosis", "medication", "physicalexam"],
        dev=True,
    )
    task = TemporalMortalityPredictionEICU()
    sample_dataset = dataset.set_task(task)
    return sample_dataset


def build_dataloaders(sample_dataset, batch_size: int = 8):
    """Splits the dataset and creates train/val/test dataloaders."""
    train_ds, val_ds, test_ds = split_by_sample(sample_dataset, [0.6, 0.2, 0.2])

    print(f"train: {len(train_ds)}")
    print(f"val: {len(val_ds)}")
    print(f"test: {len(test_ds)}")

    train_loader = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_and_evaluate(model, train_loader, val_loader, test_loader, epochs: int = 10) -> Dict[str, float]:
    """Trains a model and returns evaluation metrics on the test set."""
    trainer = Trainer(
        model=model,
        metrics=["pr_auc", "roc_auc", "f1"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 0.001},
        monitor="roc_auc",
        monitor_criterion="max",
        load_best_model_at_last=True,
    )

    results = trainer.evaluate(test_loader)
    return results


def describe_temporal_fields(sample_dataset) -> None:
    """Prints temporal metadata present in the processed samples."""
    print("sample keys:", sample_dataset[0].keys())
    years = []
    groups = {"early": 0, "late": 0}
    for i in range(len(sample_dataset)):
        sample = sample_dataset[i]
        year = sample.get("discharge_year", None)
        if hasattr(year, "item"):
            year = int(year.item())
        if isinstance(year, int) and year != -1:
            years.append(year)

        group = sample.get("split_group", None)
        if group is not None:
            groups[str(group)] = groups.get(str(group), 0) + 1

    if years:
        print("unique discharge years:", sorted(set(years)))
    else:
        print("no valid discharge_year values found in processed samples")

    print("split_group counts:", groups)


def main() -> None:
    """Runs the full baseline experiment."""
    set_seed(42)

    sample_dataset = load_sample_dataset(DATA_ROOT)
    print(sample_dataset)
    print(sample_dataset[0])
    print("num samples:", len(sample_dataset))
    describe_temporal_fields(sample_dataset)

    train_loader, val_loader, test_loader = build_dataloaders(
        sample_dataset=sample_dataset,
        batch_size=8,
    )

    print("\n=== Training RNN baseline ===")
    rnn_model = RNN(
        dataset=sample_dataset,
        embedding_dim=64,
        hidden_dim=64,
    )
    rnn_results = train_and_evaluate(
        model=rnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=10,
    )
    print("RNN results:", rnn_results)

    print("\n=== Training MLP baseline ===")
    mlp_model = MLP(
        dataset=sample_dataset,
        embedding_dim=64,
    )
    mlp_results = train_and_evaluate(
        model=mlp_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=10,
    )
    print("MLP results:", mlp_results)

    print("\n=== Summary ===")
    print({"task": "TemporalMortalityPredictionEICU", "model": "RNN", **rnn_results})
    print({"task": "TemporalMortalityPredictionEICU", "model": "MLP", **mlp_results})


if __name__ == "__main__":
    main()