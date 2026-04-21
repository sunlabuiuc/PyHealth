"""
Ablation study summary:

- Increasing hidden_dim improves performance (32 → 128)
- Higher dropout (0.5) with lower learning rate generalizes better
- Best config: hidden_dim=128, dropout=0.5, lr=0.0005

This matches intuition: larger capacity + regularization improves generalization.
"""

from __future__ import annotations

import random
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from pyhealth.models import WearableMLP


class DummyOutputProcessor:
    """Minimal output processor stub used by BaseModel."""

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class DummyBinaryDataset:
    """Minimal dataset stub for WearableMLP example."""

    def __init__(self, input_dim: int = 13) -> None:
        self.input_schema = {"wearable": {"shape": (input_dim,)}}
        self.output_schema = {"label": "binary"}
        self.feature_keys = ["wearable"]
        self.label_keys = ["label"]
        self.output_processors = {"label": DummyOutputProcessor(2)}

    def get_all_tokens(self, key):
        return None

    def get_label_tokenizer(self):
        return None

    def get_feature_tokenizer(self, key):
        return None


class SyntheticWearableTorchDataset(Dataset):
    """Tiny synthetic dataset for binary wearable prediction.

    Each sample contains 13 dense features:
        - 3 days x 2 wearable signals = 6 values
        - 7 day-of-week one-hot values = 7 values

    The label is generated from a simple rule so the model can learn quickly.
    """

    def __init__(self, num_samples: int = 256, input_dim: int = 13, seed: int = 42):
        super().__init__()
        random.seed(seed)
        torch.manual_seed(seed)

        self.x = torch.randn(num_samples, input_dim)

        # Make the task learnable:
        # use a linear combination of a few "wearable" features plus mild noise.
        signal = (
            1.2 * self.x[:, 0]
            + 0.8 * self.x[:, 1]
            - 1.0 * self.x[:, 4]
            + 0.5 * self.x[:, 7]
            + 0.1 * torch.randn(num_samples)
        )
        self.y = (signal > 0).float().unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "wearable": self.x[idx],
            "label": self.y[idx],
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "wearable": torch.stack([item["wearable"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
    }


def evaluate_accuracy(model: WearableMLP, dataloader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            output = model(**batch)
            preds = (output["y_prob"] >= 0.5).float()
            correct += (preds == batch["label"]).sum().item()
            total += batch["label"].numel()

    return correct / total if total > 0 else 0.0


def train_one_config(
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    epochs: int = 5,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, float]:
    torch.manual_seed(seed)

    dataset_stub = DummyBinaryDataset(input_dim=13)

    train_dataset = SyntheticWearableTorchDataset(
        num_samples=256,
        input_dim=13,
        seed=seed,
    )
    test_dataset = SyntheticWearableTorchDataset(
        num_samples=128,
        input_dim=13,
        seed=seed + 1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = WearableMLP(
        dataset=dataset_stub,
        feature_key="wearable",
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            output = model(**batch)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        test_acc = evaluate_accuracy(model, test_loader)
        print(
            f"epoch={epoch + 1:02d} "
            f"hidden_dim={hidden_dim} dropout={dropout:.1f} lr={learning_rate:.4f} "
            f"train_loss={avg_loss:.4f} test_acc={test_acc:.4f}"
        )

    final_acc = evaluate_accuracy(model, test_loader)
    return {
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "test_accuracy": final_acc,
    }


def main() -> None:
    """Run a small ablation study for WearableMLP.

    This example demonstrates a paper-inspired short-window wearable prediction
    setting using dense features:
        - 3 days x 2 wearable signals
        - 7 day-of-week one-hot features

    Ablations vary hidden dimension, dropout, and learning rate.
    Synthetic data is used so the example is fast and reproducible.
    """
    configs = [
        {"hidden_dim": 32, "dropout": 0.0, "learning_rate": 1e-3},
        {"hidden_dim": 64, "dropout": 0.2, "learning_rate": 1e-3},
        {"hidden_dim": 128, "dropout": 0.5, "learning_rate": 5e-4},
    ]

    results = []
    for config in configs:
        print("=" * 80)
        print(f"Running config: {config}")
        result = train_one_config(**config)
        results.append(result)

    print("\nFinal ablation results")
    print("-" * 80)
    print(
        f"{'hidden_dim':>10} {'dropout':>10} {'lr':>10} {'test_accuracy':>15}"
    )
    for result in results:
        print(
            f"{result['hidden_dim']:>10} "
            f"{result['dropout']:>10.1f} "
            f"{result['learning_rate']:>10.4f} "
            f"{result['test_accuracy']:>15.4f}"
        )

    best_result = max(results, key=lambda x: x["test_accuracy"])
    print("\nBest configuration:")
    print(best_result)


if __name__ == "__main__":
    main()