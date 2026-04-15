"""Example ablation script for EBCL on synthetic MIMIC-style data.

This script demonstrates how to:
1. instantiate EBCL,
2. run a simple training loop,
3. compare a few hyperparameter settings.

It uses synthetic data so it is lightweight and runnable without protected datasets.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pyhealth.models.ebcl import EBCL


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SyntheticEventDataset(Dataset):
    """Synthetic paired pre/post-event sequence dataset."""

    def __init__(
        self,
        num_samples: int = 128,
        seq_len: int = 16,
        input_dim: int = 16,
    ) -> None:
        self.left_x = torch.randn(num_samples, seq_len, input_dim)
        self.right_x = self.left_x + 0.1 * torch.randn(num_samples, seq_len, input_dim)

    def __len__(self) -> int:
        return self.left_x.size(0)

    def __getitem__(self, idx: int):
        return {
            "left_x": self.left_x[idx],
            "right_x": self.right_x[idx],
            "left_mask": torch.ones(self.left_x.size(1), dtype=torch.bool),
            "right_mask": torch.ones(self.right_x.size(1), dtype=torch.bool),
        }


def collate_fn(batch):
    return {
        "left_x": torch.stack([x["left_x"] for x in batch], dim=0),
        "right_x": torch.stack([x["right_x"] for x in batch], dim=0),
        "left_mask": torch.stack([x["left_mask"] for x in batch], dim=0),
        "right_mask": torch.stack([x["right_mask"] for x in batch], dim=0),
    }


def train_one_setting(
    temperature: float,
    projection_dim: int,
    hidden_dim: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
):
    dataset = SyntheticEventDataset(num_samples=128, seq_len=16, input_dim=16)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = EBCL(
        dataset=None,
        input_dim=16,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
        temperature=temperature,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            out = model(
                left_x=batch["left_x"],
                right_x=batch["right_x"],
                left_mask=batch["left_mask"],
                right_mask=batch["right_mask"],
            )
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        history.append(avg_loss)
        print(
            f"[temp={temperature}, proj={projection_dim}] "
            f"epoch={epoch + 1}, loss={avg_loss:.4f}"
        )

    return history[-1]


def main():
    set_seed(42)

    settings = [
        {"temperature": 0.07, "projection_dim": 16},
        {"temperature": 0.10, "projection_dim": 16},
        {"temperature": 0.07, "projection_dim": 32},
    ]

    results = []
    for cfg in settings:
        final_loss = train_one_setting(
            temperature=cfg["temperature"],
            projection_dim=cfg["projection_dim"],
        )
        results.append((cfg, final_loss))

    print("\n=== Ablation Summary ===")
    for cfg, final_loss in results:
        print(
            f"temperature={cfg['temperature']}, "
            f"projection_dim={cfg['projection_dim']} -> "
            f"final_loss={final_loss:.4f}"
        )


if __name__ == "__main__":
    main()