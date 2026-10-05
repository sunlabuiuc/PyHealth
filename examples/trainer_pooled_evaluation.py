"""Demonstrates that trainer.evaluate() is batch-invariant and reproducible.

Context (https://github.com/sunlabuiuc/PyHealth/issues/859): metrics such as
AUROC and AUPRC are not decomposable over batches — computing them per batch
and averaging gives batch-size-dependent, non-reproducible numbers. PyHealth's
``Trainer.evaluate`` therefore pools predictions and labels across all batches
and computes each metric exactly once on the pooled arrays, and pools the loss
as an example-weighted mean.

This script evaluates the same fixed model on the same synthetic data with
different batch sizes and shuffle orders and shows the scores are identical.

Run with: python examples/trainer_pooled_evaluation.py
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pyhealth.trainer import Trainer


class SyntheticBinaryDataset(Dataset):
    def __init__(self, n_samples=100, n_features=4, seed=7):
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(n_samples, n_features)).astype("float32")
        self.weights = rng.normal(size=(n_features,)).astype("float32")
        noise = rng.normal(scale=2.0, size=n_samples).astype("float32")
        self.y = (self.x @ self.weights + noise > 0).astype("float32")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index]}


class DeterministicBinaryModel(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.mode = "binary"
        self.linear = nn.Linear(len(weights), 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(weights).reshape(1, -1))
            self.linear.bias.zero_()

    def forward(self, x, y, **kwargs):
        logits = self.linear(x).squeeze(-1)
        y_true = y.float()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_true)
        return {"loss": loss, "y_true": y_true, "y_prob": torch.sigmoid(logits)}


def main():
    dataset = SyntheticBinaryDataset()
    trainer = Trainer(
        model=DeterministicBinaryModel(dataset.weights),
        metrics=["roc_auc", "pr_auc", "f1"],
        device="cpu",
        enable_logging=False,
    )

    configs = {
        "batch_size=16, unshuffled": DataLoader(dataset, batch_size=16),
        "batch_size=64, unshuffled": DataLoader(dataset, batch_size=64),
        "batch_size=16, shuffled(seed=0)": DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            generator=torch.Generator().manual_seed(0),
        ),
        "batch_size=16, shuffled(seed=1)": DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            generator=torch.Generator().manual_seed(1),
        ),
    }

    results = {name: trainer.evaluate(loader) for name, loader in configs.items()}
    for name, scores in results.items():
        printable = {k: round(v, 6) for k, v in scores.items()}
        print(f"{name}: {printable}")

    reference = next(iter(results.values()))
    for scores in results.values():
        for key in reference:
            np.testing.assert_allclose(scores[key], reference[key], rtol=1e-6)
    print("All configurations produced identical pooled metrics.")


if __name__ == "__main__":
    main()
