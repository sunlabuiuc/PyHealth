# examples/dynamic_survival_ablation.py

"""
Ablation Study: Effect of Observation Window Length

We vary observation window sizes (12, 24, 48 hours) and
measure performance using masked MSE.

Expected Behavior:
- Larger observation windows should improve performance
  due to more historical context.
- Results may vary due to randomness in synthetic data.

This experiment demonstrates how task design (NOT model complexity)
impacts predictive performance.
"""

import numpy as np
import torch
import torch.nn as nn
from pyhealth.tasks.dynamic_survival import DynamicSurvivalTask
from synthetic_dataset import generate_synthetic_dataset


# Simple lightweight model
class SimpleModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, horizon=24):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        _, h = self.rnn(x)
        out = self.fc(h.squeeze(0))
        return torch.sigmoid(out)


def prepare_batch(samples):
    X, Y, M = [], [], []

    for s in samples:
        X.append(s["x"])
        Y.append(s["y"])
        M.append(s["mask"])

    if len(X) == 0:
        raise ValueError("No valid samples generated.")

    max_len = max(len(x) for x in X)

    X_pad = []
    for x in X:
        pad = np.zeros((max_len - len(x), x.shape[1]))
        X_pad.append(np.vstack([x, pad]))

    return (
        torch.tensor(X_pad, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
        torch.tensor(M, dtype=torch.float32),
    )


def train_and_eval(samples):
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X, Y, M = prepare_batch(samples)

    for _ in range(5):  # VERY small training
        pred = model(X)
        loss = -(Y * torch.log(pred + 1e-8) + (1 - Y) * torch.log(1 - pred + 1e-8))
        loss = (loss * M).sum() / M.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = model(X)

        bce = -(Y * torch.log(pred + 1e-8) + (1 - Y) * torch.log(1 - pred + 1e-8))
        bce = (bce * M).sum() / M.sum()

        mse = ((pred - Y) ** 2 * M).sum() / M.sum()

    return {
        "bce": bce.item(),
        "mse": mse.item(),
    }


patients = generate_synthetic_dataset(50)

windows = [12, 24, 48]
results = {}

for w in windows:
    task = DynamicSurvivalTask(observation_window=w, horizon=24)

    samples = []
    for p in patients:
        samples.extend(task(p))

    score = train_and_eval(samples)
    results[w] = score

print("\n=== Ablation Results ===")
for w, score in results.items():
    print(f"Window={w} | BCE={score['bce']:.4f} | MSE={score['mse']:.4f}")
