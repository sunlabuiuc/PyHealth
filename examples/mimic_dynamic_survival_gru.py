"""
Ablation Study for Dynamic Survival Task

We evaluate:
- Anchor strategy (fixed vs single)
- Observation window size
- Prediction horizon

We use a GRU model on synthetic patients.

Findings:
- Anchor strategy significantly impacts performance
- Larger windows provide more context
- Prediction horizon affects difficulty

Results are printed to show how task configurations affect model performance.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from datetime import datetime, timedelta

from pyhealth.tasks.dynamic_survival import DynamicSurvivalTask

# ===== Seed (reproducibility) =====
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# ======================
# Mock MIMIC-style Dataset
# ======================

class MockEvent:
    def __init__(self, code, timestamp, vocabulary):
        self.code = code
        self.timestamp = timestamp
        self.vocabulary = vocabulary


class MockVisit:
    def __init__(self, time, diagnosis=None):
        self.encounter_time = time
        self.event_list_dict = {
            "DIAGNOSES_ICD": [
                MockEvent(c, time, "ICD9CM") for c in (diagnosis or [])
            ],
            "PROCEDURES_ICD": [],
            "PRESCRIPTIONS": [],
        }


class MockPatient:
    def __init__(self, pid, visits_data, death_time=None):
        self.patient_id = pid
        self.visits = {
            f"v{i}": MockVisit(**v) for i, v in enumerate(visits_data)
        }
        self.death_datetime = death_time


class MockDataset:
    def __init__(self, patients):
        self.patients = {p.patient_id: p for p in patients}

    def set_task(self, task):
        samples = []
        for p in self.patients.values():
            out = task(p)
            if out:
                samples.extend(out)
        return samples


# ======================
# Synthetic Patient Generator
# ======================

def generate_synthetic_patients(n=20, seed=42):
    random.seed(seed)
    base_time = datetime(2025, 4, 1)

    patients = []

    for i in range(n):

        num_visits = random.randint(5, 10)
        visit_times = sorted(random.sample(range(1, 40), num_visits))

        visits_data = [
            {
                "time": base_time + timedelta(days=t),
                "diagnosis": [str(random.randint(1000, 9999))]
            }
            for t in visit_times
        ]

        if random.random() < 0.5:
            death_time = base_time + timedelta(
                days=max(visit_times) + random.randint(5, 15)
            )
        else:
            death_time = None

        patients.append(
            MockPatient(
                pid=f"P{i}",
                visits_data=visits_data,
                death_time=death_time,
            )
        )

    return patients


# ======================
# Model
# ======================

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, horizon=24):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        _, h = self.rnn(x)
        return torch.sigmoid(self.fc(h.squeeze(0)))


# ======================
# Batch Function
# ======================

def prepare_batch(samples):
    X, Y, M = [], [], []

    for s in samples:
        X.append(s["x"])
        Y.append(s["y"])
        M.append(s["mask"])

    max_len = max(len(x) for x in X)

    X_pad = []
    for x in X:
        pad = np.zeros((max_len - len(x), x.shape[1]))
        X_pad.append(np.vstack([x, pad]))

    return (
        torch.tensor(np.array(X_pad), dtype=torch.float32),
        torch.tensor(np.array(Y), dtype=torch.float32),
        torch.tensor(np.array(M), dtype=torch.float32),
    )


# ======================
# Training Loop
# ======================

def train_model(samples, horizon, prior=None):

    X, Y, M = prepare_batch(samples)

    model = GRUModel(input_dim=X.shape[-1], horizon=horizon)

    # 🔥 Bayesian initialization
    if prior is not None:
        p = prior
        bias_init = torch.log(torch.tensor(p / (1 - p)))

        with torch.no_grad():
            model.fc.bias.fill_(bias_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5):
        pred = model(X)

        loss = -(Y * torch.log(pred + 1e-8) +
                 (1 - Y) * torch.log(1 - pred + 1e-8))
        loss = (loss * M).sum() / M.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


# ======================
# Evaluation
# ======================

def evaluate(model, samples):

    X, Y, M = prepare_batch(samples)

    with torch.no_grad():
        pred = model(X)

        bce = -(Y * torch.log(pred + 1e-8) +
                (1 - Y) * torch.log(1 - pred + 1e-8))
        bce = (bce * M).sum() / M.sum()

        mse = ((pred - Y) ** 2 * M).sum() / M.sum()

    print(f"\nFinal Performance → BCE={bce.item():.4f} | MSE={mse.item():.4f}")


# ======================
# DATASET 
# ======================

patients = generate_synthetic_patients(20)
dataset = MockDataset(patients)


# ======================
# RUN EXPERIMENT FUNCTION
# ======================

def run_experiment(dataset, horizon, window, anchor):

    task = DynamicSurvivalTask(
        dataset,
        horizon=horizon,
        observation_window=window,
        anchor_strategy=anchor
    )

    samples = dataset.set_task(task)

    if len(samples) == 0:
        return None

    model = train_model(samples, horizon=horizon)

    X, Y, M = prepare_batch(samples)

    with torch.no_grad():
        pred = model(X)

        bce = -(Y * torch.log(pred + 1e-8) +
                (1 - Y) * torch.log(1 - pred + 1e-8))
        bce = (bce * M).sum() / M.sum()

        mse = ((pred - Y) ** 2 * M).sum() / M.sum()

    return bce.item(), mse.item()


# ======================
# 1. Anchor Ablation
# ======================
print("\n=== Anchor Ablation ===")

for anchor in ["fixed", "single"]:
    result = run_experiment(dataset, horizon=10, window=12, anchor=anchor)

    if result is None:
        continue

    bce, mse = result

    print(f"{anchor} → BCE={bce:.4f} | MSE={mse:.4f}")


# ======================
# 2. Window Ablation
# ======================
print("\n=== Window Ablation ===")

for w in [6, 12, 24]:
    result = run_experiment(dataset, horizon=10, window=w, anchor="fixed")

    if result is None:
        continue

    bce, mse = result

    print(f"window={w} → BCE={bce:.4f} | MSE={mse:.4f}")

# ======================
# 3. Horizon Ablation
# ======================
print("\n=== Horizon Ablation ===")

for h in [5, 10, 20]:
    result = run_experiment(dataset, horizon=h, window=12, anchor="fixed")

    if result is None:
        continue

    bce, mse = result

    print(f"horizon={h} → BCE={bce:.4f} | MSE={mse:.4f}")
