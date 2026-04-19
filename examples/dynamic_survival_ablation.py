# examples/dynamic_survival_ablation.py

"""
Ablation Study: Effect of Observation Window Length

We vary observation window sizes (12, 24, 48 hours) and
measure performance using masked BCE and MSE.

This demonstrates how task configuration (NOT model complexity)
impacts predictive performance.
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from pyhealth.tasks.dynamic_survival import DynamicSurvivalTask
from synthetic_dataset import generate_synthetic_dataset


# ======================
# Mock EHR Classes (REQUIRED)
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
# Convert synthetic dict → MockPatient
# ======================

def convert_to_mock_patients(patients_dict):
    base_time = datetime(2025, 1, 1)

    mock_patients = []

    for p in patients_dict:
        visits_data = []

        for v in p["visits"]:
            visits_data.append({
                "time": base_time + timedelta(days=v["time"]),
                "diagnosis": ["0000"],  # dummy code for vocab
            })

        death_time = None
        if p.get("outcome_time") is not None:
            death_time = base_time + timedelta(days=p["outcome_time"])

        mock_patients.append(
            MockPatient(
                pid=p["patient_id"],
                visits_data=visits_data,
                death_time=death_time,
            )
        )

    return mock_patients


# ======================
# Model
# ======================

class SimpleModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=8, horizon=24):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        _, h = self.rnn(x)
        return torch.sigmoid(self.fc(h.squeeze(0)))


# ======================
# Utils
# ======================

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
        torch.tensor(np.array(X_pad), dtype=torch.float32),
        torch.tensor(np.array(Y), dtype=torch.float32),
        torch.tensor(np.array(M), dtype=torch.float32),
    )


def train_and_eval(samples):
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X, Y, M = prepare_batch(samples)

    for _ in range(5):
        pred = model(X)
        loss = -(Y * torch.log(pred + 1e-8) +
                 (1 - Y) * torch.log(1 - pred + 1e-8))
        loss = (loss * M).sum() / M.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = model(X)

        bce = -(Y * torch.log(pred + 1e-8) +
                (1 - Y) * torch.log(1 - pred + 1e-8))
        bce = (bce * M).sum() / M.sum()

        mse = ((pred - Y) ** 2 * M).sum() / M.sum()

    return {"bce": bce.item(), "mse": mse.item()}


# ======================
# Main Experiment
# ======================

patients_raw = generate_synthetic_dataset(50)
patients = convert_to_mock_patients(patients_raw)
dataset = MockDataset(patients)

windows = [12, 24, 48]
results = {}

print("\n=== Ablation Results ===")

for w in windows:
    task = DynamicSurvivalTask(
        dataset=dataset,
        observation_window=w,
        horizon=24,
    )

    samples = dataset.set_task(task)

    if len(samples) == 0:
        print(f"Skipping window={w}, no samples")
        continue

    score = train_and_eval(samples)
    results[w] = score

    print(f"Window={w} | BCE={score['bce']:.4f} | MSE={score['mse']:.4f}")
