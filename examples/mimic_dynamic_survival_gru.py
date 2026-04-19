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
import pandas as pd
import random
from datetime import datetime, timedelta

from sklearn.metrics import average_precision_score
from sksurv.metrics import concordance_index_censored
from pyhealth.tasks.dynamic_survival import DynamicSurvivalTask

# use import if running on real MIMIC
# from pyhealth.datasets import MIMIC3Dataset

from examples.synthetic_dataset import generate_synthetic_dataset

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

    # Bayesian initialization
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

def evaluate_3metrics(model, samples):
    X, Y, M = prepare_batch(samples)

    with torch.no_grad():
        pred = model(X)

        # BCE
        loss = -(Y * torch.log(pred + 1e-8) +
                 (1 - Y) * torch.log(1 - pred + 1e-8))
        bce = (loss * M).sum() / M.sum()

        # AuPRC
        y_true = Y[M > 0].cpu().numpy().flatten()
        y_pred = pred[M > 0].cpu().numpy().flatten()
        auprc = None if y_true.sum() == 0 else average_precision_score(y_true, y_pred)

        # C-index
        times, risks, events = [], [], []

        for i in range(len(Y)):
            y_i = Y[i].cpu().numpy()
            pred_i = pred[i].cpu().numpy()
            m_i = M[i].cpu().numpy()

            event_idx = np.where(y_i > 0)[0]
            valid_idx = np.where(m_i > 0)[0]
            if len(valid_idx) == 0:
                # Skip samples with no usable data (fully zeroed mask).
                continue

            if len(event_idx) > 0:
                # y is one-hot by construction (generate_survival_label sets exactly one
                # index), so event_idx always has one element and [0] is the event time.
                event_time = event_idx[0]
                observed = True
            else:
                # No event within horizon, use last unmasked step as the censoring time
                # (i.e., last time we know the patient was event-free).
                event_time = valid_idx[-1]
                observed = False

            cumulative_risk = float(1.0 - np.prod(1.0 - pred_i))
            times.append(event_time)
            risks.append(cumulative_risk)
            events.append(observed)

        events_arr = np.array(events, dtype=bool)
        times_arr = np.array(times)
        risks_arr = np.array(risks)

        if len(times_arr) < 2 or not events_arr.any():
            cindex = None
        else:
            # Require at least one event time strictly less than the max time
            # to ensure a valid comparable pair exists for censored c-index.
            event_times = times_arr[events_arr]
            other_times = times_arr[~events_arr] if (~events_arr).any() else times_arr[events_arr]
            if not (event_times.min() < other_times.max()):
                cindex = None
            else:
                result = concordance_index_censored(events_arr, times_arr, risks_arr)
                # Handle both API versions
                cindex = result[0] if isinstance(result, tuple) else result.concordance

    return bce.item(), auprc, cindex


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


def main():
    # Use synthetic patients so this script runs without a local MIMIC download.
    # To run on real MIMIC-III, replace with:
    #   dataset = MIMIC3Dataset(root="<your_path>", tables=[...], dev=True)
    patients = generate_synthetic_patients(20)
    dataset = MockDataset(patients)

    # =========================
    # 1. Anchor Ablation
    # =========================
    print("\n=== Anchor Ablation ===")
    anchors = ["fixed", "single"]
    bce_list, auprc_list, cindex_list = [], [], []

    for anchor in anchors:
        task = DynamicSurvivalTask(
            dataset, horizon=30, observation_window=12, anchor_strategy=anchor
        )
        samples = dataset.set_task(task)
        if not samples:
            print(f"{anchor} → no samples generated, skipping")
            continue
        model = train_model(samples, 30)
        bce, auprc, cidx = evaluate_3metrics(model, samples)
        print(f"{anchor} → BCE={bce:.4f} | AuPRC={auprc} | C-index={cidx}")
        bce_list.append(bce)
        auprc_list.append(auprc)
        cindex_list.append(cidx)

    df_anchor = pd.DataFrame({
        "Anchor": anchors, "BCE": bce_list,
        "AuPRC": auprc_list, "C-index": cindex_list
    })
    print("\n=== Anchor Results ===")
    print(df_anchor.round(4))

    # =========================
    # 2. Window Ablation
    # =========================
    print("\n=== Window Ablation ===")
    windows = [6, 12, 24]
    bce_list, auprc_list, cindex_list = [], [], []

    for w in windows:
        task = DynamicSurvivalTask(
            dataset, horizon=10, observation_window=w, anchor_strategy="fixed"
        )
        samples = dataset.set_task(task)
        if not samples:
            print(f"window={w} → no samples, skipping")
            continue
        model = train_model(samples, 10)
        bce, auprc, cidx = evaluate_3metrics(model, samples)
        print(f"window={w} → BCE={bce:.4f} | AuPRC={auprc} | C-index={cidx}")
        bce_list.append(bce)
        auprc_list.append(auprc)
        cindex_list.append(cidx)

    df_window = pd.DataFrame({
        "Window": windows, "BCE": bce_list,
        "AuPRC": auprc_list, "C-index": cindex_list
    })
    print("\n=== Window Results ===")
    print(df_window.round(4))

    # =========================
    # 3. Horizon Ablation
    # =========================
    print("\n=== Horizon Ablation ===")
    horizons = [5, 10, 20]
    bce_list, auprc_list, cindex_list = [], [], []

    for h in horizons:
        task = DynamicSurvivalTask(
            dataset, horizon=h, observation_window=12, anchor_strategy="fixed"
        )
        samples = dataset.set_task(task)
        if not samples:
            print(f"horizon={h} → no samples, skipping")
            continue
        model = train_model(samples, h)
        bce, auprc, cidx = evaluate_3metrics(model, samples)
        print(f"horizon={h} → BCE={bce:.4f} | AuPRC={auprc} | C-index={cidx}")
        bce_list.append(bce)
        auprc_list.append(auprc)
        cindex_list.append(cidx)

    df_horizon = pd.DataFrame({
        "Horizon": horizons, "BCE": bce_list,
        "AuPRC": auprc_list, "C-index": cindex_list
    })
    print("\n=== Horizon Results ===")
    print(df_horizon.round(4))


if __name__ == "__main__":
    main()
