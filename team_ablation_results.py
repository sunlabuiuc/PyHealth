
# team_ablation_results.py
"""
Dynamic Survival Analysis (DSA) Ablation Script

Run:
    python run_dsa.py

Requirements:
    pip install pyhealth torch numpy matplotlib pandas scikit-learn lifelines

Important:
    Change dataset path below before running.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import average_precision_score
from lifelines.utils import concordance_index
from pyhealth.tasks.dynamic_survival import DynamicSurvivalTask
from pyhealth.datasets import MIMIC3Dataset


# ======================
# Seed
# ======================
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


# ======================
# Model
# ======================
class GRUModel(nn.Module):
    def __init__(self, input_dim, horizon):
        super().__init__()
        self.rnn = nn.GRU(input_dim, 32, batch_first=True)
        self.fc = nn.Linear(32, horizon)

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


def train_model(samples, horizon, prior=None):
    X, Y, M = prepare_batch(samples)
    model = GRUModel(X.shape[-1], horizon)

    if prior is not None:
        bias = torch.log(torch.tensor(prior / (1 - prior)))
        with torch.no_grad():
            model.fc.bias.fill_(bias)

    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(5):
        pred = model(X)
        loss = -(Y * torch.log(pred + 1e-8) +
                 (1 - Y) * torch.log(1 - pred + 1e-8))
        loss = (loss * M).sum() / M.sum()

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


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
        times, risks = [], []

        for i in range(len(Y)):
            y_i = Y[i].cpu().numpy()
            pred_i = pred[i].cpu().numpy()

            idx = np.where(y_i > 0)[0]
            if len(idx) == 0:
                continue

            t = idx[0]
            r = pred_i[:t + 1].sum()

            times.append(t)
            risks.append(r)

        if len(times) < 2 or len(set(times)) < 2:
            cindex = None
        else:
            cindex = concordance_index(times, -np.array(risks))

    return bce.item(), auprc, cindex


# ======================
# Plot helper
# ======================
def plot_results(labels, bce, auprc, cindex, title):
    x = np.arange(len(labels))
    w = 0.25

    plt.figure(figsize=(6, 4))

    plt.bar(x - w, bce, width=w, label="BCE")
    plt.bar(x, [0 if v is None else v for v in auprc], width=w, label="AuPRC")
    plt.bar(x + w, [np.nan if v is None else v for v in cindex], width=w, label="C-index")

    plt.xticks(x, labels)
    plt.title(title)
    plt.legend()
    plt.grid(axis="y")
    plt.show()


# ======================
# Main
# ======================
def main():

    dataset = MIMIC3Dataset(
        root="D:/dl_health_project/data_zip",  # CHANGE THIS
        tables=["DIAGNOSES_ICD", "ADMISSIONS"],
        dev=True
    )

    # -------- Anchor --------
    
    # =========================
    # 1. Anchor Ablation (CORE)
    # =========================
    print("\n=== Anchor Ablation (MIMIC) ===")

    anchors = ["fixed", "single"]

    bce_list, auprc_list, cindex_list = [], [], []

    for anchor in anchors:
        task = DynamicSurvivalTask(
            dataset,
            horizon=10,
            observation_window=12,
            anchor_strategy=anchor
        )

        samples = dataset.set_task(task)

        model = train_model(samples, 10)
        bce, auprc, cidx = evaluate_3metrics(model, samples)

        print(anchor, "→", bce, auprc, cidx)

        bce_list.append(bce)
        auprc_list.append(auprc)
        cindex_list.append(cidx)

    df_anchor = pd.DataFrame({
        "Anchor": anchors,
        "BCE": bce_list,
        "AuPRC": auprc_list,
        "C-index": cindex_list
    })

    print("\n=== Anchor Results ===")
    print(df_anchor.round(4))

    plot_results(anchors, bce_list, auprc_list, cindex_list, "Anchor Ablation")


    # # -------- Window --------
    # print("\n=== Window Ablation ===")
    # windows = [6, 12, 24]

    # bce, auprc, cindex = [], [], []

    # for w in windows:
    #     task = DynamicSurvivalTask(dataset, horizon=10, observation_window=w, anchor_strategy="fixed")
    #     samples = dataset.set_task(task)

    #     model = train_model(samples, 10)
    #     b, ap, ci = evaluate_3metrics(model, samples)

    #     print(w, "→", b, ap, ci)

    #     bce.append(b)
    #     auprc.append(ap)
    #     cindex.append(ci)

    # plot_results(windows, bce, auprc, cindex, "Window Ablation")

    # # -------- Horizon --------
    # print("\n=== Horizon Ablation ===")
    # horizons = [5, 10, 20]

    # bce, auprc, cindex = [], [], []

    # for h in horizons:
    #     task = DynamicSurvivalTask(dataset, horizon=h, observation_window=12, anchor_strategy="fixed")
    #     samples = dataset.set_task(task)

    #     model = train_model(samples, h)
    #     b, ap, ci = evaluate_3metrics(model, samples)

    #     print(h, "→", b, ap, ci)

    #     bce.append(b)
    #     auprc.append(ap)
    #     cindex.append(ci)

    # plot_results(horizons, bce, auprc, cindex, "Horizon Ablation")


if __name__ == "__main__":
    main()