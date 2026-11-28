"""Biosignal forecasting on PTB-XL and Sleep-EDF: baselines, APN and MIRA.

The MIRA paper (Li et al., NeurIPS 2025, https://arxiv.org/abs/2506.07584)
reports zero-shot forecasting on held-out parts of PTB-XL and Sleep-EDF
(Table 3). This script builds those forecasting tasks with
:class:`~pyhealth.tasks.PTBXLForecasting` and
:class:`~pyhealth.tasks.SleepEDFForecasting` for the (context, horizon) pairs
of ``model_eval.py`` in the MIRA repository, and scores, on the test split:

- last value carried forward, and the record mean (0 in z-score units);
- APN (Liu et al., AAAI 2026), trained on the training split;
- MIRA zero-shot, when ``--mira`` gives a local checkpoint snapshot (the
  Hugging Face copy is no longer public, see the MIRA docs page).

Metrics are in z-score units: MSE, RMSE and MAE over all targets (pooled),
and RMSE and MAE per (window, channel) series, then averaged, as in
``model_eval.py``. PTB-XL uses the recommended folds (1-8 train, 9
validation, 10 test); Sleep-EDF splits by subject (80/10/10).

Usage:
    python examples/mira_signal_forecasting.py --dataset ptbxl --root /data/ptb-xl/1.0.3
    python examples/mira_signal_forecasting.py --dataset sleepedf \\
        --root /data/sleep-edfx/1.0.0 --mira /path/to/mira-snapshot
"""

import argparse
import copy
import json
import time

import numpy as np
import torch

from pyhealth.datasets import (
    PTBXLDataset,
    SleepEDFDataset,
    get_dataloader,
    split_by_patient,
    split_by_strat_fold,
)
from pyhealth.models import APN, MIRA
from pyhealth.tasks import PTBXLForecasting, SleepEDFForecasting

SETTINGS = [(48, 24), (72, 36), (96, 48), (128, 64)]  # model_eval.py in MIRA


def score(predict, loader, device):
    """Pooled and per-series errors of ``predict`` over the observed targets."""
    se = ae = n = 0.0
    series_rmse, series_mae = [], []
    for batch in loader:
        y = batch["y"].to(device)
        with torch.no_grad():
            pred = predict(batch).to(device)
        mask = torch.isfinite(y)
        d = torch.where(mask, pred - torch.nan_to_num(y), 0.0)
        se += float((d**2).sum())
        ae += float(d.abs().sum())
        n += int(mask.sum())
        count = mask.sum(dim=1)
        has = count > 0
        series_rmse.append(((d**2).sum(1)[has] / count[has]).sqrt().cpu())
        series_mae.append((d.abs().sum(1)[has] / count[has]).cpu())
    return {
        "MSE": se / n, "RMSE": (se / n) ** 0.5, "MAE": ae / n,
        "series RMSE": float(torch.cat(series_rmse).mean()),
        "series MAE": float(torch.cat(series_mae).mean()),
    }


def last_value(batch):
    x = batch["x"]
    last = torch.zeros(x.shape[0], x.shape[2])
    for t in range(x.shape[1]):
        last = torch.where(torch.isfinite(x[:, t]), x[:, t], last)
    return last.unsqueeze(1).expand(-1, batch["y"].shape[1], -1)


def record_mean(batch):
    return torch.zeros_like(batch["y"])


def train_apn(train, val, seed, device, max_epochs, patience=5):
    """APN with the optimizer of its paper, early-stopped on validation MSE."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Start the patches on the context span (x_time is in seconds).
    history = float(train[0]["x_time"][-1])
    model = APN(
        train, hidden_dim=24, num_patches=8, time_dim=8, dropout=0.1, history=history
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loader = get_dataloader(train, batch_size=64, shuffle=True)
    val_loader = get_dataloader(val, batch_size=256)
    best, best_state, wait = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            model(**batch)["loss"].backward()
            opt.step()
        model.eval()
        mse = score(lambda b: model(**b)["forecast"], val_loader, device)["MSE"]
        if mse < best:
            best, best_state, wait = mse, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", choices=["ptbxl", "sleepedf"], required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--dev", action="store_true", help="PTB-XL: load 1,000 patients")
    p.add_argument("--settings", default="all",
                   help="'all' or context:horizon pairs, e.g. 96:48,128:64")
    p.add_argument("--windows", type=int, default=None,
                   help="windows per record (default: 1 for PTB-XL, 100 for Sleep-EDF)")
    p.add_argument("--mira", default=None, help="local MIRA checkpoint snapshot")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--apn-epochs", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default=None, help="write the results as JSON")
    args = p.parse_args()
    settings = SETTINGS if args.settings == "all" else [
        tuple(int(v) for v in pair.split(":")) for pair in args.settings.split(",")
    ]

    if args.dataset == "ptbxl":
        dataset = PTBXLDataset(root=args.root, cache_dir=args.cache_dir, dev=args.dev)
    else:
        dataset = SleepEDFDataset(root=args.root)

    results = {}
    for context, horizon in settings:
        started = time.time()
        if args.dataset == "ptbxl":
            samples = dataset.set_task(
                PTBXLForecasting(context, horizon, num_windows=args.windows or 1)
            )
            train, val, test = split_by_strat_fold(samples)
        else:
            samples = dataset.set_task(
                SleepEDFForecasting(context, horizon, num_windows=args.windows or 100)
            )
            train, val, test = split_by_patient(samples, [0.8, 0.1, 0.1], seed=0)
        loader = get_dataloader(test, batch_size=64)
        rows = {
            "last value": score(last_value, loader, "cpu"),
            "record mean": score(record_mean, loader, "cpu"),
        }
        runs = []
        for seed in args.seeds:
            apn = train_apn(train, val, seed, args.device, args.apn_epochs)
            runs.append(score(lambda b, m=apn: m(**b)["forecast"], loader, args.device))
        rows[f"APN ({len(runs)} seed{'s' * (len(runs) > 1)})"] = {
            k: float(np.mean([r[k] for r in runs])) for k in runs[0]
        }
        if args.mira:
            mira = MIRA(test, pretrained=args.mira).to(args.device).eval()
            rows["MIRA zero-shot"] = score(
                lambda b, m=mira: m(**b)["forecast"], loader, args.device
            )
        key = f"{context}->{horizon}"
        results[key] = rows
        print(f"\n{args.dataset} {key}: {len(train)} train, {len(val)} val, "
              f"{len(test)} test windows ({time.time() - started:.0f} s)")
        for name, row in rows.items():
            print(f"  {name:<16}" + "  ".join(f"{k} {v:.4f}" for k, v in row.items()))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
