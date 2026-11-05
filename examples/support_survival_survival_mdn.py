# Contributor: Neil Hajela (nhajela2@illinois.edu)
"""Survival MDN paired transform and optional hidden-width ablation on SUPPORT.

Paper: https://proceedings.mlr.press/v182/han22a.html
Default: synthetic smoke demonstration, no downloads. Use --source support2
for PyHealth's existing Support2Dataset, or benchmark for recovered exact splits.
See survival_mdn_results/README.md for full commands and the recorded 100-run
study. Historical results used the standalone core; this uses native PyHealth.
Both transforms include the inverse Jacobian. Width 64 is an optional separate
hyperparameter demonstration, not part of the recorded 100-run study.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyhealth.models import SurvivalMDN
from survival_mdn_data import Bundle, Split, read_support2, split_frame, synthetic
from survival_mdn_metrics import paper_metrics, primary_metrics, tail_diagnostics


def tensors(part: Split, sort: bool = False) -> tuple[torch.Tensor, ...]:
    """Convert one subset to tensors, optionally ordered by duration."""
    order = np.argsort(part.duration) if sort else np.arange(len(part.duration))
    return tuple(
        torch.as_tensor(a[order], dtype=torch.float32)
        for a in (part.x, part.duration, part.event)
    )


def nll(
    model: SurvivalMDN, data: tuple[torch.Tensor, ...], selection: bool = False
) -> float:
    """Evaluate patient-weighted test NLL or historical validation selection loss."""
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(data[0]), 1024):
            x, t, e = [a[start : start + 1024] for a in data]
            params = model.network.mixture_parameters(x)
            values.append(model.network.censored_nll(params, t, e, reduction="none"))
    if selection:
        return float(np.mean([float(v.mean()) for v in values]))
    return float(torch.cat(values).double().mean())


def train(
    bundle: Bundle, transform: str, width: int, seed: int, args: argparse.Namespace
) -> tuple[dict, list[dict]]:
    """Train one arm with matched seeds, batches and stopping settings.

    Args:
        bundle: Disjoint training, validation and test data.
        transform: Softplus baseline or exponential intervention.
        width: Hidden layer width.
        seed: Paired model and minibatch seed.
        args: Optimizer, stopping and evaluation CLI settings.

    Returns:
        Best-checkpoint metrics and epoch history.
    """
    torch.manual_seed(seed)
    model = SurvivalMDN(
        bundle.train.dataset(),
        input_dim=27,
        hidden_dim=width,
        num_components=10,
        time_transform=transform,
        time_grid=torch.tensor([1.0]),
    )
    opt = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-6)
    x, t, e = tensors(bundle.train, sort=True)
    valid = tensors(bundle.valid, sort=True)
    rng = np.random.RandomState(12 + seed)
    batch = min(args.batch_size, len(x))
    best, best_epoch, state = math.inf, -1, None
    patience, history = args.patience, []
    bad_loss = bad_grad = clipped = 0
    max_grad = 0.0
    start_time = time.perf_counter()
    for epoch in range(args.epochs):
        patience -= 1
        model.train()
        order, losses = rng.permutation(len(x)), []
        for start in range(0, (len(x) // batch) * batch, batch):
            # Historical recipe drops the incomplete final minibatch.
            idx = np.sort(order[start : start + batch])
            opt.zero_grad(set_to_none=True)
            loss = model(features=x[idx], duration=t[idx], event=e[idx])["loss"]
            if not torch.isfinite(loss):
                bad_loss += 1
                continue
            loss.backward()
            if not all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                bad_grad += 1
                opt.zero_grad(set_to_none=True)
                continue
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0))
            max_grad = max(max_grad, grad)
            clipped += int(grad > 100)
            opt.step()
            losses.append(float(loss.detach()))
        val = nll(model, valid, selection=True)
        history.append(
            {
                "epoch": epoch + 1,
                "valid_loss": val,
                "train_loss": float(np.mean(losses)) if losses else None,
            }
        )
        if math.isfinite(val) and val < best:
            best, best_epoch, state = val, epoch + 1, copy.deepcopy(model.state_dict())
            patience = args.patience
        if patience <= 0:
            break
    if state is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(state)
    elapsed = time.perf_counter() - start_time
    row = {
        "seed": seed,
        "transform": transform,
        "hidden_dim": width,
        "failed": False,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_valid_nll": best,
        "test_nll": nll(model, tensors(bundle.test)),
        "seconds": elapsed,
        "nonfinite_loss_batches": bad_loss,
        "nonfinite_grad_batches": bad_grad,
        "clipped_batches": clipped,
        "max_grad_norm": max_grad,
    }
    if args.metrics != "nll":
        evaluate = paper_metrics if args.metrics == "all" else primary_metrics
        row.update(evaluate(model, bundle.test))
        row.update(tail_diagnostics(model, bundle.test, bundle.train))
    return row, history


def main() -> None:
    """Run requested ablations, print comparisons and preserve new results."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source", choices=["synthetic", "benchmark", "support2"], default="synthetic"
    )
    p.add_argument("--data", type=Path, help="Encoded or raw SUPPORT CSV")
    p.add_argument("--membership", type=Path)
    p.add_argument("--splits", type=int, nargs="+", default=[1])
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--samples", type=int, default=160)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--metrics", choices=["nll", "primary", "all"], default="nll")
    p.add_argument("--vary-hidden", action="store_true")
    p.add_argument("--output", type=Path, default=Path("survival_mdn_run"))
    args = p.parse_args()
    if min(args.epochs, args.patience, args.threads) < 1 or args.batch_size < 2:
        p.error("epochs/patience/threads must be positive; batch size >= 2")
    if not set(args.splits) <= set(range(1, 11)):
        p.error("splits must be from 1 to 10")
    if len(set(args.splits)) != len(args.splits) or len(set(args.seeds)) != len(
        args.seeds
    ):
        p.error("splits and seeds must not contain duplicates")
    if args.source != "synthetic" and args.data is None:
        p.error("--data is required for real data")
    if args.source == "benchmark" and args.membership is None:
        p.error("benchmark requires --membership")
    if (args.output / "all_runs.csv").exists():
        p.error("Choose a fresh --output directory to preserve earlier results")
    torch.set_num_threads(args.threads)
    frame = None
    if args.source == "benchmark":
        frame = pd.read_csv(args.data)
    elif args.source == "support2":
        frame = read_support2(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )
    if args.source == "support2":
        frame.to_csv(args.output / "prepared_support.csv", index=False)
    configurations = [("softplus", 32), ("exp", 32)]
    if args.vary_hidden:
        configurations.append(("softplus", 64))
    rows = []
    for split in args.splits:
        bundle = (
            synthetic(args.samples, split)
            if frame is None
            else split_frame(frame, args.membership, split, split)
        )
        for seed in args.seeds:
            for transform, width in configurations:
                row, history = train(bundle, transform, width, seed, args)
                row["split"] = split
                rows.append(row)
                pd.DataFrame(rows).to_csv(args.output / "all_runs.csv", index=False)
                name = f"{transform}_width{width}_split{split}_seed{seed}.json"
                (args.output / name).write_text(json.dumps(history, indent=2))
                print(json.dumps(row), flush=True)
    print(
        pd.DataFrame(rows)[
            ["transform", "hidden_dim", "test_nll", "best_valid_nll"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
