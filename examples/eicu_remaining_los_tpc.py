"""Try TPC with different kernel sizes and dilation rates.

Contributor: Audrey Labrie.
Paper: Temporal Pointwise Convolutional Networks for Length of Stay Prediction
in the Intensive Care Unit (Rocheteau et al., 2021).
Paper link: https://arxiv.org/abs/2007.09483
Reference code: https://github.com/EmmaRocheteau/TPC-LoS-prediction

Run: python examples/eicu_remaining_los_tpc.py
This uses made-up data on the CPU. Use --data-dir for prepared training and
validation data, and --ablation to try five settings. Data runs use two epochs
unless --epochs is set. See eicu_remaining_los_tpc.md for commands and results.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pyhealth.models.tpc import TPC
from tpc_los_support import (
    IndexedDataset, PreparedICUStays, as_pyhealth, collate_stays,
)


REFERENCE_DILATIONS = (1, 3, 6, 9, 12, 15, 18, 21, 24)
PLANS = (
    ("tpc_reference", 4, REFERENCE_DILATIONS),
    ("tpc_kernel2", 2, REFERENCE_DILATIONS),
    ("tpc_kernel3", 3, REFERENCE_DILATIONS),
    ("tpc_no_dilation", 4, (1,) * 9),
    ("tpc_gradual_dilation", 4, tuple(range(1, 10))),
)


def seed_everything(seed: int) -> None:
    """Set the random seed.

    Args:
        seed: Number used to repeat random choices.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synthetic_example() -> None:
    """Check predictions and gradients using two made-up stays."""
    seed_everything(9)
    samples = []
    for index, length in enumerate((8, 7)):
        hours = torch.arange(1, length + 1, dtype=torch.float32)
        samples.append({
            "patient_id": f"synthetic-person-{index}",
            "record_id": f"synthetic-stay-{index}",
            "x": torch.stack((
                hours / 24, 1 + hours / 10, (hours + index) / 10,
                torch.ones_like(hours), torch.ones_like(hours), hours / 23,
            )),
            "static": torch.tensor([0.2, float(index)]),
            "diagnoses": torch.tensor([1.0, 0.0, float(index)]),
            "targets": (2 + index - hours / 24).clamp_min(1 / 48),
            "valid_mask": hours > 5,
        })
    dataset = as_pyhealth(samples)
    model = TPC(
        dataset=dataset, features=2, static_dim=2, diagnosis_dim=3,
        kernel_size=2, dilation_rates=(1, 2), temporal_channels=2,
        point_channels=2, diagnosis_hidden=4, final_hidden=4,
        main_dropout=0.0, temporal_dropout=0.0,
    )
    batch = collate_stays([dataset[i] for i in range(len(dataset))])
    output = model(**batch)
    output["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if (output["predictions"].shape != (2, 8)
            or output["y_prob"].shape != (5, 1)
            or not all(bool(torch.isfinite(g).all()) for g in gradients)
            or not any(bool((g != 0).any()) for g in gradients)):
        raise RuntimeError("Synthetic output or gradient check failed")
    print("Synthetic forward/backward passed: 2 stays, 5 valid hours.")
    print(f"Loss: {output['loss'].item():.6f}; data are entirely synthetic.")


def phase(
    model: TPC, loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float | int]:
    """Run one epoch and measure error.

    Args:
        model: Model to run.
        loader: Training or validation batches.
        optimizer: Optimizer for training; None to skip training.

    Returns:
        MSLE, MAD in days, and counts of scored hours and stays.
    """
    training = optimizer is not None
    model.train(training)
    log_sum, absolute_sum, count, stays = 0.0, 0.0, 0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            batch = {key: value.to(model.device) for key, value in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            if not bool(torch.isfinite(output["loss"])):
                raise RuntimeError("Nonfinite loss")
            y = output["y_true"].detach().to("cpu", torch.float64).reshape(-1)
            p = output["y_prob"].detach().to("cpu", torch.float64).reshape(-1)
            if len(y) != int(batch["valid_mask"].sum()) or len(p) != len(y):
                raise RuntimeError("Predictions disagree with the validity mask")
            if training:
                output["loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                optimizer.step()
            log_sum += (p.log() - y.log()).square().sum().item()
            absolute_sum += (p - y).abs().sum().item()
            count += len(y)
            stays += batch["x"].shape[0]
    if not count or stays != len(loader.dataset):
        raise RuntimeError("The epoch did not cover all stays and valid hours")
    return {"msle": log_sum / count, "mad_days": absolute_sum / count,
            "prediction_hours": count, "stays": stays}


def run_one(
    args: argparse.Namespace, train: Any, val: Any,
    dimensions: dict[str, int], plan: tuple, seed: int,
) -> dict[str, Any]:
    """Train one model and save the epoch with the lowest validation MSLE.

    Args:
        args: Command-line options.
        train: PyHealth training dataset.
        val: PyHealth validation dataset.
        dimensions: Counts of measurement, static, and diagnosis features.
        plan: Model name, kernel size, and dilation rates.
        seed: Number used to repeat random choices.

    Returns:
        Model name, seed, best epoch, validation MSLE, and validation MAD.
    """
    name, kernel, dilations = plan
    folder = args.output_dir / f"{name}_seed_{seed}"
    folder.mkdir()
    seed_everything(seed)
    kwargs = {
        "kernel_size": kernel, "dilation_rates": list(dilations),
        "temporal_channels": 12, "point_channels": 13, "diagnosis_hidden": 64,
        "final_hidden": 17, "main_dropout": 0.45, "temporal_dropout": 0.05,
    }
    config = {"name": name, "seed": seed, "dimensions": dimensions,
              "model_kwargs": kwargs, "epochs": args.epochs,
              "patience": args.patience, "batch_size": 8, "learning_rate": 0.001,
              "normalization": "none", "gradient_clip": 5.0}
    (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model = TPC(dataset=train, **dimensions, **kwargs).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    common = {"batch_size": 8, "collate_fn": collate_stays, "num_workers": 0,
              "pin_memory": args.device.type == "cuda", "drop_last": False}
    train_loader = DataLoader(
        IndexedDataset(train), shuffle=True,
        generator=torch.Generator().manual_seed(seed), **common,
    )
    val_loader = DataLoader(
        IndexedDataset(val), shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1000), **common,
    )
    history, best, bad_epochs, selected = [], float("inf"), 0, {}
    for epoch in range(1, args.epochs + 1):
        training = phase(model, train_loader, optimizer)
        validation = phase(model, val_loader)
        history.append({"epoch": epoch,
                        **{f"train_{k}": v for k, v in training.items()},
                        **{f"val_{k}": v for k, v in validation.items()}})
        if validation["msle"] < best:
            best, bad_epochs = validation["msle"], 0
            selected = {"name": name, "seed": seed, "best_epoch": epoch,
                        "val_msle": best, "val_mad_days": validation["mad_days"]}
            torch.save({"config": config, "epoch": epoch,
                        "model_state_dict": {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()}},
                       folder / "best.pt")
        else:
            bad_epochs += 1
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(f"{name} seed {seed}, epoch {epoch}: "
              f"validation MSLE={validation['msle']:.4f}, best={best:.4f}",
              flush=True)
        if bad_epochs >= args.patience:
            break
    return selected


def main() -> None:
    """Read the options and run the example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[9])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.data_dir is None:
        if args.ablation or args.output_dir is not None:
            parser.error("Ablation/output options require --data-dir")
        synthetic_example()
        return
    if args.synthetic:
        parser.error("Use --synthetic or --data-dir, not both")
    if args.output_dir is None or min(args.epochs, args.patience) < 1:
        parser.error("Supply --output-dir and positive epochs/patience")
    if args.output_dir.exists():
        parser.error("Choose a new --output-dir; existing paths are preserved")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("Seeds must be distinct nonnegative integers")
    args.device = torch.device(args.device)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    prepared = {s: PreparedICUStays(args.data_dir, s) for s in ("train", "val")}
    if prepared["train"].people & prepared["val"].people:
        raise ValueError("Training and validation people overlap")
    dimensions = {k: getattr(prepared["train"], k)
                  for k in ("features", "static_dim", "diagnosis_dim")}
    if any(getattr(prepared["val"], k) != v for k, v in dimensions.items()):
        raise ValueError("Training and validation dimensions differ")
    train = as_pyhealth([prepared["train"][i] for i in range(len(prepared["train"]))])
    val = as_pyhealth([prepared["val"][i] for i in range(len(prepared["val"]))], train)
    del prepared
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for plan in PLANS if args.ablation else PLANS[:1]:
        for seed in args.seeds:
            results.append(run_one(args, train, val, dimensions, plan, seed))
            pd.DataFrame(results).to_csv(
                args.output_dir / "validation.csv", index=False
            )
    summary = pd.DataFrame(results).groupby("name")[
        ["val_msle", "val_mad_days"]
    ].agg(["mean", "std"])
    summary.to_csv(args.output_dir / "validation_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
